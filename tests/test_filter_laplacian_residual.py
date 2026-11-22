"""The phase output adds its Laplacian input after the single-channel head."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from dnqsm.model import DNQSM
from dnqsm.overfit import OverfitConfig
from dnqsm.physics import masked_rms
from dnqsm.runtime import load_trained_model
from dnqsm.training import checkpoint_training_config, model_config, parse_args
from tests.helpers import laplacian_model_config


class FilterLaplacianResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **overrides):
        return laplacian_model_config(
            (2, 4),
            filter_output_skip=False,
            filter_laplacian_residual=True,
            **overrides,
        )

    def test_sum_after_head_without_concatenation_preserves_shape_and_gradients(self):
        model = DNQSM(self.config(conditional_laplacian_threshold=0))
        field = torch.randn(1, 1, 5, 7, 9, requires_grad=True)
        mask = torch.zeros_like(field)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        inputs, raw, decoded, head = [], [], [], []
        handles = [
            model.filter.register_forward_pre_hook(lambda _, args: inputs.append(args)),
            model.filter.register_forward_hook(lambda _, args, out: raw.append(out)),
            model.filter.up_blocks[-1].register_forward_hook(
                lambda _, args, out: decoded.append(out)
            ),
            model.filter.final.register_forward_pre_hook(
                lambda _, args: head.append(args[0])
            ),
        ]
        try:
            local = model.predict_local_field(field, mask)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(inputs[0]), 1)
        self.assertEqual(model.filter.output_skip_channels, 0)
        self.assertEqual(model.filter.final.in_channels, 2)
        self.assertEqual(model.null_network.final.in_channels, 2)
        self.assertEqual(local.shape, field.shape)
        torch.testing.assert_close(head[0], decoded[0], rtol=0, atol=0)
        scale = masked_rms(field * mask, mask)
        laplacian = F.conv3d(field * mask / scale, model.laplacian_kernel, padding=1)
        expected = (laplacian + raw[0].float()) * scale * mask
        torch.testing.assert_close(local, expected, rtol=0, atol=0)
        local.square().mean().backward()
        self.assertGreater(float(model.filter.final.weight.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(field.grad).all()))
        self.assertGreater(float(field.grad.abs().sum()), 0)
        self.assertEqual(float(field.grad[mask == 0].abs().sum()), 0)

    def test_zero_raw_output_returns_selected_laplacian_in_fp32_under_bf16(self):
        model = DNQSM(self.config()).eval()
        field = torch.randn(2, 1, 12, 14, 16) * 0.01
        field[1] += 100
        mask = torch.zeros_like(field)
        mask[..., 2:-2, 2:-2, 2:-2] = 1
        with torch.no_grad():
            model.filter.final.weight.zero_()
            model.filter.final.bias.zero_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            local = model.predict_local_field(field, mask)
        self.assertEqual(local.dtype, torch.float32)
        scale = masked_rms(field * mask, mask)
        laplacian = F.conv3d(field * mask / scale, model.laplacian_kernel, padding=1)
        valid = (F.avg_pool3d(mask, 3, 1, 1) > 1 - 1e-6).float()
        interior_scale = masked_rms(laplacian * scale, valid)
        selected = scale / interior_scale >= 32
        self.assertEqual(selected.flatten().tolist(), [False, True])
        selected_scale = torch.where(selected, interior_scale, scale)
        selected_laplacian = torch.where(
            selected, valid * laplacian * scale / interior_scale, laplacian
        )
        torch.testing.assert_close(
            local, selected_laplacian * selected_scale * mask, rtol=0, atol=0
        )

    def test_defaults_checkpoint_roundtrip_and_architecture_guards(self):
        for config in (model_config(parse_args([])), OverfitConfig()):
            self.assertFalse(config.filter_laplacian_residual)
            self.assertTrue(config.filter_output_skip)
            self.assertEqual(config.filter_refinement_features, 0)
        self.assertEqual(OverfitConfig().lambda_physics, 1)
        self.assertFalse(checkpoint_training_config({})["filter_laplacian_residual"])
        model = DNQSM(self.config()).eval()
        field = torch.randn(1, 1, 9, 11, 13)
        mask = torch.ones_like(field)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save(
                dict(
                    version=1,
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                ),
                path,
            )
            loaded = load_trained_model(path)
            self.assertTrue(loaded.config.filter_laplacian_residual)
            with torch.no_grad():
                torch.testing.assert_close(
                    loaded.predict_local_field(field, mask),
                    model.predict_local_field(field, mask),
                    rtol=0,
                    atol=0,
                )
        old = DNQSM(laplacian_model_config((2, 4), filter_laplacian_residual=False))
        with self.assertRaisesRegex(ValueError, "different filtering architecture"):
            model.initialize_from(old)
        with self.assertRaisesRegex(ValueError, "Laplacian residual"):
            laplacian_model_config(filter_laplacian_residual=True)


if __name__ == "__main__":
    unittest.main()
