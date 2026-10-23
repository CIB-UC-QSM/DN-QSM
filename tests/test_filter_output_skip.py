"""The field bypass joins decoder features only at the filtering output head."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from dnqsm.model import DNQSM, ModelConfig, UNet3D
from dnqsm.overfit import OverfitConfig
from dnqsm.physics import masked_rms
from dnqsm.runtime import load_trained_model
from dnqsm.training import checkpoint_training_config, model_config, parse_args


class FilterOutputSkipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fields(self):
        field = (
            torch.randn(2, 1, 5, 7, 9, generator=torch.Generator().manual_seed(7))
            * 0.01
        )
        field[1] += 100
        mask = torch.zeros_like(field)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        field[mask == 0] = 999
        return field, mask

    def test_concatenates_raw_field_after_decoder_with_selected_scale_and_padding(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        field, mask = self.fields()
        field.requires_grad_()
        features, head_inputs = [], []
        decoder = model.filter.up_blocks[-1].register_forward_hook(
            lambda _, args, output: features.append(output)
        )
        head = model.filter.final.register_forward_pre_hook(
            lambda _, args: head_inputs.append(args[0])
        )
        with torch.no_grad():
            model.filter.final.weight.zero_()
            model.filter.final.weight[:, -1].fill_(1)
            model.filter.final.bias.zero_()
        try:
            local = model.predict_local_field(field, mask)
        finally:
            decoder.remove()
            head.remove()
        scale = masked_rms(field * mask, mask)
        lap = F.conv3d(field * mask / scale, model.laplacian_kernel, padding=1)
        valid = (F.avg_pool3d(mask, 3, 1, 1) > 1 - 1e-6).float()
        interior_scale = masked_rms(lap * scale, valid)
        selected = scale / interior_scale >= 32
        self.assertEqual(selected.flatten().tolist(), [False, True])
        scale = torch.where(selected, interior_scale, scale)
        # Two-level U-Net internally pads 5x7x9 to 8x8x9.
        expected_raw = F.pad(field * mask / scale, (0, 0, 0, 1, 0, 3))
        torch.testing.assert_close(head_inputs[0][:, :-1], features[0], rtol=0, atol=0)
        torch.testing.assert_close(head_inputs[0][:, -1:], expected_raw, rtol=0, atol=0)
        torch.testing.assert_close(local, field * mask)
        local.sum().backward()
        torch.testing.assert_close(field.grad, mask)

    def test_new_weight_starts_at_zero_and_receives_gradients(self):
        model = DNQSM(ModelConfig((2, 4)))
        self.assertEqual(model.filter.final.in_channels, 3)
        self.assertEqual(model.null_network.final.in_channels, 2)
        self.assertEqual(
            float(model.filter.final.weight[:, -1].detach().abs().sum()), 0
        )
        field, mask = self.fields()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            local = model.predict_local_field(field, mask)
        local.sum().backward()
        self.assertTrue(bool(torch.isfinite(model.filter.final.weight.grad).all()))
        self.assertGreater(float(model.filter.final.weight.grad[:, -1].abs().sum()), 0)

    def test_warm_start_zero_extends_head_and_preserves_old_weights(self):
        old = DNQSM(
            ModelConfig(
                (2, 4), filter_output_skip=False, filter_laplacian_residual=False
            )
        ).eval()
        new = DNQSM(
            ModelConfig(
                (2, 4), filter_output_skip=True, filter_laplacian_residual=False
            )
        ).eval()
        original = old.filter.final.weight.detach().clone()
        new.initialize_from(old)
        torch.testing.assert_close(
            new.filter.final.weight[:, :-1], original, rtol=0, atol=0
        )
        torch.testing.assert_close(old.filter.final.weight, original, rtol=0, atol=0)
        field, mask = self.fields()
        with torch.no_grad():
            torch.testing.assert_close(
                new.predict_local_field(field, mask),
                old.predict_local_field(field, mask),
            )
        with self.assertRaisesRegex(ValueError, "cannot discard"):
            old.initialize_from(new)

    def test_checkpoint_roundtrip_and_historical_loading(self):
        field, mask = self.fields()
        with tempfile.TemporaryDirectory() as temporary:
            for enabled in (False, True):
                model = DNQSM(
                    ModelConfig(
                        (2, 4),
                        filter_output_skip=enabled,
                        filter_laplacian_residual=False,
                    )
                ).eval()
                config = asdict(model.config)
                config.pop("filter_laplacian_residual")
                if not enabled:
                    config.pop("filter_output_skip")
                else:
                    with torch.no_grad():
                        model.filter.final.weight[:, -1].fill_(0.25)
                path = Path(temporary) / f"{enabled}.pt"
                torch.save(
                    dict(version=1, model_config=config, model=model.state_dict()), path
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.filter_output_skip, enabled)
                with torch.no_grad():
                    torch.testing.assert_close(
                        loaded.predict_local_field(field, mask),
                        model.predict_local_field(field, mask),
                        rtol=0,
                        atol=0,
                    )
        self.assertTrue(model_config(parse_args([])).filter_output_skip)
        self.assertTrue(OverfitConfig().filter_output_skip)
        self.assertFalse(
            model_config(parse_args(["--no-filter-output-skip"])).filter_output_skip
        )
        self.assertFalse(checkpoint_training_config({})["filter_output_skip"])

    def test_missing_or_misaligned_skip_is_rejected(self):
        network = UNet3D((2, 4), output_skip_channels=1)
        x = torch.randn(1, 1, 8, 8, 8)
        with self.assertRaisesRegex(ValueError, "Output skip"):
            network(x)
        with self.assertRaisesRegex(ValueError, "Output skip"):
            network(x, x[..., :-1])


if __name__ == "__main__":
    unittest.main()
