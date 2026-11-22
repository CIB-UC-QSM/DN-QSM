"""Conditional scaling must preserve ordinary inputs and checkpoint semantics."""

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


class ConditionalScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_samplewise_selection_feature_output_units_and_gradients(self):
        model = DNQSM(laplacian_model_config((2, 4)))
        self.assertEqual(model.config.conditional_laplacian_threshold, 32)
        noise = (
            torch.randn(2, 1, 24, 24, 24, generator=torch.Generator().manual_seed(5))
            * 0.01
        )
        noise[1] += 100
        field = noise.requires_grad_()
        mask = torch.zeros_like(field)
        mask[..., 2:-2, 2:-2, 2:-2] = 1
        captured, network_outputs = [], []
        first = model.filter.register_forward_pre_hook(
            lambda _, args: captured.append(args[0])
        )
        second = model.filter.register_forward_hook(
            lambda _, args, output: network_outputs.append(output)
        )
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model.predict_local_field(field, mask)
        finally:
            first.remove()
            second.remove()
        total_scale = masked_rms(field * mask, mask)
        lap = F.conv3d(field * mask / total_scale, model.laplacian_kernel, padding=1)
        valid = (F.avg_pool3d(mask, 3, 1, 1) > 1 - 1e-6).float()
        interior_scale = masked_rms(lap * total_scale, valid)
        selected = total_scale / interior_scale >= 32
        self.assertEqual(selected.flatten().tolist(), [False, True])
        expected = torch.where(
            selected, valid * lap * total_scale / interior_scale, lap
        )
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
        scale = torch.where(selected, interior_scale, total_scale)
        torch.testing.assert_close(
            output,
            network_outputs[0].float() * scale * mask,
            rtol=0,
            atol=0,
        )
        output.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(field.grad).all()))
        self.assertGreater(float(field.grad.abs().sum()), 0)
        self.assertEqual(float(field.grad[mask == 0].abs().sum()), 0)

    def test_no_valid_stencil_falls_back_to_total_rms(self):
        base = DNQSM(
            laplacian_model_config((2, 4), conditional_laplacian_threshold=0)
        ).eval()
        conditional = DNQSM(
            laplacian_model_config((2, 4), conditional_laplacian_threshold=32)
        ).eval()
        conditional.load_state_dict(base.state_dict())
        field = torch.randn(1, 1, 12, 12, 12)
        mask = torch.zeros_like(field)
        mask[..., 5:7, 5:7, 5:7] = 1
        with torch.no_grad():
            torch.testing.assert_close(
                conditional.predict_local_field(field, mask),
                base.predict_local_field(field, mask),
                rtol=0,
                atol=0,
            )

    def test_checkpoint_cli_and_legacy_defaults(self):
        field = torch.randn(1, 1, 12, 12, 12) * 0.01 + 1
        mask = torch.ones_like(field)
        with tempfile.TemporaryDirectory() as temporary:
            for threshold in (0.0, 32.0):
                model = DNQSM(
                    laplacian_model_config(
                        (2, 4), conditional_laplacian_threshold=threshold
                    )
                ).eval()
                config = asdict(model.config)
                if not threshold:
                    config.pop("conditional_laplacian_threshold")
                path = Path(temporary) / f"model_{threshold}.pt"
                torch.save(
                    dict(version=1, model_config=config, model=model.state_dict()), path
                )
                loaded = load_trained_model(path)
                self.assertEqual(
                    loaded.config.conditional_laplacian_threshold, threshold
                )
                with torch.no_grad():
                    torch.testing.assert_close(
                        loaded.predict_local_field(field, mask),
                        model.predict_local_field(field, mask),
                        rtol=0,
                        atol=0,
                    )
        self.assertEqual(
            model_config(
                parse_args(
                    [
                        "--filter-input-version",
                        "2",
                        "--laplacian-input",
                        "--conditional-laplacian-threshold",
                        "32",
                    ]
                )
            ).conditional_laplacian_threshold,
            32,
        )
        self.assertEqual(
            checkpoint_training_config({})["conditional_laplacian_threshold"], 0
        )
        self.assertEqual(
            model_config(parse_args([])).conditional_laplacian_threshold, 32
        )
        self.assertEqual(OverfitConfig().conditional_laplacian_threshold, 32)
        self.assertEqual(
            model_config(
                parse_args(["--conditional-laplacian-threshold", "0"])
            ).conditional_laplacian_threshold,
            0,
        )
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                laplacian_model_config(conditional_laplacian_threshold=value)
        with self.assertRaises(ValueError):
            laplacian_model_config(
                conditional_laplacian_threshold=32, laplacian_fp32=False
            )


if __name__ == "__main__":
    unittest.main()
