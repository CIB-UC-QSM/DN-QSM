"""Masked total-field input, independent decoders, and harmonic supervision."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

from dnqsm.evaluation import EvaluationCase, evaluate_model
from dnqsm.experiment_metrics import harmonic_laplacian_mse
from dnqsm.model import DNQSM, DualFieldUNet3D
from dnqsm.model import ModelConfig as DefaultModelConfig
from dnqsm.physics import (
    dipole_tensor,
    laplacian_stencil,
    masked_mean,
    masked_rms,
    spherical_erosion,
)
from dnqsm.runtime import load_trained_model
from tests.helpers import dual_field_model_config as ModelConfig
from tests.helpers import laplacian_model_config


class DualFieldFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fields(self):
        field = (
            torch.randn(2, 1, 13, 15, 17, generator=torch.Generator().manual_seed(31))
            * 0.02
        )
        mask = torch.zeros_like(field)
        mask[..., 2:-2, 2:-2, 2:-2] = 1
        return field, mask

    def test_raw_masked_input_shared_bottleneck_and_independent_decoders(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        self.assertIsInstance(model.filter, DualFieldUNet3D)
        self.assertIsNone(model.laplacian_kernel)
        field, mask = self.fields()
        captured, bottlenecks = [], []
        handles = [
            model.filter.register_forward_pre_hook(
                lambda _, args: captured.append(args[0])
            ),
            model.filter.bottleneck.register_forward_hook(
                lambda _, args, out: bottlenecks.append(out)
            ),
        ]
        with (
            torch.no_grad(),
            patch(
                "dnqsm.model.laplacian_stencil",
                side_effect=AssertionError("input Laplacian"),
            ),
        ):
            prediction = model.predict_fields(field, mask)
            changed = model.predict_fields(field + 1000 * (1 - mask), mask)
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(
            captured[0], field * mask / masked_rms(field * mask, mask), rtol=0, atol=0
        )
        self.assertEqual(len(bottlenecks), 2)  # One per call, shared by both decoders.
        for name in ("local_field", "harmonic_field"):
            value = getattr(prediction, name)
            self.assertEqual(value.shape, field.shape)
            torch.testing.assert_close(value, getattr(changed, name), rtol=0, atol=0)
            self.assertEqual(float(value[mask == 0].abs().sum()), 0)
        local_parameters = {id(p) for p in model.filter.local_decoder.parameters()}
        self.assertTrue(
            local_parameters.isdisjoint(
                id(p) for p in model.filter.harmonic_decoder.parameters()
            )
        )
        # Constant input survives the raw-input path instead of differentiating to zero.
        handle = model.filter.register_forward_pre_hook(
            lambda _, args: captured.append(args[0])
        )
        with torch.no_grad():
            model.predict_fields(torch.ones_like(field), torch.ones_like(mask))
        handle.remove()
        torch.testing.assert_close(captured[-1], torch.ones_like(field), rtol=0, atol=0)

    def test_changing_harmonic_head_does_not_change_local_tkd_or_chi(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        field, mask = self.fields()
        dipole = dipole_tensor(field.shape[-3:])
        with torch.no_grad():
            before = model(field, mask, dipole)
            model.filter.harmonic_decoder.final.bias.add_(1)
            after = model(field, mask, dipole)
        for name in ("local_field", "tkd", "chi"):
            torch.testing.assert_close(
                getattr(before, name), getattr(after, name), rtol=0, atol=0
            )
        self.assertFalse(torch.equal(before.harmonic_field, after.harmonic_field))

    def test_new_checkpoints_roundtrip_and_legacy_models_still_load(self):
        field, mask = self.fields()
        with tempfile.TemporaryDirectory() as temp:
            for config in (ModelConfig((2, 4)), laplacian_model_config((2, 4))):
                model = DNQSM(config).eval()
                path = Path(temp) / f"v{config.filter_input_version}.pt"
                torch.save(
                    dict(
                        version=1, model_config=asdict(config), model=model.state_dict()
                    ),
                    path,
                )
                loaded = load_trained_model(path)
                with torch.no_grad():
                    expected = model.predict_fields(field, mask)
                    actual = loaded.predict_fields(field, mask)
                torch.testing.assert_close(
                    actual.local_field, expected.local_field, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual.harmonic_field, expected.harmonic_field, rtol=0, atol=0
                )
            with self.assertRaisesRegex(ValueError, "different filtering architecture"):
                DNQSM(ModelConfig((2, 4))).initialize_from(model)
        # Compatible dual-branch initialization can extend both output heads.
        old = DNQSM(ModelConfig((2, 4), filter_output_skip=False)).eval()
        new = DNQSM(ModelConfig((2, 4))).eval()
        new.initialize_from(old)
        with torch.no_grad():
            a, b = old.predict_fields(field, mask), new.predict_fields(field, mask)
        torch.testing.assert_close(a.local_field, b.local_field, rtol=0, atol=0)
        torch.testing.assert_close(a.harmonic_field, b.harmonic_field, rtol=0, atol=0)

    def test_default_filter_architecture_and_dual_field_guards(self):
        config = DefaultModelConfig()
        self.assertTrue(config.laplacian_input)
        self.assertEqual(config.filter_input_version, 2)
        self.assertEqual(config.conditional_laplacian_threshold, 32)
        for options in (
            dict(laplacian_input=True),
            dict(residual_filter=True),
            dict(conditional_laplacian_threshold=32),
            dict(filter_laplacian_residual=True),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ModelConfig(**options)

    def test_evaluation_exports_and_reports_the_harmonic_branch(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        field, mask = self.fields()
        field, mask = field[:1], mask[:1]
        case = EvaluationCase(
            "fixture", field, mask, dipole_tensor(field.shape[-3:]), field * mask
        )
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            metrics = evaluate_model(model, [case], directory, bf16=False)["fixture"]
            self.assertGreaterEqual(metrics["harmonic_laplacian_mse"], 0)
            self.assertGreaterEqual(metrics["field_decomposition_mae"], 0)
            saved = torch.load(
                directory / "fixture" / "padded_predictions.pt", weights_only=True
            )
            self.assertTrue((directory / "fixture" / "harmonic_field.nii.gz").is_file())
            with torch.no_grad():
                expected = model.predict_fields(field, mask).harmonic_field
            torch.testing.assert_close(
                saved["harmonic_field"], expected, rtol=0, atol=0
            )


class HarmonicMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_spherical_erosion_matches_reference_and_is_not_cubic(self):
        mask = torch.ones(2, 1, 17, 19, 21, dtype=torch.bool)
        mask[0, 0, 8, 9, 10] = False
        mask[1, 0, :3] = False
        original = mask.clone()
        z, y, x = np.meshgrid(
            np.arange(-3, 4), np.arange(-3, 4), np.arange(-3, 4), indexing="ij"
        )
        sphere = z * z + y * y + x * x <= 9
        self.assertEqual(int(sphere.sum()), 123)
        expected = torch.from_numpy(
            np.stack(
                [
                    binary_erosion(sample[0].numpy(), structure=sphere, border_value=0)
                    for sample in mask
                ]
            )
        )[:, None]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = spherical_erosion(mask, radius=3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(mask, original, rtol=0, atol=0)
        self.assertTrue(bool(actual[0, 0, 11, 12, 10]))  # Hole farther than radius 3.
        self.assertFalse(bool(actual[0, 0, 11, 9, 10]))  # Exactly radius 3 from hole.
        self.assertFalse(bool(actual[..., :3, :, :].any()))
        torch.testing.assert_close(
            spherical_erosion(mask.float()), expected, rtol=0, atol=0
        )

    def test_empty_eroded_samples_have_zero_mse_and_finite_zero_gradients(self):
        harmonic = torch.randn(2, 1, 11, 11, 11, requires_grad=True)
        mask = torch.zeros_like(harmonic)
        mask[0, 0, 4:7, 4:7, 4:7] = 1
        mask[1] = 1
        mse = harmonic_laplacian_mse(harmonic, mask)
        self.assertEqual(float(mse[0].detach()), 0)
        self.assertGreater(float(mse[1].detach()), 0)
        self.assertTrue(bool(mse.isfinite().all()))
        mse.sum().backward()
        torch.testing.assert_close(
            harmonic.grad[0], torch.zeros_like(harmonic.grad[0]), rtol=0, atol=0
        )
        self.assertGreater(float(harmonic.grad[1].abs().sum()), 0)
        tiny = torch.ones(1, 1, 1, 1, 1, requires_grad=True)
        zero = harmonic_laplacian_mse(tiny, torch.ones_like(tiny))
        zero.sum().backward()
        self.assertEqual(float(zero.detach()), 0)
        self.assertEqual(float(tiny.grad.abs().sum()), 0)

    def test_laplacian_is_applied_after_masking_in_fp32_and_has_no_exterior_gradients(
        self,
    ):
        harmonic = torch.randn(2, 1, 13, 14, 15, requires_grad=True)
        mask = torch.zeros_like(harmonic)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mse = harmonic_laplacian_mse(harmonic, mask)
        self.assertEqual(mse.dtype, torch.float32)
        expected = masked_mean(
            F.conv3d(harmonic * mask, laplacian_stencil(), padding=1).square(),
            spherical_erosion(mask),
        )
        torch.testing.assert_close(mse, expected, rtol=0, atol=0)
        changed = harmonic_laplacian_mse(harmonic + (1 - mask) * 1000, mask)
        torch.testing.assert_close(mse, changed, rtol=0, atol=0)
        mse.mean().backward()
        self.assertEqual(float(harmonic.grad[mask == 0].abs().sum()), 0)
        self.assertGreater(float(harmonic.grad.abs().sum()), 0)
