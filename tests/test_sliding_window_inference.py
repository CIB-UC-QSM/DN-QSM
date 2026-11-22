import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from dnqsm.evaluation import EvaluationCase, evaluate_checkpoint
from dnqsm.inference import (
    SlidingWindowConfig,
    gaussian_sliding_window,
    gaussian_window,
)
from dnqsm.model import DNQSM
from dnqsm.physics import apply_multiplier, cutoff_kernels, dipole_tensor
from dnqsm.runtime import configure_inference, load_trained_model
from tests.helpers import dual_field_model_config as ModelConfig


class SlidingWindowInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self, shape=(11, 13, 15), batch=1):
        generator = torch.Generator().manual_seed(45)
        field = torch.randn(batch, 1, *shape, generator=generator) * 0.03
        mask = torch.zeros_like(field)
        mask[..., 2:-2, 2:-2, 2:-2] = 1
        return field, mask, dipole_tensor(shape)

    def test_gaussian_is_positive_symmetric_and_prefers_window_center(self):
        weight = gaussian_window(64)
        self.assertEqual(weight.dtype, torch.float32)
        self.assertTrue(bool((weight > 0).all()))
        self.assertEqual(float(weight.max()), 1)
        torch.testing.assert_close(weight, weight.flip((-3, -2, -1)), rtol=0, atol=0)
        self.assertLess(float(weight[..., 0, 0, 0]), float(weight[..., 32, 32, 32]))

    def test_identity_and_constant_cover_small_odd_rectangular_batched_volumes(self):
        for shape in ((3, 5, 7), (8, 8, 8), (9, 13, 17)):
            for overlap in (0.0, 0.5, 0.75):
                inputs = torch.randn(2, 2, *shape)
                config = SlidingWindowConfig(8, overlap, window_batch_size=5)
                output = gaussian_sliding_window(inputs, lambda x: x, config)
                torch.testing.assert_close(output, inputs, rtol=1e-6, atol=1e-6)
                constant = gaussian_sliding_window(
                    inputs, lambda x: torch.ones_like(x), config
                )
                torch.testing.assert_close(
                    constant, torch.ones_like(inputs), rtol=1e-6, atol=1e-6
                )

    def test_gaussian_reduces_synthetic_window_edge_artifacts(self):
        inputs = torch.ones(1, 1, 17, 19, 21)
        artifact = torch.ones(1, 1, 8, 8, 8)
        artifact[..., 1:-1, 1:-1, 1:-1] = 0

        def predict(x):
            return x + artifact

        gaussian = gaussian_sliding_window(
            inputs, predict, SlidingWindowConfig(8, window_batch_size=8)
        )
        uniform = gaussian_sliding_window(
            inputs,
            predict,
            SlidingWindowConfig(8, sigma_scale=1e6, window_batch_size=8),
        )
        self.assertLess(
            float((gaussian - inputs).abs().mean()),
            0.25 * float((uniform - inputs).abs().mean()),
        )

    def test_single_window_matches_direct_network_and_batching_is_equivalent(self):
        torch.manual_seed(8)
        model = DNQSM(ModelConfig((2, 4), full_grid_field=True)).eval()
        field, mask, dipole = self.inputs((8, 8, 8))
        with torch.no_grad():
            direct = model(field, mask, dipole)
        tiled = model.infer(field, mask, dipole, window_size=8)
        for name in ("local_field", "harmonic_field", "tkd", "correction", "chi"):
            torch.testing.assert_close(
                getattr(tiled, name), getattr(direct, name), rtol=2e-5, atol=1e-7
            )
        field, mask, dipole = self.inputs(batch=2)
        one = model.infer(field, mask, dipole, window_size=8)
        many = model.infer(field, mask, dipole, window_size=8, window_batch_size=5)
        for name in ("local_field", "harmonic_field", "chi"):
            torch.testing.assert_close(
                getattr(one, name), getattr(many, name), rtol=2e-4, atol=2e-7
            )
        self.assertFalse(many.chi.requires_grad)

    def test_empty_windows_masks_zero_input_and_global_fourier_projection(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        field, mask, dipole = self.inputs((17, 19, 21))
        mask.zero_()
        mask[..., 6:10, 8:12, 9:13] = 1
        output = model.infer(field, mask, dipole, window_size=8, window_batch_size=7)
        self.assertEqual(output.chi.shape, field.shape)
        for value in (output.chi, output.local_field, output.harmonic_field):
            self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertEqual(float((output.local_field * (1 - mask)).abs().max()), 0)
        self.assertEqual(float((output.harmonic_field * (1 - mask)).abs().max()), 0)
        inverse, projection = cutoff_kernels(dipole, model.config.threshold)
        torch.testing.assert_close(
            output.tkd, apply_multiplier(output.local_field, inverse)
        )
        retained = apply_multiplier(output.correction, 1 - projection)
        self.assertLess(float(retained.abs().max()), 1e-7)
        torch.testing.assert_close(
            output.chi, output.tkd + output.correction, rtol=0, atol=0
        )
        zero = model.infer(
            torch.zeros_like(field), mask, dipole, window_size=8, window_batch_size=7
        )
        self.assertTrue(bool(torch.isfinite(zero.chi).all()))
        changed = model.infer(
            field + 1000 * (1 - mask), mask, dipole, window_size=8, window_batch_size=7
        )
        torch.testing.assert_close(changed.chi, output.chi, rtol=0, atol=0)

    def test_legacy_checkpoint_policy_and_patch_policy_roundtrip(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = dict(
                version=1, model_config=asdict(model.config), model=model.state_dict()
            )
            path = root / "model.pt"
            torch.save(base, path)
            self.assertEqual(load_trained_model(path).inference_config.window_size, 0)
            torch.save(
                dict(base, config=dict(patch_size=8, inference_overlap=0.25)), path
            )
            self.assertEqual(
                load_trained_model(path).inference_config, SlidingWindowConfig(8, 0.25)
            )
            policy = SlidingWindowConfig(8, 0.75, 0.2, 3)
            torch.save(dict(base, inference_config=asdict(policy)), path)
            loaded = load_trained_model(path)
            self.assertEqual(loaded.inference_config, policy)
            field, mask, dipole = self.inputs((8, 8, 8))
            with torch.no_grad():
                full = loaded(field, mask, dipole)
            overridden = loaded.infer(field, mask, dipole, window_size=0)
            torch.testing.assert_close(overridden.chi, full.chi, rtol=0, atol=0)

    def test_historical_single_field_model_can_use_explicit_windows(self):
        model = DNQSM(ModelConfig((2, 4), filter_input_version=1)).eval()
        field, mask, dipole = self.inputs((9, 11, 13))
        result = model.infer(field, mask, dipole, window_size=8, window_batch_size=4)
        self.assertIsNone(result.harmonic_field)
        self.assertTrue(bool(torch.isfinite(result.chi).all()))
        self.assertEqual(result.chi.shape, field.shape)

    def test_evaluation_and_exports_use_saved_policy_and_record_overrides(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        configure_inference(model, dict(patch_size=8, inference_batch_size=4))
        field, mask, dipole = self.inputs((9, 11, 13))
        case = EvaluationCase("fixture", field, mask, dipole, field * mask)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "model.pt"
            torch.save(
                dict(
                    version=1,
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                    inference_config=asdict(model.inference_config),
                ),
                checkpoint,
            )
            evaluate_checkpoint(
                checkpoint,
                [case],
                root / "output",
                bf16=False,
                inference_options=dict(overlap=0.25),
            )
            metadata = json.loads((root / "output/fixture/metadata.json").read_text())
            self.assertEqual(metadata["inference"]["window_size"], 8)
            self.assertEqual(metadata["inference"]["overlap"], 0.25)
            self.assertEqual(metadata["inference"]["blending"], "gaussian")
            exported = torch.load(
                root / "output/fixture/padded_predictions.pt", weights_only=True
            )
            expected = model.infer(field, mask, dipole, overlap=0.25)
            torch.testing.assert_close(exported["chi"], expected.chi)
            self.assertEqual(exported["chi"].shape, field.shape)

    def test_invalid_options_training_mode_and_too_deep_network_are_rejected(self):
        config = SlidingWindowConfig()
        for change in (
            dict(window_size=-1),
            dict(window_size=1),
            dict(overlap=1),
            dict(overlap=-0.1),
            dict(sigma_scale=0),
            dict(sigma_scale=float("nan")),
            dict(window_batch_size=0),
        ):
            with self.assertRaises(ValueError):
                replace(config, **change)
        field, mask, dipole = self.inputs()
        model = DNQSM(ModelConfig((2, 4)))
        with self.assertRaisesRegex(ValueError, "eval"):
            model.infer(field, mask, dipole, window_size=8)
        with self.assertRaisesRegex(ValueError, "too small"):
            model.eval().infer(field, mask, dipole, window_size=4)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            model.infer(field, torch.zeros_like(mask), dipole, window_size=8)
