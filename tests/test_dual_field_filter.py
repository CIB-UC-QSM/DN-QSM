"""Masked total-field input, independent decoders, and harmonic supervision."""

import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

from dnqsm.evaluation import EvaluationCase, evaluate_model
from dnqsm.losses import (
    LossNormalization,
    LossWeights,
    harmonic_field_loss_terms,
    harmonic_laplacian_loss,
    normalize_loss_terms,
    reconstruction_loss,
)
from dnqsm.model import DNQSM, DualFieldUNet3D
from dnqsm.model import ModelConfig as DefaultModelConfig
from dnqsm.overfit import OverfitConfig
from dnqsm.physics import (
    dipole_tensor,
    laplacian_stencil,
    masked_mean,
    masked_rms,
    spherical_erosion,
)
from dnqsm.runtime import load_trained_model
from dnqsm.training import (
    _predict_training_batch,
    checkpoint_training_config,
    model_config,
    parse_args,
    set_training_branch,
)
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

    def test_filter_only_and_joint_losses_train_both_decoders_and_shared_encoder(self):
        field, mask = self.fields()
        data = dict(
            phase=field,
            mask=mask,
            local=field * mask * 0.4,
            chi=field * mask,
            local_full=field * 0.4,
            dipole=dipole_tensor(field.shape[-3:]),
        )
        for filtering_only in (True, False):
            model = DNQSM(ModelConfig((2, 4), checkpoint_blocks=True))
            set_training_branch(model, "phase" if filtering_only else "joint")
            with patch.object(
                model.null_network, "forward", wraps=model.null_network.forward
            ) as null:
                _, raw, _ = _predict_training_batch(
                    model, data, torch.device("cpu"), False, filtering_only, False
                )
            self.assertEqual(null.call_count, 0 if filtering_only else 1)
            self.assertIn("harmonic_laplacian", raw)
            self.assertIn("field_decomposition", raw)
            shared, local_only, harmonic_only = torch.autograd.grad(
                raw["harmonic_laplacian"].sum(),
                [
                    model.filter.downs[0][0].weight,
                    model.filter.local_decoder.final.weight,
                    model.filter.harmonic_decoder.final.weight,
                ],
                retain_graph=True,
                allow_unused=True,
            )
            self.assertIsNone(local_only)
            self.assertGreater(float(shared.abs().sum()), 0)
            self.assertGreater(float(harmonic_only.abs().sum()), 0)
            weights = LossWeights(
                extension=0, harmonic_laplacian=100, field_decomposition=1
            )
            sum(
                getattr(weights, name) * term.mean() for name, term in raw.items()
            ).backward()
            for parameter in (
                model.filter.downs[0][0].weight,
                model.filter.bottleneck[0].weight,
                model.filter.local_decoder.final.weight,
                model.filter.harmonic_decoder.final.weight,
            ):
                self.assertTrue(bool(parameter.grad.isfinite().all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            set_training_branch(model, "chi")
            self.assertTrue(all(not p.requires_grad for p in model.filter.parameters()))
            self.assertTrue(
                all(p.requires_grad for p in model.null_network.parameters())
            )
            _, without, _ = _predict_training_batch(
                model,
                data,
                torch.device("cpu"),
                False,
                False,
                False,
                include_harmonic=False,
                include_decomposition=False,
            )
            self.assertNotIn("harmonic_laplacian", without)
            self.assertNotIn("field_decomposition", without)

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

    def test_defaults_cli_weights_and_architecture_guards(self):
        for config in (
            DefaultModelConfig(),
            model_config(parse_args([])),
            OverfitConfig(),
        ):
            self.assertTrue(config.laplacian_input)
            self.assertEqual(config.filter_input_version, 2)
            self.assertEqual(config.conditional_laplacian_threshold, 32)
        config = parse_args(
            [
                "--lambda-harmonic-laplacian",
                "0.2",
                "--lambda-field-decomposition",
                "0.5",
            ]
        )
        self.assertEqual(config.lambda_harmonic_laplacian, 0.2)
        self.assertEqual(config.lambda_field_decomposition, 0.5)
        for key in ("lambda_harmonic_laplacian", "lambda_field_decomposition"):
            self.assertEqual(checkpoint_training_config({})[key], 0)
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


class HarmonicLossTests(unittest.TestCase):
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

    def test_empty_eroded_samples_have_zero_loss_and_finite_zero_gradients(self):
        harmonic = torch.randn(2, 1, 11, 11, 11, requires_grad=True)
        mask = torch.zeros_like(harmonic)
        mask[0, 0, 4:7, 4:7, 4:7] = 1
        mask[1] = 1
        loss = harmonic_laplacian_loss(harmonic, mask)
        self.assertEqual(float(loss[0].detach()), 0)
        self.assertGreater(float(loss[1].detach()), 0)
        self.assertTrue(bool(loss.isfinite().all()))
        loss.sum().backward()
        torch.testing.assert_close(
            harmonic.grad[0], torch.zeros_like(harmonic.grad[0]), rtol=0, atol=0
        )
        self.assertGreater(float(harmonic.grad[1].abs().sum()), 0)
        tiny = torch.ones(1, 1, 1, 1, 1, requires_grad=True)
        zero = harmonic_laplacian_loss(tiny, torch.ones_like(tiny))
        zero.sum().backward()
        self.assertEqual(float(zero.detach()), 0)
        self.assertEqual(float(tiny.grad.abs().sum()), 0)

    def test_laplacian_is_applied_after_masking_in_fp32_and_has_no_exterior_gradients(
        self,
    ):
        harmonic = torch.randn(2, 1, 13, 14, 15, requires_grad=True)
        mask = torch.zeros_like(harmonic)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        local = torch.randn_like(harmonic, requires_grad=True)
        total = torch.randn_like(harmonic, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            terms = harmonic_field_loss_terms(local, harmonic, mask, total_field=total)
        self.assertEqual(terms["harmonic_laplacian"].dtype, torch.float32)
        expected = masked_mean(
            F.conv3d(harmonic * mask, laplacian_stencil(), padding=1).square(),
            spherical_erosion(mask),
        )
        torch.testing.assert_close(
            terms["harmonic_laplacian"], expected, rtol=0, atol=0
        )
        changed = harmonic_field_loss_terms(
            local, harmonic + (1 - mask) * 1000, mask, total_field=total
        )
        for name in terms:
            torch.testing.assert_close(terms[name], changed[name], rtol=0, atol=0)
        sum(term.mean() for term in terms.values()).backward()
        self.assertIsNone(total.grad)
        self.assertEqual(float(harmonic.grad[mask == 0].abs().sum()), 0)
        self.assertGreater(float(harmonic.grad.abs().sum()), 0)
        self.assertGreater(float(local.grad.abs().sum()), 0)

    def test_field_sum_penalty_removes_the_zero_harmonic_shortcut(self):
        mask = torch.zeros(1, 1, 13, 13, 13)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        local = mask * 0.01
        total = local + mask * 0.2
        zero = harmonic_field_loss_terms(
            local, torch.zeros_like(mask), mask, total_field=total
        )
        matched = harmonic_field_loss_terms(
            local, total - local, mask, total_field=total
        )
        self.assertEqual(float(zero["harmonic_laplacian"]), 0)
        self.assertGreater(float(zero["field_decomposition"]), 0)
        self.assertEqual(float(matched["field_decomposition"]), 0)
        self.assertTrue(bool(spherical_erosion(mask).any()))
        # The original mask creates edge responses, but the eroded loss excludes them.
        boundary_response = F.conv3d(
            (total - local) * mask, laplacian_stencil(), padding=1
        )
        self.assertGreater(float(masked_mean(boundary_response.square(), mask)), 0)
        torch.testing.assert_close(
            matched["harmonic_laplacian"],
            torch.zeros_like(matched["harmonic_laplacian"]),
            atol=1e-12,
            rtol=0,
        )
        with self.assertRaisesRegex(ValueError, "aligned total field"):
            harmonic_field_loss_terms(local, total - local, mask)

    def test_per_sample_normalization_is_amplitude_invariant_and_detached(self):
        local = torch.randn(2, 1, 11, 12, 13, requires_grad=True)
        harmonic = torch.randn_like(local, requires_grad=True)
        total = torch.randn_like(local, requires_grad=True)
        target = torch.randn_like(local, requires_grad=True)
        mask = torch.ones_like(local)
        terms = harmonic_field_loss_terms(local, harmonic, mask, total_field=total)
        normalized = normalize_loss_terms(
            terms, target, target, mask, LossNormalization()
        )
        factors = torch.tensor([0.2, 12.0]).reshape(2, 1, 1, 1, 1)
        scaled = harmonic_field_loss_terms(
            local * factors, harmonic * factors, mask, total_field=total * factors
        )
        scaled = normalize_loss_terms(
            scaled, target * factors, target * factors, mask, LossNormalization()
        )
        for name in normalized:
            torch.testing.assert_close(normalized[name], scaled[name])
        sum(term.mean() for term in normalized.values()).backward()
        self.assertIsNone(total.grad)
        self.assertIsNone(target.grad)

    def test_zero_weights_skip_both_new_terms_and_laplacian_computation(self):
        model = DNQSM(ModelConfig((2, 4)))
        field = torch.randn(1, 1, 8, 8, 8) * 0.01
        mask = torch.ones_like(field)
        dipole = dipole_tensor(field.shape[-3:])
        prediction = model(field, mask, dipole)
        weights = replace(LossWeights(), harmonic_laplacian=0, field_decomposition=0)
        with patch(
            "dnqsm.losses.spatial_laplacian", side_effect=AssertionError("disabled")
        ):
            loss, terms = reconstruction_loss(
                prediction, field, field, mask, dipole, weights
            )
        self.assertNotIn("harmonic_laplacian", terms)
        self.assertNotIn("field_decomposition", terms)
        self.assertTrue(bool(loss.isfinite()))
        for name in ("harmonic_laplacian", "field_decomposition"):
            with self.assertRaises(ValueError):
                replace(weights, **{name: -1})


if __name__ == "__main__":
    unittest.main()
