"""TKD supervision weights the filter's error by the inversion's conditioning."""

import unittest

import torch

from dnqsm.losses import (
    LOSS_VERSION,
    LossNormalization,
    LossWeights,
    local_field_loss_terms,
    local_tkd_loss,
    normalize_loss_terms,
    reconstruction_loss,
)
from dnqsm.model import DNQSM, ModelConfig, Reconstruction
from dnqsm.physics import (
    apply_multiplier,
    cutoff_kernels,
    dipole_tensor,
    masked_mean,
)
from dnqsm.training import TrainConfig, checkpoint_training_config, training_weights


def _case(seed=5, shape=(8, 9, 10)):
    generator = torch.Generator().manual_seed(seed)
    local = torch.randn(2, 1, *shape, generator=generator) * 0.01
    chi = torch.randn(2, 1, *shape, generator=generator) * 0.03
    mask = torch.zeros_like(local)
    mask[..., 1:-1, 1:-1, 1:-1] = 1
    return local, chi * mask, mask, dipole_tensor(shape)


class LocalTkdLossTests(unittest.TestCase):
    def test_matches_explicit_truncated_inversion(self):
        local, chi, mask, dipole = _case()
        inverse, _ = cutoff_kernels(dipole, 0.2)
        expected = masked_mean((apply_multiplier(local, inverse) - chi).abs(), mask)
        torch.testing.assert_close(
            local_tkd_loss(local, chi, mask, dipole, 0.2), expected
        )

    def test_penalty_grows_as_the_threshold_amplifies_the_near_null_band(self):
        local, chi, mask, dipole = _case()
        errors = [
            float(local_tkd_loss(local, chi, mask, dipole, tau).mean())
            for tau in (0.3, 0.1, 0.02)
        ]
        self.assertEqual(errors, sorted(errors))

    def test_target_is_detached_and_prediction_keeps_its_gradient(self):
        local, chi, mask, dipole = _case()
        local, chi = local.requires_grad_(), chi.requires_grad_()
        local_tkd_loss(local, chi, mask, dipole, 0.15).sum().backward()
        self.assertIsNone(chi.grad)
        self.assertGreater(float(local.grad.abs().sum()), 0)

    def test_filter_terms_are_opt_in_and_need_a_complete_specification(self):
        local, chi, mask, dipole = _case()
        self.assertNotIn("local_tkd", local_field_loss_terms(local, local, mask))
        terms = local_field_loss_terms(
            local, local, mask, chi_target=chi, dipole=dipole, threshold=0.2
        )
        torch.testing.assert_close(
            terms["local_tkd"], local_tkd_loss(local, chi, mask, dipole, 0.2)
        )
        with self.assertRaises(ValueError):
            local_field_loss_terms(local, local, mask, chi_target=chi)

    def test_reconstruction_reuses_its_own_inversion_for_the_same_value(self):
        local, chi, mask, dipole = _case()
        inverse, projection = cutoff_kernels(dipole, 0.2)
        tkd = apply_multiplier(local, inverse)
        prediction = Reconstruction(local, tkd, torch.zeros_like(tkd), tkd)
        _, terms = reconstruction_loss(
            prediction,
            local,
            chi,
            mask,
            dipole,
            LossWeights(extension=0.0, local_tkd=1.0),
        )
        # Same quantity as the standalone helper, normalized by the chi scale.
        scale = masked_mean(chi.square(), mask).clamp_min(1e-8).sqrt()
        torch.testing.assert_close(
            terms["local_tkd"],
            (local_tkd_loss(local, chi, mask, dipole, 0.2) / scale).mean(),
        )

    def test_zero_weight_leaves_the_term_uncomputed_and_the_total_unchanged(self):
        local, chi, mask, dipole = _case()
        tkd = apply_multiplier(local, cutoff_kernels(dipole, 0.2)[0])
        prediction = Reconstruction(local, tkd, torch.zeros_like(tkd), tkd)
        total, terms = reconstruction_loss(
            prediction, local, chi, mask, dipole, LossWeights(extension=0.0)
        )
        self.assertNotIn("local_tkd", terms)
        with_zero, _ = reconstruction_loss(
            prediction,
            local,
            chi,
            mask,
            dipole,
            LossWeights(extension=0.0, local_tkd=0.0),
        )
        torch.testing.assert_close(total, with_zero)

    def test_normalization_scales_the_term_by_susceptibility(self):
        local, chi, mask, _ = _case()
        raw = {"local_tkd": torch.full((2, 1), 0.004)}
        scaled = normalize_loss_terms(raw, local, chi, mask, LossNormalization())[
            "local_tkd"
        ]
        scale = masked_mean(chi.square(), mask).clamp_min(1e-8).sqrt()
        torch.testing.assert_close(scaled, raw["local_tkd"] / scale)

    def test_filtering_batches_supervise_the_filter_through_the_inversion(self):
        from dnqsm.training import _predict_training_batch

        model = DNQSM(ModelConfig(features=(2, 4), threshold=0.2))
        local, chi, mask, dipole = _case(shape=(12, 12, 12))
        data = dict(
            phase=local,
            local=local,
            chi=chi,
            mask=mask,
            dipole=dipole,
            local_full=local,
        )
        device = torch.device("cpu")
        _, raw, _ = _predict_training_batch(
            model, data, device, False, True, False, include_tkd=True
        )
        self.assertIn("local_tkd", raw)
        self.assertGreater(float(raw["local_tkd"].detach().sum()), 0)
        _, without, _ = _predict_training_batch(model, data, device, False, True, False)
        self.assertNotIn("local_tkd", without)


class ObjectiveVersionTests(unittest.TestCase):
    def test_tkd_supervision_is_off_by_default_everywhere(self):
        self.assertEqual(LossWeights().local_tkd, 0.0)
        self.assertEqual(TrainConfig().lambda_local_tkd, 0.0)
        self.assertEqual(training_weights(TrainConfig()).local_tkd, 0.0)

    def test_configured_weight_reaches_the_training_objective(self):
        config = TrainConfig(lambda_local_tkd=0.25)
        config.validate()
        self.assertEqual(training_weights(config).local_tkd, 0.25)

    def test_checkpoints_without_the_term_default_to_disabled(self):
        restored = checkpoint_training_config({"features": [2, 4]})
        self.assertEqual(restored["lambda_local_tkd"], 0.0)

    def test_objective_version_advanced_for_the_new_term(self):
        self.assertEqual(LOSS_VERSION, 8)


if __name__ == "__main__":
    unittest.main()
