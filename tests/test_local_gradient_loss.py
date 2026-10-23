"""Gradient supervision uses in-mask voxel edges and detached target scales."""

import unittest

import torch

from dnqsm.losses import (
    LossNormalization,
    LossWeights,
    local_field_gradient_loss,
    local_field_loss_terms,
    normalize_loss_terms,
    reconstruction_loss,
)
from dnqsm.model import Reconstruction
from dnqsm.overfit import OverfitConfig
from dnqsm.physics import dipole_tensor, masked_mean
from dnqsm.training import TrainConfig, checkpoint_training_config, training_weights


class LocalGradientLossTests(unittest.TestCase):
    def test_known_three_axis_ramps_pool_valid_edges_per_sample(self):
        x, y, z = torch.meshgrid(
            torch.arange(3.0), torch.arange(4.0), torch.arange(5.0), indexing="ij"
        )
        ramp = x + 2 * y + 3 * z
        predicted = torch.stack((ramp, ramp * 5))[:, None]
        target = torch.zeros_like(predicted)
        mask = torch.ones_like(predicted)
        loss = local_field_gradient_loss(predicted, target, mask)
        expected = (40 * 1 + 45 * 2 + 48 * 3) / (40 + 45 + 48)
        torch.testing.assert_close(loss, torch.tensor([[expected], [5 * expected]]))
        torch.testing.assert_close(
            local_field_gradient_loss(predicted, predicted.detach() + 7, mask),
            torch.zeros(2, 1),
            atol=1e-6,
            rtol=0,
        )

    def test_mask_boundary_and_exterior_are_excluded_and_targets_detached(self):
        generator = torch.Generator().manual_seed(9)
        predicted = torch.randn(2, 1, 5, 6, 7, generator=generator).requires_grad_()
        target = torch.randn(2, 1, 5, 6, 7, generator=generator).requires_grad_()
        mask = torch.zeros_like(predicted)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        loss = local_field_gradient_loss(predicted, target, mask)
        changed = local_field_gradient_loss(
            predicted + (1 - mask) * 1000, target - (1 - mask) * 2000, mask
        )
        torch.testing.assert_close(loss, changed, rtol=0, atol=0)
        loss.sum().backward()
        self.assertIsNone(target.grad)
        self.assertEqual(float(predicted.grad[mask == 0].abs().sum()), 0)
        self.assertGreater(float(predicted.grad[mask == 1].abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_no_valid_edges_returns_finite_differentiable_zero(self):
        for shape in ((2, 1, 1, 1, 1), (2, 1, 3, 4, 5)):
            predicted = torch.randn(shape, requires_grad=True)
            mask = torch.zeros_like(predicted)
            mask[..., 0, 0, 0] = 1
            loss = local_field_gradient_loss(
                predicted, torch.zeros_like(predicted), mask
            )
            torch.testing.assert_close(loss, torch.zeros(2, 1), rtol=0, atol=0)
            loss.sum().backward()
            torch.testing.assert_close(
                predicted.grad, torch.zeros_like(predicted), rtol=0, atol=0
            )

    def test_normalization_is_per_sample_amplitude_invariant_and_fp32(self):
        generator = torch.Generator().manual_seed(20)
        predicted = torch.randn(2, 1, 5, 6, 7, generator=generator).requires_grad_()
        target = torch.randn(2, 1, 5, 6, 7, generator=generator).requires_grad_()
        mask = torch.ones_like(predicted)
        chi = torch.ones_like(predicted)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            raw = local_field_loss_terms(predicted, target, mask)
            normalized = normalize_loss_terms(
                raw, target, chi, mask, LossNormalization()
            )
        self.assertEqual(normalized["local_gradient"].dtype, torch.float32)
        expected = (
            raw["local_gradient"] / masked_mean(target.detach().square(), mask).sqrt()
        )
        torch.testing.assert_close(normalized["local_gradient"], expected)
        factors = torch.tensor([0.1, 20.0]).reshape(2, 1, 1, 1, 1)
        scaled = local_field_loss_terms(predicted * factors, target * factors, mask)
        scaled_normalized = normalize_loss_terms(
            scaled, target * factors, chi, mask, LossNormalization()
        )
        torch.testing.assert_close(
            normalized["local_gradient"], scaled_normalized["local_gradient"]
        )
        normalized["local_gradient"].sum().backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(predicted.grad.abs().sum()), 0)

    def test_weight_changes_total_and_disabling_skips_the_new_term(self):
        generator = torch.Generator().manual_seed(21)
        chi = torch.randn(1, 1, 5, 6, 7, generator=generator) * 0.01
        local = torch.randn(1, 1, 5, 6, 7, generator=generator, requires_grad=True)
        target = torch.zeros_like(local)
        mask = torch.ones_like(local)
        prediction = Reconstruction(local, chi, chi * 0, chi)
        dipole = dipole_tensor(local.shape[-3:])
        without, old_terms = reconstruction_loss(
            prediction,
            target,
            chi,
            mask,
            dipole,
            LossWeights(local_gradient=0, extension=0),
        )
        with_gradient, terms = reconstruction_loss(
            prediction,
            target,
            chi,
            mask,
            dipole,
            LossWeights(local_gradient=0.7, extension=0),
        )
        self.assertNotIn("local_gradient", old_terms)
        self.assertGreater(float(terms["local_gradient"].detach()), 0)
        torch.testing.assert_close(
            with_gradient, without + 0.7 * terms["local_gradient"]
        )
        for config in (TrainConfig(), OverfitConfig()):
            self.assertEqual(config.lambda_local_gradient, 1)
        self.assertEqual(training_weights(TrainConfig()).local_gradient, 1)
        self.assertEqual(checkpoint_training_config({})["lambda_local_gradient"], 0)
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LossWeights(local_gradient=weight)


if __name__ == "__main__":
    unittest.main()
