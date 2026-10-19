import copy
import unittest

import numpy as np
import torch

from dnqsm.losses import LossWeights, reconstruction_loss
from dnqsm.model import DNQSM, ModelConfig, Reconstruction
from dnqsm.physics import (
    apply_dipole,
    apply_multiplier,
    continuous_dipole_kernel,
    cutoff_kernels,
    dipole_tensor,
    masked_mean,
    masked_nrmse,
)


class PhysicsAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(4)

    def test_five_level_unet_preserves_arbitrary_input_geometry_and_gradients(self):
        model = DNQSM(ModelConfig((2, 4, 8, 16, 32), checkpoint_blocks=True))
        phase = torch.randn(1, 1, 17, 19, 21) * 0.02
        mask = torch.ones_like(phase)
        output = model(phase, mask, dipole_tensor(phase.shape[-3:]))
        self.assertEqual(output.chi.shape, phase.shape)
        output.chi.square().mean().backward()
        self.assertTrue(
            all(
                p.grad is not None and bool(p.grad.isfinite().all())
                for p in model.parameters()
            )
        )

    def test_kernel_convention_and_full_fft_reference(self):
        for shape in ((12, 14, 16), (9, 11, 13)):
            kernel = continuous_dipole_kernel(shape)
            self.assertEqual(kernel[0, 0, 0], 0)
            self.assertAlmostEqual(kernel[0, 0, 1], -2 / 3)
            self.assertAlmostEqual(kernel[0, 1, 0], 1 / 3)
            volume = torch.randn(2, 1, *shape)
            expected = np.fft.ifftn(
                np.fft.fftn(volume.numpy(), axes=(-3, -2, -1)) * kernel,
                axes=(-3, -2, -1),
            ).real
            actual = apply_dipole(volume, torch.from_numpy(kernel))
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=5e-7)
            torch.testing.assert_close(
                actual, apply_dipole(volume, dipole_tensor(shape))
            )

    def test_residual_filter_preserves_input_when_learned_correction_is_zero(self):
        phase = torch.randn(1, 1, 9, 11, 13) * 0.01
        mask = torch.ones_like(phase)
        mask[..., :2, :, :] = 0
        for residual in (False, True):
            model = DNQSM(
                ModelConfig((2, 4), residual_filter=residual, filter_input_version=1)
            )
            with torch.no_grad():
                model.filter.final.weight.zero_()
                model.filter.final.bias.zero_()
            result = model(phase, mask, dipole_tensor(phase.shape[-3:]))
            torch.testing.assert_close(
                result.local_field,
                phase * mask if residual else torch.zeros_like(phase),
            )

    def test_cutoff_boundary_and_projection(self):
        dipole = torch.tensor([-0.2, -0.1, 0.0, 0.1, 0.2])
        inverse, projection = cutoff_kernels(dipole, 0.1)
        torch.testing.assert_close(inverse, torch.tensor([-5.0, 0, 0, 0, 5]))
        torch.testing.assert_close(projection, torch.tensor([0.0, 1, 1, 1, 0]))
        torch.testing.assert_close(1 - inverse * dipole, projection)
        self.assertTrue(bool(torch.isfinite(inverse).all()))
        with self.assertRaises(ValueError):
            cutoff_kernels(dipole, 0)

    def test_projection_idempotence_and_retained_frequency_preservation(self):
        shape = (13, 15, 17)
        dipole = dipole_tensor(shape)
        _, projection = cutoff_kernels(dipole, 0.1)
        volume = torch.randn(1, 1, *shape)
        projected = apply_multiplier(volume, projection)
        torch.testing.assert_close(
            apply_multiplier(projected, projection), projected, atol=4e-7, rtol=2e-5
        )
        model = DNQSM(ModelConfig((2, 4)))
        mask = torch.ones_like(volume)
        mask[..., :3, :, :] = 0
        result = model(volume * 0.01, mask, dipole)
        correction_fft = torch.fft.rfftn(result.correction.detach(), dim=(-3, -2, -1))
        self.assertLess(float(correction_fft[projection == 0].abs().max()), 2e-6)
        delta = torch.fft.rfftn((result.chi - result.tkd).detach(), dim=(-3, -2, -1))
        self.assertLess(float(delta[projection == 0].abs().max()), 2e-6)
        # Deliberately leave the result unmasked to preserve this property.
        self.assertGreater(float(result.correction.detach()[mask == 0].abs().sum()), 0)

    def test_loss_has_no_predicted_field_consistency_penalty(self):
        shape = (8, 8, 8)
        chi = (torch.randn(1, 1, *shape) * 0.01).requires_grad_()
        local = torch.full_like(chi, 0.02, requires_grad=True)
        mask = torch.zeros_like(chi)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        dipole = dipole_tensor(shape)
        prediction = Reconstruction(local, chi, chi * 0, chi)
        local_gt, chi_gt = torch.zeros_like(chi), torch.ones_like(chi) * 0.01
        weights = LossWeights(local=3, chi=4, physics=2, extension=0)
        loss, terms = reconstruction_loss(
            prediction, local_gt, chi_gt, mask, dipole, weights, normalization=None
        )
        self.assertEqual(set(terms), {"physics", "local", "chi"})
        torch.testing.assert_close(
            loss,
            sum(getattr(weights, name) * value for name, value in terms.items()),
        )
        loss.backward()
        # With chi independent here, the only local-field gradient is ROI L1.
        # No consistency gradient from A(chi) to the predicted local field remains.
        torch.testing.assert_close(local.grad, weights.local * mask / mask.sum())
        self.assertGreater(float(local.grad.abs().sum()), 0)
        self.assertGreater(float(chi.grad.abs().sum()), 0)
        self.assertTrue(bool((local.grad[mask == 0] == 0).all()))

    def test_clean_field_physics_penalizes_jointly_biased_predictions(self):
        shape = (8, 10, 12)
        dipole = dipole_tensor(shape)
        chi = (torch.randn(1, 1, *shape) * 0.02).requires_grad_()
        mask = torch.ones_like(chi)
        clean = torch.zeros_like(chi)
        # A self-consistent but wrong filter/susceptibility pair has zero old
        # consistency penalty, while clean-field physics still gives gradients.
        local = apply_dipole(chi.detach(), dipole).requires_grad_()
        prediction = Reconstruction(local, chi, chi * 0, chi)
        _, terms = reconstruction_loss(
            prediction, clean, clean, mask, dipole, normalization=None
        )
        self.assertNotIn("forward", terms)
        torch.testing.assert_close(
            terms["physics"],
            masked_mean(apply_dipole(chi, dipole).square(), mask).mean(),
        )
        terms["physics"].backward()
        self.assertGreater(float(chi.grad.abs().sum()), 0)
        self.assertIsNone(local.grad)

    def test_checkpointed_gradients_match_and_both_networks_train(self):
        model = DNQSM(ModelConfig((2, 4), checkpoint_blocks=False))
        checkpointed = copy.deepcopy(model)
        for network in (checkpointed.filter, checkpointed.null_network):
            network.checkpoint_blocks = True
        phase = torch.randn(1, 1, 9, 11, 13) * 0.02
        mask = torch.ones_like(phase)
        dipole = dipole_tensor(phase.shape[-3:])
        gradients = []
        for net in (model, checkpointed):
            result = net(phase, mask, dipole)
            loss, _ = reconstruction_loss(result, phase * 0.5, phase, mask, dipole)
            loss.backward()
            for branch in (net.filter, net.null_network):
                self.assertGreater(
                    sum(float(p.grad.abs().sum()) for p in branch.parameters()), 0
                )
            gradients.append(torch.cat([p.grad.flatten() for p in net.parameters()]))
        torch.testing.assert_close(*gradients, rtol=5e-5, atol=2e-7)

    def test_zero_input_and_invalid_masks(self):
        model = DNQSM(ModelConfig((2, 4)))
        phase = torch.zeros(1, 1, 9, 10, 11)
        mask = torch.ones_like(phase)
        d = dipole_tensor(phase.shape[-3:])
        output = model(phase, mask, d)
        loss, _ = reconstruction_loss(output, phase, phase, mask, d)
        loss.backward()
        self.assertTrue(
            all(
                p.grad is not None and bool(p.grad.isfinite().all())
                for p in model.parameters()
            )
        )
        self.assertEqual(masked_nrmse(phase, phase, mask), [None])
        for invalid in (mask * 0, mask * 0.5):
            with self.assertRaises(ValueError):
                model(phase, invalid, d)


if __name__ == "__main__":
    unittest.main()
