import copy
import unittest

import numpy as np
import torch

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import (
    apply_dipole,
    apply_multiplier,
    continuous_dipole_kernel,
    cutoff_kernels,
    dipole_tensor,
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
        self.assertIsNone(output.harmonic_field)
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
                ModelConfig(
                    (2, 4),
                    residual_filter=residual,
                    filter_input_version=1,
                    filter_laplacian_residual=False,
                    conditional_laplacian_threshold=0,
                )
            )
            with torch.no_grad():
                model.filter.final.weight.zero_()
                if model.filter.final.bias is not None:
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
        _, projection = cutoff_kernels(dipole, ModelConfig().threshold)
        volume = torch.randn(1, 1, *shape)
        projected = apply_multiplier(volume, projection)
        torch.testing.assert_close(
            apply_multiplier(projected, projection), projected, atol=4e-7, rtol=2e-5
        )
        model = DNQSM(ModelConfig((2, 4), cutoff_mode="hard"))
        mask = torch.ones_like(volume)
        mask[..., :3, :, :] = 0
        result = model(volume * 0.01, mask, dipole)
        correction_fft = torch.fft.rfftn(result.correction.detach(), dim=(-3, -2, -1))
        self.assertLess(float(correction_fft[projection == 0].abs().max()), 2e-6)
        delta = torch.fft.rfftn((result.chi - result.tkd).detach(), dim=(-3, -2, -1))
        self.assertLess(float(delta[projection == 0].abs().max()), 2e-6)
        # Deliberately leave the result unmasked to preserve this property.
        self.assertGreater(float(result.correction.detach()[mask == 0].abs().sum()), 0)

    def test_checkpointed_gradients_match_and_both_networks_train(self):
        model = DNQSM(ModelConfig((2, 4), checkpoint_blocks=False))
        checkpointed = copy.deepcopy(model)
        for network in checkpointed.modules():
            if hasattr(network, "checkpoint_blocks"):
                network.checkpoint_blocks = True
        phase = torch.randn(1, 1, 9, 11, 13) * 0.02
        mask = torch.ones_like(phase)
        dipole = dipole_tensor(phase.shape[-3:])
        gradients = []
        for net in (model, checkpointed):
            result = net(phase, mask, dipole)
            result.chi.square().mean().backward()
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
        output.chi.square().mean().backward()
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
