import unittest

import numpy as np
import torch

from dnqsm.physics import (
    continuous_dipole_kernel,
    isotropic_dipole_tensors,
)


class IsotropicDipoleTests(unittest.TestCase):
    def test_batched_kernels_match_requested_spacing_and_numerical_isotropic_invariance(
        self,
    ):
        sizes = torch.tensor([0.5, 0.65, 1.0], dtype=torch.float32)[:, None].expand(
            -1, 3
        )
        for shape in ((12, 12, 12), (9, 11, 13)):
            actual = isotropic_dipole_tensors(shape, sizes)
            self.assertEqual(actual.shape, (3, 1, *shape[:2], shape[2] // 2 + 1))
            for i in range(3):
                full = continuous_dipole_kernel(shape, voxel_size=sizes[i].tolist())
                expected = torch.from_numpy(
                    full[..., : shape[2] // 2 + 1].astype(np.float32)
                )[None]
                torch.testing.assert_close(actual[i], expected, rtol=1e-6, atol=2e-8)
            unit = torch.from_numpy(
                continuous_dipole_kernel(shape)[..., : shape[2] // 2 + 1].astype(
                    np.float32
                )
            )
            self.assertLess(float((actual[:, 0] - unit).abs().max()), 1e-10)
            self.assertTrue(bool((actual[..., 0, 0, 0] == 0).all()))
