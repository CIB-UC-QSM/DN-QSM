import unittest

import numpy as np

from dnqsm.physics import continuous_dipole_kernel


class DipoleGridTests(unittest.TestCase):
    def test_native_frequency_coordinates(self):
        shape, spacing = (7, 9, 11), (1, 2, 3)
        kernel = continuous_dipole_kernel(shape, spacing)
        frequencies = [
            np.fft.fftfreq(n, d=s) for n, s in zip(shape, spacing, strict=True)
        ]
        x, y, z = np.meshgrid(*frequencies, indexing="ij")
        squared = x * x + y * y + z * z
        expected = np.full(shape, 1 / 3)
        np.divide(z * z, squared, out=squared, where=squared > 0)
        expected -= squared
        expected[0, 0, 0] = 0
        np.testing.assert_allclose(kernel, expected, atol=1e-12)
        # Preserve the established wh_net convention on even reference grids.
        even = (8, 10, 12)
        grids = np.meshgrid(*(np.arange(-n // 2, n // 2) for n in even), indexing="ij")
        grids = [axis / np.abs(axis).max() for axis in grids]
        legacy = np.fft.ifftshift(
            1 / 3
            - grids[2] ** 2 / (sum(axis**2 for axis in grids) + np.finfo(float).eps)
        )
        legacy[0, 0, 0] = 0
        np.testing.assert_allclose(continuous_dipole_kernel(even), legacy, atol=1e-15)
