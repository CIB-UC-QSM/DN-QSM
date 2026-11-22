import unittest

import torch

from dnqsm.model import Reconstruction
from scripts.evaluate_graz import cosmos_local_field_gain, denormalize_reconstruction


class GrazNormalizationTests(unittest.TestCase):
    def test_gain_matches_clean_reference_standard_deviation(self):
        statistics = dict(
            units="ppm",
            statistics=dict(
                full_roi=dict(
                    cosmos_reference=dict(std_ppm=0.008),
                    graz1_estimated=dict(std_ppm=0.016),
                    graz2_estimated=dict(std_ppm=0.010),
                )
            ),
        )
        self.assertEqual(cosmos_local_field_gain(statistics, "graz1"), 0.5)
        self.assertEqual(cosmos_local_field_gain(statistics, "graz2"), 0.8)
        statistics["statistics"]["full_roi"]["graz1_estimated"]["std_ppm"] = 0
        with self.assertRaises(ValueError):
            cosmos_local_field_gain(statistics, "graz1")
        statistics["units"] = "radians"
        with self.assertRaises(ValueError):
            cosmos_local_field_gain(statistics, "graz2")

    def test_denormalization_restores_every_output_and_decomposition(self):
        local, initial, correction = (
            torch.full((1, 1, 2, 2, 2), v) for v in (0.03, 0.1, 0.02)
        )
        output = Reconstruction(
            local, initial, correction, initial + correction, local * 2
        )
        restored = denormalize_reconstruction(output, 0.5)
        for key in ("local_field", "tkd", "correction", "chi", "harmonic_field"):
            torch.testing.assert_close(getattr(restored, key), 2 * getattr(output, key))
        torch.testing.assert_close(restored.chi, restored.tkd + restored.correction)
        self.assertIs(denormalize_reconstruction(output, 1), output)
        output.harmonic_field = None
        self.assertIsNone(denormalize_reconstruction(output, 0.5).harmonic_field)
        for gain in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                denormalize_reconstruction(output, gain)


if __name__ == "__main__":
    unittest.main()
