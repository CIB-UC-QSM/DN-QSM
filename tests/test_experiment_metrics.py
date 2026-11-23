import unittest

import torch

from dnqsm.experiment_metrics import (
    paired_summary,
    reconstruction_metrics,
    subject_average,
)
from dnqsm.model import Reconstruction
from dnqsm.physics import apply_dipole, apply_multiplier, cutoff_kernels, dipole_tensor


class ExperimentMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metric_bands_and_regression_have_known_values(self):
        generator = torch.Generator().manual_seed(13)
        chi = torch.randn(2, 1, 12, 12, 12, generator=generator) * 0.03
        mask = torch.ones_like(chi)
        dipole = dipole_tensor(chi.shape[-3:])
        local = apply_dipole(chi, dipole)
        output = Reconstruction(local, chi, torch.zeros_like(chi), chi * 0.8)
        rows = reconstruction_metrics(output, chi, mask, dipole, local)
        for row in rows:
            self.assertAlmostEqual(row["chi_nrmse"], 20.0, places=4)
            self.assertAlmostEqual(row["slope"], 0.8, places=5)
            self.assertAlmostEqual(row["retained_nrmse"], 20.0, places=4)
            self.assertAlmostEqual(row["near_null_nrmse"], 20.0, places=4)
            self.assertAlmostEqual(row["hfen_percent"], 20.0, places=4)
        _, projection = cutoff_kernels(dipole, 0.2)
        correction = apply_multiplier(chi, projection)
        projected = Reconstruction(local, chi, correction, chi + correction)
        rows = reconstruction_metrics(projected, chi, mask, dipole, local, hfen=False)
        self.assertTrue(all(r["retained_nrmse"] < 1e-4 for r in rows))
        self.assertTrue(all(abs(r["near_null_nrmse"] - 100) < 1e-3 for r in rows))

    def test_subject_average_and_bootstrap_are_paired_by_subject(self):
        rows = [
            dict(subject="a", profile="baseline", seed=0, chi_nrmse=10.0),
            dict(subject="a", profile="bimodal", seed=1, chi_nrmse=30.0),
            dict(subject="b", profile="baseline", seed=2, chi_nrmse=40.0),
        ]
        a = subject_average(rows)
        self.assertEqual(a["a"]["chi_nrmse"], 20.0)
        b = {k: dict(chi_nrmse=v["chi_nrmse"] - 2) for k, v in a.items()}
        summary = paired_summary(a, b, samples=100)
        self.assertEqual(summary["delta_joint_minus_frozen_pp"], 2.0)
        self.assertEqual(summary["paired_subject_bootstrap_95ci"], [2.0, 2.0])
        with self.assertRaises(ValueError):
            paired_summary(a, {"a": b["a"]})


if __name__ == "__main__":
    unittest.main()
