import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dnqsm.data import (
    BalancedSourceSampler,
    FieldSynthesizer,
    TrainingVolumes,
    make_subject_split,
    training_records,
)
from dnqsm.physics import apply_dipole, masked_mean
from tests.helpers import make_dataset


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_manifest_split_and_balanced_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            records = training_records(make_dataset(Path(temp)))
            a, b = make_subject_split(records, 0.25, 0)
            self.assertEqual((a, b), make_subject_split(records, 0.25, 0))
            self.assertFalse({r["subject"] for r in a} & {r["subject"] for r in b})
            self.assertFalse(any(r["subject"].startswith("synthetic:") for r in b))
            sampler = BalancedSourceSampler(a, 0.25, 0)
            indices = list(sampler)
            count = sum(
                a[i % len(a)]["subject"].startswith("synthetic:") for i in indices
            )
            self.assertEqual(count, round(len(a) * 0.25))
            self.assertEqual(indices, list(sampler))
            sampler.set_epoch(1)
            self.assertNotEqual(indices, list(sampler))

    def test_validation_is_fixed_and_synthesis_independent_of_batch_size(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root))
            dataset = TrainingVolumes(
                records[:2], root / "cache", validation=True, views=2, seed=99
            )
            original = dataset[0]
            dataset.set_epoch(10)
            torch.testing.assert_close(
                original["chi"], dataset[0]["chi"], rtol=0, atol=0
            )
            synthesizer = FieldSynthesizer((12, 12, 12), "cpu")
            one = [
                synthesizer(batch, background_mode="ratio")
                for batch in DataLoader(dataset, batch_size=1)
            ]
            two = [
                synthesizer(batch, background_mode="ratio")
                for batch in DataLoader(dataset, batch_size=2)
            ]
            for key in ("phase", "chi", "local", "background", "noise"):
                torch.testing.assert_close(
                    torch.cat([b[key] for b in one]),
                    torch.cat([b[key] for b in two]),
                    atol=2e-8,
                    rtol=2e-5,
                )
            self.assertEqual(len(list((root / "cache").glob("*.pt"))), 2)

    def test_targets_are_clean_and_snr_matches_definition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = TrainingVolumes(
                training_records(make_dataset(root))[:2],
                root / "cache",
                seed=13,
                chi_scale_mode="rms_mixture",
            )
            data = FieldSynthesizer((12, 12, 12), "cpu")(
                next(iter(DataLoader(dataset, batch_size=2))), background_mode="ratio"
            )
            torch.testing.assert_close(
                data["local"], apply_dipole(data["chi"], data["dipole"]) * data["mask"]
            )
            torch.testing.assert_close(
                data["phase"], data["local"] + data["background"] + data["noise"]
            )
            self.assertGreater(
                float((data["phase"] - data["background"] - data["local"]).abs().sum()),
                0,
            )

            def rms(x):
                return masked_mean(x.square(), data["mask"]).sqrt()

            snr = rms(data["local"]) / rms(data["noise"])
            self.assertTrue(bool(((snr >= 40) & (snr <= 100)).all()))
            ratios = rms(data["local"]) / rms(data["background"])
            self.assertTrue(bool(((ratios >= 0.03) & (ratios <= 3)).all()))
            self.assertTrue(
                bool(((rms(data["chi"]) >= 0.018) & (rms(data["chi"]) <= 0.08)).all())
            )


if __name__ == "__main__":
    unittest.main()
