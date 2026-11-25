import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import distance_transform_edt

from dnqsm.generate_patches import (
    MAX_ZEROS,
    PADDING,
    eligible_starts,
    exterior_source,
    store_crops,
)
from dnqsm.patch_data import SCHEMA_VERSION, QSMPatchDataset, scale_positive


class PatchGenerationTests(unittest.TestCase):
    def test_exact_zero_limit_and_stride(self):
        chi = np.ones((70, 64, 64), np.float32)
        first = chi[:64]
        first.ravel()[:MAX_ZEROS] = 0
        np.testing.assert_array_equal(eligible_starts(chi), [[0, 0, 0], [6, 0, 0]])
        first.ravel()[MAX_ZEROS] = 0
        np.testing.assert_array_equal(eligible_starts(chi), [[6, 0, 0]])

    def test_background_reproducibility_and_placement(self):
        mask = np.zeros((112, 112, 112), bool)
        mask[24:88, 24:88, 24:88] = True
        distance = distance_transform_edt(~mask)
        results = []
        for option in range(4):
            seed = np.random.SeedSequence([2026, 3, option])
            exterior, settings = exterior_source(
                mask, distance, np.random.default_rng(seed)
            )
            results.append(settings["baseline_ppm"])
            self.assertGreaterEqual(settings["baseline_ppm"], 0)
            self.assertLessEqual(settings["baseline_ppm"], 1)
            self.assertEqual(len(settings["spheres"]), 12)
            core = exterior[(slice(PADDING, -PADDING),) * 3]
            np.testing.assert_array_equal(core[mask], 0)
            self.assertAlmostEqual(exterior[0, 0, 0], settings["baseline_ppm"])
            for i, sphere in enumerate(settings["spheres"]):
                self.assertGreater(
                    distance[tuple(sphere["center"])], sphere["radius"] + 2
                )
                for other in settings["spheres"][:i]:
                    delta = np.subtract(sphere["center"], other["center"])
                    self.assertGreater(
                        np.linalg.norm(delta), sphere["radius"] + other["radius"]
                    )
            replay, parameters = exterior_source(
                mask, distance, np.random.default_rng(seed)
            )
            self.assertEqual(parameters, settings)
            np.testing.assert_array_equal(exterior, replay)
        self.assertEqual(len(set(results)), 4)

    def test_dataset_variants_and_random_background_and_noise_reads(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            chi = np.ones((64,) * 3, dtype=np.float32)
            chi[0] = -2
            local = chi * np.float32(0.01)
            mask = np.ones_like(chi, dtype=bool)
            mask[-1] = False
            local[~mask] = 0
            backgrounds = [np.where(mask, np.float32(i / 10), 0) for i in range(4)]
            starts = [[0, 0, 0]]
            for source_id, split in (("0", "train"), ("1", "validation")):
                folder = root / "patches" / source_id
                folder.mkdir(parents=True)
                sources.append(
                    {"source_id": source_id, "split": split, "starts": starts}
                )
                store_crops(folder, "chi", chi, starts)
                store_crops(folder, "mask", mask, starts)
                store_crops(folder, "local", local, starts)
                store_crops(folder, "background", None, starts, backgrounds)
                if split == "train":
                    store_crops(folder, "local_scaled", 2 * local, starts)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "positive_scale": 4,
                "sources": sources,
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            train = QSMPatchDataset(root)
            val = QSMPatchDataset(root, "validation")
            local_variants = (local, 2 * local, -local, -2 * local)
            self.assertEqual((len(train), len(val)), (4, 1))
            self.assertEqual(val[0]["source_id"], "1")
            for index, expected_chi in enumerate(
                (chi, scale_positive(chi, 4), -chi, -scale_positive(chi, 4))
            ):
                np.testing.assert_array_equal(train[index]["chi"][0], expected_chi)
            for dataset in (train, val):
                seen = set()
                noise_levels = set()
                for read in range(24):
                    index = read % len(dataset)
                    item = dataset[index]
                    option = item["background_option"]
                    seen.add(option)
                    expected_local = local_variants[index]
                    np.testing.assert_array_equal(
                        item["local_field"][0], expected_local
                    )
                    np.testing.assert_array_equal(
                        item["background_field"][0], backgrounds[option]
                    )
                    noise = item["total_field"][0].numpy() - (
                        expected_local + backgrounds[option]
                    )
                    np.testing.assert_array_equal(noise[~mask], 0)
                    std = noise[mask].std()
                    measured_snr = np.abs(expected_local).max() / std
                    # The measured standard deviation fluctuates around its target.
                    self.assertGreaterEqual(measured_snr, 49)
                    self.assertLessEqual(measured_snr, 201)
                    noise_levels.add(round(float(measured_snr), 2))
                self.assertEqual(seen, set(range(4)))
                self.assertGreater(len(noise_levels), 1)

            fixed = QSMPatchDataset(root, "validation", fixed_backgrounds=True)
            np.testing.assert_array_equal(
                fixed[0]["total_field"], fixed[0]["total_field"]
            )
