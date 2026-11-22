import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import default_collate

from dnqsm.data import FieldSynthesizer, TrainingVolumes, training_records
from dnqsm.lesion_augmentation import (
    LesionAugmentationConfig,
    _opacity,
    augment_lesions,
)
from dnqsm.physics import apply_dipole, masked_mean
from dnqsm.training import TrainConfig, parse_args, run_training
from tests.helpers import make_dataset


class LesionAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def brain(self, size=40):
        mask = torch.zeros(1, size, size, size, dtype=torch.bool)
        mask[:, 3:-3, 3:-3, 3:-3] = True
        chi = torch.full(mask.shape, 0.03) * mask
        return chi, mask

    def geometry(self, shape, lesion):
        coordinates = np.stack(
            np.meshgrid(
                *(
                    np.arange(n) - c
                    for n, c in zip(shape, lesion["center_voxels"], strict=True)
                ),
                indexing="ij",
            )
        )
        principal = np.einsum(
            "ij,j...->i...", np.array(lesion["rotation"]).T, coordinates
        )
        axes = np.array(lesion["semiaxes_voxels"])[:, None, None, None]
        core = np.sum((principal / axes) ** 2, axis=0) <= 1
        outer = (
            np.sum((principal / (axes + lesion["transition_voxels"])) ** 2, axis=0) < 1
        )
        return torch.from_numpy(core), torch.from_numpy(outer)

    def test_spheres_have_extreme_positive_and_negative_cores_inside_the_brain(self):
        chi, mask = self.brain()
        original = chi.clone()
        for positive in (0.0, 1.0):
            config = LesionAugmentationConfig(
                probability=1,
                count=(1, 1),
                sphere_probability=1,
                sphere_radius_voxels=(3, 3),
                positive_probability=positive,
            )
            result, meta = augment_lesions(chi, mask, config, 9, "qsm:test")
            self.assertEqual(
                (meta["count"], meta["spheres"], meta["skipped"]), (1, 1, 0)
            )
            lesion = meta["parameters"][0]
            core, _ = self.geometry(mask.shape[-3:], lesion)
            self.assertTrue(bool((result[0][core].abs() > 0.2).all()))
            self.assertTrue(
                bool((result[0][core] > 0).all())
                if positive
                else bool((result[0][core] < 0).all())
            )
            self.assertTrue(bool(mask[0][core].all()))
            torch.testing.assert_close(result[0][~core], chi[0][~core], rtol=0, atol=0)
            self.assertEqual(float(result[~mask].abs().sum()), 0)
        torch.testing.assert_close(chi, original, rtol=0, atol=0)

    def test_rotated_ellipsoid_has_an_extreme_core_and_smooth_local_tissue_blend(self):
        chi, mask = self.brain()
        config = LesionAugmentationConfig(
            probability=1,
            count=(1, 1),
            sphere_probability=0,
            ellipsoid_semiaxes_voxels=(3, 6),
            ellipsoid_transition_voxels=(2, 2),
        )
        result, meta = augment_lesions(chi, mask, config, 13, "qsm:test")
        self.assertEqual((meta["count"], meta["ellipsoids"]), (1, 1))
        lesion = meta["parameters"][0]
        core, outer = self.geometry(mask.shape[-3:], lesion)
        self.assertTrue(bool((result[0][core].abs() > 0.2).all()))
        torch.testing.assert_close(
            result[0][core],
            torch.full_like(result[0][core], lesion["susceptibility_ppm"]),
        )
        self.assertTrue(bool(mask[0][outer].all()))
        torch.testing.assert_close(result[0][~outer], chi[0][~outer], rtol=0, atol=1e-8)
        shell = outer & ~core
        self.assertGreater(int(shell.sum()), 0)
        low, high = sorted((0.03, lesion["susceptibility_ppm"]))
        self.assertTrue(
            bool(((result[0][shell] > low) & (result[0][shell] < high)).any())
        )
        rotation = np.array(lesion["rotation"])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_blending_profile_is_monotone_with_flat_core_and_outer_boundaries(self):
        x = np.linspace(0, 5.1, 1021)
        coordinates = np.stack((x, np.zeros_like(x), np.zeros_like(x)))
        opacity, core = _opacity(coordinates, (3, 4, 5), 2)
        self.assertTrue(np.all(opacity[core] == 1))
        self.assertTrue(np.all(opacity[x >= 5] == 0))
        self.assertTrue(np.all(np.diff(opacity) <= 1e-7))
        edge_coordinates = np.zeros((3, 4))
        edge_coordinates[0] = [3, 3.001, 4.999, 5]
        edges, _ = _opacity(edge_coordinates, (3, 4, 5), 2)
        self.assertLess(abs(float(edges[0] - edges[1])), 1e-6)
        self.assertLess(abs(float(edges[2] - edges[3])), 1e-6)

    def test_determinism_independent_rng_nonoverlap_and_unplaceable_masks(self):
        chi, mask = self.brain()
        config = LesionAugmentationConfig(probability=1, count=(3, 3))
        torch_rng = torch.get_rng_state().clone()
        a, ma = augment_lesions(chi, mask, config, 17, "qsm:test")
        b, mb = augment_lesions(chi, mask, config, 17, "qsm:test")
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(ma, mb)
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        c, mc = augment_lesions(chi, mask, config, 18, "qsm:test")
        self.assertFalse(torch.equal(a, c))
        self.assertNotEqual(ma["parameters"], mc["parameters"])
        occupancy = torch.zeros_like(mask[0], dtype=torch.int32)
        for lesion in ma["parameters"]:
            core, outer = self.geometry(mask.shape[-3:], lesion)
            occupancy += outer if lesion["kind"] == "ellipsoid" else core
        self.assertLessEqual(int(occupancy.max()), 1)
        tiny = torch.zeros_like(mask)
        tiny[..., 20, 20, 20] = 1
        result, meta = augment_lesions(
            chi * tiny, tiny, replace(config, max_placement_attempts=4), 17, "qsm:test"
        )
        self.assertEqual(meta["count"], 0)
        self.assertEqual(meta["skipped"], 3)
        torch.testing.assert_close(result, chi * tiny, rtol=0, atol=0)

    def test_dataset_lesions_follow_final_rms_and_enter_clean_field_simulation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root, shape=(32, 32, 32)))
            original_hashes = {
                p: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / "sources").glob("*.pt")
            }
            policy = LesionAugmentationConfig(
                probability=1,
                count=(1, 1),
                sphere_probability=0,
                ellipsoid_semiaxes_voxels=(2, 3),
                ellipsoid_transition_voxels=(1, 1),
            )
            common = dict(
                seed=8,
                chi_scale_mode="rms_mixture",
                chi_augmentation={"final_grid_rms": True, "bimodal_probability": 1},
            )
            ordinary = TrainingVolumes(records, root / "cache", **common)
            augmented = TrainingVolumes(
                records, root / "cache", lesion_augmentation=policy, **common
            )
            before, after = ordinary[0], augmented[0]
            self.assertEqual(after["lesion_count"], 1)
            self.assertEqual(after["chi_rms_before_lesions"], before["chi_rms_final"])
            lesion = json.loads(after["lesion_parameters"])[0]
            core, _ = self.geometry(after["mask"].shape[-3:], lesion)
            self.assertTrue(bool((after["chi"][0][core].abs() > 0.2).all()))
            self.assertGreater(after["chi_rms_final"], after["chi_rms_before_lesions"])
            for key in ("mask", "dilated", "crop_start", "seed"):
                torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
            batch = default_collate([after])
            data = FieldSynthesizer((32, 32, 32), "cpu")(
                batch, background_mode="ratio", snr_min=40, snr_max=40
            )
            torch.testing.assert_close(
                data["local"],
                apply_dipole(data["chi"] * data["mask"], data["dipole"]) * data["mask"],
            )
            torch.testing.assert_close(
                data["phase"], data["local"] + data["background"] + data["noise"]
            )
            snr = (
                masked_mean(data["local"].square(), data["mask"]).sqrt()
                / masked_mean(data["noise"].square(), data["mask"]).sqrt()
            )
            torch.testing.assert_close(snr, torch.full_like(snr, 40))
            validation = TrainingVolumes(
                records[:1],
                root / "cache",
                validation=True,
                views=1,
                lesion_augmentation=policy,
                **common,
            )
            reference = TrainingVolumes(
                records[:1], root / "cache", validation=True, views=1, **common
            )
            self.assertEqual(validation[0]["lesion_count"], 0)
            torch.testing.assert_close(
                validation[0]["chi"], reference[0]["chi"], rtol=0, atol=0
            )
            synthetic = next(
                i
                for i, r in enumerate(records)
                if r["subject"].startswith("synthetic:")
            )
            self.assertEqual(augmented[synthetic]["lesion_count"], 0)
            for p, digest in original_hashes.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)

    def test_resume_reproduces_lesion_views_and_augmentation_counts(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = TrainConfig(
                data_dir=str(make_dataset(root, shape=(24, 24, 24))),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "whole"),
                device="cpu",
                epochs=2,
                filter_pretrain_epochs=0,
                training_schedule="joint",
                features=(2, 4),
                batch_size=3,
                num_workers=0,
                threads=2,
                bf16=False,
                evaluate_external=False,
                val_views=1,
                chi_augmentation={"final_grid_rms": True},
                lesion_augmentation=LesionAugmentationConfig(
                    probability=1,
                    count=(1, 2),
                    sphere_radius_voxels=(1, 2),
                    ellipsoid_semiaxes_voxels=(1, 2),
                    ellipsoid_transition_voxels=(0.5, 0.5),
                ).to_dict(),
            )
            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                whole = run_training(config)
                part = run_training(
                    replace(config, run_dir=str(root / "part"), max_steps=1)
                )
                # A saved lesion run predating dephasing must keep that policy off.
                historical = torch.load(part / "training_last.pt", weights_only=True)
                historical["config"]["lesion_augmentation"] = {
                    key: value
                    for key, value in historical["config"][
                        "lesion_augmentation"
                    ].items()
                    if not key.startswith("dephasing_")
                }
                torch.save(historical, part / "training_last.pt")
                resumed = parse_args(
                    ["--resume", str(part / "training_last.pt"), "--max-steps", "0"]
                )
                with self.assertRaisesRegex(ValueError, "lesion_augmentation"):
                    run_training(replace(resumed, lesion_augmentation=None))
                run_training(resumed)
            a = torch.load(whole / "training_last.pt", weights_only=True)
            b = torch.load(part / "training_last.pt", weights_only=True)
            for name in a["model"]:
                torch.testing.assert_close(
                    a["model"][name], b["model"][name], rtol=0, atol=0
                )
            self.assertEqual(a["scheduler"], b["scheduler"])

            def counts(directory):
                rows = [
                    json.loads(line)
                    for line in (directory / "metrics.jsonl").read_text().splitlines()
                ]
                return [
                    (r["lesion_count"], r["lesion_samples"])
                    for r in rows
                    if r["event"] == "augmentation"
                ]

            self.assertEqual(counts(whole), counts(part))
            self.assertTrue(all(count > 0 for count, samples in counts(whole)))

    def test_configuration_validation_roundtrip_and_disabled_identity(self):
        for invalid in (
            {"probability": 2},
            {"count": [0, 1]},
            {"count": [1.0, 2]},
            {"susceptibility_abs_ppm": [0.2, 0.5]},
            {"ellipsoid_transition_voxels": [0, 1]},
            {"sphere_radius_voxels": [3, 2]},
            {"positive_probability": float("nan")},
            {"version": 2},
            {"unknown": 1},
        ):
            with self.assertRaises(ValueError):
                LesionAugmentationConfig.from_dict(invalid)
        config = LesionAugmentationConfig(probability=1)
        self.assertEqual(
            LesionAugmentationConfig.from_dict(json.loads(json.dumps(asdict(config)))),
            config,
        )
        chi, mask = self.brain()
        disabled, meta = augment_lesions(chi, mask, None, 1, "qsm:test")
        self.assertIs(disabled, chi)
        self.assertEqual(meta["count"], 0)
        self.assertIs(augment_lesions(chi, mask, config, 1, "synthetic:test")[0], chi)
