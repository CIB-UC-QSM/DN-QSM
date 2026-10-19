import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from dnqsm.data import (
    ShapeFieldSynthesizer,
    TrainingVolumes,
    collate_shapes,
    separable_dilation,
    training_records,
)
from dnqsm.losses import LossWeights, reconstruction_loss
from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import apply_dipole, continuous_dipole_kernel
from dnqsm.training import TrainConfig, backward_batch, parse_args, run_training
from tests.helpers import make_dataset


class CropTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def datasets(self, root):
        source = make_dataset(root)
        mask = torch.zeros(1, 12, 12, 12, dtype=torch.bool)
        mask[:, 2:9, 3:8, 1:10] = True
        torch.save(mask, source / "mask_0000.pt")
        records = training_records(source)
        full = TrainingVolumes(records, root / "cache", seed=12, crop_resize=False)
        cropped = TrainingVolumes(
            records, root / "cache", seed=12, crop_probability=1, crop_resize=False
        )
        return source, full, cropped

    def test_crop_interpolation_and_forward_targets_use_160_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            _, full, native = self.datasets(Path(temp))
            resized_set = TrainingVolumes(
                full.records, full.cache_dir, seed=12, crop_probability=1
            )
            before, after = native[0], resized_set[0]
            self.assertEqual(tuple(after["chi"].shape), (1, 160, 160, 160))
            self.assertEqual(
                tuple(after["crop_shape"].tolist()), tuple(before["chi"].shape[-3:])
            )
            torch.testing.assert_close(after["crop_start"], before["crop_start"])
            self.assertTrue(after["resize_applied"])
            expected_mask = F.interpolate(
                before["mask"][None].float(), size=(160, 160, 160), mode="nearest"
            )[0].bool()
            expected_chi = (
                F.interpolate(
                    before["chi"][None],
                    size=(160, 160, 160),
                    mode="trilinear",
                    align_corners=False,
                )[0]
                * expected_mask
            )
            torch.testing.assert_close(after["chi"], expected_chi, rtol=0, atol=0)
            torch.testing.assert_close(after["mask"], expected_mask, rtol=0, atol=0)
            torch.testing.assert_close(
                after["dilated"], separable_dilation(expected_mask)
            )
            self.assertEqual(after["mask"].dtype, torch.bool)
            self.assertEqual(after["dilated"].dtype, torch.bool)
            torch.testing.assert_close(
                resized_set[0]["chi"], after["chi"], rtol=0, atol=0
            )
            data = ShapeFieldSynthesizer("cpu")(
                default_collate([after]), background_mode="ratio"
            )
            self.assertEqual(tuple(data["dipole"].shape[-3:]), (160, 160, 81))
            torch.testing.assert_close(
                data["local"], apply_dipole(data["chi"], data["dipole"]) * data["mask"]
            )
            torch.testing.assert_close(
                data["phase"], data["local"] + data["background"] + data["noise"]
            )
            # Unselected samples and validation retain their original voxels.
            unselected = TrainingVolumes(full.records, full.cache_dir, seed=12)[0]
            torch.testing.assert_close(
                unselected["chi"], full[0]["chi"], rtol=0, atol=0
            )
            self.assertFalse(unselected["resize_applied"])
            validation = TrainingVolumes(
                full.records, full.cache_dir, validation=True, crop_probability=1
            )
            self.assertFalse(validation[0]["resize_applied"])
            self.assertEqual(tuple(validation[0]["chi"].shape[-3:]), (12, 12, 12))

    def test_resized_crops_and_uncropped_volumes_stack_in_one_group(self):
        with tempfile.TemporaryDirectory() as temp:
            _, full, _ = self.datasets(Path(temp))
            small = full._load(full.records[0])
            loaded = {
                key: F.pad(value, (74, 74, 74, 74, 74, 74))
                for key, value in small.items()
            }
            loaded["dilated"] = separable_dilation(loaded["mask"])
            dataset = TrainingVolumes(
                full.records, full.cache_dir, seed=12, crop_probability=1
            )
            with patch.object(dataset, "_load", return_value=loaded):
                cropped = dataset[0]
                dataset.crop_probability = 0
                uncropped = dataset[1]
            groups = collate_shapes([cropped, uncropped])
            self.assertEqual(len(groups), 1)
            self.assertEqual(tuple(groups[0]["chi"].shape), (2, 1, 160, 160, 160))
            self.assertEqual(groups[0]["resize_applied"].tolist(), [True, False])

    def test_tight_crop_preserves_values_and_index_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            _, full, cropped = self.datasets(Path(temp))
            original, result = full[0], cropped[0]
            bounds = tuple(
                slice(int(s), int(s) + n)
                for s, n in zip(
                    result["crop_start"], result["chi"].shape[-3:], strict=True
                )
            )
            self.assertEqual(sorted(result["chi"].shape[-3:]), [5, 7, 9])
            for key in ("chi", "mask", "dilated"):
                torch.testing.assert_close(
                    result[key], original[key][(slice(None), *bounds)], rtol=0, atol=0
                )
            self.assertEqual(int(result["mask"].sum()), int(original["mask"].sum()))
            torch.testing.assert_close(cropped[0]["chi"], result["chi"], rtol=0, atol=0)
            for axis in (1, 2, 3):
                self.assertTrue(bool(result["mask"].select(axis, 0).any()))
                self.assertTrue(bool(result["mask"].select(axis, -1).any()))
            data = ShapeFieldSynthesizer("cpu")(
                default_collate([result]), background_mode="ratio"
            )
            torch.testing.assert_close(
                data["local"], apply_dipole(data["chi"], data["dipole"]) * data["mask"]
            )
            validation = TrainingVolumes(
                full.records, full.cache_dir, validation=True, crop_probability=1
            )
            self.assertFalse(validation[0]["crop_applied"])
            self.assertEqual(tuple(validation[0]["chi"].shape[-3:]), (12, 12, 12))

    def test_probability_is_reproducible_and_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            _, full, _ = self.datasets(Path(temp))
            full.crop_probability = 0.2
            loaded = full._load(full.records[0])
            with patch.object(full, "_load", return_value=loaded):
                selected = [full[i]["crop_applied"] for i in range(1000)]
                self.assertEqual(
                    selected[:20], [full[i]["crop_applied"] for i in range(20)]
                )
                self.assertTrue(0.16 < sum(selected) / 1000 < 0.24)
                full.set_epoch(1)
                self.assertNotEqual(
                    selected[:20], [full[i]["crop_applied"] for i in range(20)]
                )
            synthesizer = ShapeFieldSynthesizer("cpu", capacity=2)
            for shape in ((8, 9, 10), (9, 10, 11), (10, 11, 12)):
                batch = dict(
                    chi=torch.randn(1, 1, *shape) * 0.01,
                    mask=torch.ones(1, 1, *shape),
                    dilated=torch.ones(1, 1, *shape, dtype=torch.bool),
                    seed=torch.tensor([3]),
                )
                synthesizer(batch)
            self.assertEqual(list(synthesizer.cache), [(9, 10, 11), (10, 11, 12)])

    def test_grouped_gradients_equal_per_sample_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            _, full, cropped = self.datasets(Path(temp))
            samples = [full[0], full[1], cropped[0]]
            torch.manual_seed(17)
            grouped = DNQSM(ModelConfig((2, 4), residual_filter=True))
            reference = copy.deepcopy(grouped)
            config = TrainConfig(device="cpu", bf16=False)
            cache = ShapeFieldSynthesizer("cpu")
            _, _, count, cropped_count = backward_batch(
                grouped,
                collate_shapes(samples),
                cache,
                torch.device("cpu"),
                LossWeights(),
                config,
                3,
            )
            self.assertEqual((count, cropped_count), (3, 1))
            for sample in samples:
                data = cache(default_collate([sample]), background_mode="mixture")
                output = reference(data["phase"], data["mask"], data["dipole"])
                loss, _ = reconstruction_loss(
                    output, data["local"], data["chi"], data["mask"], data["dipole"]
                )
                (loss / 3).backward()
            for actual, expected in zip(
                grouped.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    actual.grad, expected.grad, rtol=4e-4, atol=2e-6
                )

    def test_cropped_training_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source, _, _ = self.datasets(root)
            config = TrainConfig(
                data_dir=str(source),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "whole"),
                device="cpu",
                epochs=2,
                batch_size=3,
                accumulation_steps=1,
                features=(2, 4),
                crop_probability=0.5,
                crop_resize=False,
                num_workers=0,
                threads=2,
                bf16=False,
                evaluate_external=False,
                val_views=1,
            )
            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                whole = run_training(config)
                part = run_training(
                    replace(config, run_dir=str(root / "partial"), max_steps=1)
                )
                legacy = torch.load(part / "training_last.pt", weights_only=True)
                legacy["config"].pop("crop_resize")
                torch.save(legacy, part / "training_last.pt")
                resumed = parse_args(
                    ["--resume", str(part / "training_last.pt"), "--max-steps", "0"]
                )
                self.assertFalse(resumed.crop_resize)
                with self.assertRaisesRegex(ValueError, "crop_resize"):
                    run_training(replace(resumed, crop_resize=True))
                run_training(resumed)
            a = torch.load(whole / "training_last.pt", weights_only=True)
            b = torch.load(part / "training_last.pt", weights_only=True)
            for key in a["model"]:
                torch.testing.assert_close(
                    a["model"][key], b["model"][key], rtol=0, atol=0
                )
            self.assertEqual(a["scheduler"], b["scheduler"])

            def crops(path):
                return [
                    (r["samples"], r["cropped_samples"])
                    for line in (path / "metrics.jsonl").read_text().splitlines()
                    if (r := json.loads(line))["event"] == "augmentation"
                ]

            self.assertEqual(crops(whole), crops(part))

    def test_capacity_and_native_frequency_coordinates(self):
        with torch.device("meta"):
            model = DNQSM(ModelConfig((23, 47, 90)))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6421406)
        self.assertLess(abs(6421406 / (2 * 3212514) - 1), 0.001)
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
