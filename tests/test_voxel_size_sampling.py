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
from torch.utils.data import default_collate

from dnqsm.data import (
    ShapeFieldSynthesizer,
    TrainingVolumes,
    sample_voxel_sizes,
    simulated_forward_field,
    training_records,
    validate_voxel_size_range,
)
from dnqsm.physics import (
    apply_dipole,
    continuous_dipole_kernel,
    isotropic_dipole_tensors,
)
from dnqsm.training import (
    TrainConfig,
    checkpoint_training_config,
    parse_args,
    run_training,
)
from tests.helpers import make_dataset


class VoxelSizeSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def dataset(self, root, **kwargs):
        return TrainingVolumes(
            training_records(make_dataset(root))[:3],
            root / "cache",
            chi_scale_mode="rms_mixture",
            **kwargs,
        )

    def test_isotropic_uniform_draws_are_seeded_independently(self):
        seeds = list(range(2048))
        before = torch.random.get_rng_state().clone()
        sizes = sample_voxel_sizes(seeds, 0.5, 1)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(sizes.shape, (2048, 3))
        self.assertTrue(bool(((sizes >= 0.5) & (sizes <= 1)).all()))
        self.assertTrue(bool((sizes == sizes[:, :1]).all()))
        self.assertLess(abs(float(sizes[:, 0].mean()) - 0.75), 0.02)
        torch.testing.assert_close(
            sample_voxel_sizes(seeds, 0.5, 1), sizes, rtol=0, atol=0
        )
        torch.testing.assert_close(
            sample_voxel_sizes([42], 0.5, 1)[0], sizes[42], rtol=0, atol=0
        )
        torch.testing.assert_close(
            sample_voxel_sizes(seeds, 1, 1), torch.ones_like(sizes), rtol=0, atol=0
        )

    def test_batched_kernels_match_requested_spacing_and_numerical_isotropic_invariance(
        self,
    ):
        sizes = sample_voxel_sizes([3, 5, 8], 0.5, 1)
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

    def test_no_resampling_or_shared_rng_changes_and_batch_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.dataset(Path(temp), seed=9)
            batch = default_collate([dataset[0], dataset[1], dataset[2]])
            synth = ShapeFieldSynthesizer("cpu", capacity=2)
            baseline = synth(batch, background_mode="ratio")
            random = synth(
                batch, background_mode="ratio", voxel_size_min=0.5, voxel_size_max=1
            )
            self.assertEqual(random["chi"].shape, baseline["chi"].shape)
            for key in ("chi", "mask"):
                torch.testing.assert_close(random[key], baseline[key], rtol=0, atol=0)
            for key in ("local", "local_full", "noise", "background", "phase"):
                torch.testing.assert_close(
                    random[key], baseline[key], rtol=2e-5, atol=1e-8
                )
            self.assertEqual(random["voxel_size"].shape, (3, 3))
            self.assertEqual(random["dipole"].shape[0], 3)
            torch.testing.assert_close(
                random["local"],
                apply_dipole(random["chi"], random["dipole"]) * random["mask"],
            )
            for index in range(3):
                single = synth(
                    default_collate([dataset[index]]),
                    background_mode="ratio",
                    voxel_size_min=0.5,
                    voxel_size_max=1,
                )
                for key in ("voxel_size", "dipole", "phase", "noise", "background"):
                    torch.testing.assert_close(
                        single[key][0], random[key][index], rtol=2e-5, atol=1e-8
                    )
            for epoch in range(10):
                dataset.set_epoch(epoch)
                synth(
                    default_collate([dataset[0]]), voxel_size_min=0.5, voxel_size_max=1
                )
            self.assertEqual(len(synth.cache), 1)  # No cache entry per random spacing.

    def test_patch_sampling_retains_spacing_and_matching_full_and_patch_kernels(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.dataset(Path(temp), seed=11)
            batch = default_collate([dataset[0], dataset[1]])
            synth = ShapeFieldSynthesizer("cpu")
            options = dict(voxel_size_min=0.5, voxel_size_max=1)
            full = synth(batch, **options)
            patch = synth(batch, patch_size=8, **options)
            torch.testing.assert_close(
                patch["voxel_size"], full["voxel_size"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                patch["source_dipole"], full["dipole"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                patch["dipole"],
                isotropic_dipole_tensors((8, 8, 8), patch["voxel_size"]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                simulated_forward_field(patch["chi"], patch),
                patch["local_full"],
                rtol=0,
                atol=0,
            )
            for i, (d, h, w) in enumerate(patch["patch_start"]):
                torch.testing.assert_close(
                    patch["phase"][i],
                    full["phase"][i, :, d : d + 8, h : h + 8, w : w + 8],
                    rtol=0,
                    atol=0,
                )

    def test_epoch_variation_and_fixed_validation_view_draws(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root))[:2]
            train = TrainingVolumes(
                records, root / "cache", seed=13, chi_scale_mode="rms_mixture"
            )
            validation = TrainingVolumes(
                records, root / "cache", seed=13, validation=True, views=2
            )
            synth = ShapeFieldSynthesizer("cpu")

            def draw(dataset):
                return synth(
                    default_collate([dataset[0]]), voxel_size_min=0.5, voxel_size_max=1
                )["voxel_size"]

            a, b = draw(train), draw(validation)
            train.set_epoch(3)
            validation.set_epoch(3)
            self.assertFalse(torch.equal(draw(train), a))
            torch.testing.assert_close(draw(validation), b, rtol=0, atol=0)

    def test_configuration_validation_and_legacy_defaults(self):
        active = parse_args([], use_default_profile=True)
        self.assertEqual((active.voxel_size_min, active.voxel_size_max), (0.5, 1))
        legacy = checkpoint_training_config({})
        self.assertEqual((legacy["voxel_size_min"], legacy["voxel_size_max"]), (1, 1))
        self.assertEqual(
            parse_args(
                ["--voxel-size-min", "0.65", "--voxel-size-max", "0.65"]
            ).voxel_size_min,
            0.65,
        )
        for bounds in (
            (0, 1),
            (-1, 1),
            (1, 0.5),
            (float("nan"), 1),
            (0.5, float("inf")),
        ):
            with self.assertRaises(ValueError):
                validate_voxel_size_range(*bounds)
        for sizes in (
            torch.tensor([[0.5, 0.6, 0.5]]),
            torch.ones(3),
            torch.tensor([[0.0, 0.0, 0.0]]),
        ):
            with self.assertRaises(ValueError):
                isotropic_dipole_tensors((8, 8, 8), sizes)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.json"
            path.write_text(json.dumps({"epochs": 2}))
            old = parse_args(["--config", str(path)])
            self.assertEqual((old.voxel_size_min, old.voxel_size_max), (1, 1))

    def test_random_spacing_training_resumes_exactly_and_logs_policy(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = make_dataset(root)
            config = TrainConfig(
                data_dir=str(data),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "whole"),
                device="cpu",
                cuda_memory_gb=0,
                epochs=2,
                filter_pretrain_epochs=0,
                training_schedule="joint",
                features=(2, 4),
                batch_size=3,
                num_workers=0,
                threads=2,
                bf16=False,
                comet=False,
                evaluate_external=False,
                val_views=1,
                voxel_size_min=0.5,
                voxel_size_max=1,
            )
            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                whole = run_training(config)
                part = run_training(
                    replace(config, run_dir=str(root / "part"), max_steps=1)
                )
                resumed = parse_args(
                    ["--resume", str(part / "training_last.pt"), "--max-steps", "0"]
                )
                run_training(resumed)
            self.assertEqual((resumed.voxel_size_min, resumed.voxel_size_max), (0.5, 1))
            a, b = [
                torch.load(p / "training_last.pt", weights_only=True)
                for p in (whole, part)
            ]
            for name in a["model"]:
                torch.testing.assert_close(
                    a["model"][name], b["model"][name], rtol=0, atol=0
                )
            events = [
                json.loads(line)
                for line in (whole / "metrics.jsonl").read_text().splitlines()
            ]
            started = next(row for row in events if row["event"] == "started")
            self.assertEqual(
                (started["voxel_size_min"], started["voxel_size_max"]), (0.5, 1)
            )
            self.assertTrue(
                all(
                    0.5 <= row["voxel_size_mm"] <= 1
                    for row in events
                    if row["event"] == "train"
                )
            )
            with self.assertRaisesRegex(ValueError, "Resume configuration changed"):
                run_training(replace(resumed, voxel_size_min=0.6))
