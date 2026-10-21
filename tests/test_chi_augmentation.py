import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from dnqsm.chi_augmentation import (
    PROFILES,
    ChiAugmentationConfig,
    augment_chi,
    monotone_transport,
)
from dnqsm.data import FieldSynthesizer, TrainingVolumes, training_records
from dnqsm.physics import apply_dipole
from dnqsm.training import (
    TrainConfig,
    build_datasets,
    build_shape_stress_dataset,
    checkpoint_training_config,
    parse_args,
    run_training,
)
from tests.helpers import make_dataset


class ChiAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_configuration_validation_and_json_round_trip(self):
        for invalid in (
            {"version": 2},
            {"unknown": 1},
            {"bimodal_probability": 0.2},
            {
                "final_grid_rms": True,
                "bimodal_probability": 0.7,
                "central_probability": 0.4,
            },
            {"max_slope": float("nan")},
            {"cdf_samples": 2},
            {"central_std": [0, 0.1]},
            {"central_alpha": [-0.1, 1]},
            {"bimodal_negative_mean": [0.1, -0.1]},
            {"final_grid_rms": "false"},
            {"bimodal_negative_weight": [0.8, 1.0]},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ChiAugmentationConfig.from_dict(invalid)
        config = ChiAugmentationConfig(final_grid_rms=True, bimodal_probability=1 / 3)
        self.assertEqual(
            config,
            ChiAugmentationConfig.from_dict(json.loads(json.dumps(asdict(config)))),
        )
        cli = parse_args(
            ["--chi-augmentation", json.dumps(asdict(config)), "--chi-shape-validation"]
        )
        self.assertEqual(ChiAugmentationConfig.from_dict(cli.chi_augmentation), config)
        self.assertTrue(cli.chi_shape_validation)
        historical = checkpoint_training_config({})
        self.assertIsNone(historical["chi_augmentation"])
        self.assertFalse(historical["chi_shape_validation"])

    def test_disabled_policy_is_exact_and_source_files_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root))
            before = {
                p: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / "sources").glob("*.pt")
            }
            ordinary = TrainingVolumes(
                records,
                root / "cache",
                seed=37,
                chi_scale_mode="rms_mixture",
                crop_probability=0.5,
            )
            disabled = TrainingVolumes(
                records,
                root / "cache",
                seed=37,
                chi_scale_mode="rms_mixture",
                crop_probability=0.5,
                chi_augmentation={},
            )
            for epoch in [0, 2]:
                ordinary.set_epoch(epoch)
                disabled.set_epoch(epoch)
                for i in range(4):
                    a, b = ordinary[i], disabled[i]
                    for key in ["chi", "mask", "dilated", "crop_start"]:
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                    self.assertEqual(a["seed"], b["seed"])
            for p, digest in before.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)

    def test_monotone_continuous_transport_extends_and_preserves_tails(self):
        config = ChiAugmentationConfig(final_grid_rms=True)
        p = np.linspace(0.001, 0.999, 257)
        x = np.linspace(-0.08, 0.08, len(p))
        target = dict(
            weights=[0.35, 0.65, 0.0],
            means_ppm=[-0.035, 0.016, 0.1],
            stds_ppm=[0.006, 0.007, 0.02],
            alpha=1.0,
        )
        values = np.linspace(-2.0, 2.0, 10001)
        output, details = monotone_transport(values, p, x, target, config)
        self.assertTrue(np.all(np.diff(output) > 0))
        self.assertGreater(output[-1], 0.5)
        self.assertLess(output[0], -0.5)
        delta = 1e-7
        for edge in [x[0], x[-1]]:
            mapped, _ = monotone_transport(
                np.array([edge - delta, edge, edge + delta]), p, x, target, config
            )
            self.assertLess(float(np.max(np.abs(np.diff(mapped)))), 1e-5)
        self.assertEqual(details["source_unique_knots"], 257)
        # Quantile transport must not invent an ordering among tied voxels.
        tied, _ = monotone_transport(
            np.array([-0.01, -0.01, 0.02, 0.02]), p, x, target, config
        )
        self.assertEqual(tied[0], tied[1])
        self.assertEqual(tied[2], tied[3])

    def test_constant_map_falls_back_without_clipping_or_nonfinite_values(self):
        chi = torch.full((1, 12, 12, 12), 0.03)
        mask = torch.ones_like(chi, dtype=torch.bool)
        config = ChiAugmentationConfig(final_grid_rms=True, bimodal_probability=1.0)
        mapped, meta = augment_chi(chi, mask, config, 1, "qsm:a")
        torch.testing.assert_close(mapped, chi, rtol=0, atol=0)
        self.assertTrue(meta["fallback"])
        self.assertEqual(meta["profile"], "baseline")
        self.assertEqual(meta["requested_profile"], "bimodal")

    def test_final_grid_rms_masks_and_paired_field_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root, shape=(20, 20, 20)))
            policy = ChiAugmentationConfig(final_grid_rms=True, bimodal_probability=1.0)
            baseline = TrainingVolumes(
                records,
                root / "cache",
                seed=19,
                chi_scale_mode="rms_mixture",
                crop_probability=1.0,
            )
            dataset = TrainingVolumes(
                records,
                root / "cache",
                seed=19,
                chi_scale_mode="rms_mixture",
                crop_probability=1.0,
                chi_augmentation=policy,
            )
            before, sample = baseline[0], dataset[0]
            self.assertEqual(tuple(sample["chi"].shape), (1, 160, 160, 160))
            self.assertEqual(sample["seed"], before["seed"])
            self.assertEqual(sample["chi_rms_target"], before["chi_rms_target"])
            torch.testing.assert_close(sample["mask"], before["mask"], rtol=0, atol=0)
            self.assertAlmostEqual(
                sample["chi_rms_final"], sample["chi_rms_target"], places=7
            )
            self.assertTrue(bool((sample["chi"][~sample["mask"]] == 0).all()))
            self.assertEqual(sample["chi_profile"], "bimodal")
            batch = default_collate([sample])
            data = FieldSynthesizer((160, 160, 160), "cpu")(
                batch,
                background_mode="mixture",
                clean_probability=1.0,
                strong_probability=0.0,
                weak_probability=0.0,
            )
            torch.testing.assert_close(
                data["local"], apply_dipole(data["chi"], data["dipole"]) * data["mask"]
            )
            torch.testing.assert_close(
                data["phase"], data["local"] + data["background"] + data["noise"]
            )

            def rms(x):
                return (x[data["mask"].bool()].square().mean()).sqrt()

            self.assertTrue(40 <= float(rms(data["local"]) / rms(data["noise"])) <= 100)
            # Synthetic maps retain their shape, with only the final scalar changed.
            synthetic = dataset[6]
            self.assertEqual(synthetic["chi_profile"], "baseline")
            self.assertAlmostEqual(
                synthetic["chi_rms_final"], synthetic["chi_rms_target"], places=7
            )

    def test_worker_epoch_and_global_rng_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root, shape=(20, 20, 20)))
            config = ChiAugmentationConfig(
                final_grid_rms=True, bimodal_probability=0.5, central_probability=0.5
            )
            dataset = TrainingVolumes(
                records,
                root / "cache",
                seed=11,
                chi_scale_mode="rms_mixture",
                chi_augmentation=config,
            )
            expected = dataset[0]
            np.random.seed(999)
            torch.manual_seed(999)
            torch.testing.assert_close(
                dataset[0]["chi"], expected["chi"], rtol=0, atol=0
            )
            sequential = list(DataLoader(dataset, batch_size=2, num_workers=0))
            workers = list(DataLoader(dataset, batch_size=2, num_workers=2))
            for a, b in zip(sequential, workers, strict=True):
                torch.testing.assert_close(a["chi"], b["chi"], rtol=0, atol=0)
                self.assertEqual(a["chi_parameters"], b["chi_parameters"])
            dataset.set_epoch(1)
            self.assertFalse(torch.equal(dataset[0]["chi"], expected["chi"]))

    def test_fixed_validation_and_separate_shape_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_dataset(root, shape=(20, 20, 20))
            run = root / "run"
            run.mkdir()
            cfg = TrainConfig(
                data_dir=str(source),
                cache_dir=str(root / "cache"),
                val_fraction=0.3,
                chi_augmentation={"final_grid_rms": True, "bimodal_probability": 1.0},
            )
            training, validation, _ = build_datasets(cfg, run)
            control = TrainingVolumes(
                validation.records,
                root / "cache",
                validation=True,
                views=cfg.val_views,
                seed=cfg.seed + 100000,
                chi_scale_mode="rms_mixture",
            )
            for i in range(3):
                torch.testing.assert_close(
                    validation[i]["chi"], control[i]["chi"], rtol=0, atol=0
                )
            stress = build_shape_stress_dataset(cfg, validation)
            self.assertEqual(
                [stress[i]["chi_profile"] for i in range(3)], list(PROFILES)
            )
            first = stress[1]
            stress.set_epoch(9)
            torch.testing.assert_close(stress[1]["chi"], first["chi"], rtol=0, atol=0)
            self.assertTrue(training.chi_augmentation.enabled)

    def test_profile_probabilities_and_synthetic_exclusion(self):
        config = ChiAugmentationConfig(
            final_grid_rms=True, bimodal_probability=1 / 3, central_probability=1 / 3
        )
        chi = torch.linspace(-0.06, 0.06, 1024).reshape(1, 8, 8, 16)
        mask = torch.ones_like(chi, dtype=torch.bool)
        profiles = []
        # Isolate cheap profile selection; transport itself is tested above.
        with patch(
            "dnqsm.chi_augmentation.monotone_transport",
            side_effect=lambda v, *args: (v, {}),
        ):
            for seed in range(600):
                _, meta = augment_chi(chi, mask, config, seed, "qsm:a")
                profiles.append(meta["profile"])
        for profile in PROFILES:
            self.assertTrue(0.27 < profiles.count(profile) / 600 < 0.4)
        unchanged, meta = augment_chi(chi, mask, config, 0, "synthetic:a")
        torch.testing.assert_close(unchanged, chi, rtol=0, atol=0)
        self.assertEqual(meta["profile"], "baseline")

    def test_experiment_configs_differ_only_in_chi_policy(self):
        root = Path(__file__).resolve().parents[1]
        configs = [
            parse_args(["--config", str(root / f"configs/chi_augmentation_A{i}.json")])
            for i in range(4)
        ]
        common = asdict(configs[0])
        common.pop("chi_augmentation")
        for config in configs[1:]:
            other = asdict(config)
            other.pop("chi_augmentation")
            self.assertEqual(other, common)
        self.assertIsNone(configs[0].chi_augmentation)
        self.assertEqual(configs[3].chi_augmentation["bimodal_probability"], 1 / 3)

    def test_augmented_training_resume_and_shape_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_dataset(root, shape=(16, 16, 16))
            settings = dict(
                data_dir=str(source),
                cache_dir=str(root / "cache"),
                device="cpu",
                bf16=False,
                threads=2,
                num_workers=0,
                features=(2, 4),
                epochs=1,
                filter_pretrain_epochs=0,
                batch_size=2,
                accumulation_steps=1,
                val_fraction=0.3,
                evaluate_external=False,
                gradient_every=0,
                chi_shape_validation=True,
                chi_augmentation={
                    "final_grid_rms": True,
                    "bimodal_probability": 0.5,
                    "central_probability": 0.5,
                },
            )
            whole, part = root / "whole", root / "part"
            with (
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                redirect_stdout(io.StringIO()),
            ):
                run_training(TrainConfig(**settings, run_dir=str(whole)))
                run_training(TrainConfig(**settings, run_dir=str(part), max_steps=1))
                resume = parse_args(
                    ["--resume", str(part / "training_last.pt"), "--max-steps", "0"]
                )
                with self.assertRaisesRegex(ValueError, "chi_augmentation"):
                    changed = TrainConfig(
                        **{**asdict(resume), "chi_augmentation": None}
                    )
                    run_training(changed)
                run_training(resume)
            a, b = [
                torch.load(p / "training_last.pt", weights_only=True)
                for p in (whole, part)
            ]
            for key in a["model"]:
                torch.testing.assert_close(
                    a["model"][key], b["model"][key], rtol=0, atol=0
                )
            self.assertEqual(a["scheduler"], b["scheduler"])

            def events(path, event):
                return [
                    {k: v for k, v in r.items() if k != "timestamp"}
                    for line in (path / "metrics.jsonl").read_text().splitlines()
                    if (r := json.loads(line))["event"] == event
                ]

            augmentation = events(whole, "augmentation")
            self.assertEqual(augmentation, events(part, "augmentation"))
            self.assertEqual(
                sum(augmentation[0][f"chi_{p}_samples"] for p in PROFILES),
                augmentation[0]["samples"],
            )
            shape = events(whole, "validation_shape")[0]
            self.assertTrue(all(f"{p}_chi_nrmse" in shape for p in PROFILES))

    def test_historical_mid_epoch_resume_recovers_baseline_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = make_dataset(root, shape=(16, 16, 16))
            run = root / "run"
            config = TrainConfig(
                data_dir=str(source),
                cache_dir=str(root / "cache"),
                run_dir=str(run),
                device="cpu",
                bf16=False,
                threads=2,
                num_workers=0,
                features=(2, 4),
                epochs=1,
                filter_pretrain_epochs=0,
                batch_size=2,
                max_steps=1,
                val_fraction=0.3,
                evaluate_external=False,
                gradient_every=0,
            )
            with (
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                redirect_stdout(io.StringIO()),
            ):
                run_training(config)
                checkpoint = run / "training_last.pt"
                saved = torch.load(checkpoint, weights_only=True)
                saved.pop("epoch_chi_counts")
                saved["config"].pop("chi_augmentation")
                saved["config"].pop("chi_shape_validation")
                torch.save(saved, checkpoint)
                run_training(
                    parse_args(["--resume", str(checkpoint), "--max-steps", "0"])
                )
            event = next(
                r
                for line in (run / "metrics.jsonl").read_text().splitlines()
                if (r := json.loads(line))["event"] == "augmentation"
            )
            self.assertEqual(event["chi_baseline_samples"], event["samples"])
            self.assertEqual(event["chi_bimodal_samples"], 0)


if __name__ == "__main__":
    unittest.main()
