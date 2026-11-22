import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dnqsm.data import ShapeFieldSynthesizer, TrainingVolumes, training_records
from dnqsm.lesion_augmentation import (
    LesionAugmentationConfig,
    apply_lesion_dephasing,
    augment_lesions,
)
from dnqsm.training import (
    TrainConfig,
    checkpoint_training_config,
    parse_args,
    run_training,
)
from tests.helpers import make_dataset


class LesionDephasingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def brain(self):
        mask = torch.zeros(1, 24, 24, 24, dtype=torch.bool)
        mask[:, 2:-2, 2:-2, 2:-2] = 1
        return mask.float() * 0.03, mask

    def policy(self, **kwargs):
        config = LesionAugmentationConfig(
            probability=1,
            count=(1, 1),
            sphere_radius_voxels=(1.5, 2),
            ellipsoid_semiaxes_voxels=(1.5, 2),
            ellipsoid_transition_voxels=(0.5, 0.5),
            dephasing_probability=1,
        )
        return replace(config, **kwargs)

    def test_thirty_percent_mixture_selects_one_core_voxel_per_affected_lesion(self):
        chi, mask = self.brain()
        policy = self.policy(dephasing_probability=0.3)
        count, loss, spikes = 0, 0, 0
        for seed in range(600):
            result, meta = augment_lesions(chi, mask, policy, seed, "qsm:test")
            self.assertEqual(meta["count"], 1)
            self.assertLessEqual(len(meta["dephasing_events"]), 1)
            for event in meta["dephasing_events"]:
                count += 1
                loss += event["kind"] == "signal_loss"
                spikes += event["kind"] == "phase_spike"
                voxel = event["voxel"]
                lesion = meta["parameters"][0]
                delta = np.array(lesion["rotation"]).T @ (
                    np.array(voxel) - np.array(lesion["center_voxels"])
                )
                self.assertLessEqual(
                    np.sum((delta / lesion["semiaxes_voxels"]) ** 2), 1 + 1e-12
                )
                self.assertTrue(bool(result[(0, *voxel)].abs() > 0.2))
                self.assertTrue(bool(mask[(0, *voxel)]))
        self.assertTrue(0.25 <= count / 600 <= 0.35, (count, loss, spikes))
        self.assertTrue(0.35 <= loss / count <= 0.65, (count, loss, spikes))
        self.assertEqual(loss + spikes, count)

    def test_selection_never_changes_geometry_susceptibility_or_other_seed_streams(
        self,
    ):
        chi, mask = self.brain()
        off = self.policy(count=(3, 3), dephasing_probability=0)
        on = replace(off, dephasing_probability=1)
        a, ma = augment_lesions(chi, mask, off, 42, "qsm:test")
        rng = torch.get_rng_state().clone()
        b, mb = augment_lesions(chi, mask, on, 42, "qsm:test")
        c, mc = augment_lesions(chi, mask, on, 42, "qsm:test")
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(b, c, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(mb, mc)
        self.assertEqual(len(mb["dephasing_events"]), 3)
        for plain, corrupted in zip(ma["parameters"], mb["parameters"], strict=True):
            self.assertEqual(
                plain, {k: v for k, v in corrupted.items() if k != "dephasing"}
            )

    def batch(self, seed=19, signal_loss=0.5):
        chi, mask = self.brain()
        result, meta = augment_lesions(
            chi,
            mask,
            self.policy(dephasing_signal_loss_probability=signal_loss),
            seed,
            "qsm:test",
        )
        return dict(
            chi=result[None],
            mask=mask[None],
            dilated=mask[None],
            seed=torch.tensor([seed]),
            dephasing_parameters=[json.dumps(meta["dephasing_events"])],
        ), meta["dephasing_events"][0]

    def test_both_signal_types_corrupt_only_measurements_and_preserve_clean_targets(
        self,
    ):
        for mode in (0, 1):
            batch, event = self.batch(signal_loss=mode)
            synthesizer = ShapeFieldSynthesizer("cpu")
            options = dict(
                background_mode="mixture",
                clean_probability=1,
                strong_probability=0,
                weak_probability=0,
            )
            expected = synthesizer(dict(batch, dephasing_parameters=["[]"]), **options)
            actual = synthesizer(batch, **options)
            for key in (
                "chi",
                "mask",
                "dipole",
                "local",
                "local_full",
                "noise",
                "background",
            ):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            error = actual["phase"] - expected["phase"]
            self.assertEqual(int((error != 0).sum()), 1)
            index = (0, 0, *event["voxel"])
            self.assertAlmostEqual(
                float(error[index]), event["field_error_ppm"], places=7
            )
            self.assertAlmostEqual(
                math.atan2(
                    event["relative_signal_imag"], event["relative_signal_real"]
                ),
                event["phase_error_radians"],
            )
            self.assertAlmostEqual(
                math.hypot(
                    event["relative_signal_real"], event["relative_signal_imag"]
                ),
                event["relative_magnitude"],
            )
            factor = self.policy().phase_radians_per_ppm
            self.assertAlmostEqual(
                event["field_error_ppm"] * factor, event["phase_error_radians"]
            )
            if mode:
                self.assertEqual(event["kind"], "signal_loss")
                self.assertTrue(0.01 <= event["relative_magnitude"] <= 0.1)
            else:
                self.assertEqual(event["kind"], "phase_spike")
                self.assertEqual(event["relative_magnitude"], 1)
                self.assertTrue(
                    math.pi / 2 <= abs(event["phase_error_radians"]) <= math.pi
                )
            torch.testing.assert_close(
                actual["phase"],
                actual["local"]
                + actual["background"]
                + actual["noise"]
                + actual["dephasing"],
            )

    def test_disabled_identity_patch_alignment_and_invalid_coordinates(self):
        batch, event = self.batch()
        synth = ShapeFieldSynthesizer("cpu")
        full = synth(batch)
        patch_data = synth(batch, patch_size=8)
        d, h, w = patch_data["patch_start"][0]
        for key in ("phase", "dephasing"):
            torch.testing.assert_close(
                patch_data[key],
                full[key][..., d : d + 8, h : h + 8, w : w + 8],
                rtol=0,
                atol=0,
            )
        unchanged, artifact = apply_lesion_dephasing(
            full["phase"], full["chi"], full["mask"], ["[]"]
        )
        self.assertIs(unchanged, full["phase"])
        self.assertIsNone(artifact)
        for voxel in ([99, 0, 0], [0, 0, 0]):
            with self.assertRaises(ValueError):
                apply_lesion_dephasing(
                    full["phase"],
                    full["chi"],
                    full["mask"],
                    [json.dumps([dict(event, voxel=voxel)])],
                )

    def test_validation_is_unchanged_and_historical_configuration_defaults_to_off(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root, shape=(24, 24, 24)))
            common = dict(
                validation=True, views=1, seed=9, chi_scale_mode="rms_mixture"
            )
            plain = TrainingVolumes(records, root / "cache", **common)
            enabled = TrainingVolumes(
                records, root / "cache", lesion_augmentation=self.policy(), **common
            )
            for index in range(2):
                a, b = plain[index], enabled[index]
                torch.testing.assert_close(a["chi"], b["chi"], rtol=0, atol=0)
                self.assertEqual(b["dephasing_parameters"], "[]")
                self.assertEqual(b["lesion_dephasing_count"], 0)
        legacy = checkpoint_training_config(
            {"lesion_augmentation": {"probability": 0.5}}
        )
        self.assertEqual(legacy["lesion_augmentation"]["dephasing_probability"], 0)
        self.assertEqual(
            parse_args([], use_default_profile=True).lesion_augmentation[
                "dephasing_probability"
            ],
            0.3,
        )
        for kwargs in (
            dict(dephasing_probability=2),
            dict(dephasing_signal_loss_probability=-1),
            dict(dephasing_relative_magnitude=(0, 0.1)),
            dict(dephasing_relative_magnitude=(0.1, 2)),
            dict(dephasing_spike_radians=(1, 4)),
            dict(dephasing_b0_tesla=0),
            dict(dephasing_echo_time_ms=float("nan")),
        ):
            with self.assertRaises(ValueError):
                self.policy(**kwargs)

    def test_training_resume_reproduces_corruptions_and_lesion_counts(self):
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
                lesion_augmentation=self.policy().to_dict(),
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
            a, b = [
                torch.load(p / "training_last.pt", weights_only=True)
                for p in (whole, part)
            ]
            for name in a["model"]:
                torch.testing.assert_close(
                    a["model"][name], b["model"][name], rtol=0, atol=0
                )

            def events(directory):
                rows = [
                    json.loads(line)
                    for line in (directory / "metrics.jsonl").read_text().splitlines()
                ]
                return [
                    row["lesion_dephasing_count"]
                    for row in rows
                    if row["event"] == "augmentation"
                ]

            self.assertEqual(events(whole), events(part))
            self.assertTrue(all(count > 0 for count in events(whole)))
