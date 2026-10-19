import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import default_collate

from dnqsm.data import FieldSynthesizer, TrainingVolumes, training_records
from dnqsm.losses import (
    LossNormalization,
    normalize_loss_terms,
    reconstruction_loss,
)
from dnqsm.model import Reconstruction
from dnqsm.physics import apply_dipole, dipole_tensor, masked_mean
from dnqsm.training import (
    TrainConfig,
    background_options,
    parse_args,
    run_training,
    validate,
)
from tests.helpers import make_dataset

PROFILE = (
    Path(__file__).resolve().parents[1] / "configs/balanced_local_total_100epochs.json"
)


class BalancedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cap_is_per_sample_detached_and_changes_only_mse(self):
        local = torch.full((2, 1, 4, 4, 4), 0.01, requires_grad=True)
        chi = torch.full_like(local, 0.03, requires_grad=True)
        mask = torch.ones_like(local)
        field = (
            local.detach() * torch.tensor([1.0, 1000.0]).reshape(2, 1, 1, 1, 1)
        ).requires_grad_()
        raw = {
            k: torch.ones(2, 1, requires_grad=True)
            for k in ("physics", "local", "chi", "extension")
        }
        baseline = normalize_loss_terms(raw, local, chi, mask, LossNormalization())
        actual = normalize_loss_terms(
            raw,
            local,
            chi,
            mask,
            LossNormalization(mse_input_ratio_cap=10),
            input_field=field,
        )
        torch.testing.assert_close(actual["physics"][0], baseline["physics"][0])
        torch.testing.assert_close(actual["physics"][1], baseline["physics"][1] / 10000)
        for key in ("local", "chi", "extension"):
            torch.testing.assert_close(actual[key], baseline[key], rtol=0, atol=0)
        factor = 7
        scaled_raw = {
            k: v * factor ** (2 if k == "physics" else 1) for k, v in raw.items()
        }
        scaled = normalize_loss_terms(
            scaled_raw,
            local * factor,
            chi * factor,
            mask,
            LossNormalization(mse_input_ratio_cap=10),
            input_field=field * factor,
        )
        for key in actual:
            torch.testing.assert_close(scaled[key], actual[key])
        sum(v.mean() for v in actual.values()).backward()
        self.assertIsNone(local.grad)
        self.assertIsNone(chi.grad)
        self.assertIsNone(field.grad)
        self.assertTrue(
            all(
                v.grad is not None and bool(v.grad.isfinite().all())
                for v in raw.values()
            )
        )
        with self.assertRaisesRegex(ValueError, "input field"):
            normalize_loss_terms(
                raw, local, chi, mask, LossNormalization(mse_input_ratio_cap=10)
            )
        for invalid in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LossNormalization(mse_input_ratio_cap=invalid)

    def test_reconstruction_api_caps_mse_and_remains_batch_equivalent(self):
        g = torch.Generator().manual_seed(17)
        chi = torch.randn(2, 1, 8, 8, 8, generator=g) * 0.03
        mask = torch.ones_like(chi)
        dipole = dipole_tensor(chi.shape[-3:])
        local = apply_dipole(chi, dipole)
        field = local * torch.tensor([1.0, 1000.0]).reshape(2, 1, 1, 1, 1)
        predicted_local = (local * 0.8).requires_grad_()
        predicted_chi = (chi * 0.7).requires_grad_()
        output = Reconstruction(
            predicted_local, predicted_chi, predicted_chi * 0, predicted_chi
        )
        norm = LossNormalization(mse_input_ratio_cap=10)
        loss, terms = reconstruction_loss(
            output, local, chi, mask, dipole, normalization=norm, input_field=field
        )
        singles = []
        for i in range(2):
            single = Reconstruction(*(v[i : i + 1] for v in vars(output).values()))
            singles.append(
                reconstruction_loss(
                    single,
                    local[i : i + 1],
                    chi[i : i + 1],
                    mask[i : i + 1],
                    dipole,
                    normalization=norm,
                    input_field=field[i : i + 1],
                )[0]
            )
        torch.testing.assert_close(loss, torch.stack(singles).mean())
        loss.backward()
        self.assertGreater(float(predicted_local.grad.abs().sum()), 0)
        self.assertGreater(float(predicted_chi.grad.abs().sum()), 0)
        zero = torch.zeros_like(local)
        empty_loss, _ = reconstruction_loss(
            Reconstruction(zero, zero, zero, zero),
            zero,
            zero,
            mask,
            dipole,
            normalization=norm,
            input_field=zero,
        )
        self.assertTrue(bool(empty_loss.isfinite()))

    def test_noise_range_changes_only_noise_and_is_batch_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = TrainingVolumes(
                training_records(make_dataset(root, shape=(16, 16, 16))),
                root / "cache",
                chi_scale_mode="rms_mixture",
            )
            batch = default_collate([dataset[i] for i in range(8)])
            synth = FieldSynthesizer((16, 16, 16), "cpu")
            options = background_options(TrainConfig())
            original = synth(batch, **options)
            changed = synth(batch, **{**options, "snr_min": 5.0})
            for key in ("chi", "local", "local_full", "background"):
                torch.testing.assert_close(changed[key], original[key], rtol=0, atol=0)

            def snr(data):
                return (
                    masked_mean(data["local"].square(), data["mask"])
                    / masked_mean(data["noise"].square(), data["mask"])
                ).sqrt()

            expected = 5 + 95 * (snr(original) - 40) / 60
            torch.testing.assert_close(snr(changed), expected, rtol=2e-5, atol=2e-5)
            self.assertTrue(bool(((snr(changed) >= 5) & (snr(changed) <= 100)).all()))
            fixed = synth(batch, **{**options, "snr_min": 5.0, "snr_max": 5.0})
            torch.testing.assert_close(snr(fixed), torch.full((8, 1), 5.0))
            split = []
            for i in range(8):
                part = {k: v[i : i + 1] for k, v in batch.items()}
                split.append(synth(part, **{**options, "snr_min": 5.0})["phase"])
            delta = (torch.cat(split) - changed["phase"]).flatten(1).norm(dim=1)
            self.assertTrue(
                bool((delta / changed["phase"].flatten(1).norm(dim=1) < 3e-5).all())
            )
            replay = synth(batch, **options)
            torch.testing.assert_close(
                replay["phase"], original["phase"], rtol=0, atol=0
            )
            for low, high in (
                (0, 100),
                (10, 5),
                (float("nan"), 100),
                (5, float("inf")),
            ):
                with self.assertRaisesRegex(ValueError, "SNR"):
                    synth(batch, snr_min=low, snr_max=high)

    def test_profile_keeps_reference_validation_and_historical_defaults(self):
        config = parse_args(["--config", str(PROFILE)])
        self.assertEqual(
            (
                config.clean_probability,
                config.weak_probability,
                config.strong_probability,
            ),
            (0.5, 0.1, 0.05),
        )
        self.assertAlmostEqual(
            1
            - config.clean_probability
            - config.weak_probability
            - config.strong_probability,
            0.35,
        )
        self.assertEqual(
            (
                config.background_curriculum_epochs,
                config.snr_min,
                config.snr_max,
                config.mse_input_ratio_cap,
            ),
            (20, 5, 100, 10),
        )
        reference = background_options(TrainConfig())
        for key, value in config.validation_synthesis.items():
            self.assertEqual(value, reference[key])
        legacy = asdict(TrainConfig())
        for name in (
            "mse_input_ratio_cap",
            "snr_min",
            "snr_max",
            "validation_synthesis",
        ):
            legacy.pop(name)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "old.pt"
            torch.save(dict(config=legacy), checkpoint)
            resumed = parse_args(["--resume", str(checkpoint)])
            self.assertEqual(
                (
                    resumed.mse_input_ratio_cap,
                    resumed.snr_min,
                    resumed.snr_max,
                    resumed.validation_synthesis,
                ),
                (0, 40, 100, None),
            )
        for invalid in (
            replace(config, normalize_loss=False),
            replace(config, snr_min=0),
            replace(config, snr_min=101),
            replace(config, validation_synthesis="invalid"),
        ):
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_capped_training_and_fixed_validation_resume_exactly(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = replace(
                parse_args(["--config", str(PROFILE)]),
                data_dir=str(make_dataset(root)),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "whole"),
                device="cpu",
                features=(2, 4),
                epochs=2,
                filter_pretrain_epochs=0,
                batch_size=2,
                max_train_volumes=2,
                val_views=1,
                accumulation_steps=1,
                crop_probability=0,
                num_workers=0,
                threads=2,
                comet=False,
                evaluate_external=False,
                external_eval_every=0,
                bf16=False,
                gradient_every=1,
            )
            calls = []

            def record_validation(*args, **kwargs):
                calls.append(kwargs["synthesis_options"])
                return validate(*args, **kwargs)

            with (
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                patch("dnqsm.training.validate", side_effect=record_validation),
            ):
                whole = run_training(config)
                partial = run_training(
                    replace(config, run_dir=str(root / "partial"), max_steps=1)
                )
                resumed = parse_args(
                    ["--resume", str(partial / "training_last.pt"), "--max-steps", "0"]
                )
                run_training(resumed)
            first = torch.load(whole / "training_last.pt", weights_only=True)
            second = torch.load(partial / "training_last.pt", weights_only=True)
            self.assertEqual(first["validation_profile"], "reference")
            self.assertEqual(first["config"]["mse_input_ratio_cap"], 10)
            for name in first["model"]:
                torch.testing.assert_close(
                    first["model"][name], second["model"][name], rtol=0, atol=0
                )
            self.assertEqual(first["scheduler"], second["scheduler"])
            self.assertTrue(
                all(
                    c == config.validation_synthesis
                    for c in calls
                    if c["background_mode"] == "mixture"
                )
            )
            rows = [
                json.loads(s)
                for s in (whole / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len([r for r in rows if r["event"] == "train"]), 2)
            self.assertTrue(
                all("local_nrmse" in r for r in rows if r["event"] == "validation")
            )
            for change in (
                dict(mse_input_ratio_cap=0),
                dict(snr_min=6),
                dict(validation_synthesis=None),
            ):
                with self.assertRaisesRegex(ValueError, "Resume configuration changed"):
                    run_training(replace(resumed, **change))


if __name__ == "__main__":
    unittest.main()
