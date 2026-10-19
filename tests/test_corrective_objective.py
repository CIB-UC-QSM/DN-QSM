import copy
import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import default_collate

from dnqsm.data import FieldSynthesizer, TrainingVolumes, training_records
from dnqsm.diagnostics import loss_gradient_metrics, quality_totals, summarize_quality
from dnqsm.losses import (
    LOSS_VERSION,
    LossNormalization,
    LossWeights,
    normalize_loss_terms,
    per_sample_loss_terms,
    reconstruction_loss,
)
from dnqsm.model import DNQSM, ModelConfig, Reconstruction
from dnqsm.overfit import OverfitConfig
from dnqsm.physics import (
    apply_dipole,
    apply_multiplier,
    cutoff_kernels,
    dipole_tensor,
    masked_mean,
)
from dnqsm.runtime import load_trained_model
from dnqsm.training import (
    TrainConfig,
    parse_args,
    run_training,
    training_weights,
    validate,
)
from tests.helpers import make_dataset


class CorrectiveObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def example(self, count=2):
        generator = torch.Generator().manual_seed(71)
        chi = torch.randn(count, 1, 12, 12, 12, generator=generator) * 0.03
        mask = torch.zeros_like(chi)
        mask[..., 3:-3, 3:-3, 3:-3] = 1
        chi *= mask
        dipole = dipole_tensor(chi.shape[-3:])
        local_full = apply_dipole(chi, dipole)
        local = local_full * mask
        prediction = Reconstruction(local_full * 0.7, chi * 0.8, chi * 0.1, chi * 0.9)
        return prediction, chi, local, local_full, mask, dipole

    def test_local_loss_coefficient_is_fifty_percent_higher(self):
        weights = LossWeights()
        self.assertEqual(weights.local, 1.5)
        for name in ("forward", "physics", "chi", "extension"):
            self.assertEqual(getattr(weights, name), 1.0)
            self.assertEqual(weights.local, 1.5 * getattr(weights, name))
        root = Path(__file__).resolve().parents[1]
        configs = [TrainConfig(), OverfitConfig()]
        configs.extend(
            parse_args(["--config", str(root / "configs" / name)])
            for name in ("corrective_200epochs.json", "corrective_ablation.json")
        )
        for config in configs:
            for name, weight in vars(weights).items():
                self.assertEqual(getattr(config, "lambda_" + name), weight)
            self.assertEqual(getattr(config, "physics_warmup_epochs", 0), 0)
            self.assertEqual(getattr(config, "physics_warmup_steps", 0), 0)
        prediction, chi, local, full, mask, dipole = self.example()
        actual, terms = reconstruction_loss(
            prediction, local, chi, mask, dipole, local_full_target=full
        )
        equal_weight, _ = reconstruction_loss(
            prediction,
            local,
            chi,
            mask,
            dipole,
            replace(weights, local=1.0),
            local_full_target=full,
        )
        self.assertGreater(float(terms["local"]), 0)
        torch.testing.assert_close(actual, equal_weight + 0.5 * terms["local"])

    def test_normalized_losses_are_per_sample_and_amplitude_invariant(self):
        prediction, chi, local, full, mask, dipole = self.example()
        raw = per_sample_loss_terms(
            prediction, local, chi, mask, dipole, local_full_target=full
        )
        norm = LossNormalization()
        expected = normalize_loss_terms(raw, local, chi, mask, norm)
        field_energy = masked_mean(local.square(), mask)
        chi_rms = masked_mean(chi.square(), mask).sqrt()
        torch.testing.assert_close(expected["chi"], raw["chi"] / chi_rms)
        torch.testing.assert_close(expected["physics"], raw["physics"] / field_energy)
        factors = torch.tensor([0.2, 10.0]).reshape(2, 1, 1, 1, 1)
        scaled = Reconstruction(
            *(value * factors for value in vars(prediction).values())
        )
        scaled_raw = per_sample_loss_terms(
            scaled,
            local * factors,
            chi * factors,
            mask,
            dipole,
            local_full_target=full * factors,
        )
        actual = normalize_loss_terms(
            scaled_raw, local * factors, chi * factors, mask, norm
        )
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
        batch_loss, _ = reconstruction_loss(prediction, local, chi, mask, dipole)
        single_losses = []
        for index in range(2):
            single = Reconstruction(
                *(value[index : index + 1] for value in vars(prediction).values())
            )
            single_losses.append(
                reconstruction_loss(
                    single,
                    local[index : index + 1],
                    chi[index : index + 1],
                    mask[index : index + 1],
                    dipole,
                )[0]
            )
        torch.testing.assert_close(batch_loss, torch.stack(single_losses).mean())

    def test_floors_detach_targets_and_empty_exterior_is_finite(self):
        chi = torch.zeros(1, 1, 8, 8, 8, requires_grad=True)
        target = torch.zeros_like(chi, requires_grad=True)
        local = torch.ones_like(chi, requires_grad=True)
        output = Reconstruction(local, chi, chi * 0, chi)
        loss, terms = reconstruction_loss(
            output, target, target, torch.ones_like(chi), dipole_tensor((8, 8, 8))
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(terms["extension"].detach()), 0)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(bool(torch.isfinite(local.grad).all()))
        for kwargs in (dict(chi_floor=0), dict(field_floor=float("nan"))):
            with self.assertRaises(ValueError):
                LossNormalization(**kwargs)

    def test_field_extension_supervises_only_exterior(self):
        prediction, chi, local, full, mask, dipole = self.example(1)
        predicted_local = (full * mask).detach().requires_grad_()
        prediction.local_field = predicted_local
        raw = per_sample_loss_terms(
            prediction, local, chi, mask, dipole, local_full_target=full
        )
        self.assertGreater(float(raw["extension"].detach()), 0)
        raw["extension"].mean().backward()
        self.assertEqual(float(predicted_local.grad[mask == 1].abs().sum()), 0)
        self.assertGreater(float(predicted_local.grad[mask == 0].abs().sum()), 0)

    def test_tkd_uses_full_grid_field_and_legacy_loading_preserves_mask(self):
        _, chi, _, full, mask, dipole = self.example(1)
        model = DNQSM(ModelConfig((2, 4), full_grid_field=True))
        with torch.no_grad():
            model.filter.final.weight.zero_()
            model.filter.final.bias.fill_(1)
        result = model(full, mask, dipole)
        self.assertGreater(float(result.local_field[mask == 0].abs().sum()), 0)
        inverse, _ = cutoff_kernels(dipole, model.config.threshold)
        torch.testing.assert_close(
            result.tkd, apply_multiplier(result.local_field, inverse)
        )
        legacy = DNQSM(replace(model.config, full_grid_field=False)).eval()
        legacy.load_state_dict(model.state_dict())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            old_config = asdict(legacy.config)
            old_config.pop("full_grid_field")
            torch.save(
                dict(version=1, model_config=old_config, model=legacy.state_dict()),
                path,
            )
            loaded = load_trained_model(path)
            self.assertFalse(loaded.config.full_grid_field)
            torch.testing.assert_close(
                loaded(full, mask, dipole).chi, legacy(full, mask, dipole).chi
            )
        # Precision policy stays as requested: convolution outputs remain BF16
        # under autocast, with FP32 FFT/rescaling as in the existing pipeline.
        dtypes = []
        handle = model.filter.final.register_forward_hook(
            lambda module, args, value: dtypes.append(value.dtype)
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(full, mask, dipole)
        handle.remove()
        self.assertEqual(dtypes, [torch.bfloat16])

    def test_mixture_clean_strong_weak_and_curriculum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = training_records(make_dataset(root, shape=(16, 16, 16)))
            dataset = TrainingVolumes(
                records, root / "cache", chi_scale_mode="rms_mixture"
            )
            batch = default_collate([dataset[i] for i in range(8)])
            synth = FieldSynthesizer((16, 16, 16), "cpu")
            options = dict(
                background_mode="mixture",
                clean_probability=0,
                weak_probability=0,
                strong_probability=1,
            )
            strong = synth(batch, **options)
            ratio = (
                masked_mean(strong["local"].square(), strong["mask"]).sqrt()
                / masked_mean(strong["background"].square(), strong["mask"]).sqrt()
            )
            self.assertTrue(bool(((ratio >= 0.000999) & (ratio <= 0.03001)).all()))
            clean = synth(
                batch, **dict(options, clean_probability=1, strong_probability=0)
            )
            self.assertEqual(float(clean["background"].abs().sum()), 0)
            self.assertGreater(float(clean["noise"].abs().sum()), 0)
            weak = synth(
                batch, **dict(options, weak_probability=1, strong_probability=0)
            )
            ratio = (
                masked_mean(weak["local"].square(), weak["mask"]).sqrt()
                / masked_mean(weak["background"].square(), weak["mask"]).sqrt()
            )
            self.assertTrue(bool(((ratio >= 2.999) & (ratio <= 100.01)).all()))
            early = synth(batch, **options, curriculum_progress=0)
            ratio = (
                masked_mean(early["local"].square(), early["mask"]).sqrt()
                / masked_mean(early["background"].square(), early["mask"]).sqrt()
            )
            self.assertTrue(bool(((ratio >= 0.02999) & (ratio <= 3.001)).all()))
            torch.testing.assert_close(
                strong["local_full"], apply_dipole(strong["chi"], strong["dipole"])
            )
            for field in ("local", "local_full"):
                torch.testing.assert_close(strong[field], clean[field])
            split = []
            for index in range(8):
                part = {
                    key: value[index : index + 1]
                    if isinstance(value, torch.Tensor)
                    else value[index : index + 1]
                    for key, value in batch.items()
                }
                split.append(synth(part, **options)["phase"])
            # At SBR=0.001 tiny batched-FFT rounding differences are amplified
            # by up to 1000. Compare relative volume error, not near-zero voxels.
            actual = torch.cat(split)
            delta = (actual - strong["phase"]).flatten(1).norm(dim=1)
            scale = strong["phase"].flatten(1).norm(dim=1)
            self.assertTrue(bool((delta / scale < 2e-5).all()))

    def test_gradient_diagnostics_do_not_change_update_or_rng(self):
        prediction, chi, local, full, mask, dipole = self.example(1)
        torch.manual_seed(82)
        first = DNQSM(ModelConfig((2, 4), checkpoint_blocks=True)).train()
        second = copy.deepcopy(first)
        optimizer_a = torch.optim.AdamW(first.parameters(), foreach=False)
        optimizer_b = torch.optim.AdamW(second.parameters(), foreach=False)
        for model, optimizer, diagnostic in (
            (first, optimizer_a, True),
            (second, optimizer_b, False),
        ):
            output = model(local, mask, dipole)
            loss, terms = reconstruction_loss(
                output, local, chi, mask, dipole, local_full_target=full
            )
            if diagnostic:
                rng = torch.get_rng_state()
                metrics = loss_gradient_metrics(model, terms, LossWeights())
                self.assertGreater(metrics["grad_filter_physics"], 0)
                self.assertEqual(metrics["grad_null_network_local"], 0)
                self.assertTrue(
                    all(parameter.grad is None for parameter in model.parameters())
                )
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            loss.backward()
            optimizer.step()
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_warmup_and_old_objective_resume_guard(self):
        config = TrainConfig()
        self.assertEqual(
            training_weights(config, 0, 10).physics, config.lambda_physics / 10
        )
        self.assertEqual(training_weights(config, 9, 10).physics, config.lambda_physics)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "old.pt"
            torch.save(dict(version=1, loss_version=2), checkpoint)
            with self.assertRaisesRegex(ValueError, "older loss"):
                run_training(replace(config, device="cpu", resume=str(checkpoint)))
            profile = Path(temporary) / "profile.json"
            profile.write_text(
                json.dumps(dict(epochs=3, features=[2, 4], normalize_loss=True))
            )
            parsed = parse_args(["--config", str(profile), "--epochs", "4"])
            self.assertEqual((parsed.epochs, parsed.features), (4, (2, 4)))

    def test_matched_and_stress_validation_logging_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = TrainConfig(
                data_dir=str(make_dataset(root)),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "run"),
                device="cpu",
                features=(2, 4),
                num_workers=0,
                threads=2,
                epochs=2,
                max_train_volumes=2,
                val_views=1,
                accumulation_steps=1,
                gradient_every=1,
                physics_warmup_epochs=3,
                log_every=999,
                comet=True,
                evaluate_external=False,
                bf16=False,
            )
            experiment = MagicMock()
            experiment.get_key.return_value = "corrective-test"
            fake = types.SimpleNamespace(
                start=MagicMock(return_value=experiment), ExperimentConfig=MagicMock()
            )
            with (
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                patch.dict(os.environ, {"COMET_API_KEY": "unit-test-key"}),
                patch.dict("sys.modules", {"comet_ml": fake}),
            ):
                directory = run_training(config)
            rows = [
                json.loads(line)
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            primary = [r for r in rows if r["event"] == "validation"]
            stress = [r for r in rows if r["event"] == "validation_stress"]
            self.assertEqual([r["epoch"] for r in primary], [1, 2])
            self.assertEqual([r["epoch"] for r in stress], [1, 2])
            best = torch.load(directory / "model_best.pt", weights_only=True)
            self.assertEqual(best["loss_version"], LOSS_VERSION)
            self.assertEqual(best["validation_profile"], "matched")
            self.assertEqual(best["best"], min(r["chi_nrmse"] for r in primary))
            updates = [r for r in rows if r["event"] == "train"]
            self.assertIn("raw_physics", updates[0])
            self.assertIn("chi_nrmse", updates[0])
            self.assertIn("grad_filter_physics", updates[0])
            self.assertLess(
                updates[0]["active_physics_weight"],
                updates[-1]["active_physics_weight"],
            )
            self.assertIn("bucket_clean_samples", primary[-1])
            for name in ("validation", "validation_stress"):
                calls = [
                    call
                    for call in experiment.log_metrics.call_args_list
                    if call.kwargs["prefix"] == name
                ]
                self.assertEqual([call.kwargs["step"] for call in calls], [1, 2])
                self.assertTrue(
                    all(isinstance(call.args[0]["chi_nrmse"], float) for call in calls)
                )

    def test_quality_summaries_weight_samples_across_groups(self):
        result = summarize_quality(
            dict(
                quality_chi_nrmse_sum=120,
                quality_chi_nrmse_count=3,
                quality_local_nrmse_sum=0,
                quality_local_nrmse_count=0,
                bucket_clean_samples=2,
            )
        )
        self.assertEqual(result, dict(chi_nrmse=40, bucket_clean_samples=2))
        prediction, chi, local, full, mask, dipole = self.example(1)
        totals = quality_totals(
            prediction,
            dict(
                mask=mask,
                chi=chi,
                local=local,
                dipole=dipole,
                background=torch.zeros_like(chi),
            ),
        )
        result = summarize_quality(totals)
        self.assertEqual(result["bucket_clean_samples"], 1)
        self.assertAlmostEqual(result["chi_nrmse"], 10, places=4)
        self.assertNotIn("bucket_strong_chi_nrmse", result)

    def test_stress_validation_failure_resumes_without_repeating_updates(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = TrainConfig(
                data_dir=str(make_dataset(root)),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "run"),
                device="cpu",
                features=(2, 4),
                num_workers=0,
                threads=2,
                epochs=1,
                max_train_volumes=2,
                batch_size=2,
                val_views=1,
                accumulation_steps=1,
                evaluate_external=False,
                bf16=False,
            )

            def fail_stress(*args, **kwargs):
                if kwargs["synthesis_options"]["background_mode"] == "legacy":
                    raise RuntimeError("stress interrupted")
                return validate(*args, **kwargs)

            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                with patch("dnqsm.training.validate", side_effect=fail_stress):
                    with self.assertRaisesRegex(RuntimeError, "stress interrupted"):
                        run_training(config)
                checkpoint = Path(config.run_dir) / "training_last.pt"
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(
                    (saved["step"], saved["epoch"], saved["next_batch"]), (1, 0, 1)
                )
                directory = run_training(parse_args(["--resume", str(checkpoint)]))
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual((saved["step"], saved["epoch"]), (1, 1))
            self.assertTrue(
                all(
                    int(state["step"]) == 1
                    for state in saved["optimizer"]["state"].values()
                )
            )
            events = [
                json.loads(line)
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len([r for r in events if r["event"] == "train"]), 1)
