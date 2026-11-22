import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from dnqsm.losses import LossWeights, per_sample_loss_terms, reconstruction_loss
from dnqsm.model import DNQSM, ModelConfig
from dnqsm.overfit import OverfitConfig
from dnqsm.physics import dipole_tensor
from dnqsm.runtime import load_trained_model
from dnqsm.training import (
    TrainConfig,
    parse_args,
    phase_metadata,
    phased_learning_rate_multiplier,
    run_training,
    validate,
)
from tests.helpers import make_dataset


class FilterPretrainingTests(unittest.TestCase):
    def config(self, root, data, name, **options):
        return TrainConfig(
            data_dir=str(data),
            cache_dir=str(root / "cache"),
            run_dir=str(root / name),
            device="cpu",
            features=(2, 4),
            num_workers=0,
            threads=2,
            epochs=2,
            filter_pretrain_epochs=2,
            training_schedule="joint",
            max_train_volumes=2,
            batch_size=1,
            accumulation_steps=1,
            val_views=1,
            checkpoint_blocks=False,
            bf16=False,
            evaluate_external=False,
            gradient_every=1,
            save_every_steps=1,
            **options,
        )

    def assert_same(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_same(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected, strict=True):
                self.assert_same(a, b)
        elif isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            self.assertEqual(actual, expected)

    def test_default_ten_epochs_then_alternate_without_updating_frozen_weights(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = replace(
                self.config(root, make_dataset(root), "run"),
                training_schedule="alternating",
                epochs=2,
                filter_pretrain_epochs=10,
                max_train_volumes=3,
                weight_decay=0.1,
            )
            adam_step = torch.optim.AdamW.step
            model_forward = DNQSM.forward
            snapshots = []
            split = len(list(DNQSM(ModelConfig(config.features)).filter.parameters()))

            def step(optimizer, *args, **kwargs):
                parameters = optimizer.param_groups[0]["params"]
                phase_active = len(snapshots) < 30 or (len(snapshots) - 30) % 2 == 0
                # Both weights and existing Adam moments must remain bitwise fixed.
                frozen = parameters[split:] if phase_active else parameters[:split]
                before = [p.detach().clone() for p in frozen]
                states = [
                    {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in optimizer.state.get(p, {}).items()
                    }
                    for p in frozen
                ]
                for i, p in enumerate(parameters):
                    active = i < split if phase_active else i >= split
                    self.assertEqual(p.requires_grad, active)
                    self.assertEqual(p.grad is not None, active)
                result = adam_step(optimizer, *args, **kwargs)
                for p, value, state in zip(frozen, before, states, strict=True):
                    torch.testing.assert_close(p, value, rtol=0, atol=0)
                    self.assert_same(optimizer.state.get(p, {}), state)
                snapshots.append(phase_active)
                return result

            def forward(model, *args, **kwargs):
                if model.training:
                    self.assertFalse(model.filter.training)
                    self.assertTrue(model.null_network.training)
                output = model_forward(model, *args, **kwargs)
                if model.training:
                    self.assertFalse(output.local_field.requires_grad)
                    self.assertFalse(output.tkd.requires_grad)
                    self.assertTrue(output.chi.requires_grad)
                return output

            with (
                patch("dnqsm.training.filtering_figure", return_value=None),
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                patch.object(torch.optim.AdamW, "step", step),
                patch.object(DNQSM, "forward", forward),
            ):
                directory = run_training(config)
            saved = torch.load(directory / "training_last.pt", weights_only=True)
            self.assertEqual((saved["epoch"], saved["step"]), (12, 36))
            self.assertEqual(snapshots, [True] * 30 + [True, False] * 3)
            rows = [
                json.loads(line)
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            training = [row for row in rows if row["event"] == "train"]
            self.assertEqual(
                [r["training_branch"] for r in training],
                ["phase"] * 30 + ["phase", "chi"] * 3,
            )
            for row in training:
                if row["training_branch"] == "phase":
                    self.assertNotIn("chi", row)
                    self.assertEqual(row["active_physics_weight"], 0)
                    self.assertAlmostEqual(
                        row["loss"],
                        config.lambda_local * row["local"]
                        + config.lambda_local_gradient * row["local_gradient"]
                        + config.lambda_harmonic_laplacian
                        * row.get("harmonic_laplacian", 0)
                        + config.lambda_field_decomposition
                        * row.get("field_decomposition", 0),
                        delta=1e-6 * max(1, abs(row["loss"])),
                    )
                else:
                    self.assertNotIn("local_gradient", row)
                    self.assertAlmostEqual(
                        row["loss"],
                        config.lambda_chi * row["chi"]
                        + config.lambda_physics * row["physics"],
                        delta=1e-6 * max(1, abs(row["loss"])),
                    )
                    self.assertEqual(row["active_physics_weight"], 1)
                    self.assertEqual(row["grad_filter_local"], 0)
                    self.assertGreater(row["grad_null_network_chi"], 0)
            validation = [row for row in rows if row["event"] == "validation"]
            self.assertEqual(
                [r["phase"] for r in validation],
                ["filter_pretrain"] * 10 + ["alternating"] * 2,
            )
            self.assertTrue(all("chi_nrmse" not in row for row in validation[:10]))
            self.assertTrue(all("chi_nrmse" in row for row in validation[10:]))
            split = len(list(DNQSM(ModelConfig(config.features)).filter.parameters()))
            states = list(saved["optimizer"]["state"].values())
            self.assertTrue(all(int(s["step"]) == 33 for s in states[:split]))
            self.assertTrue(all(int(s["step"]) == 3 for s in states[split:]))
            best = torch.load(directory / "model_best.pt", weights_only=True)
            self.assertEqual(best["phase"], "alternating")

    def test_alternating_resume_preserves_batch_parity_across_odd_epochs(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = replace(
                self.config(root, make_dataset(root), "whole"),
                training_schedule="alternating",
                filter_pretrain_epochs=1,
                max_train_volumes=3,
                crop_probability=0.5,
                crop_resize=False,
                checkpoint_blocks=True,
            )
            with (
                patch("dnqsm.training.filtering_figure", return_value=None),
                patch("dnqsm.training.reconstruction_figure", return_value=None),
            ):
                whole = run_training(config)
                expected = torch.load(whole / "training_last.pt", weights_only=True)
                for limit in (1, 3, 4, 5, 6, 7):
                    partial = run_training(
                        replace(
                            config, run_dir=str(root / f"part{limit}"), max_steps=limit
                        )
                    )
                    path = partial / "training_last.pt"
                    run_training(
                        parse_args(["--resume", str(path), "--max-steps", "0"])
                    )
                    actual = torch.load(path, weights_only=True)
                    for key in (
                        "model",
                        "optimizer",
                        "scheduler",
                        "rng",
                        "step",
                        "epoch",
                    ):
                        self.assert_same(actual[key], expected[key])
                with self.assertRaisesRegex(ValueError, "Resume configuration changed"):
                    run_training(
                        replace(
                            parse_args(["--resume", str(path)]),
                            training_schedule="joint",
                        )
                    )

    def test_resume_inside_pretraining_at_transition_and_inside_joint_is_exact(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = replace(
                self.config(root, make_dataset(root), "whole"),
                crop_probability=0.5,
                crop_resize=False,
            )
            with (
                patch("dnqsm.training.filtering_figure", return_value=None),
                patch("dnqsm.training.reconstruction_figure", return_value=None),
            ):
                whole = run_training(config)
                expected = torch.load(whole / "training_last.pt", weights_only=True)
                for limit in (1, 4, 5):
                    with self.subTest(limit=limit):
                        partial = run_training(
                            replace(
                                config,
                                run_dir=str(root / f"partial{limit}"),
                                max_steps=limit,
                            )
                        )
                        path = partial / "training_last.pt"
                        if limit < 5:
                            self.assertFalse((partial / "model_best.pt").exists())
                        run_training(
                            parse_args(["--resume", str(path), "--max-steps", "0"])
                        )
                        actual = torch.load(path, weights_only=True)
                        for key in (
                            "model",
                            "optimizer",
                            "scheduler",
                            "rng",
                            "step",
                            "epoch",
                        ):
                            self.assert_same(actual[key], expected[key])
                with self.assertRaisesRegex(ValueError, "Resume configuration changed"):
                    run_training(
                        replace(
                            parse_args(["--resume", str(path)]),
                            filter_pretrain_epochs=1,
                        )
                    )

    def test_pretraining_validation_failure_resumes_without_repeating_updates(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = replace(
                self.config(root, make_dataset(root), "run"),
                epochs=1,
                filter_pretrain_epochs=1,
            )

            def interrupt(*args, **kwargs):
                if (
                    kwargs["filter_only"]
                    and kwargs["synthesis_options"]["background_mode"] == "legacy"
                ):
                    raise RuntimeError("pretraining validation interrupted")
                return validate(*args, **kwargs)

            with (
                patch("dnqsm.training.filtering_figure", return_value=None),
                patch("dnqsm.training.reconstruction_figure", return_value=None),
            ):
                with patch("dnqsm.training.validate", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "pretraining validation"):
                        run_training(config)
                path = Path(config.run_dir) / "training_last.pt"
                interrupted = torch.load(path, weights_only=True)
                self.assertEqual(
                    (interrupted["step"], interrupted["next_batch"]), (2, 2)
                )
                self.assertEqual(interrupted["phase"], "filter_pretrain")
                run_training(parse_args(["--resume", str(path)]))
            saved = torch.load(path, weights_only=True)
            self.assertEqual((saved["step"], saved["epoch"]), (4, 2))

    def test_schedules_restart_and_zero_pretraining_is_supported(self):
        config = TrainConfig(filter_pretrain_epochs=10, epochs=20, warmup_epochs=2)
        self.assertEqual(parse_args([]).filter_pretrain_epochs, 10)
        self.assertEqual(phased_learning_rate_multiplier(config, 0, 2), 0.25)
        self.assertEqual(
            phased_learning_rate_multiplier(config, 19, 2), config.min_lr_factor
        )
        self.assertEqual(phased_learning_rate_multiplier(config, 20, 2), 0.25)
        self.assertEqual(
            phase_metadata(config, 10), dict(phase="alternating", phase_epoch=1)
        )
        disabled = parse_args(["--filter-pretrain-epochs", "0"])
        self.assertEqual(phase_metadata(disabled, 0)["phase"], "alternating")
        for value in (-1, 0.5, float("inf")):
            with self.assertRaisesRegex(ValueError, "pretraining epochs"):
                replace(config, filter_pretrain_epochs=value).validate()

    def test_default_schedule_threshold_and_accumulation_validation(self):
        default = parse_args([])
        self.assertEqual(default.training_schedule, "alternating")
        self.assertEqual(default.filter_pretrain_epochs, 10)
        self.assertEqual(default.accumulation_steps, 1)
        self.assertEqual(default.threshold, 0.1)
        self.assertEqual(ModelConfig().threshold, 0.1)
        self.assertEqual(OverfitConfig().threshold, 0.1)
        self.assertEqual(default.cutoff_mode, "soft")
        self.assertEqual(ModelConfig().cutoff_mode, "soft")
        self.assertEqual(OverfitConfig().cutoff_mode, "soft")
        with self.assertRaisesRegex(ValueError, "accumulation_steps=1"):
            replace(default, accumulation_steps=2).validate()
        replace(default, training_schedule="joint", accumulation_steps=2).validate()
        with self.assertRaisesRegex(ValueError, "Training schedule"):
            replace(default, training_schedule="invalid").validate()

    def test_disabled_extension_skips_full_field_computation(self):
        model = DNQSM(ModelConfig((2, 4)))
        field = torch.randn(1, 1, 8, 8, 8) * 0.01
        mask = torch.ones_like(field)
        dipole = dipole_tensor(field.shape[-3:])
        output = model(field, mask, dipole)
        with patch(
            "dnqsm.losses.apply_dipole", side_effect=AssertionError("unused FFT")
        ):
            terms = per_sample_loss_terms(
                output,
                field,
                field,
                mask,
                dipole,
                predicted_field=field,
                include_extension=False,
                input_field=field,
            )
        self.assertNotIn("extension", terms)
        loss, terms = reconstruction_loss(
            output,
            field,
            field,
            mask,
            dipole,
            weights=LossWeights(extension=0),
            input_field=field,
        )
        self.assertNotIn("extension", terms)
        loss.backward()

    def test_filtering_batches_supervise_the_exterior_they_alone_can_learn(self):
        # Filtering-only batches are the only ones that update the filter under
        # the alternating schedule, and chi branches zero the extension weight.
        # If this path drops the term, exterior supervision reaches no batch at
        # all and the full-grid field is trained against nothing outside the ROI.
        from dnqsm.training import _predict_training_batch

        model = DNQSM(
            ModelConfig(
                (2, 4),
                full_grid_field=True,
                filter_input_version=1,
                conditional_laplacian_threshold=0.0,
            )
        )
        field = torch.randn(1, 1, 12, 12, 12) * 0.01
        mask = torch.zeros_like(field)
        mask[..., 2:-2, 2:-2, 2:-2] = 1
        data = dict(
            phase=field * mask,
            local=field * mask,
            chi=field * mask,
            mask=mask,
            dipole=dipole_tensor(field.shape[-3:]),
            local_full=field,
        )
        device = torch.device("cpu")
        _, raw, _ = _predict_training_batch(model, data, device, False, True, True)
        self.assertIn("extension", raw)
        self.assertGreater(float(raw["extension"].detach().sum()), 0)
        _, without, _ = _predict_training_batch(model, data, device, False, True, False)
        self.assertNotIn("extension", without)

    def test_legacy_training_resume_keeps_old_model_and_zero_pretraining(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = replace(
                self.config(root, make_dataset(root), "legacy"),
                filter_input_version=1,
                filter_laplacian_residual=False,
                conditional_laplacian_threshold=0,
                filter_pretrain_epochs=0,
                threshold=0.1,
                full_grid_field=True,
                residual_filter=True,
                lambda_extension=1.0,
            )
            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                directory = run_training(replace(config, max_steps=1))
                path = directory / "training_last.pt"
                saved = torch.load(path, weights_only=True)
                for key in (
                    "filter_input_version",
                    "filter_pretrain_epochs",
                    "training_schedule",
                    "threshold",
                ):
                    saved["config"].pop(key)
                saved["model_config"].pop("filter_input_version")
                saved["model_config"].pop("threshold")
                torch.save(saved, path)
                resumed = parse_args(["--resume", str(path), "--max-steps", "0"])
                self.assertEqual(resumed.filter_pretrain_epochs, 0)
                self.assertEqual(resumed.training_schedule, "joint")
                self.assertEqual(resumed.threshold, 0.1)
                self.assertEqual(load_trained_model(path).config.threshold, 0.1)
                run_training(resumed)
                complete = run_training(
                    replace(config, run_dir=str(root / "legacy_whole"))
                )
                expected = torch.load(complete / "training_last.pt", weights_only=True)
                actual = torch.load(path, weights_only=True)
                for key in ("model", "optimizer", "scheduler", "rng"):
                    self.assert_same(actual[key], expected[key])
                with self.assertRaisesRegex(
                    ValueError, "different filtering architecture"
                ):
                    run_training(
                        replace(
                            config,
                            run_dir=str(root / "new"),
                            filter_input_version=2,
                            laplacian_input=True,
                            full_grid_field=False,
                            residual_filter=False,
                            lambda_extension=0,
                            init_checkpoint=str(directory / "model_last.pt"),
                        )
                    )
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved["epoch"], 2)
            self.assertEqual(saved["model_config"]["filter_input_version"], 1)


if __name__ == "__main__":
    unittest.main()
