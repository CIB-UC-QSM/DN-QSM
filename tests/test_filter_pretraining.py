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
from dnqsm.physics import dipole_tensor
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

    def test_default_ten_epochs_train_filter_only_then_both_networks(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = make_dataset(root)
            config = replace(
                self.config(root, data, "run"),
                epochs=1,
                filter_pretrain_epochs=10,
                max_train_volumes=1,
                external_eval_every=1,
            )
            torch.manual_seed(config.seed)
            initial = DNQSM(ModelConfig(features=config.features)).state_dict()
            parameter_names = list(
                dict(DNQSM(ModelConfig(config.features)).named_parameters())
            )
            snapshots, forwards = [], []
            adam_step = torch.optim.AdamW.step
            model_forward = DNQSM.forward

            def step(optimizer, *args, **kwargs):
                # All parameters stay in the optimizer so filter moments survive.
                parameters = optimizer.param_groups[0]["params"]
                snapshots.append([p.grad is not None for p in parameters])
                if len(snapshots) <= 10:
                    for name, parameter in zip(
                        parameter_names, parameters, strict=True
                    ):
                        if name.startswith("null_network."):
                            torch.testing.assert_close(
                                parameter, initial[name], rtol=0, atol=0
                            )
                return adam_step(optimizer, *args, **kwargs)

            def forward(model, *args, **kwargs):
                self.assertTrue(
                    all(p.requires_grad for p in model.null_network.parameters())
                )
                forwards.append(model.training)
                return model_forward(model, *args, **kwargs)

            with (
                patch("dnqsm.training.filtering_figure", return_value=None),
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                patch.object(torch.optim.AdamW, "step", step),
                patch.object(DNQSM, "forward", forward),
                patch(
                    "dnqsm.training.evaluation.load_cases", return_value=[]
                ) as load_cases,
                patch("dnqsm.training.evaluation.evaluate_training_epoch") as external,
            ):
                directory = run_training(config)
            saved = torch.load(directory / "training_last.pt", weights_only=True)
            load_cases.assert_called_once()
            external.assert_called_once()
            self.assertEqual(external.call_args.args[6], 11)
            self.assertEqual((saved["epoch"], saved["step"]), (11, 11))
            # Only joint training and its two validation suites use full forward.
            self.assertEqual(sum(forwards), 1)
            subjects = {
                r["subject"]
                for r in json.loads((directory / "split.json").read_text())[
                    "validation"
                ]
            }
            self.assertEqual(len(forwards), 1 + 2 * len(subjects))
            split = len(list(DNQSM(ModelConfig(config.features)).filter.parameters()))
            for grads in snapshots[:10]:
                self.assertTrue(all(grads[:split]))
                self.assertFalse(any(grads[split:]))
            self.assertTrue(all(snapshots[10]))
            states = list(saved["optimizer"]["state"].values())
            self.assertTrue(all(int(s["step"]) == 11 for s in states[:split]))
            self.assertTrue(all(int(s["step"]) == 1 for s in states[split:]))
            for branch in ("filter.", "null_network."):
                self.assertTrue(
                    any(
                        not torch.equal(value, initial[name])
                        for name, value in saved["model"].items()
                        if name.startswith(branch)
                    )
                )
            rows = [
                json.loads(line)
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            validation = [row for row in rows if row["event"] == "validation"]
            self.assertEqual(
                [r["phase"] for r in validation],
                ["filter_pretrain"] * 10 + ["end_to_end"],
            )
            for row in validation[:10]:
                self.assertIn("local_nrmse", row)
                self.assertNotIn("chi_nrmse", row)
                self.assertAlmostEqual(
                    row["loss"], config.lambda_local * row["local"], places=6
                )
            self.assertTrue(all("extension" not in row for row in validation))
            best = torch.load(directory / "model_best.pt", weights_only=True)
            self.assertEqual((best["phase"], best["epoch"]), ("end_to_end", 11))

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
            phase_metadata(config, 10), dict(phase="end_to_end", phase_epoch=1)
        )
        disabled = parse_args(["--filter-pretrain-epochs", "0"])
        self.assertEqual(phase_metadata(disabled, 0)["phase"], "end_to_end")
        for value in (-1, 0.5, float("inf")):
            with self.assertRaisesRegex(ValueError, "pretraining epochs"):
                replace(config, filter_pretrain_epochs=value).validate()

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
            )
        self.assertNotIn("extension", terms)
        loss, terms = reconstruction_loss(
            output, field, field, mask, dipole, weights=LossWeights(extension=0)
        )
        self.assertNotIn("extension", terms)
        loss.backward()

    def test_legacy_training_resume_keeps_old_model_and_zero_pretraining(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = replace(
                self.config(root, make_dataset(root), "legacy"),
                filter_input_version=1,
                filter_pretrain_epochs=0,
                full_grid_field=True,
                residual_filter=True,
                lambda_extension=1.0,
            )
            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                directory = run_training(replace(config, max_steps=1))
                path = directory / "training_last.pt"
                saved = torch.load(path, weights_only=True)
                for key in ("filter_input_version", "filter_pretrain_epochs"):
                    saved["config"].pop(key)
                saved["model_config"].pop("filter_input_version")
                torch.save(saved, path)
                resumed = parse_args(["--resume", str(path), "--max-steps", "0"])
                self.assertEqual(resumed.filter_pretrain_epochs, 0)
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
