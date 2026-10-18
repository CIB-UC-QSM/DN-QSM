import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from dnqsm.logging import RunLogger, _pyplot, comparison_figure
from dnqsm.training import (
    TrainConfig,
    atomic_save,
    load_trained_model,
    parse_args,
    run_training,
)
from tests.helpers import make_dataset


class TrainingAndLoggingTests(unittest.TestCase):
    def config(self, root: Path, data: Path, name: str) -> TrainConfig:
        return TrainConfig(
            data_dir=str(data),
            cache_dir=str(root / "cache"),
            run_dir=str(root / name),
            epochs=2,
            device="cpu",
            num_workers=0,
            threads=2,
            features=(2, 4),
            accumulation_steps=2,
            val_views=1,
            evaluate_external=False,
            checkpoint_blocks=False,
            bf16=False,
            save_every_steps=1,
        )

    def test_resume_matches_uninterrupted_training_and_restores_model(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = make_dataset(root)
            config = self.config(root, data, "whole")
            # Plot styling has its own test; suppress repeated expensive figures.
            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                complete = run_training(config)
                partial_config = self.config(root, data, "resumed")
                partial = run_training(replace(partial_config, max_steps=1))
                checkpoint = partial / "training_last.pt"
                resumed = parse_args(["--resume", str(checkpoint), "--max-steps", "0"])
                run_training(resumed)
            whole = torch.load(complete / "training_last.pt", weights_only=True)
            split = torch.load(partial / "training_last.pt", weights_only=True)
            self.assertEqual(
                (whole["step"], whole["epoch"]), (split["step"], split["epoch"])
            )
            for key in whole["model"]:
                torch.testing.assert_close(
                    whole["model"][key], split["model"][key], rtol=0, atol=0
                )
            self.assertEqual(whole["scheduler"], split["scheduler"])
            for state_a, state_b in zip(
                whole["optimizer"]["state"].values(),
                split["optimizer"]["state"].values(),
                strict=True,
            ):
                for key in state_a:
                    torch.testing.assert_close(
                        state_a[key], state_b[key], rtol=0, atol=0
                    )
            self.assertEqual(
                json.loads((partial / "status.json").read_text())["status"], "complete"
            )
            model = load_trained_model(partial / "model_best.pt")
            self.assertFalse(model.training)
            self.assertEqual(model.config.features, (2, 4))
            events = [
                json.loads(line)
                for line in (partial / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len([r for r in events if r["event"] == "validation"]), 2)
            with self.assertRaisesRegex(ValueError, "Resume configuration changed"):
                run_training(replace(resumed, threshold=0.2))

    def test_partial_accumulation_group_equals_full_batch(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = make_dataset(root)
            config = replace(
                self.config(root, data, "micro"),
                max_steps=1,
                accumulation_steps=4,
                max_train_volumes=3,
            )
            with (
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                patch.object(RunLogger, "figure"),
            ):
                micro = run_training(config)
                full = run_training(
                    replace(
                        config,
                        run_dir=str(root / "full"),
                        batch_size=3,
                        accumulation_steps=1,
                    )
                )
            a = torch.load(micro / "model_last.pt", weights_only=True)["model"]
            b = torch.load(full / "model_last.pt", weights_only=True)["model"]
            # InstanceNorm is per sample, so accumulation is batch-equivalent.
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=5e-4, atol=3e-6)

    def test_comet_logging_and_resume_use_saved_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            experiment = MagicMock()
            experiment.get_key.return_value = "test-experiment"
            fake = types.SimpleNamespace(
                start=MagicMock(return_value=experiment), ExperimentConfig=MagicMock()
            )
            config = TrainConfig(comet=True)
            with (
                patch.dict(os.environ, {"COMET_API_KEY": "unit-test-key"}),
                patch.dict("sys.modules", {"comet_ml": fake}),
                redirect_stdout(io.StringIO()),
            ):
                logger = RunLogger(root, config)
                logger.log("train", 7, 1, loss=0.1)
                figure = comparison_figure(
                    [("chi", torch.zeros(1, 1, 4, 5, 6), True)], "test"
                )
                logger.figure(figure, "chi", 7, 1)
                logger.close()
                resumed = RunLogger(root, config, resume=True)
                resumed.close()
            self.assertEqual(
                fake.start.call_args.kwargs["experiment_key"], "test-experiment"
            )
            self.assertEqual(fake.start.call_args.kwargs["mode"], "get")
            experiment.log_metrics.assert_called_once_with(
                {"loss": 0.1}, prefix="train", step=7, epoch=1
            )
            experiment.log_figure.assert_not_called()
            experiment.log_image.assert_called_once_with(
                image_data=str(root / "images/chi/step_000007.png"),
                name="chi",
                step=7,
                metadata={"epoch": 1},
                overwrite=False,
            )
            self.assertTrue((root / "images/chi/step_000007.png").is_file())
            for file in root.rglob("*"):
                if file.suffix == ".json":
                    self.assertNotIn("unit-test-key", file.read_text())

    def test_figures_use_fixed_susceptibility_limits_and_central_planes(self):
        volume = torch.arange(4 * 5 * 6).reshape(1, 1, 4, 5, 6).float()
        figure = comparison_figure([("Susceptibility", volume, True)], "NRMSE=12.00%")
        try:
            for axis in figure.axes:
                self.assertEqual(axis.images[0].get_clim(), (-0.1, 0.1))
                self.assertEqual(axis.images[0].get_cmap().name, "gray")
            self.assertEqual(figure.axes[0].images[0].get_array().shape, (6, 5))
            self.assertEqual(figure.axes[1].images[0].get_array().shape, (6, 4))
            self.assertEqual(figure.axes[2].images[0].get_array().shape, (5, 4))
        finally:
            _pyplot().close(figure)

    def test_undefined_metrics_are_not_logged_as_zero(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            logger = RunLogger(Path(temp), TrainConfig())
            experiment = MagicMock()
            logger.experiment = experiment
            logger.log("external_cosmos", 50, 3, comet_step=3, chi_nrmse=None)
            experiment.log_metrics.assert_not_called()
            logger.log("external_sim2", 50, 3, comet_step=3, chi_nrmse=12.5)
            experiment.log_metrics.assert_called_once_with(
                {"chi_nrmse": 12.5}, prefix="external_sim2", step=3, epoch=3
            )
            logger.close()

    def test_resume_rng_payload_is_weights_only_loadable(self):
        import random

        import numpy as np

        from dnqsm.training import _restore_rng, _rng_state

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.pt"
            atomic_save(_rng_state(), path)
            expected = (torch.rand(2), np.random.rand(2), random.random())
            _restore_rng(torch.load(path, weights_only=True))
            torch.testing.assert_close(expected[0], torch.rand(2), rtol=0, atol=0)
            np.testing.assert_array_equal(expected[1], np.random.rand(2))
            self.assertEqual(expected[2], random.random())

    def test_gpu_budget_reserves_library_overhead(self):
        from dnqsm.training import configure_memory_budget

        device = torch.device("cuda:0")
        with (
            patch(
                "torch.cuda.get_device_properties",
                return_value=types.SimpleNamespace(total_memory=24 * 1024**3),
            ),
            patch("torch.cuda.set_per_process_memory_fraction") as limit,
        ):
            configure_memory_budget(device, 12)
            limit.assert_called_once_with(11 / 24, device)
        with self.assertRaises(ValueError):
            TrainConfig(cuda_memory_gb=0.5).validate()

    def test_old_loss_requires_explicit_new_run_and_warm_start_can_resume(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = make_dataset(root)
            config = self.config(root, data, "original")
            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                original = run_training(replace(config, max_steps=1))
                checkpoint = original / "training_last.pt"
                saved = torch.load(checkpoint, weights_only=True)
                saved.pop("loss_version")
                atomic_save(saved, checkpoint)
                with self.assertRaisesRegex(ValueError, "older loss"):
                    run_training(parse_args(["--resume", str(checkpoint)]))
                warm_config = replace(
                    self.config(root, data, "warm"),
                    init_checkpoint=str(original / "model_last.pt"),
                    max_steps=1,
                )
                warm = run_training(warm_config)
                resumed = parse_args(
                    ["--resume", str(warm / "training_last.pt"), "--max-steps", "0"]
                )
                self.assertIsNone(resumed.init_checkpoint)
                run_training(resumed)
                self.assertEqual(
                    json.loads((warm / "status.json").read_text())["status"], "complete"
                )


if __name__ == "__main__":
    unittest.main()
