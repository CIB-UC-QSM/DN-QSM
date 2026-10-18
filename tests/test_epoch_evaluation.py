import copy
import io
import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from dnqsm.evaluation import (
    evaluate_training_epoch,
    load_cosmos_case,
    load_sim2_case,
)
from dnqsm.logging import _pyplot
from dnqsm.model import DNQSM, ModelConfig
from dnqsm.training import TrainConfig, parse_args, run_training
from tests.helpers import make_dataset
from tests.test_evaluation import cosmos_fixture, sim2_fixture


class EpochEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cases(self, root):
        simulation, *_ = cosmos_fixture(root)
        sim2, *_ = sim2_fixture(root)
        return [load_cosmos_case(simulation), load_sim2_case(sim2, pad=2)]

    def test_current_weights_mode_epoch_axis_and_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = self.cases(root)
            model = DNQSM(ModelConfig((2, 4))).train()
            optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            before = copy.deepcopy(model.state_dict())
            moments = copy.deepcopy(optimizer.state_dict())
            rng = torch.get_rng_state()
            logger = MagicMock()
            logger.figure.side_effect = lambda figure, *args, **kwargs: _pyplot().close(
                figure
            )
            history_dir = root / "evaluation"
            history_dir.mkdir()
            (history_dir / "history.json").write_text(json.dumps({"4": {}, "7": {}}))
            evaluate_training_epoch(model, cases, root, "cpu", logger, 125, 4, False)
            self.assertTrue(model.training)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            after = optimizer.state_dict()
            for index, state in moments["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(
                        value, after["state"][index][key], rtol=0, atol=0
                    )
            history = json.loads((history_dir / "history.json").read_text())
            self.assertEqual(list(history), ["4"])
            self.assertEqual(set(history["4"]), {"cosmos", "sim2"})
            self.assertFalse((history_dir / "nrmse_vs_epoch.png").exists())
            self.assertFalse(list((history_dir / "epochs").rglob("*.pt")))
            self.assertFalse(list((history_dir / "epochs").rglob("*.npz")))
            self.assertEqual(logger.figure.call_count, 2)
            self.assertEqual(logger.log.call_count, 2)
            for call in logger.log.call_args_list:
                self.assertEqual(call.args[1:], (125, 4))
                self.assertEqual(call.kwargs["comet_step"], 4)
            names = [call.args[1] for call in logger.figure.call_args_list]
            self.assertEqual(names, ["external_cosmos", "external_sim2"])
            for call in logger.figure.call_args_list:
                self.assertEqual(call.args[2:], (4, 4))

    def test_every_epoch_comet_logging_and_failure_boundary_resume(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            cases = self.cases(root)
            source = make_dataset(root)
            config = TrainConfig(
                data_dir=str(source),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "run"),
                device="cpu",
                epochs=2,
                batch_size=2,
                accumulation_steps=1,
                features=(2, 4),
                crop_probability=0.2,
                crop_resize=False,
                num_workers=0,
                threads=2,
                bf16=False,
                evaluate_external=False,
                external_eval_every=1,
                max_train_volumes=2,
                val_views=1,
                comet=True,
            )
            experiment = MagicMock()
            experiment.get_key.return_value = "epoch-test"
            fake = types.SimpleNamespace(
                start=MagicMock(return_value=experiment), ExperimentConfig=MagicMock()
            )
            # Fail just before epoch-1 evaluation. The completed optimizer step
            # must be saved, and resume must run evaluation without repeating it.
            with (
                patch.dict(os.environ, {"COMET_API_KEY": "unit-test-key"}),
                patch.dict("sys.modules", {"comet_ml": fake}),
                patch("dnqsm.evaluation.load_cases", return_value=cases),
                patch("dnqsm.training.reconstruction_figure", return_value=None),
            ):
                with patch(
                    "dnqsm.evaluation.evaluate_training_epoch",
                    side_effect=RuntimeError("evaluation interrupted"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "evaluation interrupted"):
                        run_training(config)
                checkpoint = Path(config.run_dir) / "training_last.pt"
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(
                    (saved["step"], saved["epoch"], saved["next_batch"]), (1, 0, 1)
                )
                resumed = parse_args(["--resume", str(checkpoint)])
                directory = run_training(resumed)
            final = torch.load(checkpoint, weights_only=True)
            self.assertEqual((final["step"], final["epoch"]), (2, 2))
            history = json.loads((directory / "evaluation/history.json").read_text())
            self.assertEqual(set(history), {"1", "2"})
            external = [
                call
                for call in experiment.log_metrics.call_args_list
                if call.kwargs["prefix"].startswith("external_")
            ]
            self.assertEqual(len(external), 4)
            self.assertEqual([call.kwargs["step"] for call in external], [1, 1, 2, 2])
            for name in ("cosmos", "sim2"):
                calls = [
                    call
                    for call in external
                    if call.kwargs["prefix"] == "external_" + name
                ]
                for epoch, call in enumerate(calls, 1):
                    self.assertEqual(call.kwargs["epoch"], epoch)
                    self.assertEqual(
                        call.args[0]["chi_nrmse"],
                        history[str(epoch)][name]["chi_nrmse"],
                    )
                    self.assertIsInstance(call.args[0]["chi_nrmse"], float)
                images = [
                    call
                    for call in experiment.log_image.call_args_list
                    if call.kwargs["name"] == "external_" + name
                ]
                self.assertEqual([call.kwargs["step"] for call in images], [1, 2])
                self.assertEqual(
                    [call.kwargs["metadata"]["epoch"] for call in images], [1, 2]
                )
                for call in images:
                    self.assertFalse(call.kwargs["overwrite"])
                    self.assertTrue(Path(call.kwargs["image_data"]).is_file())
            experiment.log_figure.assert_not_called()
            self.assertFalse((directory / "evaluation/nrmse_vs_epoch.png").exists())
            self.assertEqual(experiment.log_image.call_count, 4)
            self.assertTrue(
                (
                    directory / "evaluation/epochs/epoch_0001/cosmos/comparison.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    directory / "evaluation/epochs/epoch_0002/sim2/comparison.png"
                ).is_file()
            )
