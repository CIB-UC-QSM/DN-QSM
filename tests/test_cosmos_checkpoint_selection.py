import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

from dnqsm.losses import LOSS_VERSION
from dnqsm.training import (
    TrainConfig,
    checkpoint_training_config,
    parse_args,
    run_training,
)
from tests.helpers import make_dataset


class CosmosCheckpointSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, root):
        return TrainConfig(
            data_dir=str(make_dataset(root)),
            cache_dir=str(root / "cache"),
            run_dir=str(root / "run"),
            device="cpu",
            features=(2, 4),
            threads=2,
            num_workers=0,
            bf16=False,
            epochs=3,
            filter_pretrain_epochs=0,
            training_schedule="joint",
            batch_size=2,
            max_train_volumes=2,
            val_views=1,
            checkpoint_blocks=False,
            checkpoint_metric="cosmos",
            external_eval_every=1,
            evaluate_external=False,
        )

    def run_experiment(self, config, scores, fail_once=False):
        snapshots = {}
        failed = False
        validation_calls = 0

        def validation(*args, **kwargs):
            nonlocal validation_calls
            validation_calls += 1
            # Validation improves every epoch; COSMOS intentionally does not.
            return dict(chi_nrmse=100.0 - validation_calls), None

        def external(model, cases, directory, device, logger, step, epoch, bf16):
            nonlocal failed
            if fail_once and epoch == 2 and not failed:
                failed = True
                snapshots["before_failure"] = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                raise RuntimeError("COSMOS evaluation interrupted")
            snapshots[epoch] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            return dict(cosmos=dict(chi_nrmse=scores[epoch - 1]))

        with (
            redirect_stdout(io.StringIO()),
            patch("dnqsm.training.validate", side_effect=validation),
            patch("dnqsm.training.filtering_figure", return_value=None),
            patch("dnqsm.training.reconstruction_figure", return_value=None),
            patch("dnqsm.evaluation.load_cases", return_value=[]),
            patch("dnqsm.evaluation.evaluate_training_epoch", side_effect=external),
        ):
            if fail_once:
                with self.assertRaisesRegex(
                    RuntimeError, "COSMOS evaluation interrupted"
                ):
                    run_training(config)
                checkpoint = Path(config.run_dir) / "training_last.pt"
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(
                    (saved["step"], saved["epoch"], saved["best"]), (2, 1, scores[0])
                )
                run_training(parse_args(["--resume", str(checkpoint)]))
            else:
                run_training(config)
        return snapshots

    def test_cosmos_selects_weights_independently_of_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.config(root)
            snapshots = self.run_experiment(config, [20.0, 10.0, 15.0])
            run = Path(config.run_dir)
            best = torch.load(run / "model_best.pt", weights_only=True)
            last = torch.load(run / "training_last.pt", weights_only=True)
            self.assertEqual((best["epoch"], best["best"]), (2, 10.0))
            self.assertEqual(best["selection_metric"], "cosmos_chi_nrmse")
            self.assertEqual(best["cosmos_chi_nrmse"], 10.0)
            self.assertEqual(last["best"], 10.0)
            self.assertEqual(last["checkpoint_metric"], "cosmos")
            self.assertEqual(
                json.loads((run / "status.json").read_text())["checkpoint_metric"],
                "cosmos",
            )
            self.assertEqual(
                json.loads((run / "best_checkpoint.json").read_text())["epoch"], 2
            )
            for key, value in snapshots[2].items():
                torch.testing.assert_close(best["model"][key], value, rtol=0, atol=0)
            self.assertTrue(
                any(
                    not torch.equal(best["model"][key], value)
                    for key, value in snapshots[3].items()
                )
            )

    def test_external_failure_resume_keeps_best_and_does_not_repeat_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            snapshots = self.run_experiment(config, [20.0, 10.0, 15.0], fail_once=True)
            for key, value in snapshots["before_failure"].items():
                torch.testing.assert_close(snapshots[2][key], value, rtol=0, atol=0)
            saved = torch.load(
                Path(config.run_dir) / "training_last.pt", weights_only=True
            )
            self.assertEqual(
                (saved["step"], saved["epoch"], saved["best"]), (3, 3, 10.0)
            )
            self.assertEqual(
                {int(v["step"]) for v in saved["optimizer"]["state"].values()}, {3}
            )
            best = torch.load(Path(config.run_dir) / "model_best.pt", weights_only=True)
            self.assertEqual(best["epoch"], 2)

    def test_ties_keep_the_earliest_best(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            self.run_experiment(config, [10.0, 10.0, 12.0])
            best = torch.load(Path(config.run_dir) / "model_best.pt", weights_only=True)
            self.assertEqual((best["epoch"], best["best"]), (1, 10.0))

    def test_missing_or_nonfinite_cosmos_never_falls_back_to_validation(self):
        for score in (None, float("nan"), float("inf")):
            with self.subTest(score=score), tempfile.TemporaryDirectory() as temp:
                config = replace(self.config(Path(temp)), epochs=1)
                with self.assertRaisesRegex(
                    ValueError, "cosmos checkpoint selection requires finite"
                ):
                    self.run_experiment(config, [score])
                self.assertFalse((Path(config.run_dir) / "model_best.pt").exists())

    def test_default_profile_and_legacy_resume_policy(self):
        self.assertEqual(
            parse_args([], use_default_profile=True).checkpoint_metric, "cosmos"
        )
        self.assertEqual(
            checkpoint_training_config({})["checkpoint_metric"], "validation"
        )
        for cadence in (0, 2):
            with self.assertRaisesRegex(ValueError, "external_eval_every=1"):
                TrainConfig(
                    checkpoint_metric="cosmos", external_eval_every=cadence
                ).validate()
        with self.assertRaisesRegex(ValueError, "Checkpoint metric"):
            TrainConfig(checkpoint_metric="unknown").validate()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.pt"
            config = asdict(
                TrainConfig(
                    device="cpu", features=(2, 4), threads=2, external_eval_every=1
                )
            )
            config.pop("checkpoint_metric")
            torch.save(dict(version=1, loss_version=LOSS_VERSION, config=config), path)
            resumed = parse_args(["--resume", str(path)])
            self.assertEqual(resumed.checkpoint_metric, "validation")
            with self.assertRaisesRegex(ValueError, "checkpoint_metric"):
                run_training(replace(resumed, checkpoint_metric="cosmos"))


if __name__ == "__main__":
    unittest.main()
