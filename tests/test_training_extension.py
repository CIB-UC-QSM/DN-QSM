import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from dnqsm.training import (
    TrainConfig,
    parse_args,
    phased_learning_rate_multiplier,
    run_training,
)
from tests.helpers import make_dataset


class TrainingExtensionTests(unittest.TestCase):
    def config(self, root, **changes):
        return TrainConfig(
            data_dir=str(make_dataset(root)),
            cache_dir=str(root / "cache"),
            run_dir=str(root / "run"),
            device="cpu",
            threads=2,
            num_workers=0,
            features=(2, 4),
            epochs=2,
            filter_pretrain_epochs=0,
            warmup_epochs=0,
            training_schedule="joint",
            batch_size=2,
            max_train_volumes=2,
            val_views=1,
            evaluate_external=False,
            bf16=False,
            checkpoint_blocks=False,
            **changes,
        )

    def run_quietly(self, config):
        with (
            redirect_stdout(io.StringIO()),
            patch("dnqsm.training.filtering_figure", return_value=None),
            patch("dnqsm.training.reconstruction_figure", return_value=None),
        ):
            return run_training(config)

    def test_completed_run_appends_epochs_and_preserves_optimizer_and_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            directory = self.run_quietly(config)
            checkpoint = directory / "training_last.pt"
            original = torch.load(checkpoint, weights_only=True)
            self.assertEqual((original["epoch"], original["step"]), (2, 2))
            continued = parse_args(
                ["--resume", str(checkpoint), "--additional-epochs", "3"]
            )
            self.assertEqual((continued.epochs, continued.additional_epochs), (2, 3))
            self.run_quietly(continued)
            final = torch.load(checkpoint, weights_only=True)
            self.assertEqual((final["epoch"], final["step"]), (5, 5))
            self.assertEqual(
                {int(v["step"]) for v in final["optimizer"]["state"].values()}, {5}
            )
            self.assertEqual(final["scheduler"]["last_epoch"], 5)
            self.assertEqual(
                final["scheduler"]["base_lrs"], original["scheduler"]["base_lrs"]
            )
            rows = [
                json.loads(line)
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            updates = [row for row in rows if row["event"] == "train"]
            self.assertEqual([row["step"] for row in updates], [1, 2, 3, 4, 5])
            for row in updates[2:]:
                self.assertAlmostEqual(
                    row["learning_rate"], config.learning_rate * config.min_lr_factor
                )
            resumed = parse_args(["--resume", str(checkpoint)])
            self.assertEqual(resumed.additional_epochs, 3)
            extended_again = parse_args(
                ["--resume", str(checkpoint), "--additional-epochs", "2"]
            )
            self.assertEqual(extended_again.additional_epochs, 5)
            with self.assertRaisesRegex(ValueError, "only extend a completed"):
                self.run_quietly(replace(resumed, additional_epochs=2))

    def test_incomplete_run_rejects_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp), max_steps=1)
            directory = self.run_quietly(config)
            continued = parse_args(
                [
                    "--resume",
                    str(directory / "training_last.pt"),
                    "--additional-epochs",
                    "2",
                    "--max-steps",
                    "0",
                ]
            )
            with self.assertRaisesRegex(ValueError, "only extend a completed"):
                self.run_quietly(continued)

    def test_schedule_anchor_does_not_change_when_epochs_are_appended(self):
        config = TrainConfig(epochs=50, filter_pretrain_epochs=0, warmup_epochs=3)
        extended = replace(config, additional_epochs=50)
        for step in (0, 35, 36, 599, 600, 601, 1199):
            self.assertEqual(
                phased_learning_rate_multiplier(config, step, 12),
                phased_learning_rate_multiplier(extended, step, 12),
            )
        self.assertEqual(
            phased_learning_rate_multiplier(extended, 600, 12), config.min_lr_factor
        )
        self.assertEqual(
            phased_learning_rate_multiplier(extended, 1199, 12), config.min_lr_factor
        )

    def test_invalid_additional_epochs_require_resume(self):
        for count in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "Additional epochs"):
                TrainConfig(additional_epochs=count).validate()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--additional-epochs", "50"])


if __name__ == "__main__":
    unittest.main()
