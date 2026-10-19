import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from dnqsm.overfit import OverfitConfig, main, passes_sanity, run_overfit
from tests.test_evaluation import cosmos_fixture


class OverfitTests(unittest.TestCase):
    def test_resume_keeps_optimizer_steps_and_initial_reference_when_lr_changes(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            simulation, *_ = cosmos_fixture(root)
            config = OverfitConfig(
                cosmos_simulation=str(simulation),
                run_dir=str(root / "resumed"),
                device="cpu",
                steps=2,
                features=(2, 4),
                threads=2,
                log_every=1,
                figure_every=2,
                bf16=False,
                stop_on_pass=False,
                target_nrmse=0.001,
            )
            with (
                patch("dnqsm.overfit.reconstruction_figure", return_value=None),
                patch("dnqsm.overfit.RunLogger.figure"),
            ):
                directory = run_overfit(config)
                original = json.loads((directory / "summary.json").read_text())
                continued = replace(
                    config,
                    resume=str(directory / "overfit_last.pt"),
                    steps=4,
                    learning_rate=1e-4,
                )
                run_overfit(continued)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["initial"], original["initial"])
            self.assertEqual(summary["steps"], 4)
            self.assertEqual(
                summary["training_phases"],
                [
                    {
                        "start_step": 0,
                        "learning_rate": 1e-3,
                        "residual_filter": False,
                        "min_lr_factor": 0.05,
                    },
                    {
                        "start_step": 2,
                        "learning_rate": 1e-4,
                        "residual_filter": False,
                        "min_lr_factor": 0.05,
                    },
                ],
            )
            saved = torch.load(directory / "overfit_last.pt", weights_only=True)
            self.assertTrue(
                all(
                    int(state["step"]) == 4
                    for state in saved["optimizer"]["state"].values()
                )
            )
            with self.assertRaisesRegex(ValueError, "already exhausted"):
                run_overfit(continued)

    def test_cli_fails_when_sanity_gate_is_not_met(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / "summary.json").write_text(json.dumps({"passed": False}))
            with patch("dnqsm.overfit.run_overfit", return_value=root):
                with self.assertRaises(SystemExit) as error:
                    main(["--device", "cpu"])
                self.assertEqual(error.exception.code, 2)
                (root / "summary.json").write_text(json.dumps({"passed": True}))
                main(["--device", "cpu"])

    def test_gate_requires_all_three_errors_and_loss_reduction(self):
        config = OverfitConfig()
        metrics = dict(chi_nrmse=5.0, local_nrmse=8.0, physics_nrmse=9.0, loss=0.1)
        self.assertTrue(passes_sanity(metrics, 1, config))
        for name in ("chi_nrmse", "local_nrmse", "physics_nrmse"):
            self.assertFalse(passes_sanity(dict(metrics, **{name: 11}), 1, config))
            self.assertFalse(passes_sanity(dict(metrics, **{name: None}), 1, config))
        self.assertFalse(passes_sanity(dict(metrics, loss=0.3), 1, config))

    def test_full_budget_can_continue_after_passing_without_losing_best(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            simulation, *_ = cosmos_fixture(root)
            config = OverfitConfig(
                cosmos_simulation=str(simulation),
                run_dir=str(root / "budget"),
                device="cpu",
                steps=2,
                features=(2, 4),
                threads=2,
                log_every=1,
                figure_every=2,
                bf16=False,
                stop_on_pass=False,
            )
            with (
                patch("dnqsm.overfit.passes_sanity", side_effect=[False, True, False]),
                patch("dnqsm.overfit.reconstruction_figure", return_value=None),
                patch("dnqsm.overfit.RunLogger.figure"),
            ):
                directory = run_overfit(config)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["steps"], 2)
            self.assertEqual(summary["best_step"], 1)
            self.assertTrue(summary["passed"])
            self.assertFalse(summary["final_passed"])

    def test_fixed_case_loop_saves_initial_final_and_failure_without_pass(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            simulation, *_ = cosmos_fixture(root)
            config = OverfitConfig(
                cosmos_simulation=str(simulation),
                run_dir=str(root / "overfit"),
                device="cpu",
                steps=4,
                features=(2, 4),
                threads=2,
                log_every=2,
                figure_every=4,
                bf16=False,
                target_nrmse=0.001,
            )
            with (
                patch("dnqsm.overfit.reconstruction_figure", return_value=None),
                patch("dnqsm.overfit.RunLogger.figure"),
            ):
                directory = run_overfit(config)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["status"], "failed_sanity")
            self.assertEqual(summary["steps"], 4)
            self.assertIn("physics", summary["initial"])
            self.assertIn("physics_nrmse", summary["final"])
            events = [
                json.loads(line)
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["step"] for row in events], [0, 2, 4])
            self.assertTrue((directory / "model_best.pt").is_file())
            self.assertTrue((directory / "evaluation/cosmos/comparison.png").is_file())
            self.assertTrue(
                json.loads((directory / "case.json").read_text())[
                    "fresh_initialization"
                ]
            )


if __name__ == "__main__":
    unittest.main()
