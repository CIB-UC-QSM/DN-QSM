import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import torch
from PIL import Image

from dnqsm.compare import compare, load_comparison
from dnqsm.evaluation import EvaluationCase
from dnqsm.inference import SlidingWindowConfig
from dnqsm.model import DNQSM
from dnqsm.patch_data import (
    BACKGROUND_OPTIONS,
    PATCH_SIZE,
    SCHEMA_VERSION,
    QSMPatchDataset,
)
from dnqsm.physics import (
    apply_dipole,
    apply_multiplier,
    clipped_tkd,
    cutoff_kernels,
    linear_gradient_descent,
)
from dnqsm.train_comparison import (
    TrainingConfig,
    _fit,
    _start_experiment,
    train_comparison,
)


class BaselineComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_five_epochs_include_each_partial_batch_and_log_epoch(self):
        loader = torch.utils.data.DataLoader([1, 2, 3], batch_size=2)
        experiment = MagicMock()
        model = torch.nn.Linear(1, 1)
        with (
            TemporaryDirectory() as tmp,
            patch("dnqsm.train_comparison.make_patch_dataloader") as make_loader,
            patch("dnqsm.train_comparison._update", return_value=0.25) as update,
            patch("dnqsm.train_comparison._validate", return_value=(50.0, None)),
            patch("dnqsm.train_comparison._log_validation_image"),
        ):
            train = MagicMock(wraps=loader)
            train.__iter__.side_effect = lambda: iter(loader)
            train.__len__.return_value = len(loader)
            train.dataset = Mock()
            make_loader.return_value = train
            result = _fit(
                model,
                "chi",
                None,
                None,
                Path(tmp),
                "model1",
                torch.device("cpu"),
                TrainingConfig(steps=None, epochs=5),
                {},
                experiment,
            )
            self.assertTrue((Path(tmp) / "model1.pt").is_file())
        self.assertEqual(
            [len(call.args[2]) for call in update.call_args_list], [2, 1] * 5
        )
        self.assertEqual(
            [call.kwargs for call in experiment.log_metrics.call_args_list],
            [{"step": 2 * epoch, "epoch": epoch} for epoch in range(1, 6)],
        )
        self.assertEqual([row["epoch"] for row in result["history"]], list(range(1, 6)))

    def test_clipped_tkd_keeps_weak_frequencies(self):
        field = torch.randn(1, 1, 4, 5, 6)
        dipole = torch.tensor([0, -0.01, 0.1, 0.2]).expand(4, 5, 4)
        inverse = torch.tensor([0, -10, 10, 5]).expand_as(dipole)
        torch.testing.assert_close(
            clipped_tkd(field, dipole), apply_multiplier(field, inverse)
        )

    def test_full_dipole_gradient_descent_matches_explicit_updates(self):
        field = torch.randn(3, 1, 5, 7, 9)
        dipole = torch.zeros(3, 1, 5, 7, 5)
        for sample, factor in enumerate((1, 0.5)):
            dipole[sample, ..., 1] = 0.03 * factor
            dipole[sample, ..., 2] = -0.2 * factor
        power = dipole.square().amax(dim=(-3, -2, -1), keepdim=True)
        step = torch.where(power > 0, power, 1).reciprocal()
        expected = torch.zeros_like(field)
        for _ in range(30):
            expected -= step * apply_dipole(
                apply_dipole(expected, dipole) - field, dipole
            )
        torch.testing.assert_close(linear_gradient_descent(field, dipole), expected)
        self.assertGreater(float(expected[0].abs().sum()), 0)
        self.assertEqual(float(expected[2].abs().sum()), 0)

    def test_train_reload_and_compare(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(4)
            chi = rng.normal(0, 0.03, (1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)).astype(
                np.float32
            )
            local = chi * 0.5
            sources = []
            for split in ("train", "validation"):
                sources.append(dict(source_id=split, split=split, starts=[[0, 0, 0]]))
                folder = root / "patches" / split
                folder.mkdir(parents=True)
                for name, value in dict(
                    chi=chi,
                    local=local,
                    local_scaled=local * 2,
                    mask=np.ones_like(chi, dtype=bool),
                    background=np.stack(
                        [np.full_like(chi, i / 20) for i in range(BACKGROUND_OPTIONS)],
                        axis=1,
                    ),
                ).items():
                    np.save(folder / f"{name}.npy", value)
            (root / "manifest.json").write_text(
                json.dumps(
                    dict(
                        schema_version=SCHEMA_VERSION,
                        sources=sources,
                        positive_scale=4,
                    )
                )
            )
            fixed = QSMPatchDataset(root, "validation", fixed_backgrounds=True)
            self.assertEqual(len({fixed[0]["background_option"] for _ in range(16)}), 1)
            run = root / "run"
            training = TrainingConfig(steps=1, validation_interval=2, batch_size=1)
            with patch.dict(os.environ, {"COMET_START_ONLINE": "0"}):
                train_comparison(root, run, features=(2,), training=training)
            manifest = run / "comparison.json"
            self._assert_comet_archives(run, json.loads(manifest.read_text()))

            experiment = Mock()
            with (
                patch(
                    "dnqsm.train_comparison._start_experiment", return_value=experiment
                ),
                patch(
                    "dnqsm.train_comparison._fit",
                    side_effect=RuntimeError("fit failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "fit failed"),
            ):
                train_comparison(
                    root, root / "failed", features=(2,), training=training
                )
            experiment.end.assert_called_once_with()

            models = dict(load_comparison(manifest, "cpu"))
            self.assertEqual(len(models), 8)
            dnqsm = models["model4"]
            self.assertIsInstance(dnqsm, DNQSM)
            self.assertEqual(dnqsm.inference_config.window_size, PATCH_SIZE)
            plain = models["model3_tikhonov"]
            initial = models["model2_tikhonov"]
            for inverse in ("tkd", "tikhonov", "gd"):
                self.assertIs(models[f"model2_{inverse}"].estimator, initial.estimator)
            corrected = [models[f"model3_{inv}"] for inv in ("tkd", "tikhonov", "gd")]
            trained_filters = [model.estimator.filter for model in corrected]
            trained_filters.append(dnqsm.filter)
            self.assertEqual(len({initial.estimator.filter, *trained_filters}), 5)
            torch.manual_seed(training.seed)
            initial_correction = DNQSM(dnqsm.config).null_network.state_dict()
            trained_corrections = [model.residual for model in corrected]
            trained_corrections.append(dnqsm.null_network)
            for trained_group, start in (
                (trained_filters, initial.estimator.filter.state_dict()),
                (trained_corrections, initial_correction),
            ):
                for trained in trained_group:
                    self.assertTrue(
                        any(
                            not torch.equal(value, start[key])
                            for key, value in trained.state_dict().items()
                        )
                    )
            self.assertEqual(
                plain.residual.state_dict().keys(),
                dnqsm.null_network.state_dict().keys(),
            )

            field = torch.from_numpy(local[:, None, :5, :6, :7].copy())
            target = torch.from_numpy(chi[:, None, :5, :6, :7].copy())
            roi = torch.ones_like(field)
            dipole = torch.full(field.shape[-3:], -0.3)
            dipole[0, 0, 0] = 0
            for window_size in (0, 4):
                policy = SlidingWindowConfig(window_size=window_size)
                b = plain.infer(field, roi, dipole, inference=policy)
                c = dnqsm.infer(field, roi, dipole, **asdict(policy))
                for result in (b, c):
                    torch.testing.assert_close(
                        result.chi, result.initial_chi + result.correction
                    )
            dnqsm.filter.load_state_dict(plain.estimator.filter.state_dict())
            dnqsm.null_network.load_state_dict(plain.residual.state_dict())
            b = plain.infer(field, roi, dipole, inference=policy)
            c = dnqsm.infer(field, roi, dipole, **asdict(policy))
            _, weight = cutoff_kernels(dipole, dnqsm.config.threshold, "soft")
            torch.testing.assert_close(
                c.correction, apply_multiplier(b.correction, weight)
            )

            affine = np.diag([1.2, 1.3, 1.4, 1])
            case = EvaluationCase(
                "synthetic", field, roi, dipole, target, field, affine=affine
            )
            output = root / "scores"
            summary = compare(manifest, [case], output, window_size=4)
            record = summary["cases"]["synthetic"]
            self.assertEqual(len(record["arms"]), 8)
            self.assertEqual(len(record["chi_nrmse_deltas_pp"]), 4)
            self.assertIsNone(record["arms"]["model1"]["local_nrmse"])
            direct_dir = output / "synthetic" / "model1"
            tensors = torch.load(
                direct_dir / "padded_predictions.pt", weights_only=True
            )
            self.assertEqual(set(tensors), {"chi", "mask"})
            image = nib.load(direct_dir / "chi.nii.gz")
            np.testing.assert_allclose(image.affine, affine)
            self.assertEqual(image.shape, field.shape[-3:])

    def _assert_comet_archives(self, run, manifest):
        stages = manifest["stages"]
        keys = {stage["comet_experiment_key"] for stage in stages.values()}
        self.assertEqual(len(keys), 6)
        self.assertEqual({path.stem for path in (run / "comet").glob("*.zip")}, keys)
        tags = set()
        for stage in stages.values():
            key = stage["comet_experiment_key"]
            with (
                self.subTest(experiment=key),
                ZipFile(run / "comet" / f"{key}.zip") as archive,
            ):
                tags.add(tuple(json.loads(archive.read("experiment.json"))["tags"]))
                messages = [
                    json.loads(line)
                    for line in archive.read("messages.json").splitlines()
                ]
                metrics = {}
                for message in messages:
                    if message["type"] == "metric_msg":
                        payload = message["payload"]
                        metric = payload["metric"]
                        metrics[payload.get("context"), metric["metricName"]] = metric
                self.assertEqual(metrics["train", "relative_mse"]["step"], 1)
                for name in ("nrmse_pct", "best_nrmse_pct"):
                    metric = metrics["validate", name]
                    self.assertEqual(metric["step"], 1)
                    self.assertEqual(metric["metricValue"], stage["selected"]["nrmse"])
                images = [
                    message["payload"]
                    for message in messages
                    if message["type"] == "file_upload"
                    and message["payload"].get("upload_type") == "visualization"
                ]
                (image,) = images
                self.assertEqual(image["additional_params"]["step"], 1)
                self.assertEqual(
                    image["additional_params"]["figName"],
                    "central-slices-step-000001.png",
                )
                with (
                    archive.open(image["file_path"]) as source,
                    Image.open(source) as png,
                ):
                    self.assertEqual(png.format, "PNG")
                    png.verify()
        self.assertEqual(len(tags), 1)
        (tag,) = next(iter(tags))
        self.assertTrue(tag.startswith("comparison-"))

    def test_comet_mode_uses_sdk_configuration(self):
        for key, configured_online, expected in (
            (None, None, False),
            ("configured-key", None, True),
            ("configured-key", False, False),
        ):
            config = {
                "comet.api_key": key,
                "comet.start.online": configured_online,
                "comet.project_name": "custom-project",
                "comet.experiment_key": None,
            }
            with (
                self.subTest(key=key, configured_online=configured_online),
                patch.dict(os.environ, {"COMET_DISABLE_AUTO_LOGGING": "1"}),
                patch("comet_ml.config.get_config", return_value=config),
                patch("comet_ml.ExperimentConfig"),
                patch("comet_ml.start") as start,
            ):
                _start_experiment(Path("run"), "model1", "comparison-test")
                self.assertIs(start.call_args.kwargs["online"], expected)
                self.assertEqual(
                    start.call_args.kwargs["project_name"], "custom-project"
                )
