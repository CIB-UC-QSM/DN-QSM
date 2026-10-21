import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import default_collate

from dnqsm.data import FieldSynthesizer, TrainingVolumes, training_records
from dnqsm.evaluation import EvaluationCase
from dnqsm.experiment_metrics import (
    paired_summary,
    reconstruction_metrics,
    subject_average,
)
from dnqsm.model import DNQSM, ModelConfig, Reconstruction
from dnqsm.physics import apply_dipole, apply_multiplier, cutoff_kernels, dipole_tensor
from dnqsm.runtime import CHECKPOINT_VERSION, atomic_save
from dnqsm.staged_experiment import (
    arm_step,
    configure_arm,
    forward_arm,
    make_anchor,
    phase_digest,
    run_arm,
    study_datasets,
    summarize,
    verify_draws,
)
from dnqsm.training import TrainConfig
from tests.helpers import make_dataset


class StagedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_forward_matches_model_and_only_null_gets_gradients(self):
        model = DNQSM(ModelConfig((2, 4)))
        generator = torch.Generator().manual_seed(18)
        phase = torch.randn(1, 1, 16, 16, 16, generator=generator) * 0.02
        mask = torch.ones_like(phase)
        dipole = dipole_tensor((16, 16, 16))
        batch = dict(phase=phase, mask=mask, dipole=dipole)
        model.eval()
        direct = model(phase, mask, dipole)
        configure_arm(model, "frozen")
        frozen = forward_arm(model, batch, "frozen")
        for key in ["local_field", "tkd", "chi", "correction"]:
            torch.testing.assert_close(
                getattr(direct, key), getattr(frozen, key), rtol=0, atol=0
            )
        frozen.chi.square().mean().backward()
        self.assertFalse(model.filter.training)
        self.assertTrue(all(p.grad is None for p in model.filter.parameters()))
        self.assertTrue(
            any(
                p.grad is not None and bool(p.grad.abs().sum())
                for p in model.null_network.parameters()
            )
        )
        self.assertFalse(frozen.local_field.requires_grad)
        configure_arm(model, "joint")
        self.assertTrue(model.filter.training)
        self.assertTrue(all(p.requires_grad for p in model.filter.parameters()))

    def test_one_update_matches_null_and_freezes_phase_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root, shape=(16, 16, 16)))
            dataset = TrainingVolumes(
                records[:2], root / "cache", chi_scale_mode="rms_mixture"
            )
            batch = default_collate([dataset[0], dataset[1]])
            torch.manual_seed(77)
            joint = DNQSM(ModelConfig((2, 4)))
            frozen = DNQSM(joint.config)
            frozen.load_state_dict(joint.state_dict())
            initial_phase = phase_digest(joint)
            config = TrainConfig(
                device="cpu", features=(2, 4), bf16=False, grad_clip=0.01
            )

            def optimizer(model, arm):
                configure_arm(model, arm)
                return torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=3e-5,
                    foreach=False,
                )

            opt_a, opt_b = optimizer(joint, "joint"), optimizer(frozen, "frozen")
            a = arm_step(
                joint,
                batch,
                FieldSynthesizer((16, 16, 16), "cpu"),
                opt_a,
                config,
                "joint",
                diagnostics=True,
            )
            b = arm_step(
                frozen,
                batch,
                FieldSynthesizer((16, 16, 16), "cpu"),
                opt_b,
                config,
                "frozen",
                diagnostics=True,
            )
            self.assertEqual(initial_phase, phase_digest(frozen))
            self.assertNotEqual(initial_phase, phase_digest(joint))
            for key, value in joint.null_network.state_dict().items():
                torch.testing.assert_close(
                    value, frozen.null_network.state_dict()[key], rtol=1e-5, atol=1e-8
                )
            self.assertAlmostEqual(
                a["null_gradient_norm"], b["null_gradient_norm"], places=6
            )
            self.assertEqual(b["phase_gradient_norm"], 0.0)

    def test_metric_bands_and_regression_have_known_values(self):
        generator = torch.Generator().manual_seed(13)
        chi = torch.randn(2, 1, 12, 12, 12, generator=generator) * 0.03
        mask = torch.ones_like(chi)
        dipole = dipole_tensor(chi.shape[-3:])
        local = apply_dipole(chi, dipole)
        output = Reconstruction(local, chi, torch.zeros_like(chi), chi * 0.8)
        rows = reconstruction_metrics(output, chi, mask, dipole, local)
        for row in rows:
            self.assertAlmostEqual(row["chi_nrmse"], 20.0, places=4)
            self.assertAlmostEqual(row["slope"], 0.8, places=5)
            self.assertAlmostEqual(row["retained_nrmse"], 20.0, places=4)
            self.assertAlmostEqual(row["near_null_nrmse"], 20.0, places=4)
            self.assertAlmostEqual(row["hfen_percent"], 20.0, places=4)
        _, projection = cutoff_kernels(dipole, 0.2)
        correction = apply_multiplier(chi, projection)
        projected = Reconstruction(local, chi, correction, chi + correction)
        rows = reconstruction_metrics(projected, chi, mask, dipole, local, hfen=False)
        self.assertTrue(all(r["retained_nrmse"] < 1e-4 for r in rows))
        self.assertTrue(all(abs(r["near_null_nrmse"] - 100) < 1e-3 for r in rows))

    def test_subject_average_and_bootstrap_are_paired_by_subject(self):
        rows = [
            dict(subject="a", profile="baseline", seed=0, chi_nrmse=10.0),
            dict(subject="a", profile="bimodal", seed=1, chi_nrmse=30.0),
            dict(subject="b", profile="baseline", seed=2, chi_nrmse=40.0),
        ]
        a = subject_average(rows)
        self.assertEqual(a["a"]["chi_nrmse"], 20.0)
        b = {k: dict(chi_nrmse=v["chi_nrmse"] - 2) for k, v in a.items()}
        summary = paired_summary(a, b, samples=100)
        self.assertEqual(summary["delta_joint_minus_frozen_pp"], 2.0)
        self.assertEqual(summary["paired_subject_bootstrap_95ci"], [2.0, 2.0])
        with self.assertRaises(ValueError):
            paired_summary(a, {"a": b["a"]})

    def test_seed_replicates_preserve_split_and_share_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root, shape=(16, 16, 16)))
            split = dict(train=records[:4] + records[6:], validation=records[4:6])
            (root / "split.json").write_text(json.dumps(split))
            config = TrainConfig(
                cache_dir=str(root / "cache"),
                device="cpu",
                features=(2, 4),
                chi_augmentation={"final_grid_rms": True},
            )
            train_a, suites_a = study_datasets(root, config, 17)
            train_b, suites_b = study_datasets(root, config, 23)
            self.assertEqual(train_a.records, train_b.records)
            self.assertNotEqual(train_a[0]["seed"], train_b[0]["seed"])
            torch.testing.assert_close(
                suites_a["validation"][0]["chi"],
                suites_b["validation"][0]["chi"],
                rtol=0,
                atol=0,
            )
            model = DNQSM(ModelConfig((2, 4)))
            atomic_save(
                dict(
                    version=CHECKPOINT_VERSION,
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                ),
                root / "phase_pretrain20.pt",
            )
            anchor = make_anchor(root, 17, config)
            digest = anchor.read_bytes()
            self.assertEqual(make_anchor(root, 17, config).read_bytes(), digest)
            saved_a = torch.load(anchor, weights_only=True)
            saved_b = torch.load(make_anchor(root, 23, config), weights_only=True)
            self.assertTrue(
                all(
                    torch.equal(saved_a["model"][k], saved_b["model"][k])
                    for k in saved_a["model"]
                    if k.startswith("filter.")
                )
            )
            self.assertTrue(
                any(
                    not torch.equal(saved_a["model"][k], saved_b["model"][k])
                    for k in saved_a["model"]
                    if k.startswith("null_network.")
                )
            )

    def test_paired_runner_completes_on_cpu_and_reuses_completed_endpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = training_records(make_dataset(root, shape=(16, 16, 16)))
            (root / "split.json").write_text(
                json.dumps(
                    dict(train=records[:4] + records[6:], validation=records[4:6])
                )
            )
            config = TrainConfig(
                cache_dir=str(root / "cache"),
                device="cpu",
                features=(2, 4),
                bf16=False,
                num_workers=0,
                batch_size=2,
                val_views=1,
                chi_augmentation={"final_grid_rms": True},
            )
            model = DNQSM(ModelConfig((2, 4)))
            atomic_save(
                dict(
                    version=CHECKPOINT_VERSION,
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                ),
                root / "phase_pretrain20.pt",
            )
            dataset = TrainingVolumes(
                records[:1], root / "cache", chi_scale_mode="rms_mixture"
            )
            sample = dataset[0]
            chi, mask = sample["chi"][None], sample["mask"][None].float()
            dipole = dipole_tensor(chi.shape[-3:])
            local = apply_dipole(chi * mask, dipole) * mask
            cases = [
                EvaluationCase("cosmos", local, mask, dipole, chi, local),
                EvaluationCase("sim2", local, mask, dipole, chi),
            ]
            manifest = dict(updates_per_arm=2, eval_every=1, mode="unit_test")
            with redirect_stdout(io.StringIO()):
                a = run_arm(
                    root, 17, "joint", config, manifest, torch.device("cpu"), cases
                )
                b = run_arm(
                    root, 17, "frozen", config, manifest, torch.device("cpu"), cases
                )
                again = run_arm(
                    root, 17, "frozen", config, manifest, torch.device("cpu"), cases
                )
                summary = summarize(root, {17: {"joint": a, "frozen": b}}, manifest)
            self.assertEqual(b, again)
            self.assertIsNone(b["external"]["sim2"]["local_nrmse"])
            self.assertTrue(max(b["phase_prediction_drift_nrmse"]) < 1e-4)
            verify_draws(root, 17)
            self.assertEqual(summary["pairs"]["17"]["validation"]["subjects"], 2)
            self.assertTrue((root / "learning_curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
