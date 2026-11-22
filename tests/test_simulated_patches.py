import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import default_collate

from dnqsm.data import (
    ShapeFieldSynthesizer,
    TrainingVolumes,
    simulated_forward_field,
    training_records,
)
from dnqsm.losses import LossWeights
from dnqsm.model import DNQSM
from dnqsm.physics import apply_dipole, dipole_tensor
from dnqsm.training import TrainConfig, backward_batch, parse_args, run_training
from tests.helpers import dual_field_model_config as ModelConfig
from tests.helpers import make_dataset


class SimulatedPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def batch(self, root, shape=(20, 22, 24)):
        source = make_dataset(root, shape=shape)
        dataset = TrainingVolumes(
            training_records(source)[:2],
            root / "cache",
            seed=7,
            chi_scale_mode="rms_mixture",
        )
        return default_collate([dataset[0], dataset[1]])

    def test_crop_after_background_and_noise_preserves_alignment_and_units(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = self.batch(Path(temp))
            synthesizer = ShapeFieldSynthesizer("cpu")
            full = synthesizer(batch, background_mode="ratio", snr_min=5, snr_max=5)
            cropped = synthesizer(
                batch, patch_size=8, background_mode="ratio", snr_min=5, snr_max=5
            )
            for key in (
                "phase",
                "mask",
                "chi",
                "local",
                "local_full",
                "background",
                "noise",
            ):
                self.assertEqual(cropped[key].shape, (2, 1, 8, 8, 8))
                for i, (d, h, w) in enumerate(cropped["patch_start"]):
                    torch.testing.assert_close(
                        cropped[key][i],
                        full[key][i, :, d : d + 8, h : h + 8, w : w + 8],
                        atol=0,
                        rtol=0,
                    )
            self.assertTrue(bool(cropped["mask"].flatten(1).any(1).all()))
            torch.testing.assert_close(cropped["dipole"], dipole_tensor((8, 8, 8)))
            torch.testing.assert_close(
                cropped["phase"],
                cropped["local"] + cropped["background"] + cropped["noise"],
            )
            # Tissue outside a patch matters: resimulating only its chi changes the target.
            isolated = apply_dipole(cropped["chi"], cropped["dipole"])
            self.assertGreater(
                float((isolated - cropped["local_full"]).abs().max()), 1e-5
            )

    def test_patches_are_batch_independent_and_follow_view_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = self.batch(Path(temp))
            synth = ShapeFieldSynthesizer("cpu")
            grouped = synth(batch, patch_size=8)
            singles = [
                synth(
                    {
                        k: v[i : i + 1] if isinstance(v, torch.Tensor) else v[i : i + 1]
                        for k, v in batch.items()
                    },
                    patch_size=8,
                )
                for i in range(2)
            ]
            self.assertEqual(
                grouped["patch_start"], [x["patch_start"][0] for x in singles]
            )
            torch.testing.assert_close(
                grouped["phase"], torch.cat([x["phase"] for x in singles])
            )
            self.assertEqual(
                grouped["patch_start"], synth(batch, patch_size=8)["patch_start"]
            )
            changed = dict(batch, seed=batch["seed"] + 1000)
            self.assertNotEqual(
                grouped["patch_start"], synth(changed, patch_size=8)["patch_start"]
            )
            with self.assertRaisesRegex(ValueError, "does not fit"):
                synth(batch, patch_size=64)

    def test_physics_matches_full_context_projection_and_has_gradients(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = self.batch(Path(temp))
            synth = ShapeFieldSynthesizer("cpu")
            full = synth(batch)
            data = synth(batch, patch_size=8)
            torch.testing.assert_close(
                simulated_forward_field(data["chi"], data), data["local_full"]
            )
            predicted = (
                data["chi"] + 0.01 * torch.randn_like(data["chi"])
            ).requires_grad_()
            contextual = full["chi"].clone()
            for i, (d, h, w) in enumerate(data["patch_start"]):
                contextual[i, :, d : d + 8, h : h + 8, w : w + 8] = predicted[i]
            projected = apply_dipole(contextual, full["dipole"])
            expected = torch.cat(
                [
                    projected[i : i + 1, :, d : d + 8, h : h + 8, w : w + 8]
                    for i, (d, h, w) in enumerate(data["patch_start"])
                ]
            )
            actual = simulated_forward_field(predicted, data)
            torch.testing.assert_close(actual, expected, atol=1e-8, rtol=2e-5)
            loss = (actual - data["local_full"]).square().mean()
            reference = (expected - data["local_full"]).square().mean()
            ga = torch.autograd.grad(loss, predicted)[0]
            gb = torch.autograd.grad(reference, predicted)[0]
            torch.testing.assert_close(ga, gb, atol=1e-10, rtol=2e-5)
            self.assertGreater(float(ga.abs().sum()), 0)

    def test_native_64_network_forward_backward_without_padding(self):
        # Retain five levels, reducing channel counts for a small CPU check.
        model = DNQSM(ModelConfig((1, 1, 1, 1, 1)))
        sizes = []
        hooks = [
            network.bottleneck.register_forward_pre_hook(
                lambda module, args: sizes.append(tuple(args[0].shape[-3:]))
            )
            for network in (model.filter, model.null_network)
        ]
        try:
            x = torch.randn(1, 1, 64, 64, 64)
            result = model(x, torch.ones_like(x), dipole_tensor((64, 64, 64)))
            self.assertEqual(sizes, [(2, 2, 2), (2, 2, 2)])
            self.assertEqual(result.chi.shape, x.shape)
            (
                result.chi.square().mean() + result.harmonic_field.square().mean()
            ).backward()
            for head in (
                model.filter.local_decoder.final,
                model.filter.harmonic_decoder.final,
                model.null_network.final,
            ):
                self.assertIsNotNone(head.weight.grad)
                self.assertTrue(bool(torch.isfinite(head.weight.grad).all()))
        finally:
            for hook in hooks:
                hook.remove()

    def test_joint_training_uses_patch_inputs_and_updates_both_heads(self):
        with tempfile.TemporaryDirectory() as temp:
            batch = self.batch(Path(temp))
            model = DNQSM(ModelConfig((2, 4), full_grid_field=True))
            seen = []
            hook = model.register_forward_pre_hook(
                lambda m, args: seen.append(args[0].shape)
            )
            cfg = TrainConfig(
                device="cpu",
                features=(2, 4),
                patch_size=8,
                bf16=False,
                lambda_extension=1,
                full_grid_field=True,
            )
            try:
                terms, _, count, _ = backward_batch(
                    model,
                    batch,
                    ShapeFieldSynthesizer("cpu"),
                    torch.device("cpu"),
                    LossWeights(harmonic_laplacian=100, field_decomposition=1),
                    cfg,
                    2,
                )
            finally:
                hook.remove()
            self.assertEqual(seen, [torch.Size((2, 1, 8, 8, 8))])
            self.assertEqual(count, 2)
            self.assertTrue(torch.isfinite(torch.tensor(terms["loss"])))
            for head in (
                model.filter.local_decoder.final,
                model.filter.harmonic_decoder.final,
            ):
                self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_patch_resume_matches_uninterrupted_and_validation_stays_full(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            source = make_dataset(root, shape=(16, 18, 20))
            cfg = TrainConfig(
                data_dir=str(source),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "whole"),
                device="cpu",
                epochs=2,
                filter_pretrain_epochs=0,
                training_schedule="joint",
                features=(2, 4),
                patch_size=8,
                num_workers=0,
                threads=2,
                batch_size=3,
                bf16=False,
                evaluate_external=False,
                val_views=1,
            )
            seen = []
            original = DNQSM.forward
            original_infer = DNQSM.infer

            def forward(model, phase, mask, dipole):
                seen.append((model.training, tuple(phase.shape[-3:])))
                return original(model, phase, mask, dipole)

            def infer(model, phase, mask, dipole, **kwargs):
                seen.append((model.training, tuple(phase.shape[-3:])))
                return original_infer(model, phase, mask, dipole, **kwargs)

            with (
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                patch.object(DNQSM, "forward", forward),
                patch.object(DNQSM, "infer", infer),
            ):
                whole = run_training(cfg)
                part = run_training(
                    replace(cfg, run_dir=str(root / "part"), max_steps=1)
                )
                resumed = parse_args(
                    ["--resume", str(part / "training_last.pt"), "--max-steps", "0"]
                )
                with self.assertRaisesRegex(ValueError, "patch_size"):
                    run_training(replace(resumed, patch_size=0))
                run_training(resumed)
            a = torch.load(whole / "training_last.pt", weights_only=True)
            b = torch.load(part / "training_last.pt", weights_only=True)
            for name in a["model"]:
                torch.testing.assert_close(
                    a["model"][name], b["model"][name], rtol=0, atol=0
                )
            self.assertIn((True, (8, 8, 8)), seen)
            self.assertIn((False, (16, 18, 20)), seen)

    def test_profile_original_selection_and_invalid_configurations(self):
        cfg = parse_args(["--config", "configs/dual_field_100epochs.json"])
        self.assertEqual((cfg.patch_size, len(cfg.features)), (64, 5))
        self.assertTrue(cfg.original_samples_only)
        self.assertEqual((cfg.synthetic_fraction, cfg.crop_probability), (0, 0))
        for change in (
            dict(patch_size=-1),
            dict(patch_size=32),
            dict(crop_probability=0.2),
            dict(synthetic_fraction=0.25),
        ):
            with self.assertRaises(ValueError):
                replace(cfg, **change).validate()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identities = {}
            for i in range(107):
                image = f"image_{i:04d}.pt"
                (root / image).touch()
                (root / f"mask_{i:04d}.pt").touch()
                identities[image] = f"qsm:{i % 105:04d}"
            (root / "subjects.json").write_text(json.dumps(identities))
            selected = training_records(root, original_samples_only=True)
            self.assertEqual(len(selected), 105)
            self.assertEqual(Path(selected[-1]["image"]).name, "image_0104.pt")
            (root / "image_0000.pt").unlink()
            with self.assertRaisesRegex(ValueError, "105 original"):
                training_records(root, original_samples_only=True)
