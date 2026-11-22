import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from dnqsm.model import DNQSM, ModelConfig, ResidualInstanceNorm3d, instance_norm
from dnqsm.physics import dipole_tensor, masked_rms
from dnqsm.runtime import load_trained_model
from dnqsm.training import (
    TrainConfig,
    checkpoint_training_config,
    parse_args,
    run_training,
)
from tests.helpers import make_dataset


class PhysicalNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **kwargs):
        return ModelConfig(
            (2, 4),
            field_normalization="laplacian_rms",
            normalization_layer="residual_instance",
            conditional_laplacian_threshold=0,
            **kwargs,
        )

    def inputs(self):
        torch.manual_seed(71)
        field = torch.randn(2, 1, 17, 19, 21) * 0.01
        mask = torch.zeros_like(field)
        mask[..., 3:-3, 3:-3, 3:-3] = 1
        return field, mask, dipole_tensor(field.shape[-3:])

    def test_zero_residual_norm_matches_legacy_and_exposes_learnable_amplitude(self):
        legacy = instance_norm(4, False)
        layer = instance_norm(4, False, "residual_instance")
        self.assertIsInstance(layer, ResidualInstanceNorm3d)
        layer.weight.data.copy_(legacy.weight)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(2, 4, 7, 9, 11).to(dtype)
            torch.testing.assert_close(layer(x), legacy(x), rtol=0, atol=0)
        self.assertIsNone(layer.bias)
        with torch.no_grad():
            layer.residual_gain.fill_(0.5)
        x = torch.ones(2, 4, 7, 9, 11)
        output = layer(x)
        self.assertGreater(float(output[:, 2:].abs().mean()), 0.4)
        self.assertLess(float(output[:, :2].abs().max()), 1e-4)
        layer.residual_gain.data.zero_()
        (layer(x) * torch.arange(4)[None, :, None, None, None]).sum().backward()
        self.assertGreater(float(layer.residual_gain.grad.abs()), 0)

    def test_physical_units_offset_and_gain_equivariance_with_active_residuals(self):
        field, mask, dipole = self.inputs()
        model = DNQSM(self.config(full_grid_field=True)).eval()
        for module in model.modules():
            if isinstance(module, ResidualInstanceNorm3d):
                module.residual_gain.data.fill_(0.15)
        with torch.no_grad():
            x, bypass, scale = model._filter_inputs(field * mask, mask)
            shifted = model._filter_inputs((field + 0.2) * mask, mask)
            doubled = model._filter_inputs(2 * field * mask, mask)
            self.assertEqual(x.dtype, torch.float32)
            torch.testing.assert_close(shifted[0], x, rtol=2e-3, atol=1e-5)
            torch.testing.assert_close(shifted[1], bypass, rtol=2e-3, atol=1e-5)
            torch.testing.assert_close(shifted[2], scale, rtol=2e-5, atol=1e-8)
            torch.testing.assert_close(doubled[0], x, rtol=0, atol=0)
            torch.testing.assert_close(doubled[1], bypass, rtol=0, atol=0)
            torch.testing.assert_close(doubled[2], 2 * scale, rtol=0, atol=0)
            output = model(field, mask, dipole)
            shifted_output = model(field + 0.2, mask, dipole)
            doubled_output = model(2 * field, mask, dipole)
        for key in ("local_field", "tkd", "chi"):
            a, b, c = [
                getattr(v, key) for v in (output, shifted_output, doubled_output)
            ]
            relative = float(
                (masked_rms(a - b, mask, 1e-12) / masked_rms(a, mask)).max()
            )
            self.assertLess(relative, 3e-3, key)
            torch.testing.assert_close(c, 2 * a, rtol=0, atol=0)
        valid = (torch.nn.functional.avg_pool3d(mask, 3, 1, 1) > 1 - 1e-6).float()
        torch.testing.assert_close(
            masked_rms(x, valid), torch.ones_like(scale), rtol=1e-6, atol=1e-6
        )
        self.assertGreater(float((x * (mask - valid)).abs().sum()), 0)

    def test_no_interior_falls_back_to_derivative_scale_and_zero_inputs_are_finite(
        self,
    ):
        field, mask, dipole = self.inputs()
        mask.zero_()
        mask[..., 8:10, 9:11, 10:12] = 1
        model = DNQSM(self.config()).eval()
        with torch.no_grad():
            x, bypass, scale = model._filter_inputs(field * mask, mask)
            shifted = model._filter_inputs((field + 1) * mask, mask)
            self.assertTrue(bool(x.isfinite().all() and bypass.isfinite().all()))
            torch.testing.assert_close(shifted[2], scale, rtol=1e-3, atol=1e-7)
            zero = model(torch.zeros_like(field), mask, dipole)
        self.assertEqual(float(zero.chi.abs().max()), 0)
        field.requires_grad_()
        output = model(field, mask, dipole)
        output.chi.square().mean().backward()
        self.assertTrue(bool(field.grad.isfinite().all()))
        self.assertEqual(float(field.grad[mask == 0].abs().sum()), 0)

    def test_all_networks_and_refinement_get_residual_layers_without_bias(self):
        for dual in (False, True):
            config = (
                self.config(filter_refinement_features=3)
                if not dual
                else ModelConfig(
                    (2, 4),
                    filter_input_version=3,
                    laplacian_input=False,
                    conditional_laplacian_threshold=0,
                    normalization_layer="residual_instance",
                    filter_refinement_features=3,
                )
            )
            model = DNQSM(config)
            self.assertFalse(
                any(name.endswith("bias") for name, _ in model.named_parameters())
            )
            norms = [m for m in model.modules() if isinstance(m, nn.InstanceNorm3d)]
            self.assertTrue(norms)
            self.assertTrue(all(isinstance(m, ResidualInstanceNorm3d) for m in norms))
            self.assertTrue(all(float(m.residual_gain) == 0 for m in norms))
            field, mask, dipole = self.inputs()
            output = model(field, mask, dipole)
            loss = output.chi.abs().mean()
            if output.harmonic_field is not None:
                loss = loss + output.harmonic_field.abs().mean()
            loss.backward()
            self.assertTrue(
                all(
                    m.residual_gain.grad is not None
                    and bool(m.residual_gain.grad.isfinite())
                    for m in norms
                )
            )
            self.assertTrue(any(float(m.residual_gain.grad.abs()) > 0 for m in norms))

    def test_checkpoint_compatibility_and_explicit_norm_adaptation(self):
        field, mask, dipole = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            for new in (False, True):
                config = self.config() if new else ModelConfig((2, 4))
                model = DNQSM(config).eval()
                metadata = asdict(config)
                if not new:
                    metadata.pop("field_normalization")
                    metadata.pop("normalization_layer")
                else:
                    for m in model.modules():
                        if isinstance(m, ResidualInstanceNorm3d):
                            m.residual_gain.data.fill_(0.12)
                path = Path(temp) / f"{new}.pt"
                torch.save(
                    dict(version=1, model_config=metadata, model=model.state_dict()),
                    path,
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config, config)
                with torch.no_grad():
                    torch.testing.assert_close(
                        loaded(field, mask, dipole).chi,
                        model(field, mask, dipole).chi,
                        rtol=0,
                        atol=0,
                    )
            source = DNQSM(ModelConfig((2, 4))).eval()
            target = DNQSM(
                replace(source.config, normalization_layer="residual_instance")
            ).eval()
            target.initialize_from(source)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(
                    target.state_dict()[name], value, rtol=0, atol=0
                )
            with torch.no_grad():
                torch.testing.assert_close(
                    target(field, mask, dipole).chi,
                    source(field, mask, dipole).chi,
                    rtol=0,
                    atol=0,
                )
            with self.assertRaisesRegex(
                ValueError, "cannot discard learned normalization"
            ):
                source.initialize_from(target)
        legacy = checkpoint_training_config({})
        self.assertEqual(
            (legacy["field_normalization"], legacy["normalization_layer"]),
            ("legacy", "instance"),
        )
        active = parse_args([], use_default_profile=True)
        self.assertEqual(
            (active.field_normalization, active.normalization_layer),
            ("laplacian_rms", "residual_instance"),
        )
        self.assertEqual(active.conditional_laplacian_threshold, 0)
        for kwargs in (
            dict(field_normalization="unknown"),
            dict(normalization_layer="unknown"),
            dict(field_normalization="laplacian_rms", filter_input_version=1),
        ):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)

    def test_residual_normalization_training_resumes_exactly_and_logs_policy(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = make_dataset(root)
            config = TrainConfig(
                data_dir=str(data),
                cache_dir=str(root / "cache"),
                run_dir=str(root / "whole"),
                device="cpu",
                cuda_memory_gb=0,
                epochs=2,
                filter_pretrain_epochs=0,
                training_schedule="joint",
                features=(2, 4),
                batch_size=3,
                num_workers=0,
                threads=2,
                bf16=False,
                comet=False,
                evaluate_external=False,
                val_views=1,
                field_normalization="laplacian_rms",
                normalization_layer="residual_instance",
            )
            with patch("dnqsm.training.reconstruction_figure", return_value=None):
                whole = run_training(config)
                part = run_training(
                    replace(config, run_dir=str(root / "part"), max_steps=1)
                )
                resumed = parse_args(
                    ["--resume", str(part / "training_last.pt"), "--max-steps", "0"]
                )
                run_training(resumed)
            a, b = [
                torch.load(p / "training_last.pt", weights_only=True)
                for p in (whole, part)
            ]
            for name in a["model"]:
                torch.testing.assert_close(
                    a["model"][name], b["model"][name], rtol=0, atol=0
                )
            self.assertTrue(
                any(
                    float(v.abs()) > 0
                    for n, v in a["model"].items()
                    if n.endswith("residual_gain")
                )
            )
            events = [
                json.loads(line)
                for line in (whole / "metrics.jsonl").read_text().splitlines()
            ]
            started = next(r for r in events if r["event"] == "started")
            self.assertEqual(started["field_normalization"], "laplacian_rms")
            self.assertEqual(started["normalization_layer"], "residual_instance")
            with self.assertRaisesRegex(ValueError, "Resume configuration changed"):
                run_training(replace(resumed, field_normalization="legacy"))
