"""Regression coverage for derivative precision and checkpoint compatibility."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from dnqsm.model import DNQSM
from dnqsm.runtime import load_trained_model
from tests.helpers import laplacian_model_config


class LaplacianPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def capture(self, model, field):
        captured = []
        handle = model.filter.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        try:
            with torch.autocast(field.device.type, dtype=torch.bfloat16):
                output = model.predict_local_field(field, torch.ones_like(field))
        finally:
            handle.remove()
        return captured[0], output

    def test_constant_field_rejected_under_bf16_and_input_gradients_survive(self):
        model = DNQSM(laplacian_model_config((2, 4), conditional_laplacian_threshold=0))
        field = torch.ones(1, 1, 16, 16, 16, requires_grad=True)
        feature, output = self.capture(model, field)
        self.assertEqual(feature.dtype, torch.float32)
        self.assertLess(
            float(feature[..., 2:-2, 2:-2, 2:-2].detach().abs().max()), 1e-6
        )
        output.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(field.grad).all()))
        self.assertGreater(float(field.grad.abs().sum()), 0)

    def test_historical_checkpoint_keeps_original_autocast_response(self):
        model = DNQSM(
            laplacian_model_config(
                (2, 4), laplacian_fp32=False, conditional_laplacian_threshold=0
            )
        ).eval()
        config = asdict(model.config)
        config.pop("laplacian_fp32")
        field = torch.ones(1, 1, 16, 16, 16)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.pt"
            torch.save(
                dict(version=1, model_config=config, model=model.state_dict()), path
            )
            loaded = load_trained_model(path)
            self.assertFalse(loaded.config.laplacian_fp32)
            with torch.no_grad():
                feature, output = self.capture(loaded, field)
                _, expected = self.capture(model, field)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertAlmostEqual(
                float(feature[..., 2:-2, 2:-2, 2:-2].float().mean()), -0.0078125
            )

    def test_new_checkpoint_precision_roundtrip(self):
        model = DNQSM(laplacian_model_config((2, 4)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new.pt"
            torch.save(
                dict(
                    version=1,
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                ),
                path,
            )
            self.assertTrue(load_trained_model(path).config.laplacian_fp32)
