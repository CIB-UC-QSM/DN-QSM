import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import apply_multiplier, cutoff_kernels, dipole_tensor, masked_rms
from dnqsm.runtime import atomic_save, load_trained_model


class SoftCutoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def inputs(self):
        field = (
            torch.randn(1, 1, 12, 12, 12, generator=torch.Generator().manual_seed(17))
            * 0.01
        )
        mask = torch.zeros_like(field)
        mask[..., 2:-2, 2:-2, 2:-2] = 1
        return field * mask, mask, dipole_tensor(field.shape[-3:])

    def test_smooth_inverse_and_complementary_weight(self):
        dipole = torch.tensor(
            [-2 / 3, -0.100001, -0.1, -0.099999, 0, 0.099999, 0.1, 0.100001, 1 / 3]
        )
        inverse, weight = cutoff_kernels(dipole, 0.1, "soft")
        expected = dipole.double() / (dipole.double().square() + 0.1**2)
        torch.testing.assert_close(inverse.double(), expected, rtol=2e-7, atol=5e-7)
        torch.testing.assert_close(inverse * dipole + weight, torch.ones_like(dipole))
        self.assertEqual(float(inverse[4]), 0)
        self.assertEqual(float(weight[4]), 1)
        self.assertAlmostEqual(float(weight[2]), 0.5)
        self.assertLessEqual(float(inverse.abs().max()), 5.000001)
        self.assertLess(float((inverse[1] - inverse[3]).abs()), 1e-5)
        self.assertTrue(bool(((weight > 0) & (weight <= 1)).all()))
        self.assertGreater(float((weight.square() - weight).abs().max()), 0.1)

    def test_fp32_under_autocast_and_finite_gradients(self):
        dipole = torch.linspace(-2 / 3, 1 / 3, 101, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            inverse, weight = cutoff_kernels(dipole, 0.1, "soft")
        self.assertEqual(inverse.dtype, torch.float32)
        self.assertEqual(weight.dtype, torch.float32)
        (inverse.square().mean() + weight.square().mean()).backward()
        self.assertTrue(bool(torch.isfinite(dipole.grad).all()))
        for mode, threshold in (("unknown", 0.1), ("soft", 0), ("soft", float("nan"))):
            with self.assertRaises(ValueError):
                cutoff_kernels(dipole, threshold, mode)
        with self.assertRaises(ValueError):
            ModelConfig(cutoff_mode="unknown")

    def test_full_volume_uses_soft_inverse_and_same_correction_weight(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        field, mask, dipole = self.inputs()
        output = model(field, mask, dipole)
        inverse, weight = cutoff_kernels(dipole, 0.1, "soft")
        initial = apply_multiplier(output.local_field, inverse)
        scale = masked_rms(initial, mask, model.config.scale_floor)
        raw = model.null_network(initial / scale).float() * scale
        torch.testing.assert_close(output.tkd, initial)
        torch.testing.assert_close(output.correction, apply_multiplier(raw, weight))
        torch.testing.assert_close(output.chi, initial + output.correction)
        output.chi.square().mean().backward()
        self.assertTrue(
            any(
                p.grad is not None and bool((p.grad != 0).any())
                for p in model.filter.parameters()
            )
        )
        self.assertTrue(
            any(
                p.grad is not None and bool((p.grad != 0).any())
                for p in model.null_network.parameters()
            )
        )

    def test_sliding_windows_keep_soft_fourier_policy(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        field, mask, dipole = self.inputs()
        output = model.infer(field, mask, dipole, window_size=8)
        inverse, weight = cutoff_kernels(dipole, 0.1, "soft")
        torch.testing.assert_close(
            output.tkd, apply_multiplier(output.local_field, inverse)
        )
        hard_initial = apply_multiplier(
            output.local_field, cutoff_kernels(dipole, 0.1, "hard")[0]
        )
        self.assertGreater(float((output.tkd - hard_initial).abs().max()), 1e-6)
        torch.testing.assert_close(output.chi, output.tkd + output.correction)
        self.assertGreater(float(weight.min()), 0)

    def test_checkpoint_roundtrip_and_historical_hard_fallback(self):
        field, mask, dipole = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            for mode in ("soft", "hard"):
                model = DNQSM(ModelConfig((2, 4), cutoff_mode=mode)).eval()
                config = asdict(model.config)
                if mode == "hard":
                    config.pop("cutoff_mode")
                path = Path(temp) / f"{mode}.pt"
                atomic_save(
                    dict(version=1, model_config=config, model=model.state_dict()), path
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.cutoff_mode, mode)
                torch.testing.assert_close(
                    loaded(field, mask, dipole).chi,
                    model(field, mask, dipole).chi,
                    rtol=0,
                    atol=0,
                )

    def test_default_cutoff_and_initialization_from_hard_cutoff(self):
        config = ModelConfig()
        self.assertEqual((config.cutoff_mode, config.threshold), ("soft", 0.1))
        source = DNQSM(ModelConfig((2, 4), cutoff_mode="hard", threshold=0.05))
        target = DNQSM(replace(source.config, cutoff_mode="soft", threshold=0.1))
        target.initialize_from(source)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
