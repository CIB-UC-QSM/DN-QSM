import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import dipole_tensor, masked_rms
from dnqsm.runtime import load_trained_model
from dnqsm.training import TrainConfig, model_config, parse_args


class LaplacianInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_impulse_reaches_filter_as_original_and_exact_laplacian_channels(self):
        model = DNQSM(ModelConfig((2, 4)))
        phase = torch.zeros(2, 1, 9, 11, 13)
        phase[0, 0, 4, 5, 6] = 0.02
        phase[1, 0, 4, 5, 6] = 0.1
        mask = torch.ones_like(phase)
        mask[..., :2, :, :] = 0
        # Values outside the ROI must not enter either input channel.
        phase[..., 0, :, :] = 10
        captured = []
        handle = model.filter.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        output = model(phase, mask, dipole_tensor(phase.shape[-3:]))
        handle.remove()
        self.assertEqual(captured[0].shape, (2, 2, 9, 11, 13))
        self.assertEqual(output.local_field.shape, phase.shape)
        self.assertEqual(output.chi.shape, phase.shape)
        self.assertEqual(model.filter.downs[0][0].in_channels, 2)
        self.assertEqual(model.null_network.downs[0][0].in_channels, 1)
        scale = masked_rms(phase * mask, mask, model.config.scale_floor)
        physical_channels = captured[0] * scale
        torch.testing.assert_close(physical_channels[:, :1], phase * mask)
        expected = torch.zeros_like(phase)
        coefficients = {0: -44 / 13, 1: 3 / 13, 2: 3 / 26, 3: 1 / 13}
        for d in range(-1, 2):
            for h in range(-1, 2):
                for w in range(-1, 2):
                    expected[:, 0, 4 + d, 5 + h, 6 + w] = (
                        coefficients[abs(d) + abs(h) + abs(w)] * phase[:, 0, 4, 5, 6]
                    )
        torch.testing.assert_close(physical_channels[:, 1:], expected)

    def test_laplacian_has_expected_polynomial_response_interior(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        grid = torch.arange(-4, 5, dtype=torch.float32)
        x, y, z = torch.meshgrid(grid, grid, grid, indexing="ij")
        phase = torch.stack((torch.ones_like(x), x * x + y * y + z * z, x * x - y * y))[
            :, None
        ]
        mask = torch.ones_like(phase)
        captured = []
        handle = model.filter.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach())
        )
        with torch.no_grad():
            model(phase, mask, dipole_tensor(phase.shape[-3:]))
        handle.remove()
        laplacian = captured[0][:, 1:] * masked_rms(phase, mask)
        interior = laplacian[..., 1:-1, 1:-1, 1:-1]
        torch.testing.assert_close(
            interior[0], torch.zeros_like(interior[0]), atol=3e-6, rtol=0
        )
        torch.testing.assert_close(
            interior[1], torch.full_like(interior[1], 6), atol=3e-5, rtol=0
        )
        torch.testing.assert_close(
            interior[2], torch.zeros_like(interior[2]), atol=1e-5, rtol=0
        )

    def test_kernel_is_fixed_and_laplacian_preserves_input_gradients(self):
        model = DNQSM(ModelConfig((2, 4)))
        original_kernel = model.laplacian_kernel.clone()
        self.assertEqual(original_kernel.shape, (1, 1, 3, 3, 3))
        self.assertFalse(model.laplacian_kernel.requires_grad)
        self.assertIn("laplacian_kernel", dict(model.named_buffers()))
        self.assertNotIn("laplacian_kernel", dict(model.named_parameters()))
        phase = (torch.randn(1, 1, 9, 11, 13) * 0.01).requires_grad_()
        mask = torch.ones_like(phase)
        captured = []
        handle = model.filter.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        output = model(phase, mask, dipole_tensor(phase.shape[-3:]))
        handle.remove()
        gradient = torch.autograd.grad(
            captured[0][:, 1:].square().mean(), phase, retain_graph=True
        )[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        output.chi.square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(
            model.laplacian_kernel, original_kernel, rtol=0, atol=0
        )
        self.assertIsNone(model.laplacian_kernel.grad)

    def test_checkpoint_roundtrip_and_historical_single_channel_loading(self):
        phase = torch.randn(1, 1, 9, 11, 13) * 0.01
        mask = torch.ones_like(phase)
        dipole = dipole_tensor(phase.shape[-3:])
        with tempfile.TemporaryDirectory() as temporary:
            for enabled in (True, False):
                model = DNQSM(ModelConfig((2, 4), laplacian_input=enabled)).eval()
                config = asdict(model.config)
                if not enabled:
                    config.pop("laplacian_input")
                path = Path(temporary) / f"model_{enabled}.pt"
                torch.save(
                    dict(version=1, model_config=config, model=model.state_dict()), path
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.laplacian_input, enabled)
                self.assertEqual(
                    loaded.filter.downs[0][0].in_channels, 2 if enabled else 1
                )
                with torch.no_grad():
                    actual = loaded(phase, mask, dipole)
                    expected = model(phase, mask, dipole)
                for name in vars(actual):
                    torch.testing.assert_close(
                        getattr(actual, name), getattr(expected, name), rtol=0, atol=0
                    )
                if enabled:
                    torch.testing.assert_close(
                        loaded.laplacian_kernel, model.laplacian_kernel
                    )

    def test_training_defaults_and_legacy_resume_select_correct_architecture(self):
        self.assertTrue(model_config(TrainConfig()).laplacian_input)
        self.assertFalse(
            model_config(parse_args(["--no-laplacian-input"])).laplacian_input
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.pt"
            old_config = asdict(TrainConfig(features=(2, 4)))
            old_config.pop("laplacian_input")
            torch.save(dict(config=old_config), path)
            resumed = parse_args(["--resume", str(path)])
            self.assertFalse(model_config(resumed).laplacian_input)
            profile = Path(temporary) / "new.json"
            profile.write_text(json.dumps(dict(features=[2, 4])))
            self.assertTrue(
                model_config(parse_args(["--config", str(profile)])).laplacian_input
            )


if __name__ == "__main__":
    unittest.main()
