import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import apply_multiplier, cutoff_kernels, dipole_tensor, masked_rms
from dnqsm.runtime import load_trained_model
from dnqsm.training import TrainConfig, model_config, parse_args


class LaplacianInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_impulse_reaches_filter_as_only_exact_laplacian_channel(self):
        model = DNQSM(ModelConfig((2, 4)))
        phase = torch.zeros(2, 1, 9, 11, 13)
        phase[0, 0, 4, 5, 6] = 0.02
        phase[1, 0, 4, 5, 6] = 0.1
        mask = torch.ones_like(phase)
        mask[..., :2, :, :] = 0
        # Values outside the ROI must not enter the input channel.
        phase[..., 0, :, :] = 10
        captured = []
        handle = model.filter.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        output = model(phase, mask, dipole_tensor(phase.shape[-3:]))
        handle.remove()
        self.assertEqual(captured[0].shape, (2, 1, 9, 11, 13))
        self.assertEqual(output.local_field.shape, phase.shape)
        self.assertEqual(output.chi.shape, phase.shape)
        self.assertEqual(model.filter.downs[0][0].in_channels, 1)
        self.assertEqual(model.null_network.downs[0][0].in_channels, 1)
        scale = masked_rms(phase * mask, mask, model.config.scale_floor)
        physical_channels = captured[0] * scale
        expected = torch.zeros_like(phase)
        coefficients = {0: -44 / 13, 1: 3 / 13, 2: 3 / 26, 3: 1 / 13}
        for d in range(-1, 2):
            for h in range(-1, 2):
                for w in range(-1, 2):
                    expected[:, 0, 4 + d, 5 + h, 6 + w] = (
                        coefficients[abs(d) + abs(h) + abs(w)] * phase[:, 0, 4, 5, 6]
                    )
        torch.testing.assert_close(physical_channels, expected)

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
        laplacian = captured[0] * masked_rms(phase, mask)
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
            captured[0].square().mean(), phase, retain_graph=True
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

    def test_checkpoint_roundtrip_and_both_historical_architectures(self):
        phase = torch.randn(1, 1, 9, 11, 13) * 0.01
        mask = torch.ones_like(phase)
        dipole = dipole_tensor(phase.shape[-3:])
        with tempfile.TemporaryDirectory() as temporary:
            for version, enabled in ((2, True), (1, True), (1, False)):
                model = DNQSM(
                    ModelConfig(
                        (2, 4),
                        laplacian_input=enabled,
                        filter_input_version=version,
                        full_grid_field=version == 1,
                    )
                ).eval()
                config = asdict(model.config)
                if version == 1:
                    config.pop("filter_input_version")
                if not enabled:
                    config.pop("laplacian_input")
                path = Path(temporary) / f"model_{version}_{enabled}.pt"
                torch.save(
                    dict(version=1, model_config=config, model=model.state_dict()), path
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.laplacian_input, enabled)
                self.assertEqual(
                    loaded.filter.downs[0][0].in_channels,
                    2 if version == 1 and enabled else 1,
                )
                self.assertEqual(loaded.config.filter_input_version, version)
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
            model_config(
                parse_args(["--filter-input-version", "1", "--no-laplacian-input"])
            ).laplacian_input
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.pt"
            old_config = asdict(TrainConfig(features=(2, 4)))
            old_config.pop("laplacian_input")
            old_config.pop("filter_input_version")
            old_config.pop("filter_pretrain_epochs")
            torch.save(dict(config=old_config), path)
            resumed = parse_args(["--resume", str(path)])
            self.assertFalse(model_config(resumed).laplacian_input)
            self.assertEqual(resumed.filter_input_version, 1)
            self.assertEqual(resumed.filter_pretrain_epochs, 0)
            profile = Path(temporary) / "new.json"
            profile.write_text(json.dumps(dict(features=[2, 4])))
            self.assertTrue(
                model_config(parse_args(["--config", str(profile)])).laplacian_input
            )

    def test_masked_output_is_the_exact_tkd_input_and_ignores_exterior(self):
        model = DNQSM(ModelConfig((2, 4))).eval()
        phase = (torch.randn(1, 1, 9, 11, 13) * 0.01).requires_grad_()
        mask = torch.zeros_like(phase)
        mask[..., 2:-2, 2:-2, 2:-2] = 1
        dipole = dipole_tensor(phase.shape[-3:])
        with torch.no_grad():
            model.filter.final.weight.zero_()
            model.filter.final.bias.fill_(1)
        output = model(phase, mask, dipole)
        self.assertEqual(float(output.local_field[mask == 0].abs().sum()), 0)
        torch.testing.assert_close(
            output.local_field, masked_rms(phase * mask, mask) * mask
        )
        inverse, _ = cutoff_kernels(dipole, model.config.threshold)
        torch.testing.assert_close(
            output.tkd, apply_multiplier(output.local_field, inverse)
        )
        changed = phase.detach() + (1 - mask) * 1000
        torch.testing.assert_close(
            model(changed, mask, dipole).chi, output.chi, rtol=0, atol=0
        )
        output.local_field.square().mean().backward()
        self.assertEqual(float(phase.grad[mask == 0].abs().sum()), 0)
        with torch.no_grad():
            model.filter.final.bias.zero_()
        self.assertEqual(float(model.predict_local_field(phase, mask).abs().sum()), 0)

    def test_new_architecture_rejects_incompatible_options(self):
        for options in (
            dict(residual_filter=True),
            dict(full_grid_field=True),
            dict(laplacian_input=False),
            dict(filter_input_version=3),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ModelConfig(**options)


if __name__ == "__main__":
    unittest.main()
