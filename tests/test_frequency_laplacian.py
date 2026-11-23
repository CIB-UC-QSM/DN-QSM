import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import frequency_laplacian, frequency_laplacian_kernel, masked_rms
from dnqsm.runtime import load_trained_model


class FrequencyLaplacianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def matlab_reference(self, array):
        shape = array.shape[-3:]
        grids = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
        edges = [
            1 - np.exp(2j * np.pi * k / n) for k, n in zip(grids, shape, strict=True)
        ]
        multiplier = sum(np.conj(edge) * edge for edge in edges)
        return np.fft.ifftn(
            multiplier * np.fft.fftn(array, axes=(-3, -2, -1)), axes=(-3, -2, -1)
        ).real

    def test_matches_matlab_gaussian_and_batched_odd_rectangular_inputs(self):
        grids = np.meshgrid(*(np.linspace(-1, 1, 64) for _ in range(3)), indexing="ij")
        gaussian = np.exp(-5 * sum(axis**2 for axis in grids)).astype(np.float32)
        for array in (
            gaussian,
            np.random.default_rng(3).normal(size=(2, 1, 7, 9, 11)).astype(np.float32),
        ):
            actual = frequency_laplacian(torch.from_numpy(array))
            self.assertEqual(tuple(actual.shape), array.shape)
            np.testing.assert_allclose(
                actual.numpy(), self.matlab_reference(array), rtol=2e-4, atol=5e-6
            )

    def test_matches_periodic_difference_operator_including_wraparound(self):
        volume = torch.randn(2, 1, 9, 10, 13)
        expected = sum(
            2 * volume - volume.roll(1, axis) - volume.roll(-1, axis)
            for axis in (-3, -2, -1)
        )
        torch.testing.assert_close(
            frequency_laplacian(volume), expected, rtol=2e-5, atol=5e-6
        )
        impulse = torch.zeros(1, 1, 8, 10, 12)
        impulse[..., 0, 0, 0] = 1
        actual = frequency_laplacian(impulse)
        self.assertAlmostEqual(float(actual[..., 0, 0, 0]), 6, places=5)
        self.assertAlmostEqual(float(actual[..., -1, 0, 0]), -1, places=5)
        self.assertAlmostEqual(float(actual[..., 0, 0, -1]), -1, places=5)

    def test_zero_dc_positive_spectrum_and_known_fourier_mode(self):
        kernel = frequency_laplacian_kernel((8, 10, 12))
        self.assertEqual(tuple(kernel.shape), (1, 1, 8, 10, 7))
        self.assertEqual(float(kernel[..., 0, 0, 0]), 0)
        self.assertTrue(bool((kernel >= 0).all()))
        self.assertAlmostEqual(float(kernel.max()), 12, places=5)
        shape, frequencies = (7, 9, 11), (1, 2, 3)
        grids = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
        mode = np.cos(
            2
            * np.pi
            * sum(k * x / n for k, x, n in zip(frequencies, grids, shape, strict=True))
        )
        eigenvalue = 4 * sum(
            np.sin(np.pi * k / n) ** 2 for k, n in zip(frequencies, shape, strict=True)
        )
        np.testing.assert_allclose(
            frequency_laplacian(torch.tensor(mode)).numpy(),
            eigenvalue * mode,
            rtol=2e-5,
            atol=4e-6,
        )
        for shape in ((8, 8, 8), (7, 9, 11)):
            torch.testing.assert_close(
                frequency_laplacian(torch.ones(1, 1, *shape)),
                torch.zeros(1, 1, *shape),
                rtol=0,
                atol=2e-6,
            )

    def test_fp32_autocast_and_input_gradient_match_the_self_adjoint_operator(self):
        volume = torch.randn(2, 1, 7, 9, 11, requires_grad=True)
        probe = torch.randn_like(volume)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = frequency_laplacian(volume)
            loss = (actual * probe).sum()
        self.assertEqual(actual.dtype, torch.float32)
        loss.backward()
        torch.testing.assert_close(
            volume.grad, frequency_laplacian(probe), rtol=2e-5, atol=1e-5
        )
        self.assertGreater(float(volume.grad.abs().sum()), 0)

    def test_model_input_is_the_frequency_operator_of_the_masked_normalized_field(self):
        config = ModelConfig(
            (2, 4), conditional_laplacian_threshold=0, laplacian_extension_iters=0
        )
        model = DNQSM(config)
        self.assertIsNone(model.laplacian_kernel)
        field = (torch.randn(1, 1, 9, 11, 13) * 0.02).requires_grad_()
        mask = torch.zeros_like(field)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        captured = []
        hook = model.filter.register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                local = model.predict_local_field(field, mask)
        finally:
            hook.remove()
        normalized = field * mask / masked_rms(field * mask, mask)
        torch.testing.assert_close(
            captured[0], frequency_laplacian(normalized), rtol=0, atol=0
        )
        self.assertEqual(captured[0].dtype, torch.float32)
        local.square().mean().backward()
        self.assertTrue(bool(field.grad.isfinite().all()))
        self.assertEqual(float(field.grad[mask == 0].abs().sum()), 0)

    def test_checkpoint_operator_roundtrip_legacy_defaults_and_explicit_warm_start(
        self,
    ):
        config = ModelConfig((2, 4), conditional_laplacian_threshold=0)
        field = torch.randn(1, 1, 9, 11, 13) * 0.02
        mask = torch.ones_like(field)
        with tempfile.TemporaryDirectory() as temp:
            for operator in ("frequency", "stencil"):
                model = DNQSM(replace(config, laplacian_operator=operator)).eval()
                metadata = asdict(model.config)
                if operator == "stencil":
                    metadata.pop("laplacian_operator")
                path = Path(temp) / f"{operator}.pt"
                torch.save(
                    dict(version=1, model_config=metadata, model=model.state_dict()),
                    path,
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.laplacian_operator, operator)
                with torch.no_grad():
                    torch.testing.assert_close(
                        loaded.predict_local_field(field, mask),
                        model.predict_local_field(field, mask),
                        rtol=0,
                        atol=0,
                    )
            for source_operator, target_operator in (
                ("stencil", "frequency"),
                ("frequency", "stencil"),
            ):
                source = DNQSM(replace(config, laplacian_operator=source_operator))
                target = DNQSM(replace(config, laplacian_operator=target_operator))
                target.initialize_from(source)
                for name, parameter in source.named_parameters():
                    torch.testing.assert_close(
                        dict(target.named_parameters())[name], parameter, rtol=0, atol=0
                    )
                self.assertEqual(
                    target.laplacian_kernel is None, target_operator == "frequency"
                )
        with self.assertRaisesRegex(ValueError, "Laplacian operator"):
            replace(config, laplacian_operator="invalid")
        with self.assertRaisesRegex(ValueError, "FP32"):
            replace(config, laplacian_fp32=False)
