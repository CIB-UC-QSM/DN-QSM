import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import (
    frequency_laplacian,
    masked_rms,
    quadratic_exterior_extension,
    smooth_harmonic_extension,
)
from dnqsm.runtime import load_trained_model


def boundary_laplacian(field, mask):
    return (
        frequency_laplacian(quadratic_exterior_extension(field, mask))[
            ..., 1:-1, 1:-1, 1:-1
        ]
        * mask
    )


class QuadraticExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def phantom(self):
        z, y, x = torch.meshgrid(
            *(torch.linspace(-1, 1, n) for n in (25, 27, 29)), indexing="ij"
        )
        mask = ((z / 0.8) ** 2 + (y / 0.75) ** 2 + (x / 0.7) ** 2 < 1)[None, None]
        field = (
            0.3 * z**2 + 0.2 * y**2 - 0.1 * x**2 + 0.1 * x * y + 0.7 * x - 0.3 * z + 1
        )[None, None]
        expected = -2 * (
            0.3 * (2 / 24) ** 2 + 0.2 * (2 / 26) ** 2 - 0.1 * (2 / 28) ** 2
        )
        return field, mask, expected

    def test_retains_quadratic_curvature_at_every_mask_boundary(self):
        field, mask, expected = self.phantom()
        actual = boundary_laplacian(field, mask)
        torch.testing.assert_close(
            actual[mask], torch.full_like(actual[mask], expected), rtol=0, atol=5e-6
        )
        valid = F.avg_pool3d(mask.float(), 3, 1, 1) > 1 - 1e-6
        border = mask & ~valid
        old = frequency_laplacian(smooth_harmonic_extension(field, mask)) * mask
        self.assertGreater(float((old[border] - expected).abs().mean()), 0.01)
        self.assertLess(float((actual[border] - expected).abs().mean()), 2e-6)

    def test_ghost_padding_removes_wraparound_at_all_six_volume_faces(self):
        field, _, expected = self.phantom()
        mask = torch.ones_like(field, dtype=torch.bool)
        actual = boundary_laplacian(field, mask)
        torch.testing.assert_close(
            actual, torch.full_like(actual, expected), rtol=0, atol=5e-6
        )
        affine = torch.arange(29, dtype=torch.float32).expand_as(field)
        self.assertLess(float(boundary_laplacian(affine, mask).abs().max()), 3e-5)

    def test_keeps_boundary_feature_and_exact_interior_without_tapering(self):
        z, y, x = torch.meshgrid(*(torch.arange(25.0) for _ in range(3)), indexing="ij")
        field = torch.exp(-((y - 12) ** 2 + (x - 12) ** 2) / 5)[None, None]
        mask = torch.zeros_like(field, dtype=torch.bool)
        mask[..., 5:20, 4:21, 4:21] = True
        extended = quadratic_exterior_extension(field, mask)
        self.assertEqual(extended.shape, (1, 1, 27, 27, 27))
        torch.testing.assert_close(
            extended[..., 1:-1, 1:-1, 1:-1][mask], field[mask], rtol=0, atol=0
        )
        actual, truth = (
            boundary_laplacian(field, mask),
            frequency_laplacian(field) * mask,
        )
        # This feature lies on the first retained ROI slice, with no normal slope.
        torch.testing.assert_close(
            actual[..., 5, 7:18, 7:18], truth[..., 5, 7:18, 7:18], rtol=2e-5, atol=2e-6
        )
        self.assertGreater(float(actual[..., 5, 12, 12]), 0.7)
        valid = F.avg_pool3d(mask.float(), 3, 1, 1) > 1 - 1e-6
        torch.testing.assert_close(actual[valid], truth[valid], rtol=2e-5, atol=2e-6)

    def test_outside_values_are_ignored_batch_independence_and_degenerate_fallback(
        self,
    ):
        field, mask, _ = self.phantom()
        original = field.clone()
        altered = torch.where(mask, field, torch.full_like(field, 10000))
        expected = quadratic_exterior_extension(field, mask)
        torch.testing.assert_close(
            quadratic_exterior_extension(altered, mask), expected, rtol=0, atol=0
        )
        batch = quadratic_exterior_extension(
            torch.cat((field, field * 2)), torch.cat((mask, mask))
        )
        torch.testing.assert_close(batch[:1], expected, rtol=0, atol=0)
        torch.testing.assert_close(batch[1:], expected * 2, rtol=0, atol=0)
        torch.testing.assert_close(field, original, rtol=0, atol=0)
        for slices in (
            (slice(8, 9), slice(8, 9), slice(8, 9)),
            (slice(None), slice(8, 9), slice(None)),
        ):
            thin = torch.zeros_like(mask)
            thin[(..., *slices)] = True
            actual = boundary_laplacian(torch.full_like(field, 3), thin)
            self.assertTrue(bool(actual.isfinite().all()))
            self.assertLess(float(actual.abs().max()), 5e-6)

    def test_fp32_autocast_and_gradient_through_extrapolated_border(self):
        field, mask, _ = self.phantom()
        field.requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = boundary_laplacian(field, mask)
        self.assertEqual(actual.dtype, torch.float32)
        probe = torch.zeros_like(actual)
        probe[..., 12, 13, 5] = 1  # An ROI boundary with exterior neighbor.
        self.assertTrue(bool(mask[..., 12, 13, 5]))
        self.assertFalse(bool(mask[..., 12, 13, 4]))
        gradient = torch.autograd.grad((actual * probe).sum(), field)[0]
        self.assertTrue(bool(gradient.isfinite().all()))
        self.assertGreater(float(gradient[mask].abs().sum()), 0)
        self.assertEqual(float(gradient[~mask].abs().sum()), 0)
        direction = torch.cos(field.detach())
        epsilon = 0.01
        numerical = (
            (
                boundary_laplacian(field.detach() + epsilon * direction, mask)
                - boundary_laplacian(field.detach() - epsilon * direction, mask)
            )
            * probe
        ).sum() / (2 * epsilon)
        torch.testing.assert_close(
            (gradient * direction).sum(), numerical, rtol=0.01, atol=2e-4
        )

    def test_model_input_and_raw_bypass_use_all_roi_voxels(self):
        field, mask, _ = self.phantom()
        field = field * 0.0001 + 1
        for threshold in (0, 32):
            model = DNQSM(
                ModelConfig((2, 4), conditional_laplacian_threshold=threshold)
            )
            captured = []
            hook = model.filter.register_forward_pre_hook(
                lambda module, args: captured.append(args)
            )
            try:
                model.predict_local_field(field, mask)
            finally:
                hook.remove()
            scale = masked_rms(field * mask, mask)
            lap = boundary_laplacian(field * mask / scale, mask)
            if threshold:
                valid = (F.avg_pool3d(mask.float(), 3, 1, 1) > 1 - 1e-6).float()
                interior_scale = masked_rms(lap * scale, valid)
                self.assertGreater(float(scale / interior_scale), threshold)
                lap = mask * lap * scale / interior_scale
                scale = interior_scale
            torch.testing.assert_close(captured[0][0], lap, rtol=0, atol=0)
            torch.testing.assert_close(
                captured[0][1], field * mask / scale, rtol=0, atol=0
            )

    def test_checkpoint_modes_old_jacobi_defaults_and_explicit_weight_adaptation(self):
        field, mask, _ = self.phantom()
        with tempfile.TemporaryDirectory() as temp:
            for mode in ("harmonic", "quadratic"):
                model = DNQSM(ModelConfig((2, 4), laplacian_extension_mode=mode)).eval()
                metadata = asdict(model.config)
                if mode == "harmonic":
                    metadata.pop("laplacian_extension_mode")
                path = Path(temp) / f"{mode}.pt"
                torch.save(
                    dict(version=1, model_config=metadata, model=model.state_dict()),
                    path,
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.laplacian_extension_mode, mode)
                with torch.no_grad():
                    torch.testing.assert_close(
                        loaded.predict_local_field(field, mask),
                        model.predict_local_field(field, mask),
                        rtol=0,
                        atol=0,
                    )
                target = DNQSM(
                    replace(model.config, laplacian_extension_mode="quadratic")
                )
                target.initialize_from(model)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        target.state_dict()[name], value, rtol=0, atol=0
                    )
        self.assertEqual(ModelConfig().laplacian_extension_mode, "quadratic")
        with self.assertRaisesRegex(ValueError, "extension mode"):
            replace(ModelConfig(), laplacian_extension_mode="invalid")
