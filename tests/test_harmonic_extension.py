import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.overfit import OverfitConfig
from dnqsm.physics import (
    frequency_laplacian,
    masked_rms,
    smooth_harmonic_extension,
)
from dnqsm.runtime import load_trained_model
from dnqsm.training import (
    TrainConfig,
    checkpoint_training_config,
    model_config,
    parse_args,
)


def centered_reference(field, mask, num_iters=6):
    """User's supplied convolution/Jacobi implementation, before mean restoration."""
    mean = (field * mask).sum((-3, -2, -1), keepdim=True) / mask.sum(
        (-3, -2, -1), keepdim=True
    ).clamp_min(1)
    u = (field - mean) * mask
    kernel = torch.zeros(1, 1, 3, 3, 3)
    kernel[0, 0, 0, 1, 1] = kernel[0, 0, 2, 1, 1] = 1 / 6
    kernel[0, 0, 1, 0, 1] = kernel[0, 0, 1, 2, 1] = 1 / 6
    kernel[0, 0, 1, 1, 0] = kernel[0, 0, 1, 1, 2] = 1 / 6
    count = F.conv3d(mask.float(), kernel * 6, padding=1)
    exterior = F.conv3d(u, kernel * 6, padding=1) / count.clamp_min(1)
    u = torch.where(mask.bool(), u, exterior * (count > 0))
    for _ in range(num_iters):
        u = torch.where(mask.bool(), u, F.conv3d(u, kernel, padding=1))
    return u, mean


class HarmonicExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def example(self):
        field = torch.randn(
            2, 1, 17, 19, 21, generator=torch.Generator().manual_seed(14)
        )
        mask = torch.zeros_like(field)
        mask[0, :, 3:-4, 4:-3, 2:-5] = 1
        mask[1, :, :10, 5:-5, 5:-5] = 1  # Also touch a volume boundary.
        mask[..., 6, 7, 8] = 0  # An interior hole is exterior too.
        return field, mask

    def test_matches_supplied_algorithm_and_preserves_every_interior_voxel(self):
        field, mask = self.example()
        before_field, before_mask = field.clone(), mask.clone()
        centered, mean = centered_reference(field, mask)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = smooth_harmonic_extension(field, mask)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, centered + mean, rtol=1e-6, atol=3e-7)
        torch.testing.assert_close(
            actual[mask.bool()], field[mask.bool()], rtol=0, atol=0
        )
        torch.testing.assert_close(field, before_field, rtol=0, atol=0)
        torch.testing.assert_close(mask, before_mask, rtol=0, atol=0)
        torch.testing.assert_close(
            frequency_laplacian(actual) * mask,
            frequency_laplacian(centered) * mask,
            rtol=2e-5,
            atol=4e-6,
        )

    def test_exterior_input_is_ignored_and_samples_are_independent(self):
        field, mask = self.example()
        extended = smooth_harmonic_extension(field, mask)
        changed = torch.where(mask.bool(), field, torch.full_like(field, 12345))
        torch.testing.assert_close(
            smooth_harmonic_extension(changed, mask), extended, rtol=0, atol=0
        )
        for index in range(2):
            torch.testing.assert_close(
                smooth_harmonic_extension(
                    field[index : index + 1], mask[index : index + 1]
                ),
                extended[index : index + 1],
                rtol=1e-6,
                atol=2e-7,
            )

    def test_constant_field_has_no_mask_edge_laplacian_without_erosion(self):
        _, mask = self.example()
        field = torch.full_like(mask, 4.0)
        naive = frequency_laplacian(field * mask) * mask
        extended = smooth_harmonic_extension(field, mask)
        laplacian = frequency_laplacian(extended) * mask
        self.assertGreater(float(naive.abs().max()), 3)
        self.assertLess(float(laplacian.abs().max()), 2e-5)
        torch.testing.assert_close(extended, field, rtol=0, atol=0)

    def test_gradients_match_centered_reference_and_do_not_use_exterior_input(self):
        field, mask = self.example()
        field.requires_grad_()
        probe = torch.randn_like(field)
        actual = frequency_laplacian(smooth_harmonic_extension(field, mask)) * mask
        grad = torch.autograd.grad((actual * probe).sum(), field)[0]
        centered, _ = centered_reference(field, mask)
        reference = frequency_laplacian(centered) * mask
        expected = torch.autograd.grad((reference * probe).sum(), field)[0]
        torch.testing.assert_close(grad, expected, rtol=2e-5, atol=1e-5)
        self.assertTrue(bool(grad.isfinite().all()))
        self.assertGreater(float(grad[mask.bool()].abs().sum()), 0)
        self.assertEqual(float(grad[~mask.bool()].abs().sum()), 0)

    def test_model_masks_after_extension_and_retains_border_with_conditional_scaling(
        self,
    ):
        field, mask = self.example()
        field = field[:1] * 0.0001 + 1
        mask = mask[:1]
        valid = (F.avg_pool3d(mask, 3, 1, 1) > 1 - 1e-6).float()
        border = mask - valid
        for threshold in (0, 32):
            model = DNQSM(
                ModelConfig(
                    (2, 4),
                    conditional_laplacian_threshold=threshold,
                    laplacian_extension_mode="harmonic",
                )
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
            lap = (
                frequency_laplacian(
                    smooth_harmonic_extension(field * mask / scale, mask)
                )
                * mask
            )
            if threshold:
                interior_scale = masked_rms(lap * scale, valid)
                self.assertGreater(float(scale / interior_scale), threshold)
                expected = mask * lap * scale / interior_scale
                scale = interior_scale
            else:
                expected = lap
            torch.testing.assert_close(captured[0][0], expected, rtol=0, atol=0)
            torch.testing.assert_close(
                captured[0][1], field * mask / scale, rtol=0, atol=0
            )
            self.assertEqual(float(captured[0][0][mask == 0].abs().sum()), 0)
            self.assertGreater(float((captured[0][0] * border).abs().sum()), 0)

    def test_checkpoint_roundtrip_legacy_fft_and_explicit_warm_start(self):
        field, mask = self.example()
        with tempfile.TemporaryDirectory() as temp:
            for iterations in (0, 6):
                model = DNQSM(
                    ModelConfig((2, 4), laplacian_extension_iters=iterations)
                ).eval()
                metadata = asdict(model.config)
                if not iterations:
                    metadata.pop("laplacian_extension_iters")
                path = Path(temp) / f"model_{iterations}.pt"
                torch.save(
                    dict(version=1, model_config=metadata, model=model.state_dict()),
                    path,
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.laplacian_extension_iters, iterations)
                with torch.no_grad():
                    torch.testing.assert_close(
                        loaded.predict_local_field(field, mask),
                        model.predict_local_field(field, mask),
                        rtol=0,
                        atol=0,
                    )
                if not iterations:
                    target = DNQSM(replace(model.config, laplacian_extension_iters=6))
                    target.initialize_from(model)
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(
                            dict(target.named_parameters())[name],
                            parameter,
                            rtol=0,
                            atol=0,
                        )

    def test_configuration_defaults_cli_and_resume_preserve_policy(self):
        self.assertEqual(ModelConfig().laplacian_extension_iters, 6)
        self.assertEqual(OverfitConfig().laplacian_extension_iters, 6)
        self.assertEqual(
            model_config(
                parse_args([], use_default_profile=True)
            ).laplacian_extension_iters,
            6,
        )
        self.assertEqual(checkpoint_training_config({})["laplacian_extension_iters"], 0)
        self.assertEqual(
            parse_args(["--laplacian-extension-iters", "9"]).laplacian_extension_iters,
            9,
        )
        with tempfile.TemporaryDirectory() as temp:
            profile = Path(temp) / "legacy.json"
            profile.write_text(json.dumps({"laplacian_operator": "frequency"}))
            self.assertEqual(
                parse_args(["--config", str(profile)]).laplacian_extension_iters, 0
            )
            for iterations in (0, 6):
                config = asdict(TrainConfig(laplacian_extension_iters=iterations))
                if not iterations:
                    config.pop("laplacian_extension_iters")
                path = Path(temp) / "training.pt"
                torch.save(dict(config=config), path)
                self.assertEqual(
                    parse_args(["--resume", str(path)]).laplacian_extension_iters,
                    iterations,
                )
        for invalid in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "Extension iterations"):
                replace(ModelConfig(), laplacian_extension_iters=invalid)
        field, mask = self.example()
        with self.assertRaisesRegex(ValueError, "Extension iterations"):
            smooth_harmonic_extension(field, mask, -1)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            smooth_harmonic_extension(field, torch.zeros_like(mask))
