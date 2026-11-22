"""Residual output refinement preserves units, signed fields and phase freezing."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from dnqsm.model import DNQSM, PhaseRefinementBlock
from dnqsm.overfit import OverfitConfig
from dnqsm.physics import masked_rms
from dnqsm.runtime import load_trained_model
from dnqsm.training import (
    checkpoint_training_config,
    model_config,
    parse_args,
    set_training_branch,
)
from tests.helpers import laplacian_model_config


class PhaseRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fields(self):
        field = (
            torch.randn(2, 1, 9, 11, 13, generator=torch.Generator().manual_seed(8))
            * 0.01
        )
        field[1] += 10
        mask = torch.zeros_like(field)
        mask[..., 1:-1, 1:-1, 1:-1] = 1
        return field, mask

    def test_identity_initialization_signed_output_and_trainable_residual(self):
        block = PhaseRefinementBlock(32)
        x = torch.randn(1, 1, 9, 11, 13, requires_grad=True)
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)
        self.assertEqual(block.conv1.out_channels, 32)
        self.assertEqual(block.conv2.in_channels, 32)
        self.assertEqual(sum(p.numel() for p in block.parameters()), 1760)
        optimizer = torch.optim.SGD(block.parameters(), lr=0.01)
        block(x).square().mean().backward()
        self.assertGreater(float(block.conv2.weight.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        refined = block(x)
        self.assertFalse(torch.equal(refined, x))
        self.assertTrue(bool((refined < 0).any()))
        refined.square().mean().backward()
        self.assertGreater(float(block.conv1.weight.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_refines_laplacian_sum_before_rescaling_with_checkpointing_and_bf16(self):
        model = DNQSM(
            laplacian_model_config(
                (2, 4),
                checkpoint_blocks=True,
                filter_output_skip=False,
                filter_laplacian_residual=True,
                filter_refinement_features=32,
            )
        )
        field, mask = self.fields()
        field.requires_grad_()
        inputs, raw, refinement_in, refinement_out = [], [], [], []
        with torch.no_grad():
            model.filter.refinement.conv2.weight.normal_(std=0.01)
        handles = [
            model.filter.register_forward_pre_hook(
                lambda _, args: inputs.append(args[0])
            ),
            model.filter.register_forward_hook(lambda _, args, out: raw.append(out)),
            model.filter.refinement.register_forward_pre_hook(
                lambda _, args: refinement_in.append(args[0])
            ),
            model.filter.refinement.register_forward_hook(
                lambda _, args, out: refinement_out.append(out)
            ),
        ]
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                local = model.predict_local_field(field, mask)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(local.dtype, torch.float32)
        self.assertEqual(local.shape, field.shape)
        torch.testing.assert_close(
            refinement_in[0], inputs[0] + raw[0].float(), rtol=0, atol=0
        )
        self.assertFalse(torch.equal(refinement_out[0], refinement_in[0]))
        # Recover the selected physical scale independently from the phase input.
        scale = masked_rms(field * mask, mask)
        lap = torch.nn.functional.conv3d(
            field * mask / scale, model.laplacian_kernel, padding=1
        )
        valid = (torch.nn.functional.avg_pool3d(mask, 3, 1, 1) > 1 - 1e-6).float()
        interior = masked_rms(lap * scale, valid)
        scale = torch.where(scale / interior >= 32, interior, scale)
        torch.testing.assert_close(
            local, refinement_out[0] * scale * mask, rtol=0, atol=0
        )
        local.square().mean().backward()
        for parameter in (
            model.filter.refinement.conv1.weight,
            model.filter.refinement.conv2.weight,
            model.filter.final.weight,
        ):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertEqual(float(field.grad[mask == 0].abs().sum()), 0)
        set_training_branch(model, "chi")
        self.assertFalse(model.filter.refinement.training)
        self.assertTrue(
            all(not p.requires_grad for p in model.filter.refinement.parameters())
        )
        set_training_branch(model, "phase")
        self.assertTrue(model.filter.refinement.training)
        self.assertTrue(
            all(p.requires_grad for p in model.filter.refinement.parameters())
        )

    def test_warm_start_adds_identity_and_preserves_both_networks(self):
        old = DNQSM(laplacian_model_config((2, 4), filter_refinement_features=0)).eval()
        new = DNQSM(
            laplacian_model_config((2, 4), filter_refinement_features=32)
        ).eval()
        new.initialize_from(old)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
        field, mask = self.fields()
        with torch.no_grad():
            torch.testing.assert_close(
                new.predict_local_field(field, mask),
                old.predict_local_field(field, mask),
                rtol=0,
                atol=0,
            )
        with self.assertRaisesRegex(ValueError, "refinement block"):
            old.initialize_from(new)
        resized = DNQSM(laplacian_model_config((2, 4), filter_refinement_features=16))
        with self.assertRaisesRegex(ValueError, "refinement block"):
            resized.initialize_from(new)

    def test_checkpoint_defaults_custom_width_and_legacy_inference(self):
        self.assertEqual(model_config(parse_args([])).filter_refinement_features, 0)
        self.assertEqual(OverfitConfig().filter_refinement_features, 0)
        self.assertEqual(
            model_config(
                parse_args(["--filter-refinement-features", "16"])
            ).filter_refinement_features,
            16,
        )
        self.assertEqual(
            checkpoint_training_config({})["filter_refinement_features"], 0
        )
        field, mask = self.fields()
        with tempfile.TemporaryDirectory() as temporary:
            for width in (0, 7):
                model = DNQSM(
                    laplacian_model_config((2, 4), filter_refinement_features=width)
                ).eval()
                config = asdict(model.config)
                if not width:
                    config.pop("filter_refinement_features")
                path = Path(temporary) / f"{width}.pt"
                torch.save(
                    dict(version=1, model_config=config, model=model.state_dict()), path
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.filter_refinement_features, width)
                with torch.no_grad():
                    torch.testing.assert_close(
                        loaded.predict_local_field(field, mask),
                        model.predict_local_field(field, mask),
                        rtol=0,
                        atol=0,
                    )
        for width in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                laplacian_model_config(filter_refinement_features=width)


if __name__ == "__main__":
    unittest.main()
