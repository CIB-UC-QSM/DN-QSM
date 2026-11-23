import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import (
    apply_dipole,
    apply_multiplier,
    cutoff_kernels,
    dipole_tensor,
)
from dnqsm.runtime import load_trained_model


class CheckpointCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tkd_uses_full_grid_field_and_legacy_loading_preserves_mask(self):
        generator = torch.Generator().manual_seed(71)
        chi = torch.randn(1, 1, 12, 12, 12, generator=generator) * 0.03
        mask = torch.zeros_like(chi)
        mask[..., 3:-3, 3:-3, 3:-3] = 1
        dipole = dipole_tensor(chi.shape[-3:])
        full = apply_dipole(chi * mask, dipole)
        model = DNQSM(
            ModelConfig(
                (2, 4),
                full_grid_field=True,
                filter_input_version=1,
                bias=True,
                filter_laplacian_residual=False,
                conditional_laplacian_threshold=0,
            )
        )
        with torch.no_grad():
            model.filter.final.weight.zero_()
            model.filter.final.bias.fill_(1)
        result = model(full, mask, dipole)
        self.assertGreater(float(result.local_field[mask == 0].abs().sum()), 0)
        inverse, _ = cutoff_kernels(
            dipole, model.config.threshold, model.config.cutoff_mode
        )
        torch.testing.assert_close(
            result.tkd, apply_multiplier(result.local_field, inverse)
        )
        legacy = DNQSM(replace(model.config, full_grid_field=False)).eval()
        legacy.load_state_dict(model.state_dict())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            old_config = asdict(legacy.config)
            old_config.pop("full_grid_field")
            torch.save(
                dict(version=1, model_config=old_config, model=legacy.state_dict()),
                path,
            )
            loaded = load_trained_model(path)
            self.assertFalse(loaded.config.full_grid_field)
            torch.testing.assert_close(
                loaded(full, mask, dipole).chi, legacy(full, mask, dipole).chi
            )
        # Precision policy stays as requested: convolution outputs remain BF16
        # under autocast, with FP32 FFT/rescaling as in the existing pipeline.
        dtypes = []
        handle = model.filter.final.register_forward_hook(
            lambda module, args, value: dtypes.append(value.dtype)
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(full, mask, dipole)
        handle.remove()
        self.assertEqual(dtypes, [torch.bfloat16])

    def test_weights_load_with_historical_objective_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = DNQSM(ModelConfig((2, 4)))
            config = dict(device="cpu", features=(2, 4), lambda_forward=1.0)
            checkpoint = root / "old_objective.pt"
            torch.save(
                dict(
                    version=1,
                    loss_version=3,
                    config=config,
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                ),
                checkpoint,
            )
            loaded = load_trained_model(checkpoint)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    loaded.state_dict()[name], value, rtol=0, atol=0
                )
