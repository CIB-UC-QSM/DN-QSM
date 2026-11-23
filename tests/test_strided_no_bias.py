import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import dipole_tensor
from dnqsm.runtime import load_trained_model


class StridedNoBiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, *, dual=False, refinement=0):
        return ModelConfig(
            (2, 4),
            filter_input_version=3 if dual else 2,
            laplacian_input=not dual,
            conditional_laplacian_threshold=0,
            filter_refinement_features=refinement,
        )

    def inputs(self):
        torch.manual_seed(23)
        field = torch.randn(2, 1, 9, 11, 13) * 0.02
        mask = torch.ones_like(field)
        return field, mask, dipole_tensor(field.shape[-3:])

    def test_all_convolutions_and_normalizations_omit_bias_in_both_architectures(self):
        for dual in (False, True):
            model = DNQSM(self.config(dual=dual, refinement=5))
            self.assertFalse(
                any(name.endswith("bias") for name, _ in model.named_parameters())
            )
            for layer in model.modules():
                if isinstance(layer, (nn.Conv3d, nn.InstanceNorm3d)):
                    self.assertIsNone(layer.bias)
                if isinstance(layer, nn.InstanceNorm3d):
                    self.assertIsNotNone(layer.weight)
                    self.assertTrue(layer.weight.requires_grad)
            for network in (model.filter, model.null_network):
                self.assertEqual(len(network.downsamples), 2)
                for layer in network.downsamples:
                    self.assertIsInstance(layer, nn.Conv3d)
                    self.assertEqual(layer.stride, (2, 2, 2))
                    self.assertEqual(layer.kernel_size, (2, 2, 2))
                    self.assertEqual(layer.in_channels, layer.out_channels)
            self.assertFalse(
                any(isinstance(layer, nn.MaxPool3d) for layer in model.modules())
            )

    def test_odd_volume_shapes_gradients_and_activation_checkpointing(self):
        field, mask, dipole = self.inputs()
        for dual in (False, True):
            model = DNQSM(replace(self.config(dual=dual), checkpoint_blocks=True))
            with patch(
                "torch.nn.functional.max_pool3d",
                side_effect=AssertionError("max pooling"),
            ):
                output = model(field, mask, dipole)
                loss = output.chi.square().mean()
                if output.harmonic_field is not None:
                    loss = loss + output.harmonic_field.square().mean()
                loss.backward()
            self.assertEqual(output.chi.shape, field.shape)
            for network in (model.filter, model.null_network):
                for layer in network.downsamples:
                    self.assertIsNotNone(layer.weight.grad)
                    self.assertTrue(bool(layer.weight.grad.isfinite().all()))
                    self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_zero_input_has_no_learned_additive_offsets(self):
        field, mask, dipole = self.inputs()
        for dual in (False, True):
            model = DNQSM(self.config(dual=dual, refinement=5)).eval()
            with torch.no_grad():
                result = model(torch.zeros_like(field), mask, dipole)
            for value in vars(result).values():
                if value is not None:
                    self.assertEqual(float(value.abs().max()), 0)

    def test_new_and_historical_checkpoints_load_their_exact_architectures(self):
        field, mask, dipole = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            for legacy in (False, True):
                config = (
                    replace(self.config(), downsampling="max_pool", bias=True)
                    if legacy
                    else self.config()
                )
                torch.manual_seed(9)
                model = DNQSM(config).eval()
                metadata = asdict(config)
                if legacy:
                    metadata.pop("downsampling")
                    metadata.pop("bias")
                checkpoint = Path(temp) / f"model_{legacy}.pt"
                torch.save(
                    dict(version=1, model_config=metadata, model=model.state_dict()),
                    checkpoint,
                )
                loaded = load_trained_model(checkpoint)
                self.assertEqual(loaded.config, config)
                with torch.no_grad():
                    actual = loaded(field, mask, dipole)
                    expected = model(field, mask, dipole)
                torch.testing.assert_close(actual.chi, expected.chi, rtol=0, atol=0)
                self.assertEqual(
                    bool(list(loaded.filter.downsamples[0].parameters())), not legacy
                )

    def test_default_downsampling_and_bias_validation(self):
        config = ModelConfig()
        self.assertEqual(config.downsampling, "strided_conv")
        self.assertFalse(config.bias)
        with self.assertRaisesRegex(ValueError, "Downsampling"):
            replace(self.config(), downsampling="invalid")
        with self.assertRaisesRegex(ValueError, "boolean"):
            replace(self.config(), bias=1)

    def test_warm_start_rejects_changes_in_learned_downsampling_or_bias(self):
        config = self.config()
        target = DNQSM(config)
        for source_config in (
            replace(config, downsampling="max_pool"),
            replace(config, bias=True),
        ):
            with self.assertRaisesRegex(
                ValueError, "different downsampling or layer biases"
            ):
                target.initialize_from(DNQSM(source_config))
