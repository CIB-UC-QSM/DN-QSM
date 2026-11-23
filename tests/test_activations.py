"""LeakyReLU covers every learned branch without rewriting old checkpoints."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from dnqsm.model import DNQSM, ModelConfig, UNet3D
from dnqsm.runtime import load_trained_model


class ActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_new_activations_use_leaky_relu_with_negative_gradients(self):
        model = DNQSM(ModelConfig((2, 4), filter_refinement_features=32))
        activations = [
            module
            for module in model.modules()
            if isinstance(module, (nn.LeakyReLU, nn.CELU, nn.ReLU))
        ]
        self.assertEqual(len(activations), 23)
        for activation in activations:
            self.assertIsInstance(activation, nn.LeakyReLU)
            self.assertEqual(activation.negative_slope, 0.01)
            x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
            y = activation(x)
            torch.testing.assert_close(y, torch.tensor([-0.02, -0.005, 0.5, 2.0]))
            y.sum().backward()
            torch.testing.assert_close(x.grad, torch.tensor([0.01, 0.01, 1.0, 1.0]))

    def test_decoder_upsampling_convolutions_use_leaky_relu(self):
        network = UNet3D((2, 4), bias=True).eval()
        with torch.no_grad():
            for conv in network.up_convs:
                conv.weight.zero_()
                conv.bias.fill_(-2)
        outputs = []
        handle = network.up_activation.register_forward_hook(
            lambda _, args, out: outputs.append(out)
        )
        try:
            with torch.no_grad():
                network(torch.randn(1, 1, 9, 11, 13))
        finally:
            handle.remove()
        self.assertEqual(len(outputs), 2)
        for output in outputs:
            torch.testing.assert_close(output, torch.full_like(output, -0.02))

    def test_custom_slope_reaches_both_networks_and_refinement_and_is_saved(self):
        model = DNQSM(
            ModelConfig((2, 4), leaky_relu_slope=0.3, filter_refinement_features=32)
        ).eval()
        activations = [
            module for module in model.modules() if isinstance(module, nn.LeakyReLU)
        ]
        self.assertEqual(len(activations), 23)
        for activation in activations:
            self.assertEqual(activation.negative_slope, 0.3)
            x = torch.tensor([-2.0, 0.5], requires_grad=True)
            y = activation(x)
            torch.testing.assert_close(y, torch.tensor([-0.6, 0.5]))
            y.sum().backward()
            torch.testing.assert_close(x.grad, torch.tensor([0.3, 1.0]))
        field = torch.randn(1, 1, 9, 11, 13)
        mask = torch.ones_like(field)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "slope.pt"
            torch.save(
                dict(
                    version=1,
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                ),
                path,
            )
            loaded = load_trained_model(path)
            self.assertEqual(loaded.config.leaky_relu_slope, 0.3)
            with torch.no_grad():
                torch.testing.assert_close(
                    loaded.predict_local_field(field, mask),
                    model.predict_local_field(field, mask),
                    rtol=0,
                    atol=0,
                )
        for slope in (-0.1, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "slope"):
                ModelConfig(leaky_relu_slope=slope)

    def test_new_and_legacy_checkpoints_preserve_activation_and_allow_warm_start(self):
        field = torch.randn(1, 1, 9, 11, 13) * 0.01
        mask = torch.ones_like(field)
        with tempfile.TemporaryDirectory() as temporary:
            for activation in ("celu", "leaky_relu"):
                model = DNQSM(ModelConfig((2, 4), activation=activation)).eval()
                config = asdict(model.config)
                if activation == "celu":
                    config.pop("activation")
                    config.pop("leaky_relu_slope")
                path = Path(temporary) / f"{activation}.pt"
                torch.save(
                    dict(version=1, model_config=config, model=model.state_dict()), path
                )
                loaded = load_trained_model(path)
                self.assertEqual(loaded.config.activation, activation)
                self.assertEqual(loaded.config.leaky_relu_slope, 0.01)
                with torch.no_grad():
                    torch.testing.assert_close(
                        loaded.predict_local_field(field, mask),
                        model.predict_local_field(field, mask),
                        rtol=0,
                        atol=0,
                    )
            legacy = load_trained_model(Path(temporary) / "celu.pt")
            adapted = DNQSM(ModelConfig((2, 4)))
            adapted.initialize_from(legacy)
            self.assertEqual(adapted.config.activation, "leaky_relu")
            self.assertFalse(
                any(isinstance(module, nn.CELU) for module in adapted.modules())
            )
            for name, tensor in legacy.state_dict().items():
                torch.testing.assert_close(
                    adapted.state_dict()[name], tensor, rtol=0, atol=0
                )

    def test_invalid_activation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Activation"):
            ModelConfig(activation="invalid")


if __name__ == "__main__":
    unittest.main()
