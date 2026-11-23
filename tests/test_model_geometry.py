import unittest

import torch

from dnqsm.model import DNQSM
from dnqsm.physics import dipole_tensor
from tests.helpers import dual_field_model_config as ModelConfig


class NativeModelGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_64_network_forward_backward_without_padding(self):
        # Retain five levels, reducing channel counts for a small CPU check.
        model = DNQSM(ModelConfig((1, 1, 1, 1, 1)))
        sizes = []
        hooks = [
            network.bottleneck.register_forward_pre_hook(
                lambda module, args: sizes.append(tuple(args[0].shape[-3:]))
            )
            for network in (model.filter, model.null_network)
        ]
        try:
            x = torch.randn(1, 1, 64, 64, 64)
            result = model(x, torch.ones_like(x), dipole_tensor((64, 64, 64)))
            self.assertEqual(sizes, [(2, 2, 2), (2, 2, 2)])
            self.assertEqual(result.chi.shape, x.shape)
            (
                result.chi.square().mean() + result.harmonic_field.square().mean()
            ).backward()
            for head in (
                model.filter.local_decoder.final,
                model.filter.harmonic_decoder.final,
                model.null_network.final,
            ):
                self.assertIsNotNone(head.weight.grad)
                self.assertTrue(bool(torch.isfinite(head.weight.grad).all()))
        finally:
            for hook in hooks:
                hook.remove()
