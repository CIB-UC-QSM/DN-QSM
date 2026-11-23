import unittest

import torch

from dnqsm.model import DEFAULT_FEATURES, DNQSM, ModelConfig
from tests.helpers import laplacian_model_config


class ModelDepthTests(unittest.TestCase):
    def test_defaults_use_four_level_single_field_full_volume_model(self):
        self.assertEqual(ModelConfig().features, DEFAULT_FEATURES)
        with torch.device("meta"):
            model = DNQSM()
        self.assertEqual(len(model.filter.downs), 4)
        self.assertEqual(len(model.null_network.downs), 4)
        self.assertEqual(len(model.filter.up_blocks), 4)
        self.assertFalse(hasattr(model.filter, "harmonic_decoder"))
        self.assertEqual(len(model.null_network.up_blocks), 4)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 53157377)
        for network, count in ((model.filter, 27), (model.null_network, 27)):
            self.assertEqual(len(network.downsamples), 4)
            self.assertEqual(network.bottleneck[0].out_channels, 512)
            self.assertEqual(
                sum(isinstance(m, torch.nn.Conv3d) for m in network.modules()),
                count,
            )

    def test_custom_widths_determine_parameter_count(self):
        with torch.device("meta"):
            model = DNQSM(laplacian_model_config((23, 47, 90)))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6421407)

    def test_different_depth_requires_fresh_initialization(self):
        with torch.device("meta"):
            old = DNQSM(ModelConfig((2, 4)))
            deeper = DNQSM(ModelConfig((2, 4, 8)))
        with self.assertRaisesRegex(ValueError, "different U-Net widths or depth"):
            deeper.initialize_from(old)


if __name__ == "__main__":
    unittest.main()
