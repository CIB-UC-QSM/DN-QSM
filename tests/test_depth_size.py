import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dnqsm.depth_size import deepest_fitting, features_at_depth, probe_depth
from dnqsm.model import DEFAULT_FEATURES, DNQSM, ModelConfig
from dnqsm.training import TrainConfig, backward_batch, parse_args
from tests.helpers import make_dataset


class DepthSizeTests(unittest.TestCase):
    def test_default_training_entrypoint_restores_full_volume_best_profile(self):
        config = parse_args([], use_default_profile=True)
        self.assertEqual(config.filter_input_version, 2)
        self.assertTrue(config.full_grid_field)
        self.assertTrue(config.laplacian_input)
        self.assertEqual(config.features, (32, 64, 128, 256))
        self.assertEqual(config.threshold, 0.1)
        self.assertEqual(config.cutoff_mode, "soft")
        self.assertEqual(config.checkpoint_metric, "cosmos")
        self.assertEqual(config.patch_size, 0)
        self.assertEqual(config.lambda_harmonic_laplacian, 0)
        self.assertEqual(config.lambda_field_decomposition, 0)
        self.assertTrue(config.original_samples_only)
        self.assertEqual(config.synthetic_fraction, 0)
        self.assertIsNone(config.init_checkpoint)
        self.assertEqual(config.downsampling, "strided_conv")
        self.assertFalse(config.bias)
        override = parse_args(["--epochs", "30"], use_default_profile=True)
        self.assertEqual(override.epochs, 30)
        experimental = parse_args(
            ["--config", "configs/dual_field_100epochs.json"], use_default_profile=True
        )
        self.assertEqual(
            (experimental.filter_input_version, experimental.patch_size), (3, 64)
        )

    def test_defaults_use_four_level_single_field_full_volume_model(self):
        config = TrainConfig()
        self.assertEqual(config.batch_size, 10)
        self.assertEqual(config.features, DEFAULT_FEATURES)
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

    def test_different_depth_requires_fresh_initialization(self):
        with torch.device("meta"):
            old = DNQSM(ModelConfig((2, 4)))
            deeper = DNQSM(ModelConfig((2, 4, 8)))
        with self.assertRaisesRegex(ValueError, "different U-Net widths or depth"):
            deeper.initialize_from(old)

    def test_preserves_widths_and_adds_symmetric_levels(self):
        self.assertEqual(
            features_at_depth((23, 47, 90), 6), (23, 47, 90, 180, 360, 720)
        )
        with self.assertRaises(ValueError):
            features_at_depth((23, 47, 90), 2)
        with self.assertRaises(ValueError):
            features_at_depth((0,), 2)

    def test_search_checks_next_failure_and_labels_capped_search(self):
        tested = []
        self.assertEqual(
            deepest_fitting(lambda d: tested.append(d) or d <= 6, 3, 9), (6, False)
        )
        self.assertEqual(tested, [3, 4, 5, 6, 7])
        self.assertEqual(deepest_fitting(lambda _: True, 3, 5), (5, True))
        with self.assertRaises(RuntimeError):
            deepest_fitting(lambda _: False, 3, 9)

    def test_probes_both_branches_after_adam_init_at_full_batch_size(self):
        calls, states = [], []
        adam_step = torch.optim.AdamW.step

        def step(optimizer, *args, **kwargs):
            value = adam_step(optimizer, *args, **kwargs)
            states.append(len(optimizer.state))
            return value

        def backward(*args, **kwargs):
            calls.append(
                (
                    kwargs["filter_only"],
                    kwargs["chi_only"],
                    kwargs["gradient_diagnostics"],
                )
            )
            return backward_batch(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = TrainConfig(
                data_dir=str(make_dataset(root, shape=(16, 16, 16))),
                cache_dir=str(root / "cache"),
                device="cpu",
                features=(2, 4),
                batch_size=2,
                num_workers=0,
                threads=2,
                bf16=False,
                checkpoint_blocks=True,
                val_views=1,
                max_val_subjects=1,
                gradient_every=1,
                chi_shape_validation=True,
            )
            with (
                patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(total_memory=96 * 1024**3),
                ),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
                patch("dnqsm.depth_size.evaluation.load_cases", return_value=[]),
                patch("dnqsm.depth_size.evaluation.evaluate_model", return_value={}),
                patch("dnqsm.training.filtering_figure", return_value=None),
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                patch("dnqsm.depth_size.backward_batch", side_effect=backward),
                patch.object(torch.optim.AdamW, "step", step),
            ):
                result = probe_depth(config, root / "probe")
            self.assertEqual(result["status"], "fits")
            self.assertTrue(result["all_adam_states_initialized"])
            self.assertTrue(all(r["batch_size"] == 2 for r in result["steps"]))
            self.assertEqual(
                calls,
                [(True, False, False), (False, True, False)] * 2
                + [(True, False, True), (False, True, True)],
            )
            self.assertGreater(states[1], states[0])
            self.assertTrue(all(s == states[1] for s in states[1:]))
            with self.assertRaisesRegex(ValueError, "alternating"):
                probe_depth(replace(config, training_schedule="joint"), root / "joint")


if __name__ == "__main__":
    unittest.main()
