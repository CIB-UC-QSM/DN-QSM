import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dnqsm.batch_size import largest_fitting_batch, probe_batch
from dnqsm.training import TrainConfig, backward_batch
from tests.helpers import make_dataset


class BatchSizeTests(unittest.TestCase):
    def test_selects_largest_fit_and_checks_next_integer(self):
        for capacity in (1, 3, 6, 9, 31):
            with self.subTest(capacity=capacity):
                measured = []

                def probe(batch):
                    measured.append(batch)
                    return batch <= capacity

                self.assertEqual(largest_fitting_batch(probe), capacity)
                self.assertIn(capacity, measured)
                self.assertIn(capacity + 1, measured)
                self.assertEqual(len(measured), len(set(measured)))

    def test_explicit_maximum_and_no_fit(self):
        measured = []
        self.assertEqual(
            largest_fitting_batch(lambda batch: measured.append(batch) or True, 7),
            7,
        )
        self.assertEqual(max(measured), 7)
        with self.assertRaisesRegex(RuntimeError, "batch size 1"):
            largest_fitting_batch(lambda _: False)
        with self.assertRaises(ValueError):
            largest_fitting_batch(lambda _: True, 0)

    def test_gradient_memory_probe_keeps_initialized_adam_state(self):
        updates = 0
        diagnostics = []
        original_step = torch.optim.AdamW.step

        def track_step(optimizer, *args, **kwargs):
            nonlocal updates
            result = original_step(optimizer, *args, **kwargs)
            updates += 1
            return result

        def record_backward(*args, **kwargs):
            diagnostics.append((kwargs["gradient_diagnostics"], updates))
            return backward_batch(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            config = TrainConfig(
                filter_pretrain_epochs=0,
                data_dir=str(make_dataset(root)),
                cache_dir=str(root / "cache"),
                device="cpu",
                features=(2, 4),
                bf16=False,
                num_workers=0,
                threads=2,
                val_views=1,
                max_train_volumes=3,
                max_val_subjects=1,
                checkpoint_blocks=False,
                crop_probability=0,
                gradient_every=100,
            )
            with (
                patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(total_memory=96 * 1024**3),
                ),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
                patch("dnqsm.batch_size.evaluation.load_cases", return_value=[]),
                patch("dnqsm.batch_size.evaluation.evaluate_model", return_value={}),
                patch("dnqsm.training.reconstruction_figure", return_value=None),
                patch("dnqsm.batch_size.backward_batch", side_effect=record_backward),
                patch.object(torch.optim.AdamW, "step", new=track_step),
            ):
                result = probe_batch(config, root / "probe", 0.95)
        self.assertEqual(result["status"], "fits")
        self.assertEqual(diagnostics[:3], [(False, 0), (True, 1), (False, 2)])
