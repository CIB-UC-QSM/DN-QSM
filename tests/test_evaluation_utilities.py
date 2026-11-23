import types
import unittest
from unittest.mock import patch

import torch

from dnqsm.logging import _pyplot, comparison_figure
from dnqsm.runtime import configure_memory_budget


class EvaluationUtilitiesTests(unittest.TestCase):
    def test_figures_use_fixed_susceptibility_limits_and_central_planes(self):
        volume = torch.arange(4 * 5 * 6).reshape(1, 1, 4, 5, 6).float()
        figure = comparison_figure([("Susceptibility", volume, True)], "NRMSE=12.00%")
        try:
            for axis in figure.axes:
                self.assertEqual(axis.images[0].get_clim(), (-0.1, 0.1))
                self.assertEqual(axis.images[0].get_cmap().name, "gray")
            self.assertEqual(figure.axes[0].images[0].get_array().shape, (6, 5))
            self.assertEqual(figure.axes[1].images[0].get_array().shape, (6, 4))
            self.assertEqual(figure.axes[2].images[0].get_array().shape, (5, 4))
        finally:
            _pyplot().close(figure)

    def test_gpu_budget_reserves_library_overhead(self):
        device = torch.device("cuda:0")
        with (
            patch(
                "torch.cuda.get_device_properties",
                return_value=types.SimpleNamespace(total_memory=24 * 1024**3),
            ),
            patch("torch.cuda.set_per_process_memory_fraction") as limit,
        ):
            configure_memory_budget(device, 12)
            limit.assert_called_once_with(11 / 24, device)
            with self.assertRaises(ValueError):
                configure_memory_budget(device, 0.5)


if __name__ == "__main__":
    unittest.main()
