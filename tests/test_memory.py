import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dnqsm.memory import ProcessMemoryMonitor
from dnqsm.overfit import OverfitConfig


class ProcessMemoryTests(unittest.TestCase):
    def test_process_peak_includes_all_devices_and_excludes_other_processes(self):
        monitor = ProcessMemoryMonitor(torch.device("cuda:0"), 30)
        monitor.pid = 123
        with patch(
            "dnqsm.memory.subprocess.run",
            return_value=SimpleNamespace(stdout="123, 14000\n456, 8000\n123, 12000\n"),
        ):
            first = monitor.metrics()
        self.assertEqual(first["peak_process_vram_gb"], 26000 * 1024**2 / 1e9)
        with patch(
            "dnqsm.memory.subprocess.run",
            return_value=SimpleNamespace(stdout="123, 1000\n"),
        ):
            self.assertEqual(monitor.metrics(), first)
        with patch(
            "dnqsm.memory.subprocess.run",
            return_value=SimpleNamespace(stdout="123, 29000\n"),
        ):
            with self.assertRaisesRegex(RuntimeError, "VRAM reached"):
                monitor.metrics()

    def test_cpu_needs_no_nvidia_tool(self):
        with patch("dnqsm.memory.subprocess.run") as command:
            with ProcessMemoryMonitor(torch.device("cpu")) as monitor:
                self.assertEqual(monitor.metrics(), {})
            command.assert_not_called()

    def test_decimal_gb_ceiling_cannot_be_confused_with_gib_budget(self):
        OverfitConfig(cuda_memory_gb=27, max_vram_gb=30).validate()
        with self.assertRaisesRegex(ValueError, "decimal GB"):
            OverfitConfig(cuda_memory_gb=30, max_vram_gb=30).validate()


if __name__ == "__main__":
    unittest.main()
