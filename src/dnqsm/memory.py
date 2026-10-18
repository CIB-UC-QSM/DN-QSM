"""Per-process NVIDIA memory telemetry, including CUDA library allocations."""

import os
import subprocess
import threading

import torch


class ProcessMemoryMonitor:
    """Sample nvidia-smi without depending on optional NVML Python bindings.

    The hard PyTorch allocator budget is configured separately. This monitor
    records observed process VRAM (including allocations outside that allocator)
    and checks the user's decimal-GB ceiling. CUDA caching keeps normal tensor
    allocation peaks visible between samples.
    """

    def __init__(self, device: torch.device, limit_gb: float = 30.0):
        self.enabled = device.type == "cuda"
        self.limit_gb = limit_gb
        self.pid = os.getpid()
        self.peak_bytes = 0
        self.error: Exception | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        if not self.enabled:
            return
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        # A process may have allocations on more than one GPU. Sum them so
        # accidental use of another device cannot under-report the budget.
        used = 0
        for line in result.stdout.splitlines():
            pid, separator, memory = line.partition(",")
            if separator and pid.strip() == str(self.pid):
                used += int(memory.strip()) * 1024**2
        self.record_peak(used)

    def record_peak(self, used_bytes: int) -> None:
        """Merge a sample or saved history without losing a concurrent peak."""
        with self._lock:
            self.peak_bytes = max(self.peak_bytes, used_bytes)

    def _watch(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self.sample()
            except Exception as error:
                self.error = error
                return

    def check(self) -> None:
        if self.error:
            raise RuntimeError("GPU process-memory telemetry failed") from self.error
        if self.enabled and self.peak_bytes >= self.limit_gb * 1e9:
            raise RuntimeError(
                f"Observed process VRAM reached {self.peak_bytes / 1e9:.3f} GB; limit is {self.limit_gb:g} GB"
            )

    def metrics(self) -> dict[str, float]:
        self.sample()
        self.check()
        if not self.enabled:
            return {}
        return dict(
            peak_process_vram_gb=self.peak_bytes / 1e9,
            process_vram_limit_gb=self.limit_gb,
        )

    def __enter__(self):
        self.sample()
        self.check()
        if self.enabled:
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6)
        if exc_type is None:
            self.sample()
            self.check()
