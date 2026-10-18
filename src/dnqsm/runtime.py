"""Shared device, precision, memory-budget, and checkpoint utilities."""

from contextlib import nullcontext
from pathlib import Path

import torch

from .model import DNQSM, ModelConfig

CHECKPOINT_VERSION = 1


def autocast_context(device: torch.device, enabled: bool = True):
    if device.type == "cuda" and enabled and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def require_device(name: str | torch.device) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Restore GPU access for full training; use --device cpu only for explicit CPU runs."
        )
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")
    return device


def configure_memory_budget(device: torch.device, budget_gb: float) -> None:
    """Reserve 1 GiB for CUDA context/libraries outside the PyTorch allocator."""
    if device.type == "cuda" and budget_gb:
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = (budget_gb - 1) * 1024**3 / total
        if not 0 < fraction <= 1:
            raise ValueError("CUDA memory budget exceeds the device capacity")
        torch.cuda.set_per_process_memory_fraction(fraction, device)


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_trained_model(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> DNQSM:
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    if saved.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported DN-QSM checkpoint version")
    config = dict(saved["model_config"])
    config["features"] = tuple(config["features"])
    model = DNQSM(ModelConfig(**config)).to(device)
    model.load_state_dict(saved["model"])
    return model.eval()
