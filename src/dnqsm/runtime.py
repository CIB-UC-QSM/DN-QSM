"""Shared device, precision, memory-budget, and checkpoint utilities."""

from contextlib import nullcontext
from pathlib import Path

import torch

from .inference import SlidingWindowConfig
from .model import DNQSM, ModelConfig
from .paths import DEFAULT_MODEL_CHECKPOINT

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


def configure_inference(model: DNQSM, config: dict) -> DNQSM:
    """Patch-trained runs use Gaussian windows; old runs keep their full grid."""
    model.inference_config = SlidingWindowConfig(
        window_size=config.get("patch_size", 0),
        overlap=config.get("inference_overlap", 0.5),
        sigma_scale=config.get("inference_sigma_scale", 0.125),
        window_batch_size=config.get("inference_batch_size", 1),
    )
    return model


def load_trained_model(
    checkpoint: str | Path = DEFAULT_MODEL_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> DNQSM:
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    if saved.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported DN-QSM checkpoint version")
    config = dict(saved["model_config"])
    config.setdefault("threshold", 0.1)
    config.setdefault("cutoff_mode", "hard")
    # Historical checkpoints were trained with a zeroed exterior field.
    config.setdefault("full_grid_field", False)
    config.setdefault("laplacian_input", False)
    config.setdefault("filter_input_version", 1)
    config.setdefault("laplacian_fp32", False)
    config.setdefault("conditional_laplacian_threshold", 0.0)
    config.setdefault("filter_output_skip", False)
    config.setdefault("filter_laplacian_residual", False)
    config.setdefault("filter_refinement_features", 0)
    config.setdefault("activation", "celu")
    config.setdefault("leaky_relu_slope", 0.01)
    config.setdefault("downsampling", "max_pool")
    config.setdefault("bias", True)
    config.setdefault("laplacian_operator", "stencil")
    config.setdefault("laplacian_extension_iters", 0)
    config.setdefault("laplacian_extension_mode", "harmonic")
    config.setdefault("field_normalization", "legacy")
    config.setdefault("normalization_layer", "instance")
    config["features"] = tuple(config["features"])
    model = DNQSM(ModelConfig(**config)).to(device)
    model.load_state_dict(saved["model"])
    configure_inference(model, saved.get("config", {}))
    if "inference_config" in saved:
        model.inference_config = SlidingWindowConfig(**saved["inference_config"])
    return model.eval()
