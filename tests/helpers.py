import json
from pathlib import Path

import torch

from dnqsm.model import ModelConfig


def laplacian_model_config(*args, **kwargs):
    """Explicit historical architecture for Laplacian compatibility regressions."""
    kwargs.setdefault("filter_input_version", 2)
    kwargs.setdefault("cutoff_mode", "hard")
    kwargs.setdefault("laplacian_input", True)
    kwargs.setdefault("conditional_laplacian_threshold", 32.0)
    kwargs.setdefault("downsampling", "max_pool")
    kwargs.setdefault("bias", True)
    kwargs.setdefault("laplacian_operator", "stencil")
    kwargs.setdefault("laplacian_extension_iters", 0)
    kwargs.setdefault("laplacian_extension_mode", "harmonic")
    return ModelConfig(*args, **kwargs)


def dual_field_model_config(*args, **kwargs):
    """Explicit opt-in to the experimental raw-input dual-field architecture."""
    kwargs.setdefault("filter_input_version", 3)
    kwargs.setdefault("cutoff_mode", "hard")
    kwargs.setdefault("laplacian_input", False)
    kwargs.setdefault("conditional_laplacian_threshold", 0.0)
    kwargs.setdefault("downsampling", "max_pool")
    kwargs.setdefault("bias", True)
    kwargs.setdefault("laplacian_operator", "stencil")
    kwargs.setdefault("laplacian_extension_iters", 0)
    kwargs.setdefault("laplacian_extension_mode", "harmonic")
    return ModelConfig(*args, **kwargs)


def make_dataset(directory: Path, shape=(12, 12, 12)) -> Path:
    root = directory / "sources"
    root.mkdir()
    generator = torch.Generator().manual_seed(312)
    identities = {}
    for index, subject in enumerate(
        (
            "qsm:a",
            "qsm:a",
            "qsm:b",
            "qsm:c",
            "separate:a",
            "separate:b",
            "synthetic:a",
            "synthetic:b",
        )
    ):
        mask = torch.zeros(1, *shape, dtype=torch.bool)
        mask[:, 4:-4, 4:-4, 4:-4] = True
        chi = torch.randn(1, *shape, generator=generator) * 0.03 * mask
        name = f"image_{index:04d}.pt"
        torch.save(chi, root / name)
        torch.save(mask, root / f"mask_{index:04d}.pt")
        identities[name] = subject
    (root / "subjects.json").write_text(json.dumps(identities))
    return root
