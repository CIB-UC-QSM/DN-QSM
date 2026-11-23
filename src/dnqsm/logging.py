"""Headless comparison figures and JSON output."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from .physics import masked_nrmse


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/dnqsm-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def comparison_figure(
    rows: list[tuple[str, torch.Tensor, bool]],
    title: str,
    *,
    display_limit: float | None = None,
):
    """Three central planes per row; susceptibility has fixed grayscale limits."""
    plt = _pyplot()
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(11, 2.7 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(title)
    for index, (label, tensor, susceptibility) in enumerate(rows):
        volume = tensor.detach().float().cpu().numpy().squeeze()
        if volume.ndim != 3:
            raise ValueError("Figure rows must each contain one 3D volume")
        center = [size // 2 for size in volume.shape]
        planes = (volume[center[0]], volume[:, center[1]], volume[:, :, center[2]])
        limit = display_limit
        if limit is None:
            limit = (
                0.1
                if susceptibility
                else max(float(np.percentile(np.abs(volume), 99.5)), 1e-6)
            )
        for column, (plane, orientation) in enumerate(
            zip(planes, ("Axis 0", "Axis 1", "Axis 2"), strict=True)
        ):
            ax = axes[index, column]
            ax.imshow(np.rot90(plane), cmap="gray", vmin=-limit, vmax=limit)
            ax.set_title(f"{label} — {orientation}", fontsize=9)
            ax.axis("off")
    return figure


def nrmse_label(
    prediction: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor
) -> str:
    value = masked_nrmse(prediction, reference, mask)[0]
    return "NRMSE=N/A" if value is None else f"NRMSE={value:.2f}%"
