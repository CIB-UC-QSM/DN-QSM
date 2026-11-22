"""Compare boundary derivatives against known full-field references; no training."""

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dnqsm-matplotlib")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from dnqsm.data import ShapeFieldSynthesizer
from dnqsm.physics import (
    frequency_laplacian,
    quadratic_exterior_extension,
    smooth_harmonic_extension,
)
from dnqsm.training import build_datasets, parse_args


def derivatives(field, mask):
    return (
        frequency_laplacian(smooth_harmonic_extension(field, mask)) * mask,
        frequency_laplacian(quadratic_exterior_extension(field, mask))[
            ..., 1:-1, 1:-1, 1:-1
        ]
        * mask,
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = args.directory or Path("artifacts/diagnostics") / (
        "laplacian_boundaries_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    directory.mkdir(parents=True, exist_ok=False)
    z, y, x = torch.meshgrid(
        *(torch.linspace(-1, 1, n) for n in (65, 67, 69)), indexing="ij"
    )
    mask = ((z / 0.85) ** 2 + (y / 0.8) ** 2 + (x / 0.75) ** 2 < 1)[None, None]
    field = (
        0.3 * z * z + 0.2 * y * y - 0.1 * x * x + 0.1 * x * y + 0.7 * x - 0.3 * z + 1
    )[None, None]
    truth = (
        torch.full_like(
            field,
            -2 * (0.3 * (2 / 64) ** 2 + 0.2 * (2 / 66) ** 2 - 0.1 * (2 / 68) ** 2),
        )
        * mask
    )
    cases = [("Known quadratic curvature", field, mask, truth, 32)]

    z, y, x = torch.meshgrid(*(torch.arange(65.0) for _ in range(3)), indexing="ij")
    field = torch.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 35)[None, None]
    mask = torch.zeros_like(field, dtype=torch.bool)
    mask[..., 12:53, 8:57, 8:57] = True
    cases.append(
        (
            "Feature on first ROI slice",
            field,
            mask,
            frequency_laplacian(field) * mask,
            12,
        )
    )

    config = replace(
        parse_args([], use_default_profile=True),
        lesion_augmentation=None,
        crop_probability=0,
    )
    training, _, _ = build_datasets(config, directory)
    sample = training[0]
    data = ShapeFieldSynthesizer("cpu")(
        default_collate([sample]),
        background_mode="mixture",
        clean_probability=1,
        weak_probability=0,
        strong_probability=0,
    )
    shape = data["local_full"].shape[-3:]
    z, y, x = torch.meshgrid(*(torch.linspace(-1, 1, n) for n in shape), indexing="ij")
    field = data["local_full"] + (0.1 * z + 0.2 * x)[None, None]
    mask = data["mask"].bool()
    truth = frequency_laplacian(field) * mask
    cases.append(
        ("Original brain + known linear background", field, mask, truth, shape[0] // 2)
    )

    fig, axes = plt.subplots(len(cases), 4, figsize=(14, 10), constrained_layout=True)
    reports = []
    for row, (label, field, mask, truth, slice_index) in enumerate(cases):
        old, new = derivatives(field, mask)
        valid = F.avg_pool3d(mask.float(), 3, 1, 1) > 1 - 1e-6
        border = mask & ~valid
        metrics = {
            "case": label,
            "roi_voxels": int(mask.sum()),
            "border_voxels": int(border.sum()),
        }
        for name, volume in (("harmonic", old), ("quadratic", new)):
            for region, selected in (("border", border), ("interior", valid)):
                error = volume[selected] - truth[selected]
                metrics[f"{name}_{region}_rmse"] = float(error.square().mean().sqrt())
                metrics[f"{name}_{region}_max_error"] = float(error.abs().max())
        extended = quadratic_exterior_extension(field, mask)[..., 1:-1, 1:-1, 1:-1]
        metrics["interior_field_exact"] = bool(torch.equal(extended[mask], field[mask]))
        reports.append(metrics)
        scale = float(torch.quantile(truth[mask].abs(), 0.99).clamp_min(1e-7))
        for col, (title, volume) in enumerate(
            (
                ("Total field", field * mask),
                ("Known reference", truth),
                ("Previous: harmonic", old),
                ("New: quadratic", new),
            )
        ):
            image = volume[0, 0, slice_index].numpy()
            roi = mask[0, 0, slice_index].numpy()
            if col == 0:
                handle = axes[row, col].imshow(image, cmap="coolwarm")
            else:
                handle = axes[row, col].imshow(
                    image, cmap="coolwarm", vmin=-scale, vmax=scale
                )
            axes[row, col].contour(roi, levels=[0.5], colors="black", linewidths=0.4)
            axes[row, col].set_title(title)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(label)
                fig.colorbar(handle, ax=axes[row, col], shrink=0.65)
        fig.colorbar(
            handle,
            ax=list(axes[row, 1:]),
            shrink=0.65,
            label="Positive discrete Laplacian (shared scale)",
        )
    fig.suptitle(
        "All original ROI voxels retained; identical scales for derivative comparisons"
    )
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)
    report = dict(
        status="passed",
        learning_updates=0,
        cases=reports,
        source_sample=sample["subject"],
        preview=str((directory / "comparison.png").resolve()),
        note="Analytic/simulation derivative checks, not trained-model accuracy; exterior curvature is estimated where unobserved.",
    )
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
