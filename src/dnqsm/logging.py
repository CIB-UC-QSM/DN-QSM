"""Local, headless visualization and explicitly configured Comet logging."""

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .physics import masked_nrmse


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configure_comet_key(env_file: str | None = None) -> None:
    """Load only the API key; never log or persist credentials."""
    if os.environ.get("COMET_API_KEY"):
        return
    path = Path(env_file) if env_file else Path(".env")
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            key, separator, value = line.partition("=")
            if separator and key.strip() == "COMET_API_KEY":
                value = value.strip().strip("\"'")
                if value:
                    os.environ["COMET_API_KEY"] = value
                    return
    raise ValueError("Comet requires COMET_API_KEY or --comet-env-file")


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/dnqsm-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def comparison_figure(rows: list[tuple[str, torch.Tensor, bool]], title: str):
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


def filtering_figure(batch: dict, local_field: torch.Tensor, title: str):
    """Field-only comparison for independent filter training."""
    mask = batch["mask"][:1]
    target, predicted = batch["local"][:1], local_field[:1]
    return comparison_figure(
        [
            ("Total field (ppm)", batch["phase"][:1] * mask, False),
            ("Clean local field (ppm)", target * mask, False),
            (
                "Filtered local, " + nrmse_label(predicted, target, mask),
                predicted,
                False,
            ),
            ("Local error (ppm)", (predicted - target) * mask, False),
        ],
        title,
    )


def reconstruction_figure(batch: dict, prediction, title: str):
    """Select the first sample only, without retaining training graphs."""
    mask = batch["mask"][:1]
    local = batch["local"][:1]
    chi = batch["chi"][:1]
    predicted_local = prediction.local_field[:1]
    initial = prediction.tkd[:1]
    predicted_chi = prediction.chi[:1]
    return comparison_figure(
        [
            ("Total field (ppm)", batch["phase"][:1] * mask, False),
            ("Clean local field (ppm)", local * mask, False),
            (
                "Filtered local, " + nrmse_label(predicted_local, local, mask),
                predicted_local * mask,
                False,
            ),
            ("Local error (ppm)", (predicted_local - local) * mask, False),
            ("Susceptibility ground truth (ppm)", chi * mask, True),
            ("TKD, " + nrmse_label(initial, chi, mask), initial * mask, True),
            (
                "DN-QSM, " + nrmse_label(predicted_chi, chi, mask),
                predicted_chi * mask,
                True,
            ),
            ("Susceptibility error (ppm)", (predicted_chi - chi) * mask, True),
        ],
        title,
    )


class RunLogger:
    def __init__(self, directory: Path, config, resume: bool = False):
        self.directory = directory
        self.experiment = None
        if config.comet:
            configure_comet_key(config.comet_env_file)
            import comet_ml

            key_file = directory / "comet.json"
            saved_key = (
                json.loads(key_file.read_text())["experiment_key"]
                if resume and key_file.is_file()
                else None
            )
            self.experiment = comet_ml.start(
                project_name=config.comet_project,
                experiment_key=saved_key,
                mode="get" if saved_key else "create",
                experiment_config=comet_ml.ExperimentConfig(
                    name=directory.name,
                    tags=["dn-qsm", "null-space"],
                    log_code=False,
                    log_graph=False,
                    log_git_metadata=False,
                    log_git_patch=False,
                    log_env_details=False,
                    parse_args=False,
                    auto_output_logging=False,
                    auto_param_logging=False,
                    auto_metric_logging=False,
                    auto_log_co2=False,
                ),
            )
            try:
                self.experiment.log_parameters(asdict(config))
                write_json(key_file, {"experiment_key": self.experiment.get_key()})
            except BaseException:
                self.experiment.end()
                raise
        try:
            self.handle = (directory / "metrics.jsonl").open("a", buffering=1)
        except BaseException:
            if self.experiment:
                self.experiment.end()
            raise

    def log(
        self,
        event: str,
        step: int,
        epoch: int,
        *,
        comet_step: int | None = None,
        **metrics,
    ) -> None:
        row = dict(
            event=event,
            step=step,
            epoch=epoch,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **metrics,
        )
        self.handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if self.experiment:
            numeric = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if numeric:
                self.experiment.log_metrics(
                    numeric,
                    prefix=event,
                    step=step if comet_step is None else comet_step,
                    epoch=epoch,
                )

    def figure(
        self,
        figure,
        name: str,
        step: int,
        epoch: int,
        *,
        output_path: Path | None = None,
    ) -> None:
        """Log one PNG under a stable case name, retaining each step locally."""
        path = output_path or self.directory / "images" / name / f"step_{step:06d}.png"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=100, bbox_inches="tight")
            if self.experiment:
                self.experiment.log_image(
                    image_data=str(path),
                    name=name,
                    step=step,
                    metadata={"epoch": epoch},
                    overwrite=False,
                )
        finally:
            _pyplot().close(figure)

    def close(self) -> None:
        try:
            self.handle.close()
        finally:
            if self.experiment:
                self.experiment.end()
