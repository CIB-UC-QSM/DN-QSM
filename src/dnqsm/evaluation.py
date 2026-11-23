"""COSMOS and sim2 evaluation adapted from wh_net, entirely in physical units."""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.io import loadmat
from torch import Tensor

from .experiment_metrics import harmonic_laplacian_mse
from .logging import _pyplot, comparison_figure, nrmse_label, write_json
from .paths import DEFAULT_MODEL_CHECKPOINT, EVALUATION_DATA_DIR, REFERENCE_ROOT
from .physics import (
    apply_dipole,
    continuous_dipole_kernel,
    masked_mean,
    masked_nrmse,
    validate_mask,
)
from .runtime import (
    autocast_context,
    configure_memory_budget,
    load_trained_model,
    require_device,
)


@dataclass
class EvaluationCase:
    name: str
    phase: Tensor
    mask: Tensor
    dipole: Tensor
    chi: Tensor
    local: Tensor | None = None
    original_shape: tuple[int, int, int] | None = None
    crop_start: tuple[int, int, int] = (0, 0, 0)
    crop_shape: tuple[int, int, int] | None = None
    pad_offsets: tuple[int, int, int] = (0, 0, 0)
    affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        validate_mask(self.mask, self.phase)
        if self.phase.shape[0] != 1 or self.chi.shape != self.mask.shape:
            raise ValueError("An evaluation case needs aligned single-sample volumes")
        if self.local is not None and self.local.shape != self.mask.shape:
            raise ValueError("Local-field reference shape differs")
        for value in (self.phase, self.chi, self.dipole, self.local):
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite evaluation inputs")

    def crop_volume(self, value: Tensor) -> np.ndarray:
        volume = value.detach().float().cpu().numpy().reshape(value.shape[-3:])
        shape = self.crop_shape or tuple(volume.shape)
        slices = tuple(
            slice(offset, offset + size)
            for offset, size in zip(self.pad_offsets, shape, strict=True)
        )
        return np.ascontiguousarray(volume[slices])

    def restore_volume(self, value: Tensor) -> np.ndarray:
        cropped = self.crop_volume(value)
        restored = np.zeros(self.original_shape or cropped.shape, dtype=np.float32)
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(self.crop_start, cropped.shape, strict=True)
        )
        restored[slices] = cropped
        return restored


def _tensor(array: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]


def find_cosmos_simulation(runs_dir: str | Path | None = None) -> Path:
    candidates = []
    for path in Path(runs_dir or REFERENCE_ROOT / "artifacts/runs").glob(
        "wh_tv_cosmos_*/simulation.pt"
    ):
        config_path, status_path = (
            path.parent / "config.json",
            path.parent / "status.json",
        )
        if config_path.is_file() and status_path.is_file():
            config, status = (
                json.loads(config_path.read_text()),
                json.loads(status_path.read_text()),
            )
            if (
                config.get("snr") == 100
                and config.get("bg") == 10
                and status.get("status") == "complete"
            ):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No completed COSMOS SNR=100/background=10 simulation; supply --cosmos-simulation"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def load_cosmos_case(simulation: str | Path | None = None) -> EvaluationCase:
    path = Path(simulation) if simulation else find_cosmos_simulation()
    saved = torch.load(path, weights_only=True, map_location="cpu")
    required = {"phase", "mask", "dipole", "chi"}
    if missing := required - saved.keys():
        raise ValueError(f"COSMOS simulation missing {sorted(missing)}")
    config_path = path.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"COSMOS requires scaling/padding metadata: {config_path}"
        )
    config = json.loads(config_path.read_text())
    if config.get("snr") != 100 or config.get("bg") != 10:
        raise ValueError("COSMOS benchmark requires SNR=100/background=10")
    factor = float(config.get("factor", 0.2125))
    pad = int(config.get("pad", 12))
    if not np.isfinite(factor) or factor <= 0 or pad < 0:
        raise ValueError("Invalid COSMOS scaling or padding")
    phase, chi = saved["phase"].float() * factor, saved["chi"].float() * factor
    mask, dipole = saved["mask"].float(), saved["dipole"].float()
    shape = tuple(int(n) - 2 * pad for n in phase.shape[-3:])
    if min(shape) <= 0:
        raise ValueError("COSMOS padding exceeds the grid")
    case = EvaluationCase(
        "cosmos",
        phase,
        mask,
        dipole,
        chi,
        local=apply_dipole(chi * mask, dipole) * mask,
        original_shape=shape,
        crop_shape=shape,
        pad_offsets=(pad, pad, pad),
        metadata=dict(
            simulation=str(path.resolve()),
            stored_factor=factor,
            physical_units="ppm",
            padding=pad,
            snr=100,
            background_source_amplitude=10,
            snr_definition=config.get(
                "snr_definition", "max(abs(local[mask])) / noise_std"
            ),
        ),
    )
    case.validate()
    return case


def center_pad(volume: np.ndarray, pad: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    if volume.ndim != 3 or pad < 0:
        raise ValueError("Expected a 3D volume and nonnegative padding")
    maximum = max(volume.shape)
    offsets = tuple((maximum - size) // 2 + pad for size in volume.shape)
    result = np.zeros((maximum + 2 * pad,) * 3, dtype=np.float32)
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, volume.shape, strict=True)
    )
    result[slices] = volume
    return result, offsets


def load_sim2_case(data_dir: str | Path | None = None, pad: int = 20) -> EvaluationCase:
    voxel_size = (0.65, 0.65, 0.65)
    root = Path(data_dir or EVALUATION_DATA_DIR)
    phase = np.asarray(loadmat(root / "Sim2.mat")["phase"], dtype=np.float32)
    mask = np.asarray(loadmat(root / "mask_final.mat")["mask_final"], dtype=np.float32)
    image = nib.load(root / "volumes/Sim2ChiGT.nii.gz")
    chi = image.get_fdata(dtype=np.float32)
    if (
        mask.shape != chi.shape
        or mask.ndim != 3
        or phase.ndim != 3
        or any(p < m for p, m in zip(phase.shape, mask.shape, strict=True))
    ):
        raise ValueError("sim2 phase must cover the aligned mask/reference grid")
    if not np.isin(mask, (0, 1)).all() or not mask.any():
        raise ValueError("sim2 needs a nonempty binary mask")
    if not np.isfinite(phase).all() or not np.isfinite(chi).all():
        raise ValueError("Nonfinite sim2 input")
    coordinates = np.argwhere(mask)
    start, stop = coordinates.min(axis=0), coordinates.max(axis=0) + 1
    crop = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))
    padded_phase, offsets = center_pad(phase[crop], pad)
    padded_mask, _ = center_pad(mask[crop], pad)
    padded_chi, _ = center_pad(chi[crop], pad)
    case = EvaluationCase(
        "sim2",
        _tensor(padded_phase),
        _tensor(padded_mask),
        _tensor(continuous_dipole_kernel(padded_phase.shape, voxel_size=voxel_size)),
        _tensor(padded_chi),
        original_shape=tuple(mask.shape),
        crop_start=tuple(int(v) for v in start),
        crop_shape=tuple(int(v) for v in stop - start),
        pad_offsets=offsets,
        affine=image.affine.copy(),
        metadata=dict(
            phase=str((root / "Sim2.mat").resolve()),
            chi_reference=str((root / "volumes/Sim2ChiGT.nii.gz").resolve()),
            physical_units="ppm",
            voxel_size_mm=list(voxel_size),
            factor=1.0,
            padding=pad,
            crop_start=start.tolist(),
            crop_stop=stop.tolist(),
            pad_offsets=list(offsets),
            background_reference="unavailable",
            affine_source="Sim2ChiGT.nii.gz",
        ),
    )
    case.validate()
    # The supplied phase is not the dipole forward field of the supplied
    # reference susceptibility. Record the mismatch beside every sim2 score so
    # the benchmark's own inconsistency travels with the number it bounds.
    case.metadata["forward_consistency_nrmse"] = masked_nrmse(
        case.phase,
        apply_dipole(case.chi * case.mask, case.dipole) * case.mask,
        case.mask,
    )[0]
    return case


def load_cases(
    datasets: tuple[str, ...] = ("cosmos", "sim2"),
    *,
    data_dir=None,
    cosmos_simulation=None,
    sim2_pad=20,
) -> list[EvaluationCase]:
    if not datasets or set(datasets) - {"cosmos", "sim2"}:
        raise ValueError("Datasets must be cosmos and/or sim2")
    return [
        load_cosmos_case(cosmos_simulation)
        if name == "cosmos"
        else load_sim2_case(data_dir, sim2_pad)
        for name in datasets
    ]


def evaluate_checkpoint(
    checkpoint: str | Path,
    cases: list[EvaluationCase],
    directory: Path,
    device: str = "cpu",
    *,
    bf16=True,
    inference_options=None,
) -> dict:
    target = require_device(device)
    model = load_trained_model(checkpoint, target)
    return evaluate_model(
        model,
        cases,
        directory,
        device,
        bf16=bf16,
        checkpoint=checkpoint,
        inference_options=inference_options,
    )


@torch.inference_mode()
def evaluate_model(
    model,
    cases: list[EvaluationCase],
    directory: Path,
    device: str = "cpu",
    *,
    bf16=True,
    checkpoint=None,
    export_volumes=True,
    inference_options=None,
) -> dict:
    """Evaluate a model and write case metrics, figures, and optional volumes."""
    target = require_device(device)
    inference_config = replace(model.inference_config, **(inference_options or {}))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    try:
        summaries = {
            case.name: _evaluate_case(
                model,
                case,
                directory / case.name,
                target,
                bf16=bf16,
                checkpoint=checkpoint,
                export_volumes=export_volumes,
                inference_config=inference_config,
            )
            for case in cases
        }
        write_json(directory / "metrics.json", summaries)
        return summaries
    finally:
        model.train(was_training)


def _evaluate_case(
    model,
    case,
    case_dir,
    target,
    *,
    bf16,
    checkpoint,
    export_volumes,
    inference_config,
):
    """Limit input, prediction, and FFT tensor lifetimes to a single case."""
    case_dir.mkdir(parents=True, exist_ok=True)
    case.validate()
    phase, mask, dipole, chi = (
        v.to(target) for v in (case.phase, case.mask, case.dipole, case.chi)
    )
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    started = time.perf_counter()
    with autocast_context(target, bf16):
        output = model.infer(phase, mask, dipole, **asdict(inference_config))
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    seconds = time.perf_counter() - started
    predicted_field = apply_dipole(output.chi, dipole)
    metrics = dict(
        chi_nrmse=masked_nrmse(output.chi, chi, mask)[0],
        tkd_nrmse=masked_nrmse(output.tkd, chi, mask)[0],
        forward_mse=float(
            masked_mean(
                (predicted_field - output.local_field).square(),
                mask,
            ).mean()
        ),
        seconds=seconds,
    )
    rows = [
        ("Total field (ppm)", phase * mask, False),
        ("Filtered local field (ppm)", output.local_field * mask, False),
        ("Ground truth susceptibility (ppm)", chi * mask, True),
        (
            ("Soft inverse, " if model.config.cutoff_mode == "soft" else "TKD, ")
            + nrmse_label(output.tkd, chi, mask),
            output.tkd * mask,
            True,
        ),
        ("DN-QSM, " + nrmse_label(output.chi, chi, mask), output.chi * mask, True),
        ("Susceptibility error (ppm)", (output.chi - chi) * mask, True),
    ]
    if output.harmonic_field is not None:
        harmonic = output.harmonic_field * mask
        metrics["harmonic_laplacian_mse"] = float(
            harmonic_laplacian_mse(harmonic, mask).mean()
        )
        metrics["field_decomposition_mae"] = float(
            masked_mean((output.local_field + harmonic - phase).abs(), mask).mean()
        )
        rows.append(("Estimated harmonic field (ppm)", harmonic, False))
    if case.local is not None:
        local = case.local.to(target)
        metrics["local_nrmse"] = masked_nrmse(output.local_field, local, mask)[0]
        metrics["physics_mse"] = float(
            masked_mean((predicted_field - local).square(), mask).mean()
        )
        metrics["physics_nrmse"] = masked_nrmse(predicted_field, local, mask)[0]
        rows[1] = (
            "Filtered local, " + nrmse_label(output.local_field, local, mask),
            output.local_field * mask,
            False,
        )
        rows.insert(1, ("Ground truth local field (ppm)", local * mask, False))
        rows.insert(
            3,
            ("Local field error (ppm)", (output.local_field - local) * mask, False),
        )
    figure = comparison_figure(rows, f"DN-QSM — {case.name.upper()} (physical units)")
    try:
        figure.savefig(case_dir / "comparison.png", dpi=100, bbox_inches="tight")
    finally:
        _pyplot().close(figure)
    # Raw padded outputs preserve Fourier projection properties. Masked NIfTI
    # exports restore the source frame for quantitative downstream use.
    if export_volumes:
        _export_volumes(case, case_dir, phase, mask, chi, output)
    write_json(case_dir / "metrics.json", metrics)
    write_json(
        case_dir / "metadata.json",
        dict(
            case.metadata,
            original_shape=list(case.original_shape or phase.shape[-3:]),
            crop_shape=list(case.crop_shape or phase.shape[-3:]),
            checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
            threshold=model.config.threshold,
            cutoff_mode=model.config.cutoff_mode,
            full_grid_field=model.config.full_grid_field,
            laplacian_input=model.config.laplacian_input,
            laplacian_operator=model.config.laplacian_operator,
            laplacian_extension_iters=model.config.laplacian_extension_iters,
            laplacian_extension_mode=model.config.laplacian_extension_mode,
            field_normalization=model.config.field_normalization,
            normalization_layer=model.config.normalization_layer,
            filter_input_version=model.config.filter_input_version,
            grid_shape=list(phase.shape[-3:]),
            inference=dict(
                asdict(inference_config),
                blending="gaussian" if inference_config.window_size else "full_volume",
                fourier_grid="full_volume",
            ),
            device=str(target),
            bf16=bool(
                bf16 and target.type == "cuda" and torch.cuda.is_bf16_supported()
            ),
        ),
    )
    return metrics


def _export_volumes(case, case_dir, phase, mask, chi, output, *, legacy=True):
    initial_name = "tkd" if legacy else "initial_chi"
    volumes = {
        "chi": output.chi,
        initial_name: output.initial_chi,
        "local_field": output.local_field,
        "harmonic_field": getattr(output, "harmonic_field", None),
        "correction": output.correction,
    }
    volumes = {name: value for name, value in volumes.items() if value is not None}
    arrays = dict(
        phase=case.crop_volume(phase),
        mask=case.crop_volume(mask),
        chi_reference=case.crop_volume(chi * mask),
    )
    for name, value in volumes.items():
        if legacy and name == "correction":
            continue
        array_name = {"local_field": "local", "harmonic_field": "harmonic"}.get(
            name, name
        )
        arrays[array_name] = case.crop_volume(value * mask)
    np.savez_compressed(case_dir / "volumes.npz", **arrays)
    volumes["mask"] = mask
    torch.save(
        {name: value.cpu() for name, value in volumes.items()},
        case_dir / "padded_predictions.pt",
    )
    for name, value in volumes.items():
        if legacy and name == "correction":
            continue
        nib.save(
            nib.Nifti1Image(case.restore_volume(value * mask), case.affine),
            case_dir / f"{name}.nii.gz",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument(
        "--datasets", nargs="+", choices=("cosmos", "sim2"), default=("cosmos", "sim2")
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--cosmos-simulation", type=Path)
    parser.add_argument("--sim2-pad", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-memory-gb", type=float, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--window-size",
        type=int,
        help="Override saved window size; 0 uses the full volume",
    )
    parser.add_argument(
        "--window-overlap", type=float, help="Fractional overlap, default 0.5"
    )
    parser.add_argument(
        "--gaussian-sigma-scale",
        type=float,
        help="Gaussian sigma / window size, default 0.125",
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        help="CNN windows evaluated together, default 1",
    )
    args = parser.parse_args(argv)
    device = require_device(args.device)
    configure_memory_budget(device, args.cuda_memory_gb)
    torch.set_num_threads(args.threads)
    cases = load_cases(
        tuple(args.datasets),
        data_dir=args.data_dir,
        cosmos_simulation=args.cosmos_simulation,
        sim2_pad=args.sim2_pad,
    )
    output = args.output_dir or args.checkpoint.parent / "evaluation"
    inference_options = {
        key: value
        for key, value in dict(
            window_size=args.window_size,
            overlap=args.window_overlap,
            sigma_scale=args.gaussian_sigma_scale,
            window_batch_size=args.window_batch_size,
        ).items()
        if value is not None
    }
    print(
        json.dumps(
            evaluate_checkpoint(
                args.checkpoint,
                cases,
                output,
                args.device,
                bf16=args.bf16,
                inference_options=inference_options,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
