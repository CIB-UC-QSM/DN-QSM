"""Evaluate DN-QSM on Graz 1/2 using the wh_net benchmark preprocessing."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from dnqsm.logging import _pyplot, write_json
from dnqsm.paths import REFERENCE_ROOT
from dnqsm.physics import apply_dipole, masked_mean, validate_mask
from dnqsm.runtime import autocast_context, load_trained_model, require_device


def load_reference(project: Path):
    _pyplot()
    sys.path.insert(0, str(project.resolve() / "src"))
    path = project / "scripts/benchmark_graz.py"
    spec = importlib.util.spec_from_file_location("graz_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def cropped(value, mask, reference):
    return reference.crop_array(value * mask, mask)


def cosmos_local_field_gain(statistics, name):
    """Match the previous Graz estimate's standard deviation to clean COSMOS."""
    if statistics["units"] != "ppm":
        raise ValueError("Normalization statistics must be in ppm")
    records = statistics["statistics"]["full_roi"]
    target = records["cosmos_reference"]["std_ppm"]
    source = records[f"{name}_estimated"]["std_ppm"]
    if not all(math.isfinite(value) and value > 0 for value in (source, target)):
        raise ValueError(
            "Normalization standard deviations must be positive and finite"
        )
    return target / source


def denormalize_reconstruction(output, gain):
    """Undo a positive input-field gain on all field and susceptibility outputs."""
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Normalization gain must be positive and finite")
    if gain == 1:
        return output
    return replace(
        output,
        local_field=output.local_field / gain,
        tkd=output.tkd / gain,
        correction=output.correction / gain,
        chi=output.chi / gain,
        harmonic_field=None
        if output.harmonic_field is None
        else output.harmonic_field / gain,
    )


def plot_comparison(arrays, destination, title, limit):
    plt = _pyplot()
    center = tuple(n // 2 for n in arrays[0][1].shape)
    selectors = (
        ("Axial", lambda x: x[center[0]]),
        ("Coronal", lambda x: x[:, center[1]]),
        ("Sagittal", lambda x: x[:, :, center[2]]),
    )
    figure, axes = plt.subplots(
        3,
        len(arrays),
        figsize=(3.2 * len(arrays), 8),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (label, volume) in enumerate(arrays):
        for row, (view, selector) in enumerate(selectors):
            image = axes[row, column].imshow(
                np.rot90(selector(volume)), cmap="gray", vmin=-limit, vmax=limit
            )
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(label)
            if column == 0:
                axes[row, column].text(
                    -0.03,
                    0.5,
                    view,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=axes[row, column].transAxes,
                )
    figure.suptitle(title)
    figure.colorbar(image, ax=axes.ravel().tolist(), label="ppm", shrink=0.8)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


@torch.inference_mode()
def evaluate_dataset(name, reference, model, args, device):
    source, _ = reference.DATASETS[name]
    destination = args.output_dir / name
    destination.mkdir()
    options = SimpleNamespace(
        cache_dir=args.reference_project / "artifacts/cache/graz",
        julia="/home/mathias/.juliaup/bin/julia",
        refresh_unwrapping=False,
        pad=args.pad,
    )
    # Read the established unwrapping cache; prepare_dataset uses embedded phase2
    # for Graz 1. A factor of one leaves the field in ppm for DN-QSM.
    data, preprocessing = reference.prepare_dataset(name, source, 1.0, options, device)
    validate_mask(data["mask"], data["phase"])
    assert preprocessing["target_voxel_size_mm"] == [1.0] * 3
    assert preprocessing["solver_factor"] == 1.0
    statistics = (
        json.loads(args.normalization_statistics.read_text())
        if args.normalization_statistics
        else None
    )
    gain = cosmos_local_field_gain(statistics, name) if statistics else 1.0
    model_input = data["phase"] * gain if statistics else data["phase"]
    print(json.dumps(dict(dataset=name, shape=list(data["phase"].shape))), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    context = (
        torch.autocast("cpu", dtype=torch.bfloat16)
        if device.type == "cpu" and args.bf16
        else autocast_context(device, args.bf16)
    )
    with context:
        normalized_output = model.infer(model_input, data["mask"], data["dipole"])
    output = denormalize_reconstruction(normalized_output, gain)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    background = (data["phase"] - output.local_field) * data["mask"]
    assert all(
        bool(torch.isfinite(value).all())
        for value in (output.chi, output.tkd, output.local_field, background)
    )
    predicted = apply_dipole(output.chi, data["dipole"])
    error = predicted - output.local_field
    metrics = reference.solution_metrics(output.chi, background, data, seconds)
    metrics["local_forward_residual_rms_ppm"] = float(
        masked_mean(error.square(), data["mask"]).sqrt().mean()
    )
    metrics["local_forward_normalized_residual"] = float(
        (error * data["mask"]).norm()
        / (output.local_field * data["mask"]).norm().clamp_min(1e-12)
    )
    arrays = {
        "chi": cropped(output.chi, data["mask"], reference),
        "tkd": cropped(output.tkd, data["mask"], reference),
        "correction": cropped(output.correction, data["mask"], reference),
        "local_field": cropped(output.local_field, data["mask"], reference),
        "background": cropped(background, data["mask"], reference),
        "total_field": cropped(data["phase"], data["mask"], reference),
        "mask": reference.crop_array(data["mask"], data["mask"]),
    }
    for key, array in arrays.items():
        nib.save(nib.Nifti1Image(array, np.eye(4)), destination / f"{key}.nii.gz")
    if statistics:
        for key, value in (
            ("normalized_total_field", model_input),
            ("normalized_local_field", normalized_output.local_field),
        ):
            nib.save(
                nib.Nifti1Image(cropped(value, data["mask"], reference), np.eye(4)),
                destination / f"{key}.nii.gz",
            )
    np.savez_compressed(destination / "volumes.npz", **arrays)
    comparisons = [
        (
            "Soft inverse" if model.config.cutoff_mode == "soft" else "TKD",
            arrays["tkd"],
        ),
        ("DN-QSM", arrays["chi"]),
    ]
    historical = {}
    if args.comparison_run:
        baseline_dir = args.comparison_run / name
        baseline = json.loads((baseline_dir / "metrics.json").read_text())
        for key in (
            "target_voxel_size_mm",
            "padded_shape",
            "resampled_cropped_shape",
            "pad",
        ):
            assert baseline["preprocessing"][key] == preprocessing[key], key
        for label, stem, method in (
            ("WHNet (saved)", "whnet", "whnet"),
            ("WH-TV (saved)", "wh_tv_l_curve", "wh_tv_lcurve"),
        ):
            path = baseline_dir / f"{stem}_chi.nii.gz"
            image = np.asarray(nib.load(path).dataobj, dtype=np.float32)
            if image.shape != arrays["chi"].shape:
                raise ValueError(f"Reference geometry differs: {path}")
            comparisons.append((label, image * arrays["mask"]))
            record = dict(baseline["methods"][method])
            for key in (
                "data_residual_rms",
                "chi_tv_mean",
                "background_laplacian_rms",
                "chi_rms",
                "background_rms",
            ):
                record[key] *= baseline["preprocessing"]["solver_factor"]
            historical[method] = record
    plot_comparison(
        comparisons,
        destination / "susceptibility_comparison.png",
        f"{name.upper()} — susceptibility (no ground truth)",
        0.25,
    )
    plot_comparison(
        [
            ("Total field", arrays["total_field"]),
            ("DN-QSM local field", arrays["local_field"]),
            ("Estimated background", arrays["background"]),
        ],
        destination / "field_decomposition.png",
        f"{name.upper()} — fields",
        0.1,
    )
    metadata = dict(
        preprocessing,
        physical_units="ppm",
        ground_truth_available=False,
        checkpoint=str(args.checkpoint.resolve()),
        cutoff_mode=model.config.cutoff_mode,
        threshold=model.config.threshold,
        inference=asdict(model.inference_config),
        background_definition="masked total field minus estimated local field",
        affine_source="identity, 1 mm resampled cropped grid; matches wh_net exports",
        mask_voxels=int(data["mask"].sum()),
        historical_comparison_run=str(args.comparison_run)
        if args.comparison_run
        else None,
        metric_note="Consistency metrics are not susceptibility accuracy; no NRMSE without ground truth.",
    )
    if statistics:
        selected = data["mask"].bool()
        metadata["input_normalization"] = dict(
            method="Positive scalar gain from prior local-field standard deviations; no nonlinear histogram remapping",
            gain=gain,
            reference_local_std_ppm=statistics["statistics"]["full_roi"][
                "cosmos_reference"
            ]["std_ppm"],
            original_estimated_local_std_ppm=statistics["statistics"]["full_roi"][
                f"{name}_estimated"
            ]["std_ppm"],
            normalized_estimated_local_std=float(
                normalized_output.local_field[selected].double().std(correction=0)
            ),
            denormalized_local_std_ppm=float(
                output.local_field[selected].double().std(correction=0)
            ),
            input_transform="total_field * gain",
            output_transform="local_field, TKD, correction, susceptibility and harmonic field divided by gain",
            statistics=str(args.normalization_statistics.resolve()),
            normalized_exports_units="scaled model-space ppm; all other exports and metrics restored to physical ppm",
        )
    write_json(destination / "metadata.json", metadata)
    write_json(destination / "metrics.json", metrics)
    result = dict(metrics=metrics, historical_comparisons_physical_ppm=historical)
    print(json.dumps(dict(dataset=name, **result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-project", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--comparison-run", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pad", type=int, default=20)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalization-statistics", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.pad < 0:
        parser.error("threads must be positive and padding nonnegative")
    torch.set_num_threads(args.threads)
    device = require_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference, reference_path = load_reference(args.reference_project)
    model = load_trained_model(args.checkpoint, device)
    saved = torch.load(
        args.checkpoint, weights_only=True, map_location="cpu", mmap=True
    )
    config = dict(
        checkpoint=str(args.checkpoint.resolve()),
        epoch=saved.get("epoch"),
        step=saved.get("step"),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        reference_script=str(reference_path),
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        device=str(device),
        bf16=args.bf16,
        pad=args.pad,
        threads=args.threads,
        model_config=asdict(model.config),
        ground_truth_available=False,
    )
    del saved
    if args.normalization_statistics:
        statistics = json.loads(args.normalization_statistics.read_text())
        if statistics["checkpoint_sha256"] != config["checkpoint_sha256"]:
            raise ValueError("Normalization statistics and inference checkpoint differ")
        for name in ("graz1", "graz2"):
            cosmos_local_field_gain(statistics, name)
        config["normalization_statistics"] = str(
            args.normalization_statistics.resolve()
        )
        config["normalization_statistics_sha256"] = hashlib.sha256(
            args.normalization_statistics.read_bytes()
        ).hexdigest()
    write_json(args.output_dir / "graz_config.json", config)
    results = {}
    for name in ("graz1", "graz2"):
        results[name] = evaluate_dataset(name, reference, model, args, device)
        write_json(
            args.output_dir / "summary.json", dict(config=config, datasets=results)
        )


if __name__ == "__main__":
    main()
