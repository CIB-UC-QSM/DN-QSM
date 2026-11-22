"""Prepare, reconstruct and score the fixed COSMOS/Graz NeXtQSM comparison.

Run with the DN-QSM environment. The official TensorFlow inference is performed
separately by run_nextqsm_reference.py in an isolated TensorFlow environment.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

from dnqsm.evaluation import load_cosmos_case
from dnqsm.logging import _pyplot
from dnqsm.paths import REFERENCE_ROOT
from dnqsm.physics import apply_dipole, dipole_tensor
from dnqsm.runtime import load_trained_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_graz import load_reference


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array(value):
    return value.detach().cpu().float().numpy().squeeze()


def save(path, value):
    value = np.asarray(value, dtype=np.float32)
    assert value.ndim == 3 and np.isfinite(value).all()
    nib.save(nib.Nifti1Image(value, np.eye(4)), path)


def load(path):
    return nib.load(path).get_fdata(dtype=np.float32)


def tensor(value):
    return torch.from_numpy(np.ascontiguousarray(value))[None, None]


def pad_like_nextqsm(value):
    # This is the official misc.add_padding convention, including odd inputs.
    target = tuple(((n + 63) // 64) * 64 for n in value.shape)
    before = tuple((t - n - n % 2) // 2 + n % 2 for n, t in zip(value.shape, target))
    pads = tuple((b, t - n - b) for n, t, b in zip(value.shape, target, before))
    padded = np.pad(value, pads)
    slices = tuple(slice(b, b + n) for b, n in zip(before, value.shape))
    assert np.array_equal(padded[slices], value)
    return padded, before


def prepare(args):
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = hashlib.file_digest(checkpoint.open("rb"), "sha256").hexdigest()
    if args.append:
        previous = read_json(args.output / "inputs.json")
        if checkpoint_hash != previous["checkpoint_sha256"]:
            raise ValueError("Appended cases must use the same frozen checkpoint")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        previous = None
    selected = args.cases or ["cosmos", "graz2"]
    if previous and any(name in previous["datasets"] for name in selected):
        raise ValueError("The requested case already exists in this comparison")
    reference, reference_path = load_reference(REFERENCE_ROOT)
    options = SimpleNamespace(
        cache_dir=REFERENCE_ROOT / "artifacts/cache/graz",
        julia="/home/mathias/.juliaup/bin/julia",
        refresh_unwrapping=False,
        pad=20,
    )
    cases = {}
    for name in selected:
        if name == "cosmos":
            case = load_cosmos_case()
            cases[name] = (dict(total_field=array(case.phase), mask=array(case.mask),
                               chi_reference=array(case.chi), local_reference=array(case.local)), case.metadata)
        else:
            source, _ = reference.DATASETS[name]
            data, metadata = reference.prepare_dataset(name, source, 1.0, options, torch.device("cpu"))
            cases[name] = (dict(total_field=array(data["phase"]), mask=array(data["mask"])), metadata)
    records = dict(previous["datasets"]) if previous else {}
    for name, (volumes, metadata) in cases.items():
        directory = args.output / name
        directory.mkdir()
        mask = volumes["mask"]
        assert np.isin(mask, [0, 1]).all() and mask.any()
        coords = np.argwhere(mask)
        crop_start, crop_stop = coords.min(0), coords.max(0) + 1
        for key, value in volumes.items():
            if key != "mask":
                value = value * mask
            value, before = pad_like_nextqsm(value)
            save(directory / f"{key}.nii.gz", value)
        padded_mask = load(directory / "mask.nii.gz")
        assert int(padded_mask.sum()) == int(mask.sum())
        records[name] = dict(
            preprocessing=metadata, original_grid=list(mask.shape),
            common_grid=list(padded_mask.shape), padding_before=list(before),
            crop_start=(crop_start + before).tolist(), crop_stop=(crop_stop + before).tolist(),
            mask_voxels=int(mask.sum()), ground_truth_available=name == "cosmos",
        )
        print(json.dumps(dict(dataset=name, **records[name])), flush=True)
    write_json(args.output / "inputs.json", dict(
        datasets=records, voxel_size_mm=[1.0]*3, b0_direction=[0, 0, 1], units="ppm",
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        graz_preprocessing_script=str(reference_path.resolve()),
        padding="Official NeXtQSM multiple-of-64 padding, shared by both methods",
        normalization="Physical ppm; no fitted scale or susceptibility offset correction before inference",
    ))


@torch.inference_mode()
def reconstruct(args):
    inputs = read_json(args.output / "inputs.json")
    model = load_trained_model(inputs["checkpoint"], "cpu")
    for name, record in inputs["datasets"].items():
        if args.cases and name not in args.cases:
            continue
        directory = args.output / name
        phase, mask = [tensor(load(directory / f"{key}.nii.gz")) for key in ("total_field", "mask")]
        start = time.perf_counter()
        output = model.infer(phase, mask, dipole_tensor(phase.shape[-3:]))
        seconds = time.perf_counter() - start
        # The saved full-grid policy predicts exterior values. Preserve these
        # for the forward operator; masking chi here changes the physics.
        save(directory / "dnqsm_chi.nii.gz", array(output.chi))
        save(directory / "dnqsm_local.nii.gz", array(output.local_field))
        write_json(directory / "dnqsm_runtime.json", dict(
            seconds=seconds, device="cpu", dtype="float32", torch_version=torch.__version__,
            model_config=vars(model.config),
        ))
        print(json.dumps(dict(dataset=name, method="dnqsm", seconds=seconds)), flush=True)


def error_metrics(prediction, target, inside):
    prediction = prediction[inside].astype(np.float64)
    target = target[inside].astype(np.float64)
    error = prediction - target
    centered = (prediction - prediction.mean()) - (target - target.mean())
    return dict(
        nrmse_percent=100 * np.linalg.norm(error) / np.linalg.norm(target),
        demeaned_nrmse_percent=100 * np.linalg.norm(centered) / np.linalg.norm(target - target.mean()),
        rmse_ppm=np.sqrt(np.mean(error**2)), mean_error_ppm=error.mean(),
        correlation=np.corrcoef(prediction, target)[0, 1],
    )


def figure(directory, arrays, mask, crop, destination, title, limit):
    plt = _pyplot()
    mask = mask[crop]
    arrays = [(label, value[crop] * mask) for label, value in arrays]
    # Median occupied coordinate keeps the displayed planes within the brain.
    center = np.median(np.argwhere(mask), axis=0).astype(int)
    fig, axes = plt.subplots(3, len(arrays), figsize=(3.1*len(arrays), 8), squeeze=False, constrained_layout=True)
    for column, (label, volume) in enumerate(arrays):
        for axis in range(3):
            plane = np.take(volume, center[axis], axis=axis)
            image = axes[axis, column].imshow(np.rot90(plane), cmap="gray", vmin=-limit, vmax=limit)
            axes[axis, column].axis("off")
            if axis == 0:
                axes[axis, column].set_title(label)
            if column == 0:
                axes[axis, column].text(-.02, .5, f"Axis {axis}", rotation=90, va="center", transform=axes[axis, column].transAxes)
    fig.suptitle(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="ppm", shrink=.8)
    fig.savefig(directory / destination, dpi=170)
    plt.close(fig)


@torch.inference_mode()
def report(args):
    inputs = read_json(args.output / "inputs.json")
    results = {}
    verification = {}
    for name, record in inputs["datasets"].items():
        directory = args.output / name
        total, mask = [load(directory / f"{key}.nii.gz") for key in ("total_field", "mask")]
        inside = mask > .5
        crop = tuple(slice(a, b) for a, b in zip(record["crop_start"], record["crop_stop"]))
        dipole = dipole_tensor(total.shape)
        outputs = {}
        metrics = {}
        for method in ("dnqsm", "nextqsm"):
            chi, local = [load(directory / f"{method}_{key}.nii.gz") for key in ("chi", "local")]
            assert chi.shape == local.shape == total.shape
            assert np.isfinite(chi).all() and np.isfinite(local).all()
            predicted = array(apply_dipole(tensor(chi), dipole))
            residual = (predicted - local)[inside].astype(np.float64)
            masked_prediction = array(apply_dipole(tensor(chi * mask), dipole))
            masked_residual = (masked_prediction - local)[inside].astype(np.float64)
            metrics[method] = dict(
                common_operator_local_residual_percent=100*np.linalg.norm(residual)/np.linalg.norm(local[inside].astype(np.float64)),
                common_operator_total_residual_percent=100*np.linalg.norm(residual)/np.linalg.norm(total[inside].astype(np.float64)),
                common_operator_residual_rms_ppm=np.sqrt(np.mean(residual**2)),
                brain_masked_chi_local_residual_percent=100*np.linalg.norm(masked_residual)/np.linalg.norm(local[inside].astype(np.float64)),
                brain_masked_chi_total_residual_percent=100*np.linalg.norm(masked_residual)/np.linalg.norm(total[inside].astype(np.float64)),
                chi_mean_ppm=float(chi[inside].mean()), chi_std_ppm=float(chi[inside].std()),
                runtime=read_json(directory / f"{method}_runtime.json"),
            )
            if name == "cosmos":
                metrics[method]["susceptibility"] = error_metrics(chi, load(directory / "chi_reference.nii.gz"), inside)
                metrics[method]["local_field"] = error_metrics(local, load(directory / "local_reference.nii.gz"), inside)
            outputs[method] = dict(chi=chi, local=local)
            for key, value in dict(chi=chi, local=local, background=(total-local)*mask).items():
                save(directory / f"{method}_{key}_brain_crop.nii.gz", (value * mask)[crop])
        chi_arrays = [("DN-QSM", outputs["dnqsm"]["chi"]), ("NeXtQSM", outputs["nextqsm"]["chi"])]
        local_arrays = [("Total field", total), ("DN-QSM local", outputs["dnqsm"]["local"]), ("NeXtQSM local", outputs["nextqsm"]["local"])]
        if name == "cosmos":
            target = load(directory / "chi_reference.nii.gz")
            chi_arrays.insert(0, ("COSMOS ground truth", target))
            local_arrays.insert(1, ("True local field", load(directory / "local_reference.nii.gz")))
            figure(directory, [("DN-QSM error", outputs["dnqsm"]["chi"]-target), ("NeXtQSM error", outputs["nextqsm"]["chi"]-target)], mask, crop, "susceptibility_errors.png", "COSMOS: signed susceptibility error", .10)
        figure(directory, chi_arrays, mask, crop, "susceptibility_comparison.png", f"{name.upper()}: susceptibility", .15)
        figure(directory, local_arrays, mask, crop, "local_field_comparison.png", f"{name.upper()}: fields", .1)
        write_json(directory / "metrics.json", metrics)
        results[name] = metrics
        verification[name] = dict(aligned_finite_outputs=True, mask_binary=True, mask_voxels=int(inside.sum()), common_grid=list(total.shape))
    write_json(args.output / "metrics.json", results)
    write_json(args.output / "verification.json", verification)
    grids = "; ".join(
        f"{name.upper()} {'×'.join(str(n) for n in record['common_grid'])}"
        for name, record in inputs["datasets"].items()
    )
    lines = [
        "# NeXtQSM comparison with DN-QSM", "",
        "Official pretrained [NeXtQSM](https://github.com/QSMxT/NeXtQSM) versus the COSMOS-selected DN-QSM epoch 18 checkpoint.", "",
        "Both methods receive the identical masked total field in ppm, binary mask, 1 mm isotropic spacing and B0=(0,0,1). Full-volume FP32 CPU inference; no data-specific gain tuning or retraining. NeXtQSM's native six-step solver and learned lambdas are unchanged.", "",
        f"Both use the same multiple-of-64 grids: {grids}.", "",
        "## COSMOS ground-truth accuracy", "",
        "| Method | χ NRMSE (%) | χ demeaned NRMSE (%) | Local-field NRMSE (%) |",
        "|---|---:|---:|---:|",
    ]
    for method, metric in results["cosmos"].items():
        lines.append(f"| {method} | {metric['susceptibility']['nrmse_percent']:.3f} | {metric['susceptibility']['demeaned_nrmse_percent']:.3f} | {metric['local_field']['nrmse_percent']:.3f} |")
    lines += ["", "NRMSE is 100‖prediction−reference‖₂/‖reference‖₂ over the original brain mask. Demeaned NRMSE subtracts each image's own masked mean before computing the ratio; it exposes susceptibility-reference offsets without fitting any scale.", "", "## Graz field consistency (no susceptibility ground truth)", "", "| Dataset | Method | Residual / total field (%) | Residual / estimated local field (%) | RMS (ppm) |", "|---|---|---:|---:|---:|"]
    graz_names = sorted(name for name in results if name.startswith("graz"))
    for name in graz_names:
        for method, metric in results[name].items():
            lines.append(f"| {name} | {method} | {metric['common_operator_total_residual_percent']:.3f} | {metric['common_operator_local_residual_percent']:.3f} | {metric['common_operator_residual_rms_ppm']:.6f} |")
    lines += [
        "", "The primary residual preserves each model's native output support: DN-QSM predicts the full grid, while NeXtQSM masks susceptibility after each solver step. For an additional common-support diagnostic, setting susceptibility to zero outside the brain gives:",
        "", "| Dataset | Method | Brain-masked χ: residual / total (%) | Brain-masked χ: residual / local (%) |",
        "|---|---|---:|---:|",
    ]
    for name in graz_names:
        for method, metric in results[name].items():
            lines.append(f"| {name} | {method} | {metric['brain_masked_chi_total_residual_percent']:.3f} | {metric['brain_masked_chi_local_residual_percent']:.3f} |")
    lines += [
        "", "DN-QSM was selected using this COSMOS case, whereas NeXtQSM uses the published pretrained weights. This is a comparison on the requested case, rather than an independent held-out model-ranking benchmark.",
    ]
    if "graz1" in results:
        lines += [
            "", "Graz1 uses the embedded unwrapped phase2 echoes from data_blend_1_graz.mat. The same magnitude-squared weighted echo fit converts these to a total field in ppm, followed by resampling from 0.798611×0.798611×2 mm to 1 mm isotropic. Its original 201³ processed grid is padded to 256³ using NeXtQSM's odd-grid convention (28 voxels before and 27 after each axis). All original brain voxels are retained. Both models receive exactly the same field and mask; there is no susceptibility ground truth for either Graz dataset.",
        ]
    lines += ["", "Residual = Dχ − estimated local field, inside the mask. Both methods are scored with the same continuous dipole operator (DC=0). NeXtQSM retains its native internal dipole convention (DC=1/3 and float32 epsilon), so this diagnostic is distinct from its internal objective. A lower consistency residual does not establish more accurate Graz2 susceptibility, especially when the local field is also estimated.", "", "## Geometry and provenance", "", "COSMOS uses the fixed SNR=100/background-amplitude=10 simulation, converted to physical ppm using its saved factor (0.2125). The clean reference field is kept from the original 184³ simulation operator, rather than regenerated on the padded inference grid. Graz2 follows the existing ROMEO weighted echo-fit and resampling to 1 mm; no new unwrapping or scanner registration is performed. Padding is zero padding only; original ROI voxels are retained.", "", "Full-grid NIfTIs and *_brain_crop.nii.gz exports have identity affine in the processed array frame (1 mm), not the original scanner frame. Figures label array axes rather than assuming anatomical orientation. Background exports equal total field minus the estimated local field.", "", "See inputs.json for dataset/checkpoint provenance, nextqsm_provenance.json for official source and weight hashes, verification.json for output checks, metrics.json for full raw and demeaned metrics, and per-case PNGs for comparisons. The DN-QSM checkpoint is the frozen epoch 18 snapshot used in the preceding Graz evaluation. The larger shared inference grids can change DN-QSM predictions compared with earlier evaluations on 184³/196³ grids. CPU runtimes include graph tracing for NeXtQSM and are not a hardware performance benchmark.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "dnqsm", "report"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/evaluations/graz_softcut01_best18_fp32_20261027_130316/model_snapshot.pt"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cases", nargs="+", choices=["cosmos", "graz1", "graz2"])
    parser.add_argument("--append", action="store_true", help="Append new cases to an existing comparison")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    {"prepare": prepare, "dnqsm": reconstruct, "report": report}[args.stage](args)
