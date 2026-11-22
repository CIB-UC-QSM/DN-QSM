"""Create COSMOS article figures and an exact DN-QSM correction decomposition."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import torch

from dnqsm.evaluation import load_cosmos_case
from dnqsm.physics import apply_multiplier, cutoff_kernels, masked_nrmse
from dnqsm.runtime import load_trained_model


METHODS = (
    ("iqsm", "iQSM"),
    ("nextqsm", "NeXtQSM"),
    ("soft_tkd", "DN-QSM: soft-TKD"),
    ("dnqsm_final", "DN-QSM: final"),
)
VIEWS = (("Axial", 2), ("Sagittal", 0), ("Coronal", 1))
CHI_LIMIT = .1
DIFFERENCE_LIMIT = .02


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load(path: Path) -> np.ndarray:
    image = nib.load(path)
    value = image.get_fdata(dtype=np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError(f"Expected one finite 3D image: {path}")
    return value


def save(path: Path, value: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(np.asarray(value, dtype=np.float32), np.eye(4)), path)


def score(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict:
    pred, target = prediction[mask].astype(np.float64), reference[mask].astype(np.float64)
    delta = pred-target
    return dict(
        nrmse_percent=float(100*np.linalg.norm(delta)/np.linalg.norm(target)),
        rmse_ppm=float(np.sqrt(np.mean(delta**2))),
        mean_error_ppm=float(delta.mean()),
        correlation=float(np.corrcoef(pred, target)[0, 1]),
        susceptibility_display_clipped_fraction=float(np.mean(np.abs(pred) > CHI_LIMIT)),
        difference_display_clipped_fraction=float(np.mean(np.abs(delta) > DIFFERENCE_LIMIT)),
    )


@torch.inference_mode()
def prepare(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    comparison = args.nextqsm_dir
    comparison_inputs = json.loads((comparison / "inputs.json").read_text())
    case_record = comparison_inputs["datasets"]["cosmos"]
    case = load_cosmos_case(case_record["preprocessing"]["simulation"])
    model = load_trained_model(args.checkpoint, "cpu")
    if model.config.cutoff_mode != "soft":
        raise ValueError("This figure requires a soft-cutoff DN-QSM checkpoint")
    if model.inference_config.window_size:
        raise ValueError("Use the full-volume checkpoint for this comparison")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    start = time.perf_counter()
    output = model.infer(case.phase, case.mask, case.dipole)
    seconds = time.perf_counter()-start
    assert not torch.is_autocast_enabled("cpu")
    assert all(x.dtype == torch.float32 for x in (output.tkd, output.chi, output.correction))
    torch.testing.assert_close(output.chi, output.tkd+output.correction, rtol=0, atol=0)
    inverse, weighting = cutoff_kernels(case.dipole, model.config.threshold, "soft")
    torch.testing.assert_close(output.tkd, apply_multiplier(output.local_field, inverse), rtol=0, atol=0)
    reference = case.restore_volume(case.chi*case.mask)
    mask = case.restore_volume(case.mask) > .5
    volumes = dict(
        reference=reference, mask=mask.astype(np.float32),
        soft_tkd=case.restore_volume(output.tkd*case.mask),
        dnqsm_final=case.restore_volume(output.chi*case.mask),
        learned_correction=case.restore_volume(output.correction*case.mask),
        estimated_local_field=case.restore_volume(output.local_field*case.mask),
    )
    # Undo only the official extra padding, then the simulation padding. No
    # registration, interpolation, amplitude calibration or reference fitting.
    before = case_record["padding_before"]
    original_grid = case_record["original_grid"]
    undo_extra = tuple(slice(b, b+n) for b, n in zip(before, original_grid, strict=True))
    sim_crop = tuple(slice(p, p+n) for p, n in zip(case.pad_offsets, case.original_shape, strict=True))
    native_reference = load(comparison / "cosmos/chi_reference.nii.gz")[undo_extra]
    native_mask = load(comparison / "cosmos/mask.nii.gz")[undo_extra]
    assert np.array_equal(native_mask, case.mask.numpy()[0, 0])
    assert np.array_equal(native_reference, (case.chi*case.mask).numpy()[0, 0])
    volumes["nextqsm"] = load(comparison / "cosmos/nextqsm_chi.nii.gz")[undo_extra][sim_crop]*mask
    iqsm_image = nib.load(args.iqsm)
    if iqsm_image.shape != reference.shape or not np.allclose(iqsm_image.affine, case.affine):
        raise ValueError("iQSM must already match the original COSMOS array grid")
    if not np.allclose(iqsm_image.header.get_zooms(), [1, 1, 1]):
        raise ValueError("iQSM spacing differs from the COSMOS processed grid")
    volumes["iqsm"] = load(args.iqsm)*mask
    for name, value in volumes.items():
        assert value.shape == reference.shape and np.isfinite(value).all()
        save(args.output / f"{name}.nii.gz", value)
    values = {key: score(volumes[key], reference, mask) for key, _ in METHODS}
    # Check the independent NumPy score against the training/evaluation metric.
    for key, prediction in (("soft_tkd", output.tkd), ("dnqsm_final", output.chi)):
        expected = masked_nrmse(prediction, case.chi, case.mask)[0]
        assert abs(values[key]["nrmse_percent"]-expected) < 2e-5
    existing_next = json.loads((comparison / "cosmos/metrics.json").read_text())
    assert abs(values["nextqsm"]["nrmse_percent"]-existing_next["nextqsm"]["susceptibility"]["nrmse_percent"]) < 1e-10
    delta = volumes["dnqsm_final"]-(volumes["soft_tkd"]+volumes["learned_correction"])
    assert np.max(np.abs(delta)) == 0
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    provenance = dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        checkpoint_epoch=saved.get("epoch"), checkpoint_step=saved.get("step"),
        model_config=asdict(model.config), dnqsm_inference_grid=list(case.phase.shape[-3:]),
        nextqsm_inference_grid=case_record["common_grid"],
        display_and_scoring_grid=list(reference.shape), mask_voxels=int(mask.sum()),
        cosmos=case.metadata, iqsm=str(args.iqsm.resolve()), iqsm_sha256=digest(args.iqsm),
        iqsm_input_simulation_provenance="User-supplied reconstruction; original iQSM inference inputs/settings unavailable",
        nextqsm_chi_sha256=digest(comparison / "cosmos/nextqsm_chi.nii.gz"),
        nextqsm_provenance=json.loads((comparison / "nextqsm_provenance.json").read_text()),
        nextqsm_crop=dict(undo_extra_padding=before, simulation_padding=list(case.pad_offsets)),
        interpolation="None", fitted_gain=False, fitted_offset=False, units="ppm",
        dtype="float32", device="cpu", autocast=False, dnqsm_seconds=seconds,
        nrmse="100 * ||(estimate-reference)[mask]||_2 / ||reference[mask]||_2, whole 3D ROI, no demeaning",
        correction="chi_final = chi_soft_TKD + correction; correction = IFFT[(tau^2/(D^2+tau^2))*FFT(residual_network)]",
        soft_tkd="IFFT[(D/(D^2+tau^2))*FFT(estimated_local_field)], tau=0.1",
        correction_interpretation="Soft weighting emphasizes small-|D| frequencies; it is not a strict hard null-space projection",
    )
    write_json(args.output / "metrics.json", values)
    write_json(args.output / "provenance.json", provenance)
    write_json(args.output / "verification.json", dict(
        reference_and_mask_match_previous_comparison=True, iqsm_geometry_matches=True,
        all_outputs_finite_fp32=True, decomposition_exact=True,
        decomposition_max_absolute_error_ppm=float(np.abs(delta).max()),
        numpy_nrmse_matches_torch=True, nextqsm_nrmse_matches_prior_evaluation=True,
    ))
    print(json.dumps(dict(event="prepared", metrics=values, seconds=seconds)), flush=True)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": "black", "axes.labelcolor": "black",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "image.interpolation": "nearest",
    })


def plane(value: np.ndarray, mask: np.ndarray, axis: int, index: int) -> np.ma.MaskedArray:
    image = np.rot90(np.take(value, index, axis=axis))
    support = np.rot90(np.take(mask, index, axis=axis))
    positions = np.argwhere(support)
    lo, hi = positions.min(0), positions.max(0)+1
    crop = tuple(slice(max(0, int(a)-2), min(n, int(b)+2)) for a, b, n in zip(lo, hi, image.shape, strict=True))
    return np.ma.masked_where(~support[crop], image[crop])


def panel(ax, value, mask, axis, index, limit, cmap) -> None:
    ax.imshow(plane(value, mask, axis, index), cmap=cmap,
              norm=Normalize(-limit, limit, clip=True), interpolation="nearest")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def colorbar(fig, rect, limit: float, label: str, cmap) -> None:
    ax = fig.add_axes(rect)
    bar = fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap=cmap), cax=ax, orientation="horizontal")
    bar.set_ticks(np.linspace(-limit, limit, 5))
    bar.set_label(label, fontsize=11, labelpad=5)
    bar.ax.tick_params(labelsize=10, length=3, width=.6, pad=3)
    bar.outline.set_linewidth(.6)


def export(fig, directory: Path, stem: str, dpi: int) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=dpi, facecolor="white")
    fig.savefig(directory / f"{stem}.pdf", dpi=dpi, facecolor="white")
    fig.savefig(directory / f"{stem}_preview.png", dpi=125, facecolor="white")
    plt.close(fig)


def plot(args: argparse.Namespace) -> None:
    style()
    volumes = {path.name.removesuffix(".nii.gz"): load(path) for path in args.output.glob("*.nii.gz")}
    reference, mask = volumes["reference"], volumes["mask"] > .5
    metrics = json.loads((args.output / "metrics.json").read_text())
    views = [(name, axis, reference.shape[axis]//2) for name, axis in VIEWS]
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad("black")
    columns = (("reference", "COSMOS reference"),)+METHODS
    fig = plt.figure(figsize=(11.6, 13.0))
    grid = fig.add_gridspec(6, 5, left=.052, right=.987, bottom=.113, top=.89,
                            hspace=.045, wspace=.055)
    for view_index, (view_name, axis, index) in enumerate(views):
        for column, (key, title) in enumerate(columns):
            value = volumes[key]
            reconstruction = fig.add_subplot(grid[2*view_index, column])
            difference = fig.add_subplot(grid[2*view_index+1, column])
            panel(reconstruction, value, mask, axis, index, CHI_LIMIT, cmap)
            panel(difference, (value-reference)*mask, mask, axis, index, DIFFERENCE_LIMIT, cmap)
            if view_index == 0:
                nrmse = 0.0 if key == "reference" else metrics[key]["nrmse_percent"]
                reconstruction.set_title(f"{title}\nNRMSE = {nrmse:.2f}%", fontsize=12, pad=10, linespacing=1.45)
            if column == 0:
                reconstruction.text(-.12, .5, f"{view_name}\nχ", transform=reconstruction.transAxes,
                                    rotation=90, ha="center", va="center", fontsize=11)
                difference.text(-.12, .5, "Difference", transform=difference.transAxes,
                                rotation=90, ha="center", va="center", fontsize=10)
    fig.suptitle("COSMOS susceptibility reconstruction", fontsize=17, weight="semibold", y=.979)
    fig.text(.5, .946, "Central orthogonal slices · NRMSE over the full 3D brain mask", ha="center", fontsize=11)
    colorbar(fig, [.11, .067, .34, .012], CHI_LIMIT, "Susceptibility, χ (ppm)", cmap)
    colorbar(fig, [.58, .067, .34, .012], DIFFERENCE_LIMIT, "Difference, estimate − COSMOS (ppm)", cmap)
    fig.text(.5, .014, "Gray colormap · fixed display limits · no fitted scale or offset", ha="center", fontsize=10)
    export(fig, args.output, "cosmos_method_comparison", args.dpi)

    # The middle panel is the actual post-weighting correction added by forward().
    fig = plt.figure(figsize=(9.8, 10.2))
    grid = fig.add_gridspec(3, 3, left=.07, right=.98, bottom=.20, top=.84,
                            hspace=.045, wspace=.095)
    decomposition = (("soft_tkd", "Soft-TKD", CHI_LIMIT),
                     ("learned_correction", "Added soft-null component", DIFFERENCE_LIMIT),
                     ("dnqsm_final", "Final DN-QSM", CHI_LIMIT))
    for row, (view_name, axis, index) in enumerate(views):
        for column, (key, title, limit) in enumerate(decomposition):
            ax = fig.add_subplot(grid[row, column])
            panel(ax, volumes[key], mask, axis, index, limit, cmap)
            if row == 0:
                suffix = (f"NRMSE = {metrics[key]['nrmse_percent']:.2f}%" if key != "learned_correction"
                          else r"$\Delta\chi = \chi_{\mathrm{final}} - \chi_{\mathrm{soft}}$")
                ax.set_title(f"{title}\n{suffix}", fontsize=12, pad=10, linespacing=1.4)
            if column == 0:
                ax.text(-.1, .5, view_name, transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=11)
    fig.suptitle("DN-QSM learned susceptibility correction", fontsize=17, weight="semibold", y=.978)
    fig.text(.5, .925, "Soft-TKD  +  learned soft-null component  =  final reconstruction", ha="center", fontsize=12)
    colorbar(fig, [.12, .13, .33, .015], CHI_LIMIT, "Soft-TKD and final χ (ppm)", cmap)
    colorbar(fig, [.60, .13, .30, .015], DIFFERENCE_LIMIT, "Added component, Δχ (ppm)", cmap)
    fig.text(.5, .024, "Actual additive correction; soft frequency weighting (τ = 0.1).\nSoft weighting emphasizes the near-null region of the dipole kernel.", ha="center", va="bottom", fontsize=10, linespacing=1.5)
    export(fig, args.output, "cosmos_soft_null_component", args.dpi)
    write_json(args.output / "figure_settings.json", dict(
        dpi=args.dpi, formats=["PNG", "PDF"], colormap="gray", outside_roi_color="black",
        susceptibility_limits_ppm=[-CHI_LIMIT, CHI_LIMIT],
        difference_limits_ppm=[-DIFFERENCE_LIMIT, DIFFERENCE_LIMIT],
        correction_limits_ppm=[-DIFFERENCE_LIMIT, DIFFERENCE_LIMIT],
        views=[dict(name=name, array_axis=axis, original_grid_index=index) for name, axis, index in views],
        slice_selection="Original 160³ COSMOS grid midpoint; identical across all methods",
        interpolation="nearest for display only; no volume resampling",
    ))
    summary = ["# COSMOS article figures", "", "| Method | Whole-mask 3D NRMSE (%) |", "|---|---:|"]
    for key, label in METHODS:
        summary.append(f"| {label} | {metrics[key]['nrmse_percent']:.4f} |")
    summary += ["", "Main figure: cosmos_method_comparison.png / .pdf. Complementary figure: cosmos_soft_null_component.png / .pdf. Previews use 125 dpi; publication PNG/PDF exports use the requested figure DPI.", "", "All image panels use gray. Susceptibility limits are [−0.1, 0.1] ppm. Signed differences are estimate minus COSMOS, with limits [−0.02, 0.02] ppm. The added soft-null component also uses [−0.02, 0.02] ppm. Outside-mask values are displayed in black. Display clipping is applied only by the colormap; metrics use the unclipped 3D data.", "", "DN-QSM uses the frozen COSMOS-selected epoch 18 checkpoint on the original 184³ COSMOS simulation grid, followed by removal of its 12-voxel padding. NeXtQSM is the previously evaluated official pretrained model on its native 192³ padded grid, cropped back to the same 160³ frame. iQSM is the user-provided reconstruction on that frame; its original inference inputs/settings are unavailable. No intensity fitting, registration, resampling, or mean-offset correction is applied.", "", "The soft-TKD panel uses DN-QSM's estimated local field, rather than the ground-truth field. It is the regularized inverse D/(D²+τ²), τ=0.1. The learned component is the actual correction returned by the saved model after soft weighting τ²/(D²+τ²); soft-TKD plus that component equals the final result exactly. Because the weighting is soft, this is a near-null correction rather than an exact hard null-space projection.", "", "NRMSE = 100‖(estimate−reference)[mask]‖₂ / ‖reference[mask]‖₂. It is measured over the full 3D COSMOS brain mask, rather than only the displayed slices. The checkpoint was selected using COSMOS, so this is a development-case comparison. All reconstructions, the correction, metrics, provenance and figure settings are saved alongside the figures.", ""]
    (args.output / "README.md").write_text("\n".join(summary))
    shutil.copy2(__file__, args.output / Path(__file__).name)
    print(json.dumps(dict(event="figures_saved", output=str(args.output.resolve()), metrics=metrics)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "prepare", "plot"], default="all")
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/cosmos_article_best18_20261027"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/evaluations/graz_softcut01_best18_fp32_20261027_130316/model_snapshot.pt"))
    parser.add_argument("--nextqsm-dir", type=Path, default=Path("artifacts/evaluations/nextqsm_comparison_20261027"))
    parser.add_argument("--iqsm", type=Path, default=Path("/home/mathias/Documents/PhD/wh_net/data/evaluation/volumes/iQSM.nii.gz"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    torch.set_float32_matmul_precision("highest")
    if args.stage in ("all", "prepare"):
        prepare(args)
    if args.stage in ("all", "plot"):
        plot(args)


if __name__ == "__main__":
    main()
