"""Create Graz1 method and soft-null montages using existing reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from create_cosmos_composite_figures import (
    composite, latex_style, load, montage, render, sha256, write_json,
)


COLUMNS = (("nextqsm", "Next-QSM"), ("iqsm", "iQSM"),
           ("soft_tkd", "Tikhonov regularized inverse"), ("dnqsm_final", "Final DN-QSM"))


def assert_geometry(path: Path, shape) -> None:
    image = nib.load(path)
    if image.shape != tuple(shape):
        raise ValueError(f"Unexpected shape for {path}: {image.shape} versus {shape}")
    if not np.array_equal(image.affine, np.eye(4)) or not np.allclose(image.header.get_zooms(), [1, 1, 1]):
        raise ValueError(f"Expected the identity-affine processed 1 mm grid: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/evaluations/graz_softcut01_best18_fp32_20261027_130316"))
    parser.add_argument("--comparison", type=Path, default=Path("artifacts/evaluations/nextqsm_comparison_20261027"))
    parser.add_argument("--iqsm", type=Path, default=Path("/home/mathias/Documents/PhD/wh_net/data/evaluation/volumes/iQSM2.nii.gz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/graz1_article_composites_best18_20261027"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--column-width", type=int, default=800)
    parser.add_argument("--gap", type=int, default=2)
    args = parser.parse_args()
    if min(args.dpi, args.column_width, args.gap) < 1:
        parser.error("DPI, column width and gap must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline/"graz1"
    metadata = json.loads((baseline/"metadata.json").read_text())
    if metadata["cutoff_mode"] != "soft" or metadata["threshold"] != .1:
        raise ValueError("Expected the original soft-cutoff epoch 18 evaluation")
    inputs = json.loads((args.comparison/"inputs.json").read_text())
    record = inputs["datasets"]["graz1"]
    baseline_provenance = json.loads((args.baseline/"provenance.json").read_text())
    assert baseline_provenance["checkpoint_sha256"] == inputs["checkpoint_sha256"]
    assert metadata["padded_shape"] == record["original_grid"] == [201, 201, 201]
    mask_value = load(baseline/"mask.nii.gz")
    assert np.isin(mask_value, [0, 1]).all()
    mask = mask_value>.5
    assert list(mask.shape) == metadata["resampled_cropped_shape"] == [138, 161, 118]
    # The existing comparison explicitly records where this ROI resides in its
    # 256-grid input, and how that input was padded from the 201-grid original.
    original_start = np.asarray(record["crop_start"])-record["padding_before"]
    centered_start = (np.asarray(metadata["padded_shape"])-mask.shape)//2
    assert np.array_equal(original_start, centered_start)
    original_crop = tuple(slice(int(a), int(a+n)) for a, n in zip(original_start, mask.shape, strict=True))
    common_crop = tuple(slice(a, b) for a, b in zip(record["crop_start"], record["crop_stop"], strict=True))
    assert np.array_equal(load(args.comparison/"graz1/mask.nii.gz")[common_crop], mask_value)
    assert np.array_equal(load(args.comparison/"graz1/total_field.nii.gz")[common_crop], load(baseline/"total_field.nii.gz"))
    assert_geometry(args.iqsm, record["original_grid"])
    paths = dict(
        iqsm=args.iqsm,
        nextqsm=args.comparison/"graz1/nextqsm_chi.nii.gz",
        soft_tkd=baseline/"tkd.nii.gz",
        dnqsm_final=baseline/"chi.nii.gz",
        learned_correction=baseline/"correction.nii.gz",
        mask=baseline/"mask.nii.gz",
    )
    assert_geometry(paths["nextqsm"], record["common_grid"])
    volumes = dict(
        iqsm=load(args.iqsm)[original_crop]*mask,
        nextqsm=load(paths["nextqsm"])[common_crop]*mask,
    )
    for key in ("soft_tkd", "dnqsm_final", "learned_correction"):
        assert_geometry(paths[key], mask.shape)
        volumes[key] = load(paths[key])*mask
    for key, value in volumes.items():
        assert value.shape == mask.shape and np.isfinite(value).all()
        assert value.dtype == np.float32
        nib.save(nib.Nifti1Image(value, np.eye(4)), args.output/f"{key}.nii.gz")
    nib.save(nib.Nifti1Image(mask_value, np.eye(4)), args.output/"mask.nii.gz")
    residual = volumes["dnqsm_final"]-(volumes["soft_tkd"]+volumes["learned_correction"])
    assert np.max(np.abs(residual)) == 0

    latex_style()
    columns, view_records = [], None
    # The extra 180-degree axial rotation was requested specifically for COSMOS.
    # Graz keeps the established Graz display orientation (np.rot90 only).
    for key, _ in COLUMNS:
        image, views = composite(volumes[key], mask, args.column_width, args.gap, .1, axial_rotation=0)
        if view_records is None:
            view_records = views
        assert views == view_records
        columns.append((key, image))
    image, positions = montage(columns, args.gap)
    products = dict(graz1_reconstructions=render(
        args, "graz1_reconstructions", image, positions,
        [(title, "") for _, title in COLUMNS],
        [(.22, .56, .1, r"Susceptibility $\chi$ (ppm)")],
    ))
    columns, titles = [], []
    for key, title, subtitle, limit in (
        ("soft_tkd", "Tikhonov regularized inverse", r"$\chi_{\mathrm{soft}}$", .1),
        ("learned_correction", "Soft-null correction", r"$\Delta\chi=\chi_{\mathrm{final}}-\chi_{\mathrm{soft}}$", .02),
        ("dnqsm_final", "Final DN-QSM", r"$\chi_{\mathrm{final}}$", .1),
    ):
        image, views = composite(volumes[key], mask, args.column_width, args.gap, limit, axial_rotation=0)
        assert views == view_records
        columns.append((key, image))
        titles.append((title, subtitle))
    image, positions = montage(columns, args.gap)
    products["graz1_soft_null_component"] = render(
        args, "graz1_soft_null_component", image, positions, titles,
        [(.07, .37, .1, r"Initial and final $\chi$ (ppm)"),
         (.57, .36, .02, r"Added component $\Delta\chi$ (ppm)")],
    )
    display_statistics = {}
    for key, value in volumes.items():
        roi = value[mask].astype(np.float64)
        limit = .02 if key == "learned_correction" else .1
        display_statistics[key] = dict(mean_ppm=float(roi.mean()), std_ppm=float(roi.std()),
                                       display_clipped_fraction=float(np.mean(np.abs(roi)>limit)))
    write_json(args.output/"figure_settings.json", dict(
        dpi=args.dpi, columns=[dict(key=k, title=t) for k, t in COLUMNS],
        colormap="gray", susceptibility_limits_ppm=[-.1, .1], correction_limits_ppm=[-.02, .02],
        slice_gap_pixels=args.gap, column_gap_pixels=args.gap, gap_color="black",
        column_width_pixels=args.column_width, views=view_records,
        axial_additional_rotation_degrees=0, typography="Actual LaTeX, Latin Modern",
        resize="Bilinear display-plane resizing only; nearest-neighbor masks; aspect ratios preserved",
        products=products, display_statistics=display_statistics,
    ))
    write_json(args.output/"provenance.json", dict(
        dataset="graz1", ground_truth_available=False, nrmse_available=False,
        mask_voxels=int(mask.sum()), aligned_crop_shape=list(mask.shape), physical_units="ppm",
        source_files={key: dict(path=str(path.resolve()), sha256=sha256(path)) for key, path in paths.items()},
        preprocessing=metadata, dnqsm=baseline_provenance,
        dnqsm_inference_grid=[201]*3, nextqsm_inference_grid=[256]*3,
        nextqsm=json.loads((args.comparison/"graz1/nextqsm_provenance.json").read_text()),
        iqsm_grid=[201]*3, iqsm_crop_start=original_start.tolist(), iqsm_crop_stop=(original_start+mask.shape).tolist(),
        iqsm_provenance="User-supplied iQSM2 reconstruction on the processed padded 1 mm Graz1 frame; original iQSM inference settings unavailable",
        transformations="Remove padding and apply the common brain mask; no registration, volumetric resampling, fitted scale or offset",
        correction="Actual soft-weighted correction returned by epoch 18 DN-QSM; Tikhonov regularized inverse plus correction equals final exactly",
    ))
    write_json(args.output/"verification.json", dict(
        original_and_comparison_input_masks_identical=True,
        original_and_comparison_input_fields_identical=True,
        original_dnqsm_protocol_used=True, no_new_model_inference=True,
        iqsm_padded_grid_matches_original_graz1_grid=True,
        crops_aligned=True, outputs_finite_float32=True,
        correction_decomposition_exact=True, correction_max_difference_ppm=float(np.max(np.abs(residual))),
        products=products,
    ))
    (args.output/"README.md").write_text(
        "# Graz1 article comparisons\n\n"
        "Reconstructions: graz1_reconstructions.png / .pdf. Learned correction: graz1_soft_null_component.png / .pdf. "
        "Both are exported at 600 dpi with actual LaTeX/Latin Modern typography. *_montage.png contains the raster matrix without headings or colorbars.\n\n"
        "Columns: Next-QSM (official NeXtQSM), iQSM (the supplied iQSM2.nii.gz), Tikhonov regularized inverse, Final DN-QSM. "
        "There is no ground-truth susceptibility for Graz1, so no ground-truth column, NRMSE or ground-truth error map is included.\n\n"
        "The original DN-QSM epoch 18 FP32 reconstructions are used on their original 201³ inference grid. "
        "NeXtQSM retains its native 256³ inference grid. iQSM2 matches the 201³ padded processed frame; "
        "its ROI starts at [31, 20, 41] and has shape [138, 161, 118]. Only cropping and a common brain mask are applied. "
        "The iQSM geometry is consistent with that frame; its original inference inputs/settings are unavailable.\n\n"
        "Each column stacks central axial, sagittal and coronal views, tightly cropped to the brain mask, resized to equal widths, "
        "with exactly 2 black pixels between slices and columns. Unlike the COSMOS-specific display rotation, Graz retains its established orientation "
        "without an extra 180-degree axial rotation. All methods have the same view selection and display treatment.\n\n"
        "All images use gray. Susceptibility limits are [−0.1, 0.1] ppm. The complementary soft-null correction has limits [−0.02, 0.02] ppm. "
        "This correction adds exactly to Tikhonov regularized inverse to produce the saved final DN-QSM reconstruction. It is a soft near-null correction, "
        "rather than a strict hard null-space component. Display resizing/clipping leaves original voxel values unchanged.\n\n"
        f"The supplied iQSM2 has a wider intensity distribution: {100*display_statistics['iqsm']['display_clipped_fraction']:.2f}% "
        "of its masked voxels exceed the shared ±0.1 ppm display range. Its values are displayed as supplied, without a fitted gain or offset. "
        "The common display scale makes this saturation visible; it is not introduced by modifying its reconstruction.\n"
    )
    shutil.copy2(__file__, args.output/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("create_cosmos_composite_figures.py"), args.output/"create_cosmos_composite_figures.py")
    print(json.dumps(dict(event="complete", output=str(args.output.resolve()), products=products)), flush=True)


if __name__ == "__main__":
    main()
