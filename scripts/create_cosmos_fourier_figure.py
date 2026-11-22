"""Compare COSMOS or in vivo reconstruction spectra and their Fourier correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.fft import fftn, fftshift

from create_cosmos_composite_figures import (
    latex_style, load, montage, sha256, write_json,
)
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


COSMOS_COLUMNS = (
    ("nextqsm", "Next-QSM"),
    ("iqsm", "iQSM"),
    ("soft_tkd", "Tikhonov\nregularized inverse"),
    ("dnqsm_final", "Final DN-QSM"),
    ("reference", "Ground truth"),
    ("correction", "Added correction"),
)
IN_VIVO_COLUMNS = tuple(column for column in COSMOS_COLUMNS if column[0] != "reference")
PLANES = ((2, r"$k_z=0$", r"$(k_x,k_y)$"),
          (1, r"$k_y=0$", r"$(k_x,k_z)$"),
          (0, r"$k_x=0$", r"$(k_y,k_z)$"))


def main(default_case: str = "cosmos") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("cosmos", "in_vivo"), default=default_case)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cosmos = args.case == "cosmos"
    columns_spec = COSMOS_COLUMNS if cosmos else IN_VIVO_COLUMNS
    args.source = args.source or Path("artifacts/figures/cosmos_article_best18_20261027" if cosmos else
                                     "artifacts/figures/graz1_article_composites_best18_20261027")
    args.output = args.output or Path("artifacts/figures/cosmos_fourier_best18_20261028" if cosmos else
                                     "artifacts/figures/in_vivo_fourier_best18_20261028")
    prefix = "cosmos" if cosmos else "in_vivo"
    display_name = "COSMOS" if cosmos else "In vivo"
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    mask = load(source/"mask.nii.gz") > .5
    source_shape = mask.shape
    source_provenance = json.loads((source/"provenance.json").read_text())
    metrics = json.loads((source/"metrics.json").read_text()) if cosmos else None
    if cosmos:
        assert mask.shape == (160, 160, 160)
        padding = [(0, 0)]*3
    else:
        # Restore the original DN-QSM field of view with zeros outside the
        # common article ROI. This is a Fourier-analysis operation only.
        native = np.array(source_provenance["dnqsm_inference_grid"])
        before = np.array(source_provenance["iqsm_crop_start"])
        assert np.array_equal(native, [201]*3)
        assert np.array_equal(before, (native-np.array(mask.shape))//2)
        padding = [(int(a), int(n-a-s)) for a, n, s in zip(before, native, mask.shape, strict=True)]
        iqsm_source = source_provenance["source_files"]["iqsm"]
        assert sha256(Path(iqsm_source["path"])) == iqsm_source["sha256"], "Regenerate article volumes: supplied iQSM changed"
    volumes = {key: load(source/f"{key}.nii.gz").astype(np.float64)
               for key, _ in columns_spec if key != "correction"}
    for value in volumes.values():
        assert value.shape == mask.shape
        assert np.all(value[~mask] == 0), "Expected the common masked article grid"
    volumes["correction"] = volumes["dnqsm_final"]-volumes["soft_tkd"]
    saved_correction = load(source/"learned_correction.nii.gz").astype(np.float64)
    correction_roundoff = float(np.max(np.abs(volumes["correction"]-saved_correction)))
    correction_tolerance = max(2e-8, 4*np.finfo(np.float32).eps*float(np.abs(volumes["dnqsm_final"]).max()))
    assert correction_roundoff < correction_tolerance
    if not cosmos:
        volumes = {key: np.pad(value, padding) for key, value in volumes.items()}
        mask = np.pad(mask, padding)
    assert len(set(mask.shape)) == 1
    grid_size = mask.shape[0]
    spectra = {key: fftn(value, norm="ortho") for key, value in volumes.items()}
    linearity_error = float(np.linalg.norm(
        spectra["dnqsm_final"]-spectra["soft_tkd"]-spectra["correction"]
    )/np.linalg.norm(spectra["correction"]))
    assert linearity_error < 1e-12
    amplitude_anchor = "reference" if cosmos else "soft_tkd"
    reference_peak = float(np.abs(spectra[amplitude_anchor]).max())
    amplitude_symbol = r"A_{\mathrm{ref}}" if cosmos else r"A_0"
    dB_limits = [-80., 10.]
    norm = Normalize(*dB_limits, clip=True)
    cmap = plt.get_cmap("gray")
    dpi, bin_scale, gap = 600, 5 if cosmos else 4, 2
    width = grid_size*bin_scale
    spectral_metrics, stored_planes, columns = {}, {}, []
    for key, _ in columns_spec:
        value, frequency = volumes[key], spectra[key]
        real_energy = float(np.sum(np.abs(value)**2))
        spectral_energy = float(np.sum(np.abs(frequency)**2))
        assert abs(spectral_energy-real_energy)/real_energy < 1e-12
        record = dict(spatial_energy=real_energy, spectral_energy=spectral_energy)
        if cosmos and key != "correction":
            nrmse = float(100*np.linalg.norm(frequency-spectra["reference"])
                          /np.linalg.norm(spectra["reference"]))
            if key != "reference":
                assert abs(nrmse-metrics[key]["nrmse_percent"]) < 1e-9
            record["nrmse_percent"] = nrmse
        shifted = fftshift(frequency)
        db = 20*np.log10(np.maximum(np.abs(shifted)/reference_peak, 1e-12))
        pieces = []
        for index, (axis, _, _) in enumerate(PLANES):
            # Transpose so horizontal/vertical axes match the named k pair.
            # Flip rows for a conventional upward positive vertical frequency.
            plane = np.take(db, db.shape[axis]//2, axis=axis).T[::-1]
            complex_plane = np.take(shifted, shifted.shape[axis]//2, axis=axis).T[::-1]
            stored_planes[f"{key}_plane_{index}"] = complex_plane
            pixels = cmap(norm(plane), bytes=True)
            # An integer number of display pixels per Fourier bin; no
            # interpolation in k-space.
            pixels = np.repeat(np.repeat(pixels, bin_scale, axis=0), bin_scale, axis=1)
            assert pixels.shape == (width, width, 4)
            if index:
                pieces.append(np.full((gap, width, 4), [0, 0, 0, 255], dtype=np.uint8))
            pieces.append(pixels)
        spectral_metrics[key] = record
        columns.append((key, np.concatenate(pieces)))
    mosaic, records = montage(columns, gap)
    for record in records[1:]:
        assert np.all(mosaic[:, record["start_column"]-gap:record["start_column"]] == [0, 0, 0, 255])
    for row in [width, 2*width+gap]:
        assert np.all(mosaic[row:row+gap] == [0, 0, 0, 255])

    # Quantify observed correction energy on the common comparison grid.
    # The crop and spatial mask mean this is not the native network grid.
    frequency = np.fft.fftfreq(grid_size, d=1.)
    kx2, ky2, kz2 = frequency[:, None, None]**2, frequency[None, :, None]**2, frequency[None, None, :]**2
    radius2 = kx2+ky2+kz2
    dipole = np.full(radius2.shape, 1/3., dtype=np.float64)
    np.subtract(dipole, np.divide(kz2, radius2, out=np.zeros_like(radius2), where=radius2>0), out=dipole)
    dipole[0, 0, 0] = 0
    correction_power = np.abs(spectra["correction"])**2
    spectral_metrics["correction"]["energy_fraction_at_abs_D_le_0_1"] = float(
        correction_power[np.abs(dipole)<=.1].sum()/correction_power.sum())
    stored_planes["frequency_centers_cycles_per_mm"] = np.fft.fftshift(frequency)
    np.savez_compressed(args.output/f"{prefix}_fourier_planes.npz", **stored_planes)

    latex_style()
    left, header, footer = 280, 430, 480
    height, body_width = mosaic.shape[:2]
    figure_width, figure_height = left+body_width, header+height+footer
    fig = plt.figure(figsize=((figure_width+.01)/dpi, (figure_height+.01)/dpi), dpi=dpi)
    fig.figimage(mosaic, xo=left, yo=footer, origin="upper")
    artists = []
    fig.text(.5, 1-72/figure_height, r"\textbf{"+display_name+r" Fourier-space comparison}",
             fontsize=14, ha="center", va="center", color="#263343")
    for record, (key, title) in zip(records, columns_spec, strict=True):
        center = (left+(record["start_column"]+record["stop_column"])/2)/figure_width
        artist = fig.text(center, 1-218/figure_height, title, ha="center", va="center",
                          fontsize=9.5 if "\n" in title else 10.5)
        artists.append((artist, record))
        subtitle = (r"$|\mathcal F(\Delta\chi)|$" if key == "correction" else
                    rf"$\mathrm{{NRMSE}}={spectral_metrics[key]['nrmse_percent']:.2f}\%$" if cosmos else
                    r"$|\mathcal F(\chi)|$")
        artist = fig.text(center, 1-365/figure_height, subtitle,
                          ha="center", va="center", fontsize=9)
        artists.append((artist, record))
    for index, (_, title, pair) in enumerate(PLANES):
        center = 1-(header+index*(width+gap)+width/2)/figure_height
        fig.text(left/2/figure_width, center+42/figure_height, title,
                 ha="center", va="center", fontsize=9)
        fig.text(left/2/figure_width, center-42/figure_height, pair,
                 ha="center", va="center", fontsize=8)
    axis = fig.add_axes([.245, 360/figure_height, .55, 23/figure_height])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(*dB_limits), cmap="gray"),
                            cax=axis, orientation="horizontal")
    colorbar.set_ticks([-80, -60, -40, -20, 0, 10])
    colorbar.ax.tick_params(labelsize=8, length=2, width=.5, pad=2)
    colorbar.set_label(r"Fourier magnitude $20\log_{10}(|\mathcal F(\chi)|/"+amplitude_symbol+r")$ (dB)",
                       fontsize=10, labelpad=4)
    colorbar.outline.set_linewidth(.5)
    anchor_label = r"\chi_{\mathrm{COSMOS}}" if cosmos else r"\chi_0"
    scale_note = fig.text(.52, 65/figure_height,
             r"Shared scale: $"+amplitude_symbol+r"=\max_{\mathbf k}|\mathcal F("+anchor_label+r")|$"
             r"\qquad $\Delta\chi=\chi_{\mathrm{DN-QSM}}-\chi_0$"
             +rf"\qquad $k_j\in[{frequency.min():.5g},{frequency.max():.5g}]$ cycles/mm",
             ha="center", va="center", fontsize=8.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, record in artists:
        box = artist.get_window_extent(renderer)
        available = record["stop_column"]-record["start_column"]-20
        if box.width > available:
            artist.set_fontsize(artist.get_fontsize()*available/box.width*.99)
    fig.canvas.draw()
    for artist, record in artists:
        box = artist.get_window_extent(fig.canvas.get_renderer())
        assert left+record["start_column"] <= box.x0 < box.x1 <= left+record["stop_column"]
        assert footer+height <= box.y0 < box.y1 <= figure_height
    renderer = fig.canvas.get_renderer()
    assert scale_note.get_window_extent(renderer).y1+15 < colorbar.ax.xaxis.label.get_window_extent(renderer).y0
    assert scale_note.get_window_extent(renderer).y0 >= 0
    assert 0 <= scale_note.get_window_extent(renderer).x0 < scale_note.get_window_extent(renderer).x1 <= figure_width
    stem = args.output/f"{prefix}_fourier_reconstructions"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)
    Image.fromarray(mosaic).save(args.output/f"{prefix}_fourier_montage.png", dpi=(dpi, dpi))
    with Image.open(stem.with_suffix(".png")) as image:
        assert image.size == (figure_width, figure_height)
        assert np.array_equal(np.asarray(image.convert("RGBA"))[header:header+height, left:], mosaic)
        preview = image.convert("RGB")
        preview.thumbnail((2100, 1500), Image.Resampling.LANCZOS)
        preview.save(args.output/f"{prefix}_fourier_reconstructions_preview.png")

    caption = (
        "Central Fourier planes of COSMOS susceptibility reconstructions. All columns use the same centered 3D FFT "
        "and grayscale log-magnitude range (-80 to 10 dB), referenced to the peak ground-truth spectrum. "
        "Added correction shows |F(final DN-QSM - initial QSM)|; complex corrections can reinforce or cancel existing components. "
        "Spectra use the common masked 160-cubed grid, where cropping and masking spread spectral content. "
        "NRMSE denotes whole-volume reconstruction error."
    )
    if not cosmos:
        caption = (
            "Central Fourier planes of in vivo brain hemorrhage susceptibility reconstructions. All columns share "
            "a grayscale log-magnitude scale (-80 to 10 dB), relative to the peak Tikhonov spectrum. Added correction "
            "shows |F(final DN-QSM - initial QSM)|. Common masked ROI volumes are zero-padded to the original "
            "201-cubed field of view. Cropping and masking spread spectral content. Reference susceptibility was "
            "unavailable; no accuracy metric is shown."
        )
    assert len(caption) <= 500
    (args.output/"caption.txt").write_text(caption+"\n")
    write_json(args.output/"spectral_metrics.json", spectral_metrics)
    write_json(args.output/"provenance.json", dict(
        dataset=display_name, ground_truth_available=cosmos, nrmse_available=cosmos,
        source=str(source), columns=[dict(key=k, title=t.replace("\n", " ")) for k, t in columns_spec],
        full_volume_fft=True, fft_normalization="ortho", fft_shift=True,
        source_grid=list(source_shape), common_grid=list(mask.shape), voxel_size_mm=[1., 1., 1.],
        spatial_support="Identical zero-outside-mask support in all existing article volumes",
        spatial_window=None, mean_removal=False, zero_padding=not cosmos, padding=padding,
        amplitude_anchor=amplitude_anchor, amplitude_anchor_peak=reference_peak, common_db_limits=dB_limits,
        phase_displayed=False, method_specific_scaling=False,
        planes=[dict(fixed_array_axis=a, fixed_frequency=0, horizontal_vertical_pair=p) for a, _, p in PLANES],
        frequency_range_cycles_per_mm=[float(frequency.min()), float(frequency.max())],
        correction="Complex Fourier difference: F(final DN-QSM) - F(Tikhonov initialization); not a difference of magnitudes",
        correction_source_roundoff_max_ppm=correction_roundoff,
        interpretation="Magnitude indicates the location and amplitude of the added complex correction, not the signed change in spectral power. Masking and cropping mix frequencies; original exterior network predictions have been masked out for comparison.",
        actual_native_model_projection="Smooth weighting tau^2/(D^2+tau^2), tau=0.1; not a strict null-space projector",
        no_new_model_inference=True, dpi=dpi, colormap="gray", gap_pixels=gap,
        display_interpolation=f"Nearest-neighbor: exactly {bin_scale} pixels per Fourier bin",
        source_files=[dict(path=str(source/f"{name}.nii.gz"), sha256=sha256(source/f"{name}.nii.gz"))
                      for name in [k for k, _ in columns_spec if k != "correction"]+["mask", "learned_correction"]],
    ))
    write_json(args.output/"verification.json", dict(
        complex_fft_additivity_relative_error=linearity_error,
        parseval_energy_verified=True, fourier_and_spatial_nrmse_match=True if cosmos else None,
        no_ground_truth_or_accuracy_metrics_invented=True,
        reconstruction_data_not_modified=True, output_montage_matches_png=True,
        shared_gray_db_scale=True, two_pixel_gaps_are_black=True,
        title_text_stays_inside_columns=True, caption_characters=len(caption),
    ))
    readme = (
        "# COSMOS Fourier-space comparison\n\n"
        "The publication PNG and PDF show Next-QSM, iQSM, Tikhonov regularized inverse, final DN-QSM, ground truth, "
        "and the added correction. The three rows are central planes kz=0 (horizontal kx, vertical ky), "
        "ky=0 (horizontal kx, vertical kz), and kx=0 (horizontal ky, vertical kz). "
        "Positive horizontal frequency points right; positive vertical frequency points up. "
        "All axes cover the same 160 Fourier bins at 1-mm voxel spacing: -0.5 to 0.49375 cycles/mm. "
        "These are planes of a full 3D FFT, not 2D FFTs of displayed anatomical slices.\n\n"
        "All spectra share a log-magnitude range of -80 to +10 dB, relative to the maximum magnitude in the ground-truth "
        "3D spectrum. There is no per-method rescaling, mean subtraction, Hann window, or additional spatial padding. "
        "The 2-pixel gaps are black; text is rendered with LaTeX/Latin Modern.\n\n"
        "The correction is final minus initial susceptibility, computed before the FFT. Complex FFT additivity and "
        "Parseval energy/NRMSE equivalence are verified. Its magnitude is not the difference between spectral magnitudes; "
        "the correction can either reinforce or cancel existing components depending on phase. All methods use the existing "
        "common masked 160-cubed article volumes. Cropping and masking spread energy across frequencies, so the displayed "
        "correction is not confined to a hard null-space region. The network itself uses a soft regularized weighting "
        "on its original 184-cubed inference grid.\n\n"
        "The NPZ contains complex central Fourier planes and frequency coordinates. spectral_metrics.json reports "
        "whole-grid spectral energy and reconstruction NRMSE, plus correction energy in |D|<=0.1 on this comparison grid. "
        "caption.txt provides a caption below 500 characters. No model inference or training was performed.\n"
    )
    if not cosmos:
        readme = (
            "# In vivo Fourier-space comparison\n\n"
            "Columns: Next-QSM, iQSM, Tikhonov regularized inverse, final DN-QSM, added correction. "
            "Rows: kz=0 (horizontal kx, vertical ky), ky=0 (horizontal kx, vertical kz), and kx=0 "
            "(horizontal ky, vertical kz). Positive frequency points right and up. These are planes of "
            "the full 3D FFT, rather than 2D FFTs of anatomical slices.\n\n"
            "Existing common masked 138 by 161 by 118 article volumes are restored to the original 201-cubed "
            "DN-QSM field of view by zero-padding before the FFT. All methods use identical support and "
            "padding (before: 31,20,41; after: 32,20,42). This changes spectral sampling without adding image "
            "information. The processed voxel size is 1 mm isotropic; it differs from the native acquired "
            "approximately 0.8 by 0.8 by 2 mm resolution.\n\n"
            "All columns share -80 to +10 dB, relative to the peak of the initial Tikhonov 3D spectrum. "
            "This is a shared display anchor, not reference susceptibility. There is no ground-truth column "
            "or NRMSE. There is no method-specific rescaling, mean removal, or tapering window. Text uses "
            "LaTeX/Latin Modern, grayscale displays, and 2-pixel black gaps. Each Fourier bin occupies exactly "
            "4 by 4 display pixels.\n\n"
            "The correction is final minus initial susceptibility before the FFT. Complex additivity and "
            "Parseval energy are verified. Its magnitude shows the complex correction's location and amplitude; "
            "phase determines whether it reinforces or cancels existing components. Cropping and masking "
            "spread spectral content. Although the common grid matches the network's field of view, its "
            "original exterior predictions are masked out; it is not the unmasked native projection.\n\n"
            "The supplied iQSM2 hash is verified against article provenance. Complex central planes and "
            "frequency coordinates are saved in the NPZ. Caption and provenance files are included. "
            "No training or inference was performed.\n"
        )
    (args.output/"README.md").write_text(readme)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    files = sorted(p for p in args.output.iterdir() if p.is_file() and p.name != "manifest.json")
    write_json(args.output/"manifest.json", [dict(path=p.name, bytes=p.stat().st_size, sha256=sha256(p)) for p in files])
    print(json.dumps(dict(output=str(args.output.resolve()), png_pixels=[figure_width, figure_height],
                          linearity_error=linearity_error,
                          correction_energy_fraction_abs_D_le_0_1=spectral_metrics["correction"]["energy_fraction_at_abs_D_le_0_1"])), flush=True)


if __name__ == "__main__":
    main()
