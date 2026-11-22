"""Render compact COSMOS method columns with actual LaTeX typography.

Only display arrays are resized. Reconstructions and whole-volume NRMSE are
read from the previously verified COSMOS article evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dnqsm-article-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


COLUMNS = (
    ("nextqsm", "Next-QSM"),
    ("iqsm", "iQSM"),
    ("soft_tkd", "Tikhonov regularized inverse"),
    ("dnqsm_final", "Final DN-QSM"),
    ("reference", "Ground truth"),
)
VIEWS = (("Axial", 2), ("Sagittal", 0), ("Coronal", 1))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path) -> np.ndarray:
    value = nib.load(path).get_fdata(dtype=np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError(f"Expected one finite 3D image: {path}")
    return value


def latex_style() -> None:
    # Use actual LaTeX/Latin Modern without depending on the optional type1ec
    # package. Generic serif/sans families avoid Matplotlib's type1ec preamble.
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["serif"],
        "font.sans-serif": ["sans-serif"],
        "font.monospace": ["Courier"],
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def cropped_planes(value: np.ndarray, mask: np.ndarray, *, axial_rotation: int = 180):
    if axial_rotation not in (0, 180):
        raise ValueError("Axial rotation must be 0 or 180 degrees")
    planes = []
    for name, axis in VIEWS:
        index = value.shape[axis]//2
        image = np.rot90(np.take(value, index, axis=axis))
        support = np.rot90(np.take(mask, index, axis=axis))
        rotation = axial_rotation if name == "Axial" else 0
        if rotation:
            image, support = np.rot90(image, 2), np.rot90(support, 2)
        positions = np.argwhere(support)
        if positions.size == 0:
            raise ValueError(f"Empty {name} slice")
        lower, upper = positions.min(0), positions.max(0)+1
        crop = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
        planes.append((image[crop], support[crop], dict(
            name=name, array_axis=axis, slice_index=index,
            additional_rotation_degrees=rotation,
            rotated_crop_start=lower.tolist(), rotated_crop_stop=upper.tolist(),
            cropped_shape=list(image[crop].shape),
        )))
    return planes


def composite(value: np.ndarray, mask: np.ndarray, width: int, gap: int, limit: float,
              *, axial_rotation: int = 180):
    """One RGBA image per method, with tight slice crops and exact pixel gaps."""
    cmap = matplotlib.colormaps["gray"]
    norm = Normalize(-limit, limit, clip=True)
    parts, records = [], []
    offset = 0
    for index, (image, support, record) in enumerate(cropped_planes(value, mask, axial_rotation=axial_rotation)):
        height = round(image.shape[0]*width/image.shape[1])
        factors = (height/image.shape[0], width/image.shape[1])
        resized = zoom(image, factors, order=1, mode="nearest", prefilter=False)
        resized_support = zoom(support.astype(np.uint8), factors, order=0,
                               mode="nearest", prefilter=False).astype(bool)
        assert resized.shape == resized_support.shape == (height, width)
        rgba = cmap(norm(resized), bytes=True)
        rgba[~resized_support] = [0, 0, 0, 255]
        if index:
            parts.append(np.broadcast_to(np.array([0, 0, 0, 255], dtype=np.uint8), (gap, width, 4)).copy())
            offset += gap
        record.update(resized_shape=[height, width], start_row=offset, stop_row=offset+height)
        records.append(record)
        parts.append(rgba)
        offset += height
    image = np.concatenate(parts, axis=0)
    for record in records[1:]:
        assert np.all(image[record["start_row"]-gap:record["start_row"]] == [0, 0, 0, 255])
    return image, records


def montage(columns, gap: int):
    images, records = [], []
    offset = 0
    for index, (key, image) in enumerate(columns):
        if index:
            images.append(np.broadcast_to(np.array([0, 0, 0, 255], dtype=np.uint8), (image.shape[0], gap, 4)).copy())
            offset += gap
        records.append(dict(method=key, start_column=offset, stop_column=offset+image.shape[1]))
        images.append(image)
        offset += image.shape[1]
    return np.concatenate(images, axis=1), records


def bar(fig, figure_width, figure_height, x, width, limit, label):
    # Positions in final output pixels, converted only for Matplotlib's labels.
    axis = fig.add_axes([x/figure_width, 220/figure_height, width/figure_width, 20/figure_height])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap="gray"),
                      cax=axis, orientation="horizontal")
    cb.set_ticks(np.linspace(-limit, limit, 5))
    cb.ax.tick_params(labelsize=8, length=1.7, width=.5, pad=1.5)
    cb.set_label(label, fontsize=9, labelpad=2)
    cb.outline.set_linewidth(.5)


def render(args, stem, mosaic, column_records, titles, bars):
    single_line = all(not subtitle for _, subtitle in titles)
    footer, header = 280, 140 if single_line else 210
    multiline_title = any(title == "Tikhonov regularized inverse" for title, _ in titles)
    if multiline_title:
        header = 220 if single_line else 330
    height, width = mosaic.shape[:2]
    total_height = height+footer+header
    # A small subpixel margin prevents floating-point truncation of integer
    # canvas dimensions when converting pixels to inches and back.
    fig = plt.figure(figsize=((width+.01)/args.dpi, (total_height+.01)/args.dpi), dpi=args.dpi)
    # One single image for all method columns. FigureImage draws at native
    # pixel coordinates, without subplot scaling or automatic layout margins.
    fig.figimage(mosaic, xo=0, yo=footer, origin="upper")
    title_artists = []
    for record, (title, subtitle) in zip(column_records, titles, strict=True):
        x = (record["start_column"]+record["stop_column"])/2/width
        title_center = .5 if single_line else .69
        display_title = "Tikhonov\nregularized inverse" if title == "Tikhonov regularized inverse" else title
        first = fig.text(x, (footer+height+title_center*header)/total_height, display_title,
                         ha="center", va="center", fontsize=9.5 if "\n" in display_title else 10.5)
        title_artists.append((first, record))
        if subtitle:
            second = fig.text(x, (footer+height+.25*header)/total_height, subtitle,
                              ha="center", va="center", fontsize=9)
            title_artists.append((second, record))
    for x_fraction, width_fraction, limit, label in bars:
        bar(fig, width, total_height, width*x_fraction, width*width_fraction, limit, label)
    fig.canvas.draw()
    for artist, record in title_artists:
        box = artist.get_window_extent(fig.canvas.get_renderer())
        available = record["stop_column"]-record["start_column"]-30
        if box.width > available:
            artist.set_fontsize(artist.get_fontsize()*available/box.width*.99)
    fig.canvas.draw()
    for artist, record in title_artists:
        box = artist.get_window_extent(fig.canvas.get_renderer())
        assert box.x0 >= record["start_column"] and box.x1 <= record["stop_column"]
        assert box.y0 >= footer+height and box.y1 <= total_height, "Heading extends outside its header"
    png_path, pdf_path = args.output/f"{stem}.png", args.output/f"{stem}.pdf"
    fig.savefig(png_path, dpi=args.dpi, facecolor="white", pad_inches=0)
    fig.savefig(pdf_path, dpi=args.dpi, facecolor="white", pad_inches=0)
    plt.close(fig)
    # Verify the exported pixels, not only the pre-rendered montage array.
    with Image.open(png_path) as output:
        assert output.size == (width, total_height), (output.size, width, total_height)
        pixels = np.asarray(output.convert("RGBA"))
        assert np.array_equal(pixels[header:header+height], mosaic), "Montage was rescaled or shifted"
        assert all(abs(value-args.dpi)<.1 for value in output.info["dpi"])
        preview = output.convert("RGB")
        preview.thumbnail((1800, 1800), resample=Image.Resampling.LANCZOS)
        preview.save(args.output/f"{stem}_preview.png")
    Image.fromarray(mosaic).save(args.output/f"{stem}_montage.png", dpi=(args.dpi, args.dpi))
    return dict(pixels=[width, total_height], montage_pixels=[width, height],
                montage_bounds=dict(left=0, top=header, right=width, bottom=header+height),
                columns=column_records, montage_matches_final_png_exactly=True,
                title_text_fits_within_each_column=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/figures/cosmos_article_best18_20261027"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/cosmos_article_composites_best18_20261027"))
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--column-width", type=int, default=800)
    parser.add_argument("--gap", type=int, default=2)
    args = parser.parse_args()
    if args.dpi < 1 or args.column_width < 1 or args.gap < 1:
        parser.error("dpi, column width and gap must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    latex_style()
    reference, mask = load(args.source/"reference.nii.gz"), load(args.source/"mask.nii.gz")>.5
    volumes = {key: load(args.source/f"{key}.nii.gz") for key, _ in COLUMNS}
    volumes["learned_correction"] = load(args.source/"learned_correction.nii.gz")
    metrics = json.loads((args.source/"metrics.json").read_text())
    for key, _ in COLUMNS:
        assert volumes[key].shape == reference.shape
        target = reference[mask].astype(np.float64)
        error = volumes[key][mask].astype(np.float64)-target
        value = float(100*np.linalg.norm(error)/np.linalg.norm(target))
        if key != "reference":
            assert abs(value-metrics[key]["nrmse_percent"]) < 1e-10
    assert np.array_equal(volumes["soft_tkd"]+volumes["learned_correction"], volumes["dnqsm_final"])
    titles = [(label, rf"$\mathrm{{NRMSE}} = {0 if key=='reference' else metrics[key]['nrmse_percent']:.2f}\%$")
              for key, label in COLUMNS]
    products, view_records = {}, None
    for stem, difference, limit, label in (
        ("cosmos_reconstructions", False, .1, r"Susceptibility $\chi$ (ppm)"),
        ("cosmos_differences", True, .02, r"Difference $\hat{\chi}-\chi_{\mathrm{COSMOS}}$ (ppm)"),
    ):
        columns = []
        for key, _ in COLUMNS:
            value = volumes[key]-reference if difference else volumes[key]
            image, records = composite(value, mask, args.column_width, args.gap, limit)
            if view_records is None:
                view_records = records
            assert records == view_records
            columns.append((key, image))
        image, positions = montage(columns, args.gap)
        products[stem] = render(args, stem, image, positions, titles, [(0.24, .52, limit, label)])
        print(json.dumps(dict(product=stem, status="saved", **products[stem])), flush=True)
    columns = []
    correction_titles = []
    for key, label, limit in (("soft_tkd", "Tikhonov regularized inverse", .1),
                              ("learned_correction", "Soft-null correction", .02),
                              ("dnqsm_final", "Final DN-QSM", .1)):
        image, records = composite(volumes[key], mask, args.column_width, args.gap, limit)
        assert records == view_records
        columns.append((key, image))
        subtitle = (rf"$\mathrm{{NRMSE}} = {metrics[key]['nrmse_percent']:.2f}\%$" if key != "learned_correction"
                    else r"$\Delta\chi = \chi_{\mathrm{final}}-\chi_{\mathrm{soft}}$")
        correction_titles.append((label, subtitle))
    image, positions = montage(columns, args.gap)
    products["cosmos_soft_null_component"] = render(
        args, "cosmos_soft_null_component", image, positions, correction_titles,
        [(.07, .37, .1, r"Initial and final $\chi$ (ppm)"),
         (.57, .36, .02, r"Added component $\Delta\chi$ (ppm)")],
    )
    settings = dict(
        source=str(args.source.resolve()), dpi=args.dpi,
        columns=[dict(key=key, title=label) for key, label in COLUMNS],
        layout="One single raster montage; each method column stacks axial, sagittal, coronal views",
        column_width_pixels=args.column_width, slice_gap_pixels=args.gap, column_gap_pixels=args.gap,
        gap_color="black", outside_mask_color="black", colormap="gray",
        susceptibility_limits_ppm=[-.1, .1], difference_limits_ppm=[-.02, .02],
        correction_limits_ppm=[-.02, .02], views=view_records,
        axial_additional_rotation_degrees=180,
        resize="Bilinear display-plane resizing to equal widths, preserving aspect ratio; nearest-neighbor masks",
        volumes_resampled=False, nrmse_recomputed_from_original_3d_volumes=True,
        typography="Actual LaTeX with Latin Modern and amsmath", products=products,
        source_sha256={f"{key}.nii.gz": sha256(args.source/f"{key}.nii.gz") for key in volumes},
    )
    write_json(args.output/"figure_settings.json", settings)
    write_json(args.output/"metrics.json", metrics)
    shutil.copy2(args.source/"provenance.json", args.output/"provenance.json")
    shutil.copy2(__file__, args.output/Path(__file__).name)
    (args.output/"README.md").write_text(
        "# Compact COSMOS article figures\n\n"
        "Reconstructions: cosmos_reconstructions.png / .pdf. Differences: cosmos_differences.png / .pdf. "
        "Complementary correction: cosmos_soft_null_component.png / .pdf. *_montage.png files contain just the image matrix, "
        "without typography or colorbars. Previews are resized for viewing.\n\n"
        "Each method is represented by one continuous column containing the three central orthogonal views: axial, sagittal, coronal. "
        "The axial view is additionally rotated 180 degrees in all methods, differences, and correction panels. "
        "Slice images are tightly cropped using the reference brain mask, resized to equal widths with preserved aspect ratio, "
        "and separated by exactly 2 black pixels in the publication PNG and raw montage. Columns also have 2-pixel black gaps. "
        "The original 3D images and metrics are unchanged.\n\n"
        "Requested column order: Next-QSM, iQSM, Tikhonov regularized inverse, Final DN-QSM, Ground truth. The Next-QSM column uses the previously evaluated official NeXtQSM model. "
        "All titles include whole-brain 3D NRMSE, including 0% for the reference. Text uses actual LaTeX with Latin Modern. "
        "All images use gray, susceptibility limits [−0.1, 0.1] ppm, and signed difference limits [−0.02, 0.02] ppm. "
        "Differences are reconstruction minus ground truth; the ground-truth difference is identically zero.\n\n"
        "The complementary columns show Tikhonov regularized inverse, the actual learned soft-null correction, and final DN-QSM. "
        "The correction has limits [−0.02, 0.02] ppm and adds exactly to Tikhonov regularized inverse to give the final reconstruction. "
        "The soft frequency weighting emphasizes the dipole near-null region; it is not a strict hard null-space projection. "
        "See provenance.json for inference grids, checkpoint, source reconstruction hashes, and the supplied iQSM provenance limitation.\n"
    )
    print(json.dumps(dict(event="complete", output=str(args.output.resolve()))), flush=True)


if __name__ == "__main__":
    main()
