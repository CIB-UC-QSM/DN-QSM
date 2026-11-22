"""Build a compact 3:2 LaTeX/TikZ model preview with existing MRI thumbnails."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from create_cosmos_composite_figures import cropped_planes, load


ROOT = Path(__file__).resolve().parents[1]


def build_preview(output: Path, source: Path) -> dict:
    output = output.resolve()
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets = output/"assets"
    assets.mkdir(exist_ok=True)
    comparison = ROOT/"artifacts/evaluations/nextqsm_comparison_20261027/cosmos"
    mask = load(source/"mask.nii.gz") > .5
    padded_mask = load(comparison/"mask.nii.gz") > .5
    padded_total = load(comparison/"total_field.nii.gz")
    before = (np.asarray(padded_mask.shape)-mask.shape)//2
    crop = tuple(slice(int(a), int(a+n)) for a, n in zip(before, mask.shape, strict=True))
    assert np.array_equal(padded_mask[crop], mask)
    total = padded_total[crop]
    final = load(source/"dnqsm_final.nii.gz")
    limits = dict(total_field=np.percentile(total[mask], [1, 99]).tolist(),
                  final_qsm=[-.1, .1])
    views = {}
    for name, volume in (("total_field", total), ("final_qsm", final)):
        plane, support, record = cropped_planes(volume, mask)[0]
        low, high = limits[name]
        pixels = np.rint(np.clip((plane-low)/(high-low), 0, 1)*255).astype(np.uint8)
        pixels[~support] = 0
        Image.fromarray(pixels).save(assets/f"{name}.png")
        views[name] = record
    tex = output/"dnqsm_model_preview.tex"
    shutil.copy2(ROOT/"docs/figures/dnqsm_model_preview.tex", tex)
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                            cwd=output, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:])
    assert "Overfull" not in result.stdout, result.stdout
    pdf = output/"dnqsm_model_preview.pdf"
    subprocess.run(["pdftoppm", "-png", "-scale-to-x", "1200", "-scale-to-y", "800",
                    "-singlefile", str(pdf), str(output/"dnqsm_model_preview")], check=True)
    png = output/"dnqsm_model_preview.png"
    with Image.open(png) as image:
        assert image.size == (1200, 800)
        image.save(png, optimize=True)
    assert png.stat().st_size < 1_000_000
    subprocess.run(["pdftoppm", "-png", "-scale-to-x", "600", "-scale-to-y", "400",
                    "-singlefile", str(pdf), str(output/"dnqsm_model_preview_small")], check=True)
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(output/"dnqsm_model_preview.svg")], check=True)
    page_info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    size = next(line for line in page_info.splitlines() if line.startswith("Page size:"))
    dimensions = size.split(":", 1)[1].split()
    width, height = float(dimensions[0]), float(dimensions[2])
    assert abs(width/height-1.5) < 1e-5
    provenance = dict(
        purpose="Uncaptioned model preview for program display",
        aspect_ratio="3:2", pixels=[1200, 800], bytes=png.stat().st_size,
        typography="LaTeX/TikZ with Latin Modern", page_mm=[150, 100],
        model="Laplacian field estimation, Tikhonov regularized inverse, residual network, regularized null projection, additive correction",
        thumbnails="Same central axial COSMOS slice, with the established 180-degree display rotation",
        thumbnail_limits_ppm=limits, input_window="1st to 99th percentile inside the brain mask; display only",
        output_window="[-0.1, 0.1] ppm", views=views,
        no_new_model_inference=True, external_caption=None,
        source_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in [source/"mask.nii.gz", source/"dnqsm_final.nii.gz",
                                comparison/"total_field.nii.gz", comparison/"mask.nii.gz"]],
    )
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (output/"README.md").write_text(
        "# DN-QSM model preview\n\n"
        "A compact 3:2 LaTeX/TikZ graphic for program display. PNG: 1200 by 800 pixels, below 1 MB. "
        "PDF and SVG preserve vector text and diagram elements; MRI thumbnails are embedded rasters. "
        "The editable TeX and assets are included.\n\n"
        "The main path combines Laplacian-based local-field estimation with Tikhonov inversion. "
        "The residual branch applies regularized null projection before adding its correction to the initial QSM. "
        "MRI thumbnails illustrate the total-field input and final susceptibility on the same COSMOS axial slice. "
        "Input and output use separate display windows because they represent different quantities. "
        "The complete architecture figure remains in the abstract. No external caption is required.\n"
    )
    shutil.copy2(__file__, output/Path(__file__).name)
    (output/"verification.json").write_text(json.dumps(dict(
        latex_compiles_without_errors_or_overfull_boxes=True,
        aspect_ratio="3:2", png_pixels=[1200, 800], small_png_pixels=[600, 400],
        png_below_1_mb=True, external_caption=None, pdf_points=[width, height],
    ), indent=2)+"\n")
    files = sorted(p for p in output.rglob("*") if p.is_file() and
                   p.suffix not in {".aux", ".log", ".out"} and p.name != "manifest.json")
    (output/"manifest.json").write_text(json.dumps([
        dict(path=str(p.relative_to(output)), bytes=p.stat().st_size,
             sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files
    ], indent=2)+"\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/figures/dnqsm_model_preview_20261028")
    parser.add_argument("--source", type=Path, default=ROOT/"artifacts/figures/cosmos_article_best18_20261027")
    args = parser.parse_args()
    result = build_preview(args.output, args.source)
    print(json.dumps(dict(output=str(args.output.resolve()), pixels=result["pixels"],
                          bytes=result["bytes"], aspect_ratio=result["aspect_ratio"])), flush=True)


if __name__ == "__main__":
    main()
