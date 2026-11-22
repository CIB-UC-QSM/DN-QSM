"""Compare masked FP32 local-field exports in ppm and standardized units."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from dnqsm.logging import _pyplot, write_json

LABELS = {
    "cosmos_reference": "COSMOS clean reference",
    "cosmos_estimated": "COSMOS estimated",
    "graz1_estimated": "Graz 1 estimated",
    "graz2_estimated": "Graz 2 estimated",
}
COLORS = ("#252525", "#16825d", "#2866ae", "#dc6b28")
QUANTILES = dict(
    p001=0.001,
    p01=0.01,
    p05=0.05,
    p25=0.25,
    median=0.5,
    p75=0.75,
    p95=0.95,
    p99=0.99,
    p999=0.999,
)


def describe(values):
    mean = float(values.mean())
    centered = values - mean
    variance = float(np.mean(centered**2))
    std = np.sqrt(variance)
    quantiles = dict(
        zip(
            QUANTILES,
            np.quantile(values, list(QUANTILES.values())).tolist(),
            strict=True,
        )
    )
    return dict(
        voxels=len(values),
        mean_ppm=mean,
        std_ppm=float(std),
        rms_ppm=float(np.sqrt(np.mean(values**2))),
        minimum_ppm=float(values.min()),
        maximum_ppm=float(values.max()),
        quantiles_ppm=quantiles,
        skewness=float(np.mean(centered**3) / std**3),
        excess_kurtosis=float(np.mean(centered**4) / variance**2 - 3),
        negative_fraction=float(np.mean(values < 0)),
        abs_gt_001_fraction=float(np.mean(np.abs(values) > 0.01)),
        abs_gt_002_fraction=float(np.mean(np.abs(values) > 0.02)),
        abs_gt_005_fraction=float(np.mean(np.abs(values) > 0.05)),
        abs_gt_010_fraction=float(np.mean(np.abs(values) > 0.1)),
    )


def transformed(values, mode):
    return values if mode == "physical" else (values - values.mean()) / values.std()


def probability(values, edges):
    counts = np.histogram(values, edges)[0]
    # Explicit overflow bins retain all tail mass for the JS distances.
    counts = np.r_[
        np.count_nonzero(values < edges[0]),
        counts,
        np.count_nonzero(values > edges[-1]),
    ]
    assert counts.sum() == len(values)
    result = counts / len(values)
    np.testing.assert_allclose(result.sum(), 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-evaluation", type=Path, required=True)
    parser.add_argument("--graz-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosmos_metadata = json.loads((args.cosmos_evaluation / "metadata.json").read_text())
    graz_config = json.loads((args.graz_evaluation / "graz_config.json").read_text())
    assert cosmos_metadata["checkpoint_sha256"] == graz_config["checkpoint_sha256"]
    assert cosmos_metadata["dtype"] == "float32" and not graz_config["bf16"]
    sources = {
        "cosmos_reference": (args.cosmos_evaluation, "local_reference.nii.gz"),
        "cosmos_estimated": (args.cosmos_evaluation, "estimated_local_field.nii.gz"),
        "graz1_estimated": (args.graz_evaluation / "graz1", "local_field.nii.gz"),
        "graz2_estimated": (args.graz_evaluation / "graz2", "local_field.nii.gz"),
    }
    grid = np.meshgrid(*([np.arange(-3, 4)] * 3), indexing="ij")
    sphere = sum(axis**2 for axis in grid) <= 9
    data = {"full_roi": {}, "interior_radius3": {}}
    stats = {key: {} for key in data}
    boundary = {}
    provenance = {}
    for name, (directory, filename) in sources.items():
        image = nib.load(directory / filename)
        mask_image = nib.load(directory / "mask.nii.gz")
        volume = np.asarray(image.dataobj, dtype=np.float64)
        raw_mask = np.asarray(mask_image.dataobj)
        assert volume.shape == raw_mask.shape and np.isin(raw_mask, [0, 1]).all()
        assert (
            image.get_data_dtype() == np.dtype("float32") and np.isfinite(volume).all()
        )
        mask = raw_mask.astype(bool)
        interior = binary_erosion(mask, structure=sphere, border_value=0)
        edge = mask & ~interior
        for scope, selected in (("full_roi", mask), ("interior_radius3", interior)):
            values = volume[selected]
            assert values.size and values.std() > 0
            data[scope][name] = values
            stats[scope][name] = describe(values)
        boundary[name] = dict(
            boundary_voxel_fraction=float(edge.sum() / mask.sum()),
            boundary_energy_fraction=float(
                np.sum(volume[edge] ** 2) / np.sum(volume[mask] ** 2)
            ),
            abs_gt_005_boundary_fraction_of_all_tails=float(
                np.count_nonzero((np.abs(volume) > 0.05) & edge)
                / max(np.count_nonzero((np.abs(volume) > 0.05) & mask), 1)
            ),
        )
        provenance[name] = dict(
            field=str((directory / filename).resolve()),
            mask=str((directory / "mask.nii.gz").resolve()),
            field_sha256=hashlib.sha256(
                (directory / filename).read_bytes()
            ).hexdigest(),
            shape=list(volume.shape),
            reference_available=name.startswith("cosmos"),
        )
    distances = {}
    histograms = {}
    for scope, groups in data.items():
        distances[scope] = {}
        for mode in ("physical", "standardized"):
            values = {name: transformed(v, mode) for name, v in groups.items()}
            low = min(float(np.quantile(v, 0.0005)) for v in values.values())
            high = max(float(np.quantile(v, 0.9995)) for v in values.values())
            edges = np.linspace(low, high, 601)
            probabilities = {name: probability(v, edges) for name, v in values.items()}
            histograms[f"{scope}_{mode}_edges"] = edges
            for name, p in probabilities.items():
                histograms[f"{scope}_{mode}_{name}_probability_with_overflow"] = p
            comparisons = {}
            for reference in ("cosmos_reference", "cosmos_estimated"):
                comparisons[reference] = {}
                for name in ("graz1_estimated", "graz2_estimated"):
                    comparisons[reference][name] = dict(
                        wasserstein=float(
                            wasserstein_distance(values[reference], values[name])
                        ),
                        ks_statistic=float(
                            ks_2samp(
                                values[reference], values[name], method="asymp"
                            ).statistic
                        ),
                        js_distance_base2=float(
                            jensenshannon(
                                probabilities[reference], probabilities[name], base=2
                            )
                        ),
                    )
            distances[scope][mode] = comparisons
    np.savez_compressed(args.output_dir / "histograms.npz", **histograms)
    plt = _pyplot()
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for row, (scope, groups) in enumerate(data.items()):
        for column, mode in enumerate(("physical", "physical", "standardized")):
            values = {name: transformed(v, mode) for name, v in groups.items()}
            limits = (
                (
                    min(float(np.quantile(v, 0.0005)) for v in values.values()),
                    max(float(np.quantile(v, 0.9995)) for v in values.values()),
                )
                if mode == "physical"
                else (-6, 6)
            )
            edges = np.linspace(*limits, 301)
            axis = axes[row, column]
            for (name, v), color in zip(values.items(), COLORS, strict=True):
                density = np.histogram(v, edges)[0] / len(v) / np.diff(edges)
                axis.stairs(
                    density,
                    edges,
                    color=color,
                    label=LABELS[name],
                    linestyle="--" if name == "cosmos_reference" else "-",
                    linewidth=1.6,
                )
            if column == 1:
                axis.set_yscale("log")
            axis.set(
                xlim=limits,
                xlabel="Local field (ppm)"
                if mode == "physical"
                else "(Field − ROI mean) / ROI standard deviation",
                ylabel="Probability density",
                title=(
                    "Full brain mask"
                    if row == 0
                    else "Interior: mask eroded by radius 3 voxels"
                )
                + (
                    " · standardized"
                    if column == 2
                    else " · log density"
                    if column == 1
                    else " · amplitude"
                ),
            )
            if column == 0:
                axis.set_xlim(
                    min(float(np.quantile(v, 0.01)) for v in values.values()),
                    max(float(np.quantile(v, 0.99)) for v in values.values()),
                )
            axis.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "COSMOS versus Graz local fields — epoch 503, float32\nAll masked voxels; independent density normalization per volume; Graz fields are estimates"
    )
    for extension in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"local_field_distributions.{extension}", dpi=180
        )
    plt.close(figure)
    records = []
    for scope, groups in stats.items():
        for name, record in groups.items():
            flat = dict(
                scope=scope,
                dataset=name,
                **{k: v for k, v in record.items() if k != "quantiles_ppm"},
            )
            flat.update({f"{k}_ppm": v for k, v in record["quantiles_ppm"].items()})
            records.append(flat)
    with (args.output_dir / "statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = dict(
        units="ppm",
        checkpoint_sha256=graz_config["checkpoint_sha256"],
        epoch=graz_config["epoch"],
        provenance=provenance,
        statistics=stats,
        boundary_diagnostic=boundary,
        distances=distances,
        methodology="All voxels inside each binary mask, FP64 statistics; no p-values because voxels are spatially correlated. Wasserstein and KS use exact voxel arrays; JS uses common 600-bin histograms plus explicit overflow bins. Each volume has independent probability normalization.",
        limitations="Graz has no known clean local-field reference. These compare field distributions, not voxelwise accuracy; subject anatomy and mask support differ. The radius-3 interior is a secondary analysis only and does not change inference masks or model outputs.",
    )
    write_json(args.output_dir / "results.json", report)
    shutil.copy2(__file__, args.output_dir / "compare_cosmos_graz_local_fields.py")
    print(
        json.dumps(
            dict(
                directory=str(args.output_dir.resolve()),
                statistics=stats,
                boundary_diagnostic=boundary,
                distances=distances,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
