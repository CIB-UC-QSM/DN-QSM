"""Compare stored and actually sampled training distributions with external cases.

Run from the repository root with .venv/bin/python scripts/analyze_domain_histograms.py.
Source images and checkpoints are read only; no model inference or training is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from dnqsm.data import BalancedSourceSampler, TrainingVolumes
from dnqsm.evaluation import load_cosmos_case, load_sim2_case
from dnqsm.physics import apply_dipole, dipole_tensor

DEFAULT_RUN = "artifacts/runs/dnqsm_laplacian_fp32_finetune100_20261020_001036"
FIELDS = ("chi", "local")
COLORS = {"stored": "#83909b", "augmented": "#224f8c", "cosmos": "#18865f", "sim2": "#dc6844"}
_DATASET = None
_KERNELS = {}
_SAMPLES = 0


def initialize_worker(records, config, epoch, samples):
    global _DATASET, _SAMPLES
    torch.set_num_threads(1)
    _SAMPLES = samples
    _DATASET = TrainingVolumes(
        records, config["cache_dir"], seed=config["seed"],
        chi_scale_mode="rms_mixture", crop_probability=config["crop_probability"],
        crop_resize=config["crop_resize"],
        chi_augmentation=config.get("chi_augmentation"),
    )
    _DATASET.set_epoch(epoch)


def describe(values):
    """Exact masked moments and fractions, using FP64 reductions."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty mask or nonfinite masked values")
    mean = float(np.mean(values, dtype=np.float64))
    square = float(np.mean(np.square(values, dtype=np.float64)))
    return dict(
        voxels=len(values), mean=mean, rms=float(np.sqrt(square)),
        std=float(np.sqrt(max(0, square - mean**2))),
        minimum=float(values.min()), maximum=float(values.max()),
        negative_fraction=float(np.mean(values < 0)),
        zero_fraction=float(np.mean(values == 0)),
        near_zero_fraction=float(np.mean(np.abs(values) < 0.001)),
        abs_gt_005_fraction=float(np.mean(np.abs(values) > 0.05)),
        abs_gt_010_fraction=float(np.mean(np.abs(values) > 0.1)),
    )


def summarize_volumes(chi, local, mask, seed, samples):
    mask = mask.reshape(-1).bool()
    values = [x.reshape(-1)[mask].numpy() for x in (chi, local)]
    rng = np.random.default_rng(seed)
    # Sampling with replacement gives the same count/weight to every image.
    indices = rng.integers(len(values[0]), size=samples)
    selected = np.stack([v[indices] for v in values])
    stats = {field: describe(v) for field, v in zip(FIELDS, values, strict=True)}
    return stats, selected


def susceptibility_tail_checks(case):
    values = case.chi[case.mask.bool()].numpy()
    energy = np.sum(values.astype(np.float64)**2)
    result = {"chi_quantiles_exact": dict(zip(
        ("p0001", "p001", "p01", "p05", "p50", "p95", "p99", "p999", "p9999"),
        np.quantile(values, [.0001, .001, .01, .05, .5, .95, .99, .999, .9999]).tolist(), strict=True))}
    for threshold in (.05, .1, .2, .5, 1.):
        selected = np.abs(values) > threshold
        result[str(threshold)] = dict(
            voxels=int(selected.sum()), fraction=float(selected.mean()),
            energy_fraction=float(np.sum(values[selected].astype(np.float64)**2) / energy),
            rms_excluding=float(np.sqrt(np.mean(values[~selected].astype(np.float64)**2))),
        )
    edges = np.linspace(-.1, .1, 401)
    smooth = gaussian_filter1d(np.histogram(values, edges)[0].astype(float), 2)
    peaks, _ = find_peaks(smooth, prominence=smooth.max() * .05, distance=15)
    result["major_histogram_peaks_ppm"] = ((edges[:-1] + edges[1:])[peaks] / 2).tolist()
    return result


def analyze_task(task):
    stage, index = task
    record = _DATASET.records[index % len(_DATASET.records)]
    if stage == "stored":
        sample = _DATASET._load(record)
        crop = resize = False
    else:
        sample = _DATASET[index]
        crop, resize = sample["crop_applied"], sample["resize_applied"]
    mask = sample["mask"]
    chi = sample["chi"] * mask
    shape = tuple(chi.shape[-3:])
    if shape not in _KERNELS:
        _KERNELS[shape] = dipole_tensor(shape)
    local = apply_dipole(chi.unsqueeze(0), _KERNELS[shape])[0] * mask
    stats, selected = summarize_volumes(chi, local, mask, 89173 + index, _SAMPLES)
    row = dict(stage=stage, index=index, subject=record["subject"],
               cohort=record["subject"].split(":")[0], image=record["image"],
               crop=bool(crop), resize=bool(resize), shape=list(shape), stats=stats)
    return row, selected


def transformed(samples, rows, field, mode):
    values = samples[:, FIELDS.index(field), :]
    if mode == "physical":
        return values.reshape(-1)
    scales = np.array([r["stats"][field]["rms" if mode == "rms" else "std"] for r in rows])
    means = 0 if mode == "rms" else np.array([r["stats"][field]["mean"] for r in rows])[:, None]
    return ((values - means) / scales[:, None].clip(1e-12)).reshape(-1)


def histogram(values, edges):
    # Fold overflow into edge bins, preserving probability for distances.
    counts = np.histogram(values, edges)[0].astype(np.float64)
    counts[0] += np.count_nonzero(values < edges[0])
    counts[-1] += np.count_nonzero(values > edges[-1])
    assert counts.sum() == values.size
    return counts / counts.sum()


def small_sample(values, seed=1281, maximum=500_000):
    if len(values) <= maximum:
        return values
    return values[np.random.default_rng(seed).integers(len(values), size=maximum)]


def summarize_group(rows, samples):
    result = {"images": len(rows), "cohorts": dict(Counter(r["cohort"] for r in rows)),
              "crop_count": sum(r["crop"] for r in rows),
              "resize_count": sum(r["resize"] for r in rows)}
    for field in FIELDS:
        per_image = {key: np.array([r["stats"][field][key] for r in rows])
                     for key in rows[0]["stats"][field]}
        result[field] = {
            "image_equal_mean": float(per_image["mean"].mean()),
            "image_equal_rms": float(np.sqrt(np.mean(per_image["rms"]**2))),
            "image_equal_negative_fraction": float(per_image["negative_fraction"].mean()),
            "image_equal_zero_fraction": float(per_image["zero_fraction"].mean()),
            "image_equal_near_zero_fraction": float(per_image["near_zero_fraction"].mean()),
            "voxel_weighted_rms": float(np.sqrt(np.average(per_image["rms"]**2, weights=per_image["voxels"]))),
            "sampled_voxel_quantiles": dict(zip(
                ("p001", "p01", "p05", "p50", "p95", "p99", "p999"),
                np.quantile(transformed(samples, rows, field, "physical"),
                            [.001, .01, .05, .5, .95, .99, .999]).tolist(), strict=True)),
            "per_image": {key: dict(zip(("min", "p01", "p05", "p50", "p95", "p99", "max"),
                                       np.quantile(per_image[key], [0, .01, .05, .5, .95, .99, 1]).tolist(), strict=True))
                          for key in ("rms", "std", "mean", "negative_fraction", "zero_fraction")},
        }
    return result


def analyze_distributions(groups, external, output):
    metrics, histograms = {}, {}
    for field in FIELDS:
        metrics[field] = {}
        for mode in ("physical", "rms", "standardized"):
            values = {name: transformed(samples, rows, field, mode)
                      for name, (rows, samples) in (groups | external).items()}
            # Central 99.98% union for bins; tail counts folded into edge bins.
            low = min(float(np.quantile(v, .0001)) for v in values.values())
            high = max(float(np.quantile(v, .9999)) for v in values.values())
            edges = np.linspace(low, high, 1201)
            hist = {name: histogram(v, edges) for name, v in values.items()}
            histograms[f"{field}_{mode}_edges"] = edges
            for name, h in hist.items():
                histograms[f"{field}_{mode}_{name}"] = h
            comparisons = {}
            train = small_sample(values["augmented"])
            for name in external:
                reference = small_sample(values[name])
                comparisons[name] = dict(
                    wasserstein=float(wasserstein_distance(train, reference)),
                    js_distance_base2=float(jensenshannon(hist["augmented"], hist[name], base=2)),
                    ks_statistic=float(ks_2samp(train, reference, method="asymp").statistic),
                    fraction_outside_train_p01_p99=float(np.mean(
                        (reference < np.quantile(train, .01)) | (reference > np.quantile(train, .99)))),
                )
                if mode == "physical":
                    image_rms = np.array([r["stats"][field]["rms"] for r in groups["augmented"][0]])
                    external_rms = external[name][0][0]["stats"][field]["rms"]
                    comparisons[name]["rms_percentile_among_train_images"] = float(100 * np.mean(image_rms <= external_rms))
                    comparisons[name]["rms_ratio_to_train_median"] = float(external_rms / np.median(image_rms))
            metrics[field][mode] = comparisons
    np.savez_compressed(output / "histograms.npz", **histograms)
    return metrics


def draw_figures(groups, external, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    labels = {"stored": "Stored training", "augmented": "Training after augmentation",
              "cosmos": "COSMOS", "sim2": "sim2 (local = proxy)"}
    for row, field in enumerate(FIELDS):
        for col, mode in enumerate(("physical", "rms", "standardized")):
            ax = axes[row, col]
            arrays = {name: transformed(samples, rows, field, mode)
                      for name, (rows, samples) in (groups | external).items()}
            plotted = [v for name, v in arrays.items() if name != "stored"]
            limits = (-6, 6) if mode != "physical" else (
                min(float(np.quantile(v, .001)) for v in plotted),
                max(float(np.quantile(v, .999)) for v in plotted))
            edges = np.linspace(*limits, 220)
            for name, values in arrays.items():
                # Normalize by all sampled voxels, including off-axis tails.
                density = np.histogram(values, edges)[0] / len(values) / np.diff(edges)
                ax.stairs(density, edges, color=COLORS[name], label=labels[name],
                          linewidth=1.6 if name != "stored" else 1.1,
                          linestyle="--" if name == "stored" else "-")
            ax.set(xlim=limits, ylabel="Probability density", xlabel={
                "physical": "Susceptibility (ppm)" if field == "chi" else "Clean local field (ppm)",
                "rms": "Value / per-image masked RMS",
                "standardized": "(Value − per-image mean) / per-image std",
            }[mode], title=("Susceptibility" if field == "chi" else "Local field") + " · " + mode)
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Training versus external data: amplitude, RMS-normalized shape, and centered shape\n"
                 "Masked voxels only; equal weight per training image; sim2 local field is forward modeled", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"histogram_comparison.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cohort_colors = {"qsm": "#536caa", "separate": "#a177ab", "synthetic": "#bf9137"}
    for row, field in enumerate(FIELDS):
        ax = axes[row, 0]
        rows, samples = groups["augmented"]
        limits = ((-.15, .2) if field == "chi" else (-.06, .06))
        edges = np.linspace(*limits, 250)
        for cohort, color in cohort_colors.items():
            selected = np.array([r["cohort"] == cohort for r in rows])
            values = samples[selected, FIELDS.index(field), :].reshape(-1)
            density = np.histogram(values, edges)[0] / len(values) / np.diff(edges)
            ax.stairs(density, edges, label=f"{cohort} ({selected.sum()} draws)", color=color)
        for name, (ext_rows, ext_samples) in external.items():
            values = transformed(ext_samples, ext_rows, field, "physical")
            density = np.histogram(values, edges)[0] / len(values) / np.diff(edges)
            ax.stairs(density, edges, color=COLORS[name], label=labels[name])
        ax.set(yscale="log", ylim=(.001, None), xlim=limits, xlabel=f"{field} (ppm)", ylabel="Density (log)", title="Augmented training cohorts and tails")
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        categories = []
        for stage in ("stored", "augmented"):
            for cohort in cohort_colors:
                category_rows = [r for r in groups[stage][0] if r["cohort"] == cohort]
                categories.append((f"{stage}\n{cohort}", [r["stats"][field]["rms"] for r in category_rows]))
        ax.boxplot([v for _, v in categories], tick_labels=[k for k, _ in categories], showfliers=False, whis=(1, 99))
        for name, (ext_rows, _) in external.items():
            ax.axhline(ext_rows[0]["stats"][field]["rms"], color=COLORS[name], label=labels[name])
        ax.set(yscale="log", ylabel=f"Per-image {field} RMS (ppm)", title="RMS coverage (boxes 25–75%; whiskers 1–99%)")
        ax.legend(fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"cohorts_and_rms.{extension}", dpi=180)
    plt.close(fig)


def write_report(result, output):
    lines = ["# Training / COSMOS / sim2 histogram comparison", "",
             "All values are in ppm inside the original evaluation/training masks. Background padding is excluded.", "",
             "**sim2 local field is a forward-model proxy from masked susceptibility, not an independently measured local-field reference.** "
             "COSMOS clean local field is also generated by its evaluation forward operator. Observed total fields are reported separately.", "",
             "## Distributions", "",
             "Training rows below use equal image weight. RMS is the square root of the image-averaged squared RMS; the external rows use their complete masks.", "",
             "| Dataset | χ mean | χ RMS | χ negative | χ exactly zero | Local mean | Local RMS |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in result["groups"].items():
        c, f = summary["chi"], summary["local"]
        lines.append(f"| {name} | {c['image_equal_mean']:.5f} | {c['image_equal_rms']:.5f} | {100*c['image_equal_negative_fraction']:.1f}% | {100*c['image_equal_zero_fraction']:.1f}% | {f['image_equal_mean']:.5f} | {f['image_equal_rms']:.5f} |")
    for name, ext in result["external"].items():
        c, f = ext["stats"]["chi"], ext["stats"]["local"]
        lines.append(f"| {name} | {c['mean']:.5f} | {c['rms']:.5f} | {100*c['negative_fraction']:.1f}% | {100*c['zero_fraction']:.1f}% | {f['mean']:.5f} | {f['rms']:.5f} |")
    lines += ["", "## External versus augmented training", "",
              "JS is Jensen–Shannon distance with base-2 logs (0 identical, 1 disjoint). "
              "W1 is empirical Wasserstein distance (ppm in physical mode; dimensionless otherwise). "
              "KS is the descriptive CDF gap; no independent-voxel significance test is interpreted.", "",
              "| Quantity | Dataset | RMS / train median | RMS percentile | Physical W1 | Physical JS | RMS-normalized JS | Centered/std JS | Outside train 1–99% |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for field in FIELDS:
        for name in result["external"]:
            m = result["distances"][field]
            p = m["physical"][name]
            lines.append(f"| {field} | {name} | {p['rms_ratio_to_train_median']:.2f} | {p['rms_percentile_among_train_images']:.1f} | {p['wasserstein']:.5f} | {p['js_distance_base2']:.3f} | {m['rms'][name]['js_distance_base2']:.3f} | {m['standardized'][name]['js_distance_base2']:.3f} | {100*p['fraction_outside_train_p01_p99']:.2f}% |")
    train = result["groups"]["augmented"]
    sim2 = result["external"]["sim2"]
    cosmos = result["external"]["cosmos"]
    chi_dist = result["distances"]["chi"]
    local_dist = result["distances"]["local"]
    tail = sim2["susceptibility_tail_checks"]["0.5"]
    peaks = sim2["susceptibility_tail_checks"]["major_histogram_peaks_ppm"]
    lines += ["", "## Findings", "",
              "1. **The current susceptibility amplitude augmentation covers both external cases.** "
              f"Augmented training per-image χ RMS spans {train['chi']['per_image']['rms']['p01']:.4f}–{train['chi']['per_image']['rms']['p99']:.4f} ppm (1–99%). "
              f"COSMOS and sim2 χ RMS are {cosmos['stats']['chi']['rms']:.4f} and {sim2['stats']['chi']['rms']:.4f} ppm, "
              f"at the {chi_dist['physical']['cosmos']['rms_percentile_among_train_images']:.1f} and {chi_dist['physical']['sim2']['rms_percentile_among_train_images']:.1f} percentiles. "
              f"Their local-field RMS values are at the {local_dist['physical']['cosmos']['rms_percentile_among_train_images']:.1f} and {local_dist['physical']['sim2']['rms_percentile_among_train_images']:.1f} percentiles. "
              "The histograms provide no evidence of a large clean-target global amplitude mismatch.", "",
              "2. **sim2 has a clear susceptibility histogram-shape difference.** "
              f"Its dominant peaks are near {', '.join(f'{p:+.4f}' for p in peaks)} ppm, compared with the broader central peak in the pooled training data. "
              f"Negative voxels comprise {100*sim2['stats']['chi']['negative_fraction']:.1f}% versus {100*train['chi']['image_equal_negative_fraction']:.1f}% in augmented training. "
              f"Physical χ JS distance is {chi_dist['physical']['sim2']['js_distance_base2']:.3f}; "
              f"it remains {chi_dist['rms']['sim2']['js_distance_base2']:.3f} after RMS normalization and "
              f"{chi_dist['standardized']['sim2']['js_distance_base2']:.3f} after per-image centering/std normalization. "
              "This is consistent with a difference in tissue contrast/mixture or image construction; histograms alone do not identify its cause. "
              "RMS scaling or a simple reference offset cannot account for the observed difference.", "",
              "3. **COSMOS is closer to the training distribution.** "
              f"Susceptibility JS distance is {chi_dist['physical']['cosmos']['js_distance_base2']:.3f} and "
              f"clean-local JS distance is {local_dist['physical']['cosmos']['js_distance_base2']:.3f}. "
              "Its susceptibility distribution remains somewhat more centrally concentrated than training after normalization, "
              "but the local-field density is close in physical units. These are relative comparisons, not calibrated acceptability thresholds.", "",
              "4. **sim2's local-field difference is substantially reduced by amplitude normalization.** "
              f"Its clean-local proxy JS distance drops from {local_dist['physical']['sim2']['js_distance_base2']:.3f} "
              f"to {local_dist['rms']['sim2']['js_distance_base2']:.3f} after per-image RMS normalization. "
              "The forward dipole also produces much smoother value distributions than the susceptibility image. "
              "Because the clean field is synthesized from χ, this does not independently establish agreement with the acquired field physics.", "",
              "5. **Inspect rare susceptibility extremes separately from central histograms.** "
              f"sim2 reaches {sim2['stats']['chi']['minimum']:.3f} ppm; {tail['voxels']} voxels "
              f"({100*tail['fraction']:.4f}% of its mask) exceed |χ|=0.5 ppm and contribute {100*tail['energy_fraction']:.2f}% of χ squared energy. "
              "The central plots deliberately focus on the bulk distribution; their axes do not display every extreme voxel. "
              "These voxels may represent intended structures or artifacts; no clipping is justified by a histogram alone.", "",
              "## Implications for training", "",
              "- Retain the current physical-unit RMS mixture as the baseline; increasing its amplitude range alone is unlikely to address sim2's susceptibility shape difference.",
              "- Test spatially coherent tissue-contrast and tissue-proportion augmentation using independent development data. Recompute local fields from every modified susceptibility map so the targets remain physically consistent. A global multiplier cannot generate the observed bimodal shape.",
              "- Audit susceptibility reference conventions, ROI definitions, resampling, and the origin of extreme values before changing normalization. The near-zero external means and persistent standardized shape difference argue against a simple global reference offset as the sole explanation.",
              "- Compare original-only and current 75/25 original/synthetic sampling on independent held-out subjects. The synthetic cohort has a lower clean-local RMS at similar χ RMS and a different tail shape; its optimal weight cannot be inferred from these histograms alone.",
              "- Analyze observed input-field backgrounds and noise separately before attributing external validation errors to target histograms. "
              f"COSMOS's observed total-field RMS is {cosmos['observed_total_field']['rms']:.4f} ppm versus its clean-local RMS of {cosmos['stats']['local']['rms']:.5f} ppm. "
              "sim2 has no independent clean-local reference in the current evaluation loader.",
              "- Keep COSMOS and sim2 as external checks; do not tune preprocessing to match their exact test histograms or infer performance gains without a separate validation experiment."]
    lines += ["", "## Sampling and interpretation", "", *[f"- {s}" for s in result["method"]], "",
              "- Histogram distances depend on binning and sampling and have no universal threshold for acceptable generalization.",
              "- RMS normalization and centering/std are analysis controls, not proposed preprocessing changes. Subtracting susceptibility means under a finite mask changes its forward field.",
              "- One external image per dataset cannot establish a population-wide domain shift or explain model error causally. Histograms omit anatomy, spatial correlations, orientation, resolution, background, and noise.", "",
              "![Histogram comparison](histogram_comparison.png)", "", "![Cohorts and RMS](cohorts_and_rms.png)", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnostics/domain_histograms_20261020"))
    parser.add_argument("--epoch", type=int, default=55, help="Zero-based sampler epoch; 55 is the last interrupted epoch")
    parser.add_argument("--samples", type=int, default=16384, help="Uniform masked samples per training image")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run / "config.json").read_text())
    split = json.loads((args.run / "split.json").read_text())
    records = split["train"]
    if config.get("max_train_volumes"):
        raise ValueError("Handle max_train_volumes explicitly before analysis")
    sampler = BalancedSourceSampler(records, config["synthetic_fraction"], config["seed"])
    sampler.set_epoch(args.epoch)
    tasks = [("stored", i) for i in range(len(records))] + [("augmented", i) for i in sampler]
    groups = {stage: ([], np.empty((len(records), 2, args.samples), dtype=np.float32))
              for stage in ("stored", "augmented")}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(records, config, args.epoch, args.samples)) as pool:
        for count, (row, values) in enumerate(pool.map(analyze_task, tasks, chunksize=4), 1):
            rows, samples = groups[row["stage"]]
            samples[len(rows)] = values
            rows.append(row)
            if count % 100 == 0 or count == len(tasks):
                print(f"Analyzed {count}/{len(tasks)} images", flush=True)
    external, external_metadata = {}, {}
    for case in (load_cosmos_case(config["cosmos_simulation"]), load_sim2_case(pad=config.get("sim2_pad", 20))):
        local = case.local if case.local is not None else apply_dipole(case.chi * case.mask, case.dipole) * case.mask
        stats, values = summarize_volumes(case.chi, local, case.mask, 92317, 262144)
        row = dict(stage=case.name, stats=stats)
        external[case.name] = ([row], values[None])
        external_metadata[case.name] = dict(
            stats=stats, observed_total_field=describe(case.phase[case.mask.bool()].numpy()),
            shape=list(case.phase.shape[-3:]), masked_voxels=int(case.mask.sum()),
            metadata=case.metadata,
            local_reference="evaluation forward field" if case.local is not None else "susceptibility-derived forward proxy; no measured clean local reference",
            susceptibility_tail_checks=susceptibility_tail_checks(case),
        )
    result = dict(run=str(args.run.resolve()), epoch_zero_based=args.epoch,
                  groups={name: summarize_group(*group) for name, group in groups.items()},
                  external=external_metadata)
    result["cohort_summaries"] = {
        stage: {cohort: summarize_group([r for r in rows if r["cohort"] == cohort],
                                       samples[np.array([r["cohort"] == cohort for r in rows])])
                for cohort in sorted({r["cohort"] for r in rows})}
        for stage, (rows, samples) in groups.items()
    }
    result["crop_summaries"] = {
        str(crop): summarize_group([r for r in groups["augmented"][0] if r["crop"] == crop],
                                  groups["augmented"][1][np.array([r["crop"] == crop for r in groups["augmented"][0]])])
        for crop in (False, True)
    }
    result["distances"] = analyze_distributions(groups, external, output)
    result["method"] = [
        f"All {len(records)} training source records from the latest run's persisted subject split, excluding its held-out validation subjects.",
        f"One complete balanced sampler epoch (zero-based {args.epoch}), including repeated source views, flips/permutations, RMS-mixture scaling, and configured crop/resize.",
        f"Training histograms: {args.samples} uniform masked voxel draws with replacement per image, equal image weight. External histograms: 262144 draws each. Fixed random seeds.",
        "Exact complete-mask mean/RMS/std/fractions per image; pooled histogram quantiles approximate. W1/KS use at most 500000 additional draws per compared histogram.",
        "Jensen–Shannon distance: 1200 common bins spanning the union of dataset 0.01–99.99 percentiles; overflow folded into edge bins. These bins are shared within each quantity/normalization mode.",
        "Training local fields use the actual FP32 FFT dipole, native augmented grid, isotropic voxels and B0=(0,0,1); each case uses its evaluation dipole and padding. No noise or exterior background enters clean-field histograms.",
        "COSMOS physical-unit conversion follows evaluation metadata (stored factor 0.2125); sim2 susceptibility uses the evaluation NIfTI values directly. Observed phase is total field and is never mislabeled local field.",
    ]
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "tail_checks.json").write_text(json.dumps(
        {name: case["susceptibility_tail_checks"] for name, case in external_metadata.items()}, indent=2) + "\n")
    flat_rows = []
    for rows, _ in groups.values():
        for row in rows:
            flat = {k: v for k, v in row.items() if k != "stats"}
            flat.update({f"{field}_{k}": v for field in FIELDS for k, v in row["stats"][field].items()})
            flat_rows.append(flat)
    with (output / "per_image.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    draw_figures(groups, external, output)
    write_report(result, output)
    print(f"Saved {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    if "--chi-policy-audit" in sys.argv:
        from audit_chi_augmentation import main as audit_main

        audit_main([arg for arg in sys.argv[1:] if arg != "--chi-policy-audit"])
    else:
        main()
