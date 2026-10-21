"""Data-only paired A0–A3 audit; no model weights, training, or external fitting."""

import argparse
import csv
import json
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from analyze_domain_histograms import (
    DEFAULT_RUN,
    FIELDS,
    histogram,
    small_sample,
    summarize_group,
    summarize_volumes,
    transformed,
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from dnqsm.chi_augmentation import ChiAugmentationConfig
from dnqsm.data import BalancedSourceSampler, TrainingVolumes
from dnqsm.evaluation import load_cosmos_case, load_sim2_case
from dnqsm.physics import apply_dipole, dipole_tensor

POLICIES = {
    "A0": None,
    "A1": ChiAugmentationConfig(final_grid_rms=True),
    "A2": ChiAugmentationConfig(final_grid_rms=True, bimodal_probability=1 / 3),
    "A3": ChiAugmentationConfig(
        final_grid_rms=True, bimodal_probability=1 / 3, central_probability=1 / 3
    ),
}
COLORS = {
    "A0": "#64748b",
    "A1": "#a1a1aa",
    "A2": "#7c3aed",
    "A3": "#2458a5",
    "baseline": "#64748b",
    "bimodal": "#ae4894",
    "central_tail": "#397aaa",
    "cosmos": "#18865f",
    "sim2": "#dc6844",
}
_SETS, _KERNELS, _RADII = {}, {}, {}
_SAMPLE_COUNT = 0


def initialize(records, config, policies, samples):
    global _SETS, _SAMPLE_COUNT
    torch.set_num_threads(1)
    _SAMPLE_COUNT = samples
    _SETS = {
        name: TrainingVolumes(
            records,
            config["cache_dir"],
            seed=config["seed"],
            chi_scale_mode="rms_mixture",
            crop_probability=config["crop_probability"],
            crop_resize=config["crop_resize"],
            chi_augmentation=policies[name],
        )
        for name in policies
    }


def spatial_statistics(chi, mask, spectrum):
    x, m = chi.reshape(chi.shape[-3:]), mask.reshape(mask.shape[-3:]).bool()
    gradient_square, gradient_count, sampled = 0.0, 0, []
    rng = np.random.default_rng(8171)
    for axis in range(3):
        first = [slice(None)] * 3
        second = [slice(None)] * 3
        first[axis], second[axis] = slice(1, None), slice(None, -1)
        valid = m[tuple(first)] & m[tuple(second)]
        delta = (x[tuple(first)] - x[tuple(second)])[valid].numpy()
        if not len(delta):
            continue
        gradient_square += float(np.sum(delta.astype(np.float64) ** 2))
        gradient_count += len(delta)
        sampled.extend(np.abs(delta[rng.integers(len(delta), size=2048)]).tolist())
    if not gradient_count:
        raise ValueError("ROI has no neighboring voxels for spatial diagnostics")
    shape = tuple(x.shape)
    if shape not in _RADII:
        frequencies = np.meshgrid(
            np.fft.fftfreq(shape[0]),
            np.fft.fftfreq(shape[1]),
            np.fft.rfftfreq(shape[2]),
            indexing="ij",
        )
        radii = np.sqrt(sum(f * f for f in frequencies))
        _RADII[shape] = np.minimum((radii / np.sqrt(0.75) * 32).astype(np.int16), 31)
    power = (
        np.abs(spectrum.reshape(spectrum.shape[-3:]).numpy()).astype(np.float64) ** 2
    )
    power[..., 1 : (-1 if shape[-1] % 2 == 0 else None)] *= 2
    dc = float(power[0, 0, 0])
    power[0, 0, 0] = 0
    radial = np.bincount(
        _RADII[shape].reshape(-1), weights=power.reshape(-1), minlength=32
    )
    radial /= max(float(radial.sum()), 1e-30)
    values = x[m].numpy().astype(np.float64)
    energy = max(float(np.sum(values**2)), 1e-30)
    return dict(
        gradient_rms=float(np.sqrt(gradient_square / gradient_count)),
        gradient_abs_p95=float(np.quantile(sampled, 0.95)),
        tail_energy_gt_01=float(np.sum(values[np.abs(values) > 0.1] ** 2) / energy),
        tail_energy_gt_05=float(np.sum(values[np.abs(values) > 0.5] ** 2) / energy),
        dc_power_fraction=dc / max(dc + float(np.sum(power)), 1e-30),
        radial_power=radial.tolist(),
    )


def analyze(task):
    epoch, index, position = task
    output = []
    for name, dataset in _SETS.items():
        dataset.set_epoch(epoch)
        sample = dataset[index]
        mask = sample["mask"]
        # Match FieldSynthesizer: historical source files can contain values
        # outside the ROI, which are excluded before forward synthesis.
        chi = sample["chi"] * mask
        shape = tuple(chi.shape[-3:])
        if shape not in _KERNELS:
            _KERNELS[shape] = dipole_tensor(shape)
        spectrum = torch.fft.rfftn(chi, dim=(-3, -2, -1))
        local = (
            torch.fft.irfftn(spectrum * _KERNELS[shape][0], s=shape, dim=(-3, -2, -1))
            * mask
        )
        stats, values = summarize_volumes(
            chi, local, mask, 89173 + index + epoch * 1000003, _SAMPLE_COUNT
        )
        row = dict(
            stage=name,
            epoch=epoch,
            index=index,
            subject=sample["subject"],
            cohort=sample["subject"].split(":")[0],
            crop=sample["crop_applied"],
            resize=sample["resize_applied"],
            stats=stats,
            profile=sample["chi_profile"],
            requested_profile=sample["chi_requested_profile"],
            fallback=sample["chi_fallback"],
            target_rms=sample["chi_rms_target"],
            parameters=json.loads(sample["chi_parameters"]),
            spatial=spatial_statistics(chi, mask, spectrum),
        )
        row["field_chi_rms_ratio"] = stats["local"]["rms"] / max(
            stats["chi"]["rms"], 1e-12
        )
        if dataset.chi_augmentation.enabled and not np.isclose(
            stats["chi"]["rms"], row["target_rms"], rtol=2e-6
        ):
            raise AssertionError("Final-grid RMS differs from its target")
        if not bool(torch.isfinite(chi).all()) or bool((chi[~mask] != 0).any()):
            raise AssertionError("Nonfinite chi or nonzero background")
        slices = None
        if position < 12:
            center = [
                int(
                    torch.where(
                        mask.any(dim=tuple(i for i in range(1, 4) if i != axis))
                    )[-1]
                    .float()
                    .median()
                )
                for axis in (1, 2, 3)
            ]
            slices = np.stack([chi[0, center[0]].numpy(), local[0, center[0]].numpy()])
        output.append((row, values, slices))
    return output


def compare(groups, external, directory):
    metrics, saved_hist = {}, {}
    all_groups = groups | external
    for field in FIELDS:
        for mode in ("physical", "rms", "standardized"):
            bounds = (
                ((-0.3, 0.3) if field == "chi" else (-0.1, 0.1))
                if mode == "physical"
                else (-12.0, 12.0)
            )
            edges = np.linspace(*bounds, 1201)
            key = f"{field}_{mode}"
            saved_hist[key + "_edges"] = edges
            probabilities = {}
            for name, (rows, samples) in all_groups.items():
                values = transformed(samples, rows, field, mode)
                probabilities[name] = histogram(values, edges)
                saved_hist[f"{key}_{name}"] = probabilities[name]
            for name, (rows, samples) in groups.items():
                entry = (
                    metrics.setdefault(name, {})
                    .setdefault(field, {})
                    .setdefault(mode, {})
                )
                values = small_sample(transformed(samples, rows, field, mode))
                for target, (target_rows, target_samples) in external.items():
                    reference = transformed(target_samples, target_rows, field, mode)
                    per_image = []
                    for row, sampled in zip(rows, samples, strict=True):
                        v = transformed(sampled[None], [row], field, mode)
                        distance = float(
                            jensenshannon(
                                histogram(v, edges), probabilities[target], base=2
                            )
                        )
                        per_image.append(distance)
                        row.setdefault("histogram_distances", {})[
                            f"{field}_{mode}_js_{target}"
                        ] = distance
                    entry[target] = dict(
                        js_distance=float(
                            jensenshannon(
                                probabilities[name], probabilities[target], base=2
                            )
                        ),
                        wasserstein=float(wasserstein_distance(values, reference)),
                        per_image_js_p05_p50_p95=np.quantile(
                            per_image, [0.05, 0.5, 0.95]
                        ).tolist(),
                    )
    np.savez_compressed(directory / "histograms.npz", **saved_hist)
    return metrics


def plot(groups, external, summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [
        ("chi", "physical", (-0.12, 0.18)),
        ("chi", "rms", (-5, 6)),
        ("local", "physical", (-0.04, 0.04)),
    ]
    for row in range(2):
        names = (
            [n for n in groups if "/" not in n]
            if row == 0
            else [n for n in groups if n.startswith("A3/")]
        )
        for ax, (field, mode, bounds) in zip(axes[row], panels, strict=True):
            edges = np.linspace(*bounds, 200)
            for name in names + list(external):
                rows, samples = (groups | external)[name]
                v = transformed(samples, rows, field, mode)
                color = COLORS[name.split("/")[-1]]
                ax.stairs(
                    np.histogram(v, edges)[0] / len(v) / np.diff(edges),
                    edges,
                    label=name,
                    color=color,
                )
            ax.set(
                xlim=bounds,
                xlabel=f"{field} ({'ppm' if mode == 'physical' else '/ RMS'})",
                ylabel="Density",
                title="Paired policy mixtures" if row == 0 else "A3 realized profiles",
            )
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        "Chi augmentation data audit · masked, image-equal histograms\nsim2 clean local field is a forward-model proxy"
    )
    fig.savefig(output / "policy_histograms.png", dpi=160)
    fig.savefig(output / "policy_histograms.pdf")
    plt.close(fig)
    names = [n for n in groups if "/" not in n]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(
        axes.flat,
        ("gradient_rms", "field_chi_rms_ratio", "radial_power", "tail_energy_gt_01"),
        strict=True,
    ):
        if metric == "radial_power":
            for name in names + list(external):
                rows = (groups | external)[name][0]
                values = np.mean([r["spatial"][metric] for r in rows], axis=0)
                ax.plot(
                    np.linspace(0, np.sqrt(0.75), 32),
                    values,
                    label=name,
                    color=COLORS[name],
                )
            ax.set(
                yscale="log",
                xlabel="Spatial frequency (cycles/voxel)",
                ylabel="Fraction of non-DC power",
            )
            ax.legend(fontsize=8)
        else:
            values = [
                [
                    r[metric]
                    if metric == "field_chi_rms_ratio"
                    else r["spatial"][metric]
                    for r in groups[n][0]
                ]
                for n in names
            ]
            ax.boxplot(values, tick_labels=names, whis=(5, 95), showfliers=False)
            for name in external:
                row = external[name][0][0]
                value = (
                    row[metric]
                    if metric == "field_chi_rms_ratio"
                    else row["spatial"][metric]
                )
                ax.axhline(value, color=COLORS[name], label=name)
            ax.set(ylabel=metric)
            ax.legend(fontsize=8)
    fig.savefig(output / "spatial_diagnostics.png", dpi=160)
    plt.close(fig)


def write_report(result, output):
    lines = [
        "# Chi augmentation implementation audit",
        "",
        "Data-only audit; no training or claim of improved reconstruction performance.",
        "",
        f"Epochs: {result['epochs']}; draws per epoch: {result['draws_per_epoch']}; shared source indices and existing RNG streams across policies.",
        "",
        "| Policy | Draws | Realized profiles | Fallbacks | χ RMS median | Field/χ RMS median | χ JS COSMOS | χ JS sim2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name in result["policies"]:
        group, m = result["summaries"][name], result["metrics"][name]["chi"]["physical"]
        extra = result["spatial"][name]
        lines.append(
            f"| {name} | {group['images']} | {result['profiles'][name]} | {result['fallbacks'][name]} | {group['chi']['per_image']['rms']['p50']:.5f} | {extra['field_chi_rms_ratio']['p50']:.3f} | {m['cosmos']['js_distance']:.3f} | {m['sim2']['js_distance']:.3f} |"
        )
    lines += [
        "",
        "## Shape-specific coverage",
        "",
        "Distances below use each realized profile separately, not the best single image.",
        "",
        "| Group | Images | χ negative fraction | χ JS COSMOS (RMS normalized) | χ JS sim2 (RMS normalized) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, group in result["summaries"].items():
        if "/" not in name:
            continue
        m = result["metrics"][name]["chi"]["rms"]
        lines.append(
            f"| {name} | {group['images']} | {100 * group['chi']['image_equal_negative_fraction']:.1f}% | {m['cosmos']['js_distance']:.3f} | {m['sim2']['js_distance']:.3f} |"
        )
    lines += [
        "",
        "- All histogram comparisons share fixed bins across policies (χ ±0.3 ppm; local ±0.1 ppm; normalized values ±12; 1200 bins). Overflow is included in endpoint bins.",
        "- Per-image and cohort summaries, realized target parameters, gradient RMS, radial power, tail energy, and physical/normalized distances are saved in summary.json and per_image.csv.",
        "- Spectra include the masked field of view and each case's padding; spatial comparisons can reflect mask geometry and padding as well as susceptibility texture.",
        "- Gradients use differences between neighboring voxels that are both inside the ROI; no artificial mask-edge differences enter their RMS.",
        "- Each external clean local field uses its evaluation forward operator. sim2 has no independent clean-local reference.",
        "- This bounded audit can establish implementation behavior and sampled coverage, not population coverage or reconstruction improvement. Evaluate independent held-out subjects before changing the default.",
        "",
        "![Profiles](policy_histograms.png)",
        "",
        "![Spatial checks](spatial_diagnostics.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    if (output / "examples.png").is_file():
        with (output / "report.md").open("a") as handle:
            handle.write(
                "\n![Paired susceptibility and field examples](examples.png)\n"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/chi_augmentation_audit"),
    )
    parser.add_argument(
        "--policies", nargs="+", choices=POLICIES, default=list(POLICIES)
    )
    parser.add_argument(
        "--policy-config",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Override a named policy from an experiment JSON or chi-policy JSON",
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=[53, 54, 55])
    parser.add_argument(
        "--max-per-epoch",
        type=int,
        default=0,
        help="0 audits the complete sampler epochs",
    )
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if (
        min(args.epochs) < 0
        or args.max_per_epoch < 0
        or args.samples < 64
        or args.workers < 1
    ):
        parser.error("Invalid audit counts")
    config = json.loads((args.run / "config.json").read_text())
    policy_configs = {
        name: None if POLICIES[name] is None else POLICIES[name].to_dict()
        for name in args.policies
    }
    for override in args.policy_config:
        name, separator, path = override.partition("=")
        if not separator or name not in policy_configs:
            parser.error("--policy-config must be NAME=JSON for a selected policy")
        configured = json.loads(Path(path).read_text())
        if not isinstance(configured, dict):
            parser.error("Policy JSON must be an object")
        configured = configured.get("chi_augmentation", configured)
        policy_configs[name] = ChiAugmentationConfig.from_dict(configured).to_dict()
    records = json.loads((args.run / "split.json").read_text())["train"]
    if config.get("max_train_volumes"):
        parser.error("Use a split with no max_train_volumes truncation")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    tasks, draws = [], {}
    sampler = BalancedSourceSampler(
        records, config["synthetic_fraction"], config["seed"]
    )
    for epoch in args.epochs:
        sampler.set_epoch(epoch)
        indices = list(sampler)
        # Select evenly spaced shuffled draws to avoid a special source order.
        if args.max_per_epoch and args.max_per_epoch < len(indices):
            positions = np.linspace(0, len(indices) - 1, args.max_per_epoch).astype(int)
            indices = [indices[i] for i in positions]
        draws[str(epoch)] = len(indices)
        tasks.extend((epoch, index, i) for i, index in enumerate(indices))
    groups = {
        name: ([], np.empty((len(tasks), 2, args.samples), np.float32))
        for name in args.policies
    }
    examples = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(records, config, policy_configs, args.samples),
    ) as pool:
        for count, batch in enumerate(pool.map(analyze, tasks, chunksize=2), 1):
            baseline_slices = next((v for r, _, v in batch if r["stage"] == "A0"), None)
            for row, values, slices in batch:
                rows, samples = groups[row["stage"]]
                samples[len(rows)] = values
                rows.append(row)
                if (
                    slices is not None
                    and row["stage"] == "A3"
                    and row["cohort"] != "synthetic"
                ):
                    examples.setdefault(row["profile"], (row, slices, baseline_slices))
            if count % 32 == 0 or count == len(tasks):
                print(
                    f"Audited {count}/{len(tasks)} paired draws ({len(args.policies)} policies)",
                    flush=True,
                )
    external = {}
    for case in [
        load_cosmos_case(config["cosmos_simulation"]),
        load_sim2_case(pad=config.get("sim2_pad", 20)),
    ]:
        local = (
            case.local
            if case.local is not None
            else apply_dipole(case.chi * case.mask, case.dipole) * case.mask
        )
        stats, values = summarize_volumes(case.chi, local, case.mask, 92813, 262144)
        spectrum = torch.fft.rfftn(case.chi * case.mask, dim=(-3, -2, -1))
        row = dict(
            stats=stats,
            spatial=spatial_statistics(case.chi * case.mask, case.mask, spectrum),
            field_chi_rms_ratio=stats["local"]["rms"] / stats["chi"]["rms"],
        )
        external[case.name] = ([row], values[None])
    all_rows = [r for rows, _ in groups.values() for r in rows]
    for name, (rows, samples) in list(groups.items()):
        for profile in sorted({r["profile"] for r in rows}):
            selected = np.array([r["profile"] == profile for r in rows])
            groups[f"{name}/{profile}"] = (
                [r for r, yes in zip(rows, selected, strict=True) if yes],
                samples[selected],
            )
    result = dict(
        run=str(args.run.resolve()),
        epochs=args.epochs,
        draws_per_epoch=draws,
        policies=args.policies,
        policy_configs=policy_configs,
        summaries={name: summarize_group(*group) for name, group in groups.items()},
        profiles={
            name: dict(Counter(r["profile"] for r in groups[name][0]))
            for name in args.policies
        },
        fallbacks={
            name: sum(r["fallback"] for r in groups[name][0]) for name in args.policies
        },
        cohort_summaries={
            name: {
                c: summarize_group(
                    [r for r in rows if r["cohort"] == c],
                    samples[np.array([r["cohort"] == c for r in rows])],
                )
                for c in sorted({r["cohort"] for r in rows})
            }
            for name, (rows, samples) in groups.items()
            if "/" not in name
        },
        spatial={
            name: {
                metric: dict(
                    zip(
                        ["p05", "p50", "p95"],
                        np.quantile(
                            [
                                r[metric]
                                if metric == "field_chi_rms_ratio"
                                else r["spatial"][metric]
                                for r in rows
                            ],
                            [0.05, 0.5, 0.95],
                        ).tolist(),
                        strict=True,
                    )
                )
                for metric in [
                    "gradient_rms",
                    "gradient_abs_p95",
                    "field_chi_rms_ratio",
                    "tail_energy_gt_01",
                    "tail_energy_gt_05",
                ]
            }
            for name, (rows, _) in groups.items()
        },
        external={name: group[0][0] for name, group in external.items()},
        metrics=compare(groups, external, output),
    )
    for name, (rows, _) in groups.items():
        result["spatial"][name]["radial_power_mean"] = np.mean(
            [row["spatial"]["radial_power"] for row in rows], axis=0
        ).tolist()
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "per_image.csv").open("w") as handle:
        flattened = [
            {
                **{
                    k: v
                    for k, v in row.items()
                    if k
                    not in ["stats", "spatial", "parameters", "histogram_distances"]
                },
                "parameters": json.dumps(row["parameters"], sort_keys=True),
                "radial_power": json.dumps(row["spatial"]["radial_power"]),
                **row["histogram_distances"],
                **{
                    f"{field}_{k}": v
                    for field in FIELDS
                    for k, v in row["stats"][field].items()
                },
                **{k: v for k, v in row["spatial"].items() if k != "radial_power"},
            }
            for row in all_rows
        ]
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    if examples:
        fig, axes = plt.subplots(
            len(examples),
            3,
            figsize=(10, 3 * len(examples)),
            squeeze=False,
            constrained_layout=True,
        )
        for row_axes, (profile, (row, slices, before)) in zip(
            axes, examples.items(), strict=True
        ):
            data = [
                before[0] if before is not None else slices[0],
                slices[0],
                slices[1],
            ]
            for ax, value, title, limit in zip(
                row_axes,
                data,
                ["A0 susceptibility", f"A3 {profile}", "New clean local field"],
                [0.1, 0.1, 0.03],
                strict=True,
            ):
                ax.imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit)
                ax.set_title(title)
                ax.axis("off")
        fig.savefig(output / "examples.png", dpi=150)
        plt.close(fig)
        np.savez_compressed(
            output / "examples.npz",
            **{
                f"{p}_{kind}": v
                for p, (_, slices, before) in examples.items()
                for kind, v in [
                    ("modified", slices),
                    ("baseline", before if before is not None else slices),
                ]
            },
        )
    plot(groups, external, result, output)
    write_report(result, output)
    print(f"Saved {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
