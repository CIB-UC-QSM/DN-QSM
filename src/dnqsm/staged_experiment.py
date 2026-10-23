"""Matched joint/frozen-phase experiment from an immutable phase boundary."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .chi_augmentation import PROFILES
from .data import (
    BalancedSourceSampler,
    ShapeFieldSynthesizer,
    TrainingVolumes,
    shape_groups,
)
from .diagnostics import loss_gradient_metrics, quality_totals, summarize_quality
from .evaluation import load_cases
from .experiment_metrics import paired_summary, reconstruction_metrics, subject_average
from .logging import _pyplot, write_json
from .losses import normalize_loss_terms, per_sample_loss_terms
from .model import Reconstruction, UNet3D
from .physics import (
    apply_dipole,
    apply_multiplier,
    cutoff_kernels,
    masked_nrmse,
    masked_rms,
)
from .runtime import (
    CHECKPOINT_VERSION,
    atomic_save,
    autocast_context,
    configure_memory_budget,
    load_trained_model,
    require_device,
)
from .training import (
    TrainConfig,
    _loader,
    _restore_rng,
    _rng_state,
    background_options,
    build_shape_stress_dataset,
    normalization_config,
    training_weights,
)


def configure_arm(model, arm):
    if arm not in ("joint", "frozen"):
        raise ValueError("Unknown experiment arm")
    model.train()
    model.null_network.requires_grad_(True)
    model.filter.requires_grad_(arm == "joint")
    if arm == "frozen":
        model.filter.eval()


def forward_arm(model, data, arm):
    if arm == "joint":
        return model(data["phase"], data["mask"], data["dipole"])
    if arm != "frozen":
        raise ValueError("Unknown experiment arm")
    with torch.no_grad():
        local = model.predict_local_field(data["phase"], data["mask"]).detach()
        inverse, projection = cutoff_kernels(data["dipole"], model.config.threshold)
        initial = apply_multiplier(local, inverse)
        scale = masked_rms(initial, data["mask"], model.config.scale_floor)
    residual = model.null_network(initial / scale).float() * scale
    correction = apply_multiplier(residual, projection)
    return Reconstruction(local, initial, correction, initial + correction)


def phase_digest(model):
    digest = hashlib.sha256()
    for key, value in model.filter.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def draw_digest(batch):
    digest = hashlib.sha256()
    for shaped in shape_groups(batch):
        for key in (
            "subject",
            "seed",
            "chi_profile",
            "chi_parameters",
            "chi_rms_target",
            "crop_start",
            "crop_shape",
        ):
            value = shaped[key]
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            digest.update(json.dumps({key: value}, sort_keys=True).encode())
    return digest.hexdigest()


def arm_step(model, batch, synthesizer, optimizer, config, arm, *, diagnostics=False):
    configure_arm(model, arm)
    optimizer.zero_grad(set_to_none=True)
    totals = defaultdict(float)
    count_total = sum(group["chi"].shape[0] for group in shape_groups(batch))
    weights = training_weights(config)
    for group in shape_groups(batch):
        data = synthesizer(group, **background_options(config, 1.0))
        with autocast_context(data["chi"].device, config.bf16):
            output = forward_arm(model, data, arm)
        predicted_field = apply_dipole(output.chi, data["dipole"])
        raw = per_sample_loss_terms(
            output,
            data["local"],
            data["chi"],
            data["mask"],
            data["dipole"],
            predicted_field=predicted_field,
            include_extension=False,
            include_gradient=bool(weights.local_gradient),
        )
        normalized = normalize_loss_terms(
            raw,
            data["local"],
            data["chi"],
            data["mask"],
            normalization_config(config),
            input_field=data["phase"],
        )
        terms = {name: value.mean() for name, value in normalized.items()}
        loss = sum(getattr(weights, name) * value for name, value in terms.items())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite experimental training loss")
        fraction = len(data["chi"]) / count_total
        if diagnostics:
            totals.update(
                loss_gradient_metrics(
                    model, {k: v for k, v in terms.items() if v.requires_grad}, weights
                )
            )
            diagnostics = False
        (loss * fraction).backward()
        for key, value in dict(loss=loss, **terms).items():
            totals[key] += float(value.detach()) * fraction
        for key, value in quality_totals(
            output, data, predicted_field=predicted_field
        ).items():
            totals[key] += value
        for profile in PROFILES:
            totals[f"chi_{profile}_samples"] += group["chi_profile"].count(profile)
        del data, output, predicted_field, raw, normalized, terms, loss
    for name, module in (("phase", model.filter), ("null", model.null_network)):
        parameters = [p for p in module.parameters() if p.requires_grad]
        norm = (
            float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, config.grad_clip, error_if_nonfinite=True
                )
            )
            if parameters
            else 0.0
        )
        totals[name + "_gradient_norm"] = norm
        totals[name + "_gradient_clipped"] = int(norm > config.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return summarize_quality(totals)


@torch.inference_mode()
def evaluate_suite(model, dataset, config, synthesizer, synthesis_options=None):
    model.eval()
    rows = []
    for batch in _loader(dataset, config):
        data = synthesizer(
            batch,
            **(
                synthesis_options
                or config.validation_synthesis
                or background_options(config)
            ),
        )
        with autocast_context(data["chi"].device, config.bf16):
            output = model(data["phase"], data["mask"], data["dipole"])
        values = reconstruction_metrics(
            output,
            data["chi"],
            data["mask"],
            data["dipole"],
            data["local"],
            threshold=model.config.threshold,
        )
        rows.extend(
            dict(subject=subject, profile=profile, seed=int(seed), **metrics)
            for subject, profile, seed, metrics in zip(
                batch["subject"],
                batch["chi_profile"],
                batch["seed"],
                values,
                strict=True,
            )
        )
        del data, output
    subjects = subject_average(rows)
    means = {
        name: float(
            np.mean([r[name] for r in subjects.values() if r[name] is not None])
        )
        if any(r[name] is not None for r in subjects.values())
        else None
        for name in next(iter(subjects.values()))
    }
    return dict(mean=means, subjects=subjects, views=rows)


@torch.inference_mode()
def evaluate_external(model, cases, config, device):
    model.eval()
    results = {}
    for case in cases:
        phase, mask, dipole, chi = (
            x.to(device) for x in (case.phase, case.mask, case.dipole, case.chi)
        )
        with autocast_context(device, config.bf16):
            output = model(phase, mask, dipole)
        local = case.local.to(device) if case.local is not None else None
        results[case.name] = reconstruction_metrics(
            output, chi, mask, dipole, local, threshold=model.config.threshold
        )[0]
        del phase, mask, dipole, chi, output, local
    return results


def study_datasets(study, config, seed):
    split = json.loads((study / "split.json").read_text())
    training = TrainingVolumes(
        split["train"],
        config.cache_dir,
        seed=seed,
        chi_scale_mode="rms_mixture",
        crop_probability=config.crop_probability,
        crop_resize=config.crop_resize,
        chi_augmentation=config.chi_augmentation,
    )
    representatives = {r["subject"]: r for r in reversed(split["validation"])}
    records = [representatives[s] for s in sorted(representatives)]
    validation = TrainingVolumes(
        records,
        config.cache_dir,
        validation=True,
        views=config.val_views,
        seed=config.seed + 100000,
        chi_scale_mode="rms_mixture",
    )
    stress = TrainingVolumes(
        records,
        config.cache_dir,
        validation=True,
        views=config.val_views,
        seed=config.seed + 100000,
        chi_scale_mode="legacy",
    )
    return training, {
        "validation": validation,
        "validation_stress": stress,
        "validation_shape": build_shape_stress_dataset(config, validation),
    }


def make_anchor(study, seed, config):
    directory = study / f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    anchor = directory / "anchor.pt"
    if anchor.is_file():
        return anchor
    model = load_trained_model(study / "phase_pretrain20.pt", "cpu")
    torch.manual_seed(seed)
    model.null_network = UNet3D(
        model.config.features,
        model.config.checkpoint_blocks,
        activation=model.config.activation,
        leaky_relu_slope=model.config.leaky_relu_slope,
    )
    atomic_save(
        dict(
            version=CHECKPOINT_VERSION,
            model_config=asdict(model.config),
            model=model.state_dict(),
            seed=seed,
            phase="shared_phase20_fresh_null",
        ),
        anchor,
    )
    return anchor


def run_arm(study, seed, arm, config, manifest, device, cases):
    directory = study / f"seed_{seed}" / arm
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"step_{manifest['updates_per_arm']:04d}.json"
    if (
        final.is_file()
        and (directory / "status.json").is_file()
        and json.loads((directory / "status.json").read_text())["state"] == "complete"
    ):
        return json.loads(final.read_text())
    anchor = make_anchor(study, seed, config)
    model = load_trained_model(anchor, device)
    initial_phase_digest = phase_digest(model)
    configure_arm(model, arm)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if device.type == "cuda" else None
    dataset, suites = study_datasets(study, config, seed)
    sampler = BalancedSourceSampler(dataset.records, config.synthetic_fraction, seed)
    synthesizer = ShapeFieldSynthesizer(device)
    per_epoch = math.ceil(len(dataset) / config.batch_size)
    updates = manifest["updates_per_arm"]
    step = 0
    checkpoint = directory / "training_last.pt"
    if checkpoint.is_file():
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"])
        step = saved["step"]
        del saved
    write_json(
        directory / "config.json",
        dict(
            training=asdict(config),
            arm=arm,
            paired_seed=seed,
            phase_initial_digest=initial_phase_digest,
            updates=updates,
            learning_rate_schedule="20-update warmup then constant",
            data_epoch_offset=20,
            clipping="per-network",
            background_progress=1.0,
            anchor=str(anchor),
        ),
    )
    probe_batch = default_collate(
        [
            suites["validation"][0],
            suites["validation"][min(3, len(suites["validation"]) - 1)],
        ]
    )
    probe = synthesizer(
        probe_batch, **(config.validation_synthesis or background_options(config))
    )
    probe_input = probe["phase"].cpu(), probe["mask"].cpu()
    del probe
    baseline_probe_path = directory.parent / "phase_probe.pt"
    if baseline_probe_path.is_file():
        baseline_probe = torch.load(
            baseline_probe_path, weights_only=True, map_location="cpu"
        )
    else:
        with torch.inference_mode(), autocast_context(device, config.bf16):
            model.eval()
            baseline_probe = model.predict_local_field(
                probe_input[0].to(device), probe_input[1].to(device)
            ).cpu()
        atomic_save({"local": baseline_probe}, baseline_probe_path)
        baseline_probe = {"local": baseline_probe}

    def save():
        atomic_save(
            dict(
                version=CHECKPOINT_VERSION,
                model_config=asdict(model.config),
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step,
                rng=_rng_state(),
                arm=arm,
                seed=seed,
            ),
            checkpoint,
        )

    def evaluate():
        torch.cuda.empty_cache() if device.type == "cuda" else None
        results = {
            name: evaluate_suite(
                model,
                data,
                config,
                synthesizer,
                dict(background_mode="legacy") if name == "validation_stress" else None,
            )
            for name, data in suites.items()
        }
        external = evaluate_external(model, cases, config, device)
        with torch.inference_mode(), autocast_context(device, config.bf16):
            model.eval()
            probe_prediction = (
                model.predict_local_field(
                    probe_input[0].to(device), probe_input[1].to(device)
                )
                .float()
                .cpu()
            )
        drift = masked_nrmse(probe_prediction, baseline_probe["local"], probe_input[1])
        current_digest = phase_digest(model)
        if arm == "frozen" and current_digest != initial_phase_digest:
            raise AssertionError("Frozen phase parameters changed")
        if arm == "frozen" and max(x for x in drift if x is not None) > 1e-4:
            raise AssertionError("Frozen phase predictions changed")
        result = dict(
            seed=seed,
            arm=arm,
            step=step,
            suites=results,
            external=external,
            phase_prediction_drift_nrmse=drift,
            phase_digest=current_digest,
        )
        write_json(directory / f"step_{step:04d}.json", result)
        write_json(
            directory / "status.json",
            dict(state="evaluated", seed=seed, arm=arm, step=step),
        )
        print(
            json.dumps(
                dict(
                    event="evaluation",
                    seed=seed,
                    arm=arm,
                    step=step,
                    chi_nrmse=results["validation"]["mean"]["chi_nrmse"],
                    local_nrmse=results["validation"]["mean"]["local_nrmse"],
                    external=external,
                )
            ),
            flush=True,
        )
        return result

    if not (directory / f"step_{step:04d}.json").is_file():
        evaluate()
    with (directory / "train.jsonl").open("a", buffering=1) as log:
        while step < updates:
            epoch = step // per_epoch
            dataset.set_epoch(20 + epoch)
            sampler.set_epoch(20 + epoch)
            indices = list(sampler)[(step % per_epoch) * config.batch_size :]
            loader = _loader(dataset, config, sampler=indices, epoch=20 + epoch)
            for batch in loader:
                started = time.perf_counter()
                rate = config.learning_rate * min(1.0, (step + 1) / 20)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                metrics = arm_step(
                    model,
                    batch,
                    synthesizer,
                    optimizer,
                    config,
                    arm,
                    diagnostics=step in (0, 99, 199, 299),
                )
                step += 1
                event = dict(
                    event="train",
                    arm=arm,
                    seed=seed,
                    step=step,
                    draw_digest=draw_digest(batch),
                    learning_rate=rate,
                    seconds=time.perf_counter() - started,
                    **metrics,
                )
                log.write(json.dumps(event) + "\n")
                write_json(
                    directory / "status.json",
                    dict(state="training", arm=arm, seed=seed, step=step),
                )
                if step % 10 == 0:
                    print(
                        json.dumps(
                            {
                                k: event[k]
                                for k in (
                                    "event",
                                    "arm",
                                    "seed",
                                    "step",
                                    "loss",
                                    "seconds",
                                )
                            }
                        ),
                        flush=True,
                    )
                if (
                    step % 50 == 0
                    or step % manifest["eval_every"] == 0
                    or step == updates
                ):
                    save()
                if step % manifest["eval_every"] == 0 or step == updates:
                    evaluate()
                if step == updates:
                    break
    end = json.loads((directory / f"step_{updates:04d}.json").read_text())
    atomic_save(
        dict(
            version=CHECKPOINT_VERSION,
            model_config=asdict(model.config),
            model=model.state_dict(),
            step=step,
            seed=seed,
            arm=arm,
        ),
        directory / "model_final.pt",
    )
    write_json(
        directory / "status.json", dict(state="complete", seed=seed, arm=arm, step=step)
    )
    model = optimizer = synthesizer = None
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return end


def verify_draws(study, seed):
    logs = {}
    for arm in ("joint", "frozen"):
        rows = [
            json.loads(x)
            for x in (study / f"seed_{seed}" / arm / "train.jsonl")
            .read_text()
            .splitlines()
        ]
        logs[arm] = {row["step"]: row["draw_digest"] for row in rows}
    if logs["joint"] != logs["frozen"]:
        raise AssertionError("The two arms did not use identical training draws")
    expected = json.loads(
        (study / f"seed_{seed}" / "joint" / "config.json").read_text()
    )["updates"]
    if set(logs["joint"]) != set(range(1, expected + 1)):
        raise AssertionError(
            "Training draw logs do not cover the complete update budget"
        )


def summarize(study, results, manifest):
    pairs = {}
    for seed, arms in results.items():
        verify_draws(study, seed)
        pairs[str(seed)] = {
            suite: paired_summary(
                arms["joint"]["suites"][suite]["subjects"],
                arms["frozen"]["suites"][suite]["subjects"],
            )
            for suite in arms["joint"]["suites"]
        }
    mean_delta = float(
        np.mean(
            [p["validation"]["delta_joint_minus_frozen_pp"] for p in pairs.values()]
        )
    )
    summary = dict(
        pairs=pairs,
        mean_delta_joint_minus_frozen_pp=mean_delta,
        primary="validation chi_nrmse, subject averaged",
        full_results=results,
        mode=manifest["mode"],
        updates_per_arm=manifest["updates_per_arm"],
        interpretation="Positive delta means joint training is worse at this fixed update budget.",
    )
    write_json(study / "summary.json", summary)
    lines = [
        "# Joint versus frozen phase: matched experiment",
        "",
        "Results below cover completed paired seeds. Check manifest.json for the overall study state.",
        "Shared 20-epoch phase refinement, followed by fresh, paired Deep Null initialization.",
        f"Each arm uses {manifest['updates_per_arm']} updates, matched data draws, fresh AdamW, and per-network clipping.",
        "",
        "| Seed | Joint χ NRMSE | Frozen χ NRMSE | Joint − frozen (pp) | Paired subject bootstrap 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for seed, suites in pairs.items():
        p = suites["validation"]
        lines.append(
            f"| {seed} | {p['joint_chi_nrmse']:.3f}% | {p['frozen_chi_nrmse']:.3f}% | {p['delta_joint_minus_frozen_pp']:+.3f} | {p['paired_subject_bootstrap_95ci']} |"
        )
    lines += [
        "",
        f"Mean paired effect: **{mean_delta:+.3f} percentage points** (joint minus frozen).",
        "",
        "Positive means joint is worse. These results concern this update budget and initialization, not the asymptotic winner.",
        "The phase stage was warm-started from the earlier interrupted model; Deep Null was freshly initialized at the fork.",
        "External cases were evaluation-only in this comparison. sim2 has no independent local-field reference.",
        "Per-view/subject secondary metrics, Fourier-band errors, bias, slope, HFEN, and external scores are saved in summary.json.",
        "Intervals resample subjects within each paired seed. Views and voxels are not independent replicates.",
        "",
    ]
    lines += [
        "## Intermediate and external outcomes",
        "",
        "| Seed | Arm | Local NRMSE | TKD NRMSE | χ NRMSE | HFEN | COSMOS χ NRMSE | sim2 χ NRMSE | Phase-output drift |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, arms in results.items():
        for arm, result in arms.items():
            v = result["suites"]["validation"]["mean"]
            lines.append(
                f"| {seed} | {arm} | {v['local_nrmse']:.3f} | {v['tkd_nrmse']:.3f} | {v['chi_nrmse']:.3f} | {v['hfen_percent']:.3f} | {result['external']['cosmos']['chi_nrmse']:.3f} | {result['external']['sim2']['chi_nrmse']:.3f} | {np.mean(result['phase_prediction_drift_nrmse']):.6f} |"
            )
    deltas = [p["validation"]["delta_joint_minus_frozen_pp"] for p in pairs.values()]
    threshold = manifest.get("repeat_threshold_pp", 1.0)
    if len(deltas) == 1:
        interpretation = (
            "Single paired pilot; seed variability has not been established."
        )
    elif abs(mean_delta) >= threshold and all(
        np.sign(x) == np.sign(mean_delta) for x in deltas
    ):
        interpretation = (
            "Joint training is worse" if mean_delta > 0 else "Joint training is better"
        ) + " across the paired seeds at this budget."
    else:
        interpretation = "The effect is small or seed-dependent; no consistent practical winner is established."
    lines += [
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "A worse local-field score alone does not establish worse final χ reconstruction.",
        "![Learning curves](learning_curves.png)",
        "",
    ]
    (study / "report.md").write_text("\n".join(lines))
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed in results:
        for arm, linestyle in (("joint", "-"), ("frozen", "--")):
            paths = sorted((study / f"seed_{seed}" / arm).glob("step_*.json"))
            rows = [json.loads(p.read_text()) for p in paths]
            for ax, metric in zip(axes, ("chi_nrmse", "local_nrmse"), strict=True):
                ax.plot(
                    [r["step"] for r in rows],
                    [r["suites"]["validation"]["mean"][metric] for r in rows],
                    linestyle,
                    marker=".",
                    label=f"{arm}, seed {seed}",
                )
                ax.set(xlabel="Optimizer updates", ylabel=metric + " (%)")
    axes[0].legend(fontsize=8)
    fig.savefig(study / "learning_curves.png", dpi=160)
    plt.close(fig)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args(argv)
    study = args.study.resolve()
    # The OS releases the lock after an interruption, allowing continuation.
    runner_lock = (study / ".runner.lock").open("a+")
    try:
        fcntl.flock(runner_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This study already has an active experiment runner") from None
    manifest = json.loads((study / "manifest.json").read_text())
    config_values = json.loads(Path(manifest["pretrain_config"]).read_text())
    config_values.setdefault("lambda_local_gradient", 0.0)
    config = TrainConfig(**config_values)
    config.validate()
    if config.accumulation_steps != 1 or config.lambda_extension:
        raise ValueError("The study requires accumulation=1 and no extension loss")
    if config.cuda_allocator_config:
        os.environ["PYTORCH_ALLOC_CONF"] = config.cuda_allocator_config
    device = require_device(config.device)
    configure_memory_budget(device, config.cuda_memory_gb)
    torch.set_num_threads(config.threads)
    boundary = study / "phase_pretrain20.pt"
    if not boundary.is_file():
        source = study / "pretrain/model_last.pt"
        saved = torch.load(source, weights_only=True, map_location="cpu", mmap=True)
        if (
            saved["step"] != manifest["pretrain_updates"]
            or saved["phase"] != "filter_pretrain"
        ):
            raise ValueError("The shared phase pretraining boundary is not complete")
        atomic_save(saved, boundary)
    cases = load_cases(
        data_dir=config.evaluation_data_dir,
        cosmos_simulation=config.cosmos_simulation,
        sim2_pad=config.sim2_pad,
    )
    results = {}
    manifest.update(state="paired_experiments")
    write_json(study / "manifest.json", manifest)
    try:
        for position, seed in enumerate(manifest["paired_seeds"]):
            order = ("joint", "frozen") if position % 2 == 0 else ("frozen", "joint")
            results[seed] = {
                arm: run_arm(study, seed, arm, config, manifest, device, cases)
                for arm in order
            }
            summary = summarize(study, results, manifest)
            if (
                position == 0
                and abs(summary["mean_delta_joint_minus_frozen_pp"])
                < manifest["repeat_threshold_pp"]
            ):
                manifest["repeat_decision"] = (
                    "Below prespecified 1 pp threshold; pilot complete, no additional seeds"
                )
                break
        manifest.update(
            state="complete",
            completed_seeds=list(results),
            summary=str(study / "summary.json"),
        )
        write_json(study / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(state="failed", error=repr(error))
        write_json(study / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
