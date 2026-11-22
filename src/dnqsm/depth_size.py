"""Find the deepest measured U-Net at a fixed batch size using isolated probes."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import evaluation
from .data import BalancedSourceSampler, ShapeFieldSynthesizer, TrainingVolumes
from .logging import _pyplot, write_json
from .memory import ProcessMemoryMonitor
from .model import DNQSM
from .runtime import configure_inference, configure_memory_budget, require_device
from .training import (
    TrainConfig,
    _loader,
    backward_batch,
    build_datasets,
    build_shape_stress_dataset,
    model_config,
    normalization_config,
    set_training_branch,
    training_weights,
    validate,
)


def features_at_depth(base, depth):
    """Preserve all existing widths and double only added encoder levels."""
    if not base or depth < len(base) or any(type(w) is not int or w < 1 for w in base):
        raise ValueError("Depth must include all positive base feature widths")
    result = list(base)
    while len(result) < depth:
        result.append(result[-1] * 2)
    return tuple(result)


def deepest_fitting(probe, first, maximum):
    """Stop at the first failed next level; a user cap is explicitly bounded."""
    if first < 1 or maximum < first:
        raise ValueError("Invalid depth range")
    selected = None
    for depth in range(first, maximum + 1):
        if not probe(depth):
            if selected is None:
                raise RuntimeError("Even the starting depth exceeds the memory budget")
            return selected, False
        selected = depth
    return selected, True


def probe_depth(config, directory, fraction=0.98):
    """Exercise actual alternating updates with zero LR and resident Adam state.

    Uncropped full batches bound the crop/resize path. The last phase/chi pair
    includes gradient diagnostics after both networks have initialized moments.
    Validation and external inference keep the full optimizer in memory.
    """
    config.validate()
    if config.training_schedule != "alternating" or config.accumulation_steps != 1:
        raise ValueError("Depth probes require the alternating batch schedule")
    directory.mkdir(parents=True, exist_ok=False)
    device = require_device(config.device)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    total = torch.cuda.get_device_properties(device).total_memory
    budget = fraction * total / 1024**3
    configure_memory_budget(device, budget)
    result = dict(
        depth=len(config.features),
        features=list(config.features),
        batch_size=config.batch_size,
        model_config=asdict(model_config(config)),
        target_fraction=fraction,
        total_memory_bytes=total,
        cuda_memory_gib=budget,
        optimizer="AdamW, foreach=False, both networks resident, zero LR",
        learning_updates=0,
        schedule="phase, chi, phase, chi, phase with diagnostics, chi with diagnostics",
        steps=[],
    )
    started = time.perf_counter()
    stage = "initialization"
    try:
        with ProcessMemoryMonitor(device, fraction * total / 1e9) as monitor:
            training, validation, shape = build_datasets(config, directory)
            result.update(
                shape=list(shape),
                internal_shape=[max(n, 2 ** (len(config.features) + 1)) for n in shape],
            )
            model = DNQSM(model_config(config)).to(device)
            configure_inference(model, asdict(config))
            result["parameters"] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0, weight_decay=0, foreach=False
            )
            synthesizer = ShapeFieldSynthesizer(device)
            sampler = BalancedSourceSampler(
                training.records, config.synthetic_fraction, config.seed
            )
            sampler.set_epoch(config.filter_pretrain_epochs)
            training.set_epoch(config.filter_pretrain_epochs)
            training.crop_probability = 0.0
            indices = list(sampler)
            # Repeat real sampled batches if a small fixture has fewer than six.
            needed = 6 * config.batch_size
            indices = (indices * math.ceil(needed / len(indices)))[:needed]
            loader = _loader(training, replace(config, num_workers=0), sampler=indices)
            weights = training_weights(config)
            for index, batch in enumerate(loader):
                branch = "phase" if index % 2 == 0 else "chi"
                stage = f"{branch}_update_{index + 1}"
                set_training_branch(model, branch)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                step_started = time.perf_counter()
                terms, figure, count, _ = backward_batch(
                    model,
                    batch,
                    synthesizer,
                    device,
                    replace(weights, extension=0.0, local_tkd=0.0)
                    if branch == "chi"
                    else weights,
                    config,
                    config.batch_size,
                    "Depth memory probe" if index < 2 else None,
                    filter_only=branch == "phase",
                    chi_only=branch == "chi",
                    gradient_diagnostics=bool(config.gradient_every and index >= 4),
                    background_progress=1.0,
                )
                if count != config.batch_size:
                    raise AssertionError("Probe did not use a full nominal batch")
                if figure is not None:
                    _pyplot().close(figure)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                row = dict(
                    branch=branch,
                    batch_size=count,
                    loss=terms["loss"],
                    optimizer_states=len(optimizer.state),
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)
                    / 1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
                    seconds=time.perf_counter() - step_started,
                    **monitor.metrics(),
                )
                result["steps"].append(row)
                write_json(directory / "progress.json", dict(stage=stage, **row))
                print(
                    json.dumps(dict(stage=stage, depth=len(config.features), **row)),
                    flush=True,
                )
            result["all_adam_states_initialized"] = len(optimizer.state) == sum(
                1 for _ in model.parameters()
            )
            if not result["all_adam_states_initialized"]:
                raise AssertionError("Both networks' Adam state must be resident")
            suites = {"validation": validation}
            suites["validation_stress"] = TrainingVolumes(
                validation.records,
                config.cache_dir,
                validation=True,
                views=config.val_views,
                seed=config.seed + 100000,
                chi_scale_mode="legacy",
            )
            if config.chi_shape_validation:
                suites["validation_shape"] = build_shape_stress_dataset(
                    config, validation
                )
            for stage, dataset in suites.items():
                tail = len(dataset) % config.batch_size
                selected = list(range(min(config.batch_size, len(dataset))))
                if tail and len(dataset) > config.batch_size:
                    selected.extend(range(len(dataset) - tail, len(dataset)))
                _, figure = validate(
                    model,
                    _loader(dataset, replace(config, num_workers=0), sampler=selected),
                    synthesizer,
                    device,
                    weights,
                    config.bf16,
                    normalization=normalization_config(config),
                    synthesis_options=dict(background_mode="legacy")
                    if stage == "validation_stress"
                    else config.validation_synthesis,
                )
                if figure is not None:
                    _pyplot().close(figure)
                monitor.check()
            stage = "external_inference"
            cases = evaluation.load_cases(
                data_dir=config.evaluation_data_dir or None,
                cosmos_simulation=config.cosmos_simulation,
                sim2_pad=config.sim2_pad,
            )
            result["external_shapes"] = {
                c.name: list(c.phase.shape[-3:]) for c in cases
            }
            evaluation.evaluate_model(
                model,
                cases,
                directory / "external",
                config.device,
                bf16=config.bf16,
                export_volumes=False,
            )
            torch.cuda.synchronize(device)
            result.update(status="fits", **monitor.metrics())
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error), failed_stage=stage)
    except RuntimeError as error:
        if "Observed process VRAM reached" not in str(error):
            raise
        result.update(status="over_budget", error=str(error), failed_stage=stage)
    finally:
        result["seconds"] = time.perf_counter() - started
        write_json(directory / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--base-features", nargs="+", type=int, default=[23, 47, 90])
    parser.add_argument("--maximum-depth", type=int, default=9)
    parser.add_argument("--target-fraction", type=float, default=0.98)
    parser.add_argument("--probe-depth", type=int)
    args = parser.parse_args(argv)
    if not 0 < args.target_fraction <= 1 or args.batch_size < 1:
        parser.error("Invalid batch size or memory fraction")
    config = TrainConfig(**json.loads(args.config.read_text()))
    config = replace(
        config,
        batch_size=args.batch_size,
        init_checkpoint=None,
        resume=None,
        comet=False,
    )
    if config.cuda_allocator_config:
        os.environ["PYTORCH_ALLOC_CONF"] = config.cuda_allocator_config
    if args.probe_depth is not None:
        config.features = features_at_depth(args.base_features, args.probe_depth)
        result = probe_depth(config, args.directory, args.target_fraction)
        print(json.dumps(result), flush=True)
        return
    args.directory.mkdir(parents=True, exist_ok=False)
    write_json(args.directory / "config.json", asdict(config))
    results = []

    def probe(depth):
        command = [
            sys.executable,
            "-u",
            "-m",
            "dnqsm.depth_size",
            "--config",
            str(args.directory / "config.json"),
            "--directory",
            str(args.directory / f"depth_{depth:02d}"),
            "--probe-depth",
            str(depth),
            "--batch-size",
            str(args.batch_size),
            "--base-features",
            *map(str, args.base_features),
            "--target-fraction",
            str(args.target_fraction),
        ]
        with (args.directory / f"depth_{depth:02d}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(
            (args.directory / f"depth_{depth:02d}/result.json").read_text()
        )
        results.append(result)
        write_json(args.directory / "probes.json", results)
        print(json.dumps(result), flush=True)
        return result["status"] == "fits"

    selected, bounded = deepest_fitting(
        probe, len(args.base_features), args.maximum_depth
    )
    chosen = next(r for r in results if r["depth"] == selected)
    values = asdict(
        replace(
            config,
            features=features_at_depth(args.base_features, selected),
            cuda_memory_gb=chosen["cuda_memory_gib"],
        )
    )
    write_json(args.directory / "selected_config.json", values)
    write_json(
        args.directory / "summary.json",
        dict(
            selected_depth=selected,
            bounded_by_user_cap=bounded,
            selected=chosen,
            probes=results,
            family="Keep base widths, double each added encoder level; symmetric decoders",
        ),
    )


if __name__ == "__main__":
    main()
