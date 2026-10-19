"""Measure real training batches in isolated CUDA processes before a long run."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path

import torch

from . import evaluation
from .data import (
    BalancedSourceSampler,
    ShapeFieldSynthesizer,
    TrainingVolumes,
    shape_groups,
)
from .logging import _pyplot, write_json
from .losses import LOSS_VERSION
from .memory import ProcessMemoryMonitor
from .model import DNQSM
from .runtime import configure_memory_budget, require_device
from .training import (
    TrainConfig,
    _loader,
    background_options,
    backward_batch,
    build_datasets,
    model_config,
    normalization_config,
    training_weights,
    validate,
)


def largest_fitting_batch(probe, maximum: int = 128) -> int:
    """Bracket then binary-search the largest measured stable integer batch."""
    if maximum < 1:
        raise ValueError("Maximum batch size must be positive")
    low, high = 0, 1
    while probe(high):
        low = high
        if high == maximum:
            return high
        high = min(high * 2, maximum)
    while high - low > 1:
        middle = (low + high) // 2
        if probe(middle):
            low = middle
        else:
            high = middle
    if not low:
        raise RuntimeError("Even batch size 1 exceeds the memory target")
    return low


def probe_batch(config: TrainConfig, directory: Path, fraction: float) -> dict:
    """Include field synthesis, backward, initialized Adam, figures and validation.

    Zero learning rate keeps the probe from performing training. Each candidate
    runs in a new process so allocator history and failed allocations cannot
    influence subsequent measurements.
    """
    directory.mkdir(parents=True, exist_ok=False)
    device = require_device(config.device)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    total = torch.cuda.get_device_properties(device).total_memory
    budget = fraction * total / 1024**3
    configure_memory_budget(device, budget)
    result = dict(
        batch_size=config.batch_size,
        target_fraction=fraction,
        total_memory_bytes=total,
        cuda_memory_gib=budget,
        model_config=asdict(model_config(config)),
        optimizer_foreach=False,
        allocator_config=os.environ.get("PYTORCH_ALLOC_CONF", ""),
        learning_updates=0,
        training_repeats=3,
        crop_probability=config.crop_probability,
        crop_resize=config.crop_resize,
        loss_version=LOSS_VERSION,
        normalize_loss=config.normalize_loss,
        background_profile=config.background_profile,
        gradient_every=config.gradient_every,
    )
    started = time.perf_counter()
    try:
        with ProcessMemoryMonitor(device, fraction * total / 1e9) as monitor:
            training, validation, shape = build_datasets(config, directory)
            result["shape"] = list(shape)
            synthesizer = ShapeFieldSynthesizer(device)
            model = DNQSM(model_config(config)).to(device).train()
            result["parameters"] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0, weight_decay=0, foreach=False
            )
            sampler = BalancedSourceSampler(
                training.records, config.synthetic_fraction, config.seed
            )
            weights = training_weights(config)
            tested_shapes = set()

            def step_batch(batch, title=None, *, diagnostics=False):
                groups = shape_groups(batch)
                tested_shapes.update(tuple(group["chi"].shape[-3:]) for group in groups)
                count = sum(group["chi"].shape[0] for group in groups)
                _, figure, _, _ = backward_batch(
                    model,
                    batch,
                    synthesizer,
                    device,
                    weights,
                    config,
                    count,
                    title,
                    gradient_diagnostics=bool(diagnostics and config.gradient_every),
                )
                if figure is not None:
                    _pyplot().close(figure)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                monitor.check()

            # Worst case is an entire uncropped nominal batch, even though
            # actual training crops each sample independently with p=0.2.
            training.crop_probability = 0.0
            loader = _loader(training, replace(config, num_workers=0), sampler)
            for index, batch in enumerate(islice(loader, 3)):
                # Probe gradients with Adam moments already resident. A fresh
                # process can fit a larger batch only before optimizer init.
                step_batch(
                    batch,
                    "Batch memory probe" if index == 0 else None,
                    diagnostics=index == 1,
                )
                print(f"batch={config.batch_size} repeat={index + 1}/3", flush=True)
            del loader, batch
            if config.crop_probability:
                training.crop_probability = config.crop_probability
                for batch in islice(
                    _loader(training, replace(config, num_workers=0), sampler), 2
                ):
                    step_batch(batch)
                # Guarantee crop/resize augmentation is exercised even at B=1.
                training.crop_probability = 1.0
                step_batch(
                    next(
                        iter(
                            _loader(
                                training, replace(config, batch_size=1, num_workers=0)
                            )
                        )
                    )
                )
            training.crop_probability = config.crop_probability
            tail = len(training) % config.batch_size
            if tail:
                step_batch(
                    next(
                        iter(
                            _loader(
                                training,
                                replace(config, num_workers=0),
                                sampler=list(sampler)[-tail:],
                            )
                        )
                    )
                )
            result["tested_native_shapes"] = [list(s) for s in sorted(tested_shapes)]
            # Cover full and final partial validation batches with Adam retained.
            validation_indices = list(range(min(config.batch_size, len(validation))))
            tail = len(validation) % config.batch_size
            if tail and len(validation) > config.batch_size:
                validation_indices.extend(
                    range(len(validation) - tail, len(validation))
                )
            _, figure = validate(
                model,
                _loader(
                    validation,
                    replace(config, num_workers=0),
                    sampler=validation_indices,
                ),
                synthesizer,
                device,
                weights,
                config.bf16,
                normalization=normalization_config(config),
                synthesis_options=background_options(config),
            )
            if figure is not None:
                _pyplot().close(figure)
            stress = TrainingVolumes(
                validation.records,
                config.cache_dir,
                validation=True,
                views=config.val_views,
                seed=config.seed + 100000,
            )
            _, figure = validate(
                model,
                _loader(
                    stress, replace(config, num_workers=0), sampler=validation_indices
                ),
                synthesizer,
                device,
                weights,
                config.bf16,
                normalization=normalization_config(config),
                synthesis_options=dict(background_mode="legacy"),
            )
            if figure is not None:
                _pyplot().close(figure)
            torch.cuda.empty_cache()
            evaluation.evaluate_model(
                model,
                evaluation.load_cases(
                    data_dir=config.evaluation_data_dir or None,
                    cosmos_simulation=config.cosmos_simulation,
                    sim2_pad=config.sim2_pad,
                ),
                directory / "external",
                config.device,
                bf16=config.bf16,
                export_volumes=False,
            )
            torch.cuda.synchronize(device)
            result.update(status="fits", **monitor.metrics())
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    finally:
        result.update(
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
            seconds=time.perf_counter() - started,
        )
        if "peak_process_vram_gb" in result:
            result["peak_process_vram_percent"] = (
                100 * result["peak_process_vram_gb"] * 1e9 / total
            )
        write_json(directory / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--target-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-batch", type=int, default=128)
    parser.add_argument("--probe-batch", type=int)
    parser.add_argument("--expandable-segments", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--features", type=int, nargs="+")
    parser.add_argument("--crop-probability", type=float)
    parser.add_argument(
        "--crop-resize", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args(argv)
    if not 0 < args.target_fraction < 1:
        parser.error("Target fraction must be between zero and one")
    config = TrainConfig(
        features=(23, 47, 90),
        residual_filter=False,
        accumulation_steps=1,
        crop_probability=0.2,
    )
    values = asdict(config)
    if args.config:
        values.update(json.loads(args.config.read_text()))
    if args.features:
        values["features"] = tuple(args.features)
    if args.crop_probability is not None:
        values["crop_probability"] = args.crop_probability
    if args.crop_resize is not None:
        values["crop_resize"] = args.crop_resize
    if args.expandable_segments:
        values["cuda_allocator_config"] = "expandable_segments:True"
    config = TrainConfig(**values)
    config.validate()
    if config.cuda_allocator_config:
        os.environ["PYTORCH_ALLOC_CONF"] = config.cuda_allocator_config
    if args.probe_batch:
        result = probe_batch(
            replace(config, batch_size=args.probe_batch),
            args.directory,
            args.target_fraction,
        )
        print(json.dumps(result), flush=True)
        return
    args.directory.mkdir(parents=True, exist_ok=False)
    write_json(args.directory / "config.json", asdict(config))
    results = []

    def probe(batch):
        directory = args.directory / f"batch_{batch:03d}"
        command = [
            sys.executable,
            "-m",
            "dnqsm.batch_size",
            "--directory",
            str(directory),
            "--target-fraction",
            str(args.target_fraction),
            "--probe-batch",
            str(batch),
            "--config",
            str(args.directory / "config.json"),
        ]
        with (args.directory / f"batch_{batch:03d}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((directory / "result.json").read_text())
        results.append(result)
        print(json.dumps(result), flush=True)
        return result["status"] == "fits"

    selected = largest_fitting_batch(probe, args.maximum_batch)
    chosen = next(row for row in results if row["batch_size"] == selected)
    write_json(
        args.directory / "summary.json",
        dict(selected_batch_size=selected, selected=chosen, probes=results),
    )
    print(f"Selected batch size: {selected}", flush=True)


if __name__ == "__main__":
    main()
