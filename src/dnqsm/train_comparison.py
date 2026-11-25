"""Train the eight-arm DN-QSM comparison on materialized AHEAD patches."""

import argparse
import hashlib
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from math import isfinite
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image

from .baselines import INVERSES, DirectSusceptibility, LocalFieldBaseline
from .inference import SlidingWindowConfig
from .logging import _pyplot, comparison_figure, write_json
from .model import DEFAULT_FEATURES, DNQSM, LocalFieldEstimator, ModelConfig
from .patch_data import PATCH_SIZE, make_patch_dataloader
from .physics import masked_mean
from .runtime import (
    CHECKPOINT_VERSION,
    atomic_save,
    autocast_context,
    configure_memory_budget,
    require_device,
)

LEGACY_MANIFEST_VERSION = 1
MANIFEST_VERSION = 2
DIRECT = "model1"
DNQSM_ARM = "model4"


@dataclass(frozen=True)
class TrainingConfig:
    steps: int | None = 10_000
    validation_interval: int = 1_000
    batch_size: int = 2
    workers: int = 0
    seed: int = 2026
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    bf16: bool = True
    epochs: int | None = None


def _start_experiment(output, stage, comparison_tag, *, archive_dir=None):
    # PyTorch is already loaded. Use Comet's manual logging mode at SDK import.
    setting = "COMET_DISABLE_AUTO_LOGGING"
    previous = os.environ.get(setting)
    os.environ[setting] = "1"
    try:
        import comet_ml
        from comet_ml.config import get_config
    finally:
        if previous is None:
            os.environ.pop(setting, None)
        else:
            os.environ[setting] = previous

    config = get_config()
    online = bool(config["comet.api_key"]) and config["comet.start.online"] is not False
    return comet_ml.start(
        mode="create",
        online=online,
        project_name=config["comet.project_name"] or "dn-qsm",
        experiment_key=uuid4().hex if config["comet.experiment_key"] else None,
        experiment_config=comet_ml.ExperimentConfig(
            name=f"{output.name}-{stage}",
            tags=[comparison_tag],
            offline_directory=str(archive_dir or output / "comet"),
            auto_metric_logging=False,
            auto_param_logging=False,
            disabled=False,
        ),
    )


def _predict_target(model, batch, target_name, dipole, device, bf16):
    total, mask, target = (
        batch[key].to(device) for key in ("total_field", "mask", target_name)
    )
    with autocast_context(device, bf16):
        if target_name == "local_field":
            prediction = model.predict_local_field(total, mask)
        else:
            prediction = model(total, mask, dipole).chi
    return total, mask, target, prediction.float()


def _relative_error(prediction, target, mask):
    error = masked_mean((prediction - target).square(), mask)
    energy = masked_mean(target.square(), mask)
    return error.flatten(), energy.flatten()


def _optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _update(model, optimizer, batch, target_name, dipole, device, bf16):
    optimizer.zero_grad(set_to_none=True)
    _, mask, target, prediction = _predict_target(
        model, batch, target_name, dipole, device, bf16
    )
    error, energy = _relative_error(prediction, target, mask)
    loss = (error / energy.clamp_min(1e-8)).mean()
    value = loss.item()
    if not isfinite(value):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    optimizer.step()
    return value


def _probe_batch_size(model, target, dataset, stage, device, config):
    """Find the largest batch that fits the configured CUDA allocator budget."""
    data = make_patch_dataloader(dataset, seed=config.seed).dataset
    dipole = data.dipole(device)
    initial = {
        name: tensor.detach().to("cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }
    model.train()

    def trial(size):
        loader = make_patch_dataloader(dataset, batch_size=size, seed=config.seed)
        batch = next(iter(loader))
        optimizer = _optimizer(model, config)
        for _ in range(2):
            _update(model, optimizer, batch, target, dipole, device, config.bf16)
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_reserved(device) / 1024**3

    def measure(size):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            peak = trial(size)
        except torch.cuda.OutOfMemoryError:
            peak = None
        finally:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(initial)
            torch.cuda.empty_cache()
        result = "OOM" if peak is None else f"{peak:.2f} GiB reserved"
        print(f"{stage} batch probe {size}: {result}", flush=True)
        return peak

    best, peak = 0, None
    candidate = 1
    while True:
        result = measure(candidate)
        if result is None:
            break
        best, peak = candidate, result
        if best == len(data):
            return best, peak
        candidate = min(2 * candidate, len(data))
    if best == 0:
        raise RuntimeError(f"{stage} cannot fit a batch of one")
    while candidate - best > 1:
        middle = (best + candidate) // 2
        result = measure(middle)
        if result is None:
            candidate = middle
        else:
            best, peak = middle, result
    return best, peak


@torch.no_grad()
def _validate(model, loader, target_name, dipole, device, bf16):
    model.eval()
    totals = defaultdict(float)
    counts = defaultdict(int)
    preview = None
    for batch in loader:
        total, mask, target, prediction = _predict_target(
            model, batch, target_name, dipole, device, bf16
        )
        error, energy = _relative_error(prediction, target, mask)
        if not bool(torch.isfinite(error).all() & torch.isfinite(energy).all()):
            raise FloatingPointError("Nonfinite validation prediction or target")
        if preview is None:
            roi = mask[0].bool()
            preview = tuple(
                torch.where(roi, volume[0], 0).cpu()
                for volume in (total, target, prediction)
            )
        scores = 100 * (error / energy.clamp_min(torch.finfo(energy.dtype).tiny)).sqrt()
        for subject, score, valid in zip(
            batch["source_id"], scores.tolist(), (energy > 0).tolist(), strict=True
        ):
            if valid:
                totals[subject] += score
                counts[subject] += 1
    if not counts:
        raise ValueError(f"No validation patches have nonzero {target_name} energy")
    score = sum(totals[key] / count for key, count in counts.items()) / len(counts)
    return score, preview


def _log_validation_image(experiment, preview, target_name, step):
    plt = _pyplot()
    from .patch_qc import DISPLAY_LIMIT

    susceptibility = target_name == "chi"
    target_label = "Susceptibility" if susceptibility else "Local field"
    total, target, prediction = preview
    figure = comparison_figure(
        [
            ("Total field", total, False),
            (f"{target_label} reference", target, susceptibility),
            (f"{target_label} prediction", prediction, susceptibility),
        ],
        f"Validation step {step} • grayscale ±{DISPLAY_LIMIT:g} ppm",
        display_limit=DISPLAY_LIMIT,
    )
    try:
        with BytesIO() as buffer:
            figure.savefig(buffer, format="png", dpi=100)
            buffer.seek(0)
            with Image.open(buffer) as image:
                experiment.log_image(
                    image_data=image,
                    name=f"central-slices-step-{step:06d}.png",
                    step=step,
                )
    finally:
        plt.close(figure)


def _fit(
    model,
    target_name,
    dataset,
    validation,
    output,
    stage,
    device,
    config,
    checkpoint_metadata,
    experiment,
):
    train = make_patch_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.workers,
        seed=config.seed,
    )
    dipole = train.dataset.dipole(device)
    optimizer = _optimizer(model, config)
    steps_per_epoch = len(train)
    if not steps_per_epoch:
        raise ValueError("Training split has no batches")
    if config.epochs is None:
        steps = config.steps
        validation_interval = config.validation_interval
    else:
        steps = config.epochs * steps_per_epoch
        validation_interval = steps_per_epoch
        experiment.log_parameters(
            {"steps_per_epoch": steps_per_epoch, "total_optimizer_steps": steps}
        )
    batches = iter(train)
    history = []
    best = float("inf")
    checkpoint = output / f"{stage}.pt"

    model.train()
    for step in range(1, steps + 1):
        epoch = (step - 1) // steps_per_epoch + 1 if config.epochs is not None else None
        batch = next(batches, None)
        if batch is None:
            batches = iter(train)
            batch = next(batches)
        try:
            loss_value = _update(
                model, optimizer, batch, target_name, dipole, device, config.bf16
            )
        except FloatingPointError as error:
            raise FloatingPointError(f"{error} at {stage} step {step}") from error
        with experiment.train():
            experiment.log_metric("relative_mse", loss_value, step=step, epoch=epoch)
        if step % validation_interval == 0 or step == steps:
            score, preview = _validate(
                model, validation, target_name, dipole, device, config.bf16
            )
            record = {"step": step, "nrmse": score}
            if epoch is not None:
                record["epoch"] = epoch
            history.append(record)
            progress = (
                f"epoch {epoch} step {step}" if epoch is not None else f"step {step}"
            )
            print(f"{stage} {progress} validation NRMSE {score:.4f}%", flush=True)
            if score < best:
                best = score
                atomic_save(
                    dict(
                        checkpoint_metadata,
                        model=model.state_dict(),
                        step=step,
                        validation_nrmse=score,
                    ),
                    checkpoint,
                )
            with experiment.validate():
                experiment.log_metrics(
                    {"nrmse_pct": score, "best_nrmse_pct": best}, step=step, epoch=epoch
                )
                _log_validation_image(experiment, preview, target_name, step)
            model.train()
    return dict(
        checkpoint=checkpoint.name,
        target=target_name,
        history=history,
        selected=min(history, key=lambda row: row["nrmse"]),
    )


def train_comparison(
    dataset,
    output,
    *,
    device="cpu",
    training=TrainingConfig(),
    features=DEFAULT_FEATURES,
    tkd_threshold=0.1,
    tikhonov_threshold=0.1,
    gd_iterations=30,
    cuda_memory_gb=None,
    probe_batch_sizes=False,
    batch_sizes=None,
    comparison_tag=None,
):
    """Fit the direct model, one shared field estimator, and four correction arms."""
    if training.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if training.epochs is not None:
        if training.epochs < 1:
            raise ValueError("Epochs must be positive")
    elif (
        training.steps is None or min(training.steps, training.validation_interval) < 1
    ):
        raise ValueError("Steps and validation interval must be positive")
    if not isfinite(tkd_threshold) or tkd_threshold <= 0:
        raise ValueError("TKD threshold must be finite and positive")
    if type(gd_iterations) is not int or gd_iterations < 1:
        raise ValueError("GD iterations must be a positive integer")
    device = require_device(device)
    if probe_batch_sizes and (cuda_memory_gb is None or batch_sizes is not None):
        raise ValueError(
            "Batch probing requires a CUDA budget and no fixed batch sizes"
        )
    if cuda_memory_gb is not None:
        if device.type != "cuda":
            raise ValueError("CUDA batch tuning requires a CUDA device")
        if not isfinite(cuda_memory_gb) or cuda_memory_gb <= 1:
            raise ValueError("CUDA memory budget must be finite and greater than 1 GiB")
        configure_memory_budget(device, cuda_memory_gb)
    dataset, output = Path(dataset).resolve(), Path(output)
    model_config = ModelConfig(
        features=tuple(features),
        threshold=tikhonov_threshold,
        checkpoint_blocks=True,
        field_normalization="laplacian_rms",
        normalization_layer="residual_instance",
        conditional_laplacian_threshold=0,
    )
    if 2 ** (len(features) + 1) > PATCH_SIZE:
        raise ValueError(
            "U-Net depth exceeds the training patch and inference window size"
        )
    inference = SlidingWindowConfig(window_size=PATCH_SIZE)
    metadata = dict(
        model_config=asdict(model_config), inference_config=asdict(inference)
    )
    inverses = dict(tkd_threshold=tkd_threshold, gd_iterations=gd_iterations)
    manifest_file = dataset / "manifest.json"
    manifest = dict(
        version=MANIFEST_VERSION,
        dataset=dict(
            path=str(manifest_file),
            sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        ),
        training=asdict(training),
        device=str(device),
        inverses=inverses,
        **metadata,
        stages={},
    )
    output.mkdir(parents=True, exist_ok=False)
    comparison_tag = comparison_tag or f"comparison-{uuid4().hex}"
    manifest["comparison_tag"] = comparison_tag

    def fit(stage, model, target, *, legacy=False):
        experiment = _start_experiment(output, stage, comparison_tag)
        try:
            peak = None
            batch_size = training.batch_size
            if batch_sizes is not None:
                batch_size = batch_sizes[stage]
            elif probe_batch_sizes:
                batch_size, peak = _probe_batch_size(
                    model, target, dataset, stage, device, training
                )
                print(f"{stage} selected batch size {batch_size}", flush=True)
            stage_training = replace(training, batch_size=batch_size)
            validation = make_patch_dataloader(
                dataset,
                "validation",
                batch_size=batch_size,
                num_workers=training.workers,
                seed=training.seed,
                fixed_backgrounds=True,
            )
            experiment.log_parameters(
                dict(
                    training=asdict(stage_training),
                    cuda_memory_budget_gib=cuda_memory_gb,
                    batch_probe_peak_reserved_gib=peak,
                    device=str(device),
                    inverses=inverses,
                    dataset_sha256=manifest["dataset"]["sha256"],
                    stage=stage,
                    target=target,
                    **metadata,
                ),
                nested_support=True,
            )
            record = _fit(
                model,
                target,
                dataset,
                validation,
                output,
                stage,
                device,
                stage_training,
                dict(metadata, version=CHECKPOINT_VERSION, stage=stage)
                if legacy
                else dict(metadata, comparison_version=MANIFEST_VERSION, stage=stage),
                experiment,
            )
            record["comet_experiment_key"] = experiment.get_key()
            record["batch_size"] = batch_size
            if peak is not None:
                record["batch_probe_peak_reserved_gib"] = peak
        finally:
            experiment.end()
        manifest["stages"][stage] = record
        write_json(output / "validation.json", manifest)

    torch.manual_seed(training.seed)
    direct = DirectSusceptibility(model_config).to(device)
    fit(DIRECT, direct, "chi")
    del direct

    torch.manual_seed(training.seed)
    estimator = LocalFieldEstimator(model_config).to(device)
    fit("local_field", estimator, "local_field")
    field_checkpoint = output / manifest["stages"]["local_field"]["checkpoint"]
    estimator.zero_grad(set_to_none=True)
    estimator.cpu()
    estimator.load_state_dict(
        torch.load(field_checkpoint, weights_only=True, map_location="cpu")["model"]
    )
    manifest["shared_field_checkpoint"] = field_checkpoint.name

    torch.manual_seed(training.seed)
    dnqsm = DNQSM(model_config)
    dnqsm.filter.load_state_dict(estimator.filter.state_dict())
    residual_start = dnqsm.null_network.state_dict()
    for inverse in INVERSES:
        baseline = LocalFieldBaseline(
            deepcopy(estimator), inverse, correction=True, **inverses
        )
        baseline.residual.load_state_dict(residual_start)
        fit(f"model3_{inverse}", baseline.to(device), "chi")
        del baseline
    del residual_start, estimator

    fit(DNQSM_ARM, dnqsm.to(device), "chi", legacy=True)
    write_json(output / "comparison.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--features", nargs="+", type=int, default=DEFAULT_FEATURES)
    parser.add_argument("--tkd-threshold", type=float, default=0.1)
    parser.add_argument("--tikhonov-threshold", type=float, default=0.1)
    parser.add_argument("--gd-iterations", type=int, default=30)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, default=10_000)
    duration.add_argument("--epochs", type=int, help="Full passes per stage")
    parser.add_argument("--validation-interval", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    training = TrainingConfig(
        **{key: getattr(args, key) for key in TrainingConfig.__dataclass_fields__}
    )
    train_comparison(
        args.patch_dir,
        args.output_dir,
        device=args.device,
        training=training,
        features=args.features,
        tkd_threshold=args.tkd_threshold,
        tikhonov_threshold=args.tikhonov_threshold,
        gd_iterations=args.gd_iterations,
    )


if __name__ == "__main__":
    main()
