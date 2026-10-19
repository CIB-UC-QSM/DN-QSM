"""Filter pretraining and end-to-end DN-QSM with resumable deterministic views."""

import argparse
import json
import math
import os
import random
import signal
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import evaluation
from .data import (
    BalancedSourceSampler,
    ShapeFieldSynthesizer,
    TrainingVolumes,
    collate_shapes,
    make_subject_split,
    shape_groups,
    training_records,
)
from .diagnostics import loss_gradient_metrics, quality_totals, summarize_quality
from .logging import RunLogger, filtering_figure, reconstruction_figure, write_json
from .losses import (
    LOSS_VERSION,
    LossNormalization,
    LossWeights,
    local_field_loss_terms,
    normalize_loss_terms,
    per_sample_loss_terms,
)
from .memory import ProcessMemoryMonitor
from .model import DNQSM, ModelConfig
from .paths import RUNS_DIR, TRAINING_CACHE_DIR, TRAINING_DATA_DIR
from .physics import apply_dipole, masked_nrmse
from .runtime import (
    CHECKPOINT_VERSION,
    atomic_save,
    autocast_context,
    configure_memory_budget,
    load_trained_model,
    require_device,
)

STOP_REQUESTED = False


@dataclass
class TrainConfig:
    data_dir: str = str(TRAINING_DATA_DIR)
    source_manifest: str | None = None
    cache_dir: str = str(TRAINING_CACHE_DIR)
    run_dir: str = ""
    resume: str | None = None
    init_checkpoint: str | None = None
    device: str = "cuda:0"
    cuda_memory_gb: float = 0.0
    cuda_allocator_config: str = ""
    epochs: int = 50
    filter_pretrain_epochs: int = 10
    batch_size: int = 1
    accumulation_steps: int = 5
    num_workers: int = 4
    threads: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    min_lr_factor: float = 0.05
    grad_clip: float = 10.0
    val_fraction: float = 0.1
    val_views: int = 3
    synthetic_fraction: float = 0.25
    crop_probability: float = 0.0
    crop_resize: bool = True
    seed: int = 0
    features: tuple[int, ...] = (16, 32, 64)
    threshold: float = 0.1
    scale_floor: float = 1e-6
    checkpoint_blocks: bool = True
    residual_filter: bool = False
    full_grid_field: bool = False
    laplacian_input: bool = True
    filter_input_version: int = 2
    normalize_loss: bool = True
    chi_loss_floor: float = 1e-4
    field_loss_floor: float = 1e-4
    mse_input_ratio_cap: float = 0.0
    lambda_local: float = 1.5
    lambda_chi: float = 1.0
    lambda_physics: float = 1.0
    lambda_extension: float = 0.0
    physics_warmup_epochs: int = 0
    background_profile: str = "mixture"
    background_curriculum_epochs: int = 5
    strong_min_ratio: float = 0.001
    strong_probability: float = 0.25
    clean_probability: float = 0.15
    weak_probability: float = 0.1
    snr_min: float = 40.0
    snr_max: float = 100.0
    validation_synthesis: dict | None = None
    gradient_every: int = 100
    bf16: bool = True
    comet: bool = False
    comet_env_file: str | None = None
    comet_project: str = os.environ.get("COMET_PROJECT_NAME", "dn-qsm")
    save_every_steps: int = 50
    log_every: int = 1
    max_steps: int = 0
    max_train_volumes: int = 0
    max_val_subjects: int = 0
    evaluate_external: bool = True
    external_eval_every: int = 0
    evaluation_data_dir: str = ""
    cosmos_simulation: str | None = None
    sim2_pad: int = 20

    def validate(self) -> None:
        if (
            type(self.filter_pretrain_epochs) is not int
            or self.filter_pretrain_epochs < 0
        ):
            raise ValueError("Filter pretraining epochs must be a nonnegative integer")
        positive = (
            self.epochs,
            self.batch_size,
            self.accumulation_steps,
            self.threads,
            self.learning_rate,
            self.grad_clip,
            self.val_views,
            self.save_every_steps,
            self.log_every,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError(
                "Counts, learning rate, and gradient clipping must be positive"
            )
        if (
            min(
                self.num_workers,
                self.warmup_epochs,
                self.max_steps,
                self.max_train_volumes,
                self.max_val_subjects,
                self.sim2_pad,
                self.external_eval_every,
                self.physics_warmup_epochs,
                self.background_curriculum_epochs,
                self.gradient_every,
            )
            < 0
        ):
            raise ValueError(
                "Workers, warmup, padding, and diagnostic limits must be nonnegative"
            )
        if (
            not 0 < self.val_fraction < 1
            or not 0 <= self.synthetic_fraction <= 1
            or not 0 < self.min_lr_factor <= 1
            or not 0 <= self.crop_probability <= 1
        ):
            raise ValueError("Invalid split, synthetic fraction, or LR floor")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if (
            not math.isfinite(self.cuda_memory_gb)
            or self.cuda_memory_gb < 0
            or 0 < self.cuda_memory_gb <= 1
        ):
            raise ValueError(
                "CUDA memory budget must be zero (unlimited) or greater than 1 GiB"
            )
        ModelConfig(
            tuple(self.features),
            self.threshold,
            self.scale_floor,
            self.checkpoint_blocks,
            self.residual_filter,
            self.full_grid_field,
            self.laplacian_input,
            self.filter_input_version,
        )
        LossWeights(
            local=self.lambda_local,
            chi=self.lambda_chi,
            physics=self.lambda_physics,
            extension=self.lambda_extension,
        )
        LossNormalization(
            self.chi_loss_floor, self.field_loss_floor, self.mse_input_ratio_cap
        )
        if self.mse_input_ratio_cap and not self.normalize_loss:
            raise ValueError("MSE input-ratio cap requires normalized losses")
        if not all(math.isfinite(v) for v in (self.snr_min, self.snr_max)) or not (
            0 < self.snr_min <= self.snr_max
        ):
            raise ValueError("SNR bounds must be finite, positive, and ordered")
        if self.validation_synthesis is not None and (
            not isinstance(self.validation_synthesis, dict)
            or "background_mode" not in self.validation_synthesis
        ):
            raise ValueError(
                "Validation synthesis must be an object with background_mode"
            )
        if (
            self.background_profile not in ("mixture", "ratio", "legacy")
            or not math.isfinite(self.strong_min_ratio)
            or not 0 < self.strong_min_ratio <= 0.03
            or any(
                not 0 <= p <= 1
                for p in (
                    self.strong_probability,
                    self.clean_probability,
                    self.weak_probability,
                )
            )
            or self.strong_probability + self.clean_probability + self.weak_probability
            > 1
        ):
            raise ValueError("Invalid background sampling configuration")
        if not self.full_grid_field and self.lambda_extension:
            raise ValueError("Masked-field ablations require --lambda-extension 0")
        if self.resume and self.init_checkpoint:
            raise ValueError("Choose --resume or --init-checkpoint, not both")


def model_config(config: TrainConfig) -> ModelConfig:
    return ModelConfig(
        tuple(config.features),
        config.threshold,
        config.scale_floor,
        config.checkpoint_blocks,
        config.residual_filter,
        config.full_grid_field,
        config.laplacian_input,
        config.filter_input_version,
    )


def normalization_config(config):
    return (
        LossNormalization(
            config.chi_loss_floor, config.field_loss_floor, config.mse_input_ratio_cap
        )
        if config.normalize_loss
        else None
    )


def training_weights(config, step=0, warmup_steps=0):
    weights = LossWeights(
        local=config.lambda_local,
        chi=config.lambda_chi,
        physics=config.lambda_physics,
        extension=config.lambda_extension,
    )
    if warmup_steps:
        weights = replace(
            weights, physics=weights.physics * min(1.0, (step + 1) / warmup_steps)
        )
    return weights


def background_options(config, progress=1.0):
    return dict(
        background_mode=config.background_profile,
        curriculum_progress=progress,
        strong_min_ratio=config.strong_min_ratio,
        strong_probability=config.strong_probability,
        clean_probability=config.clean_probability,
        weak_probability=config.weak_probability,
        snr_min=config.snr_min,
        snr_max=config.snr_max,
    )


def learning_rate_multiplier(step: int, warmup: int, total: int, floor: float) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def phase_metadata(config: TrainConfig, epoch: int) -> dict:
    """Describe the next/current epoch using zero-based checkpoint positions."""
    pretraining = epoch < config.filter_pretrain_epochs
    return dict(
        phase="filter_pretrain" if pretraining else "end_to_end",
        phase_epoch=epoch + 1
        if pretraining
        else epoch - config.filter_pretrain_epochs + 1,
    )


def phased_learning_rate_multiplier(config, step, updates_per_epoch):
    pretrain_steps = config.filter_pretrain_epochs * updates_per_epoch
    if step < pretrain_steps:
        total = pretrain_steps
    else:
        step -= pretrain_steps
        total = config.epochs * updates_per_epoch
    warmup = min(config.warmup_epochs * updates_per_epoch, total)
    return learning_rate_multiplier(step, warmup, total, config.min_lr_factor)


def checkpoint_training_config(saved: dict) -> dict:
    """Fill historical defaults without changing an existing run's behavior."""
    values = dict(saved)
    # Resume reaches the loss-version guard rather than failing on an old key.
    values.pop("lambda_forward", None)
    values.setdefault("filter_input_version", 1)
    values.setdefault("filter_pretrain_epochs", 0)
    values.setdefault("full_grid_field", False)
    values.setdefault("lambda_extension", 1.0 if values["full_grid_field"] else 0.0)
    values.setdefault("residual_filter", False)
    values.setdefault("crop_resize", False)
    values.setdefault("laplacian_input", False)
    return values


def _predict_training_batch(model, data, device, bf16, filter_only, include_extension):
    with autocast_context(device, bf16):
        output = (
            model.predict_local_field(data["phase"], data["mask"])
            if filter_only
            else model(data["phase"], data["mask"], data["dipole"])
        )
    if filter_only:
        return output, local_field_loss_terms(output, data["local"], data["mask"]), None
    predicted_field = apply_dipole(output.chi, data["dipole"])
    raw = per_sample_loss_terms(
        output,
        data["local"],
        data["chi"],
        data["mask"],
        data["dipole"],
        local_full_target=data["local_full"],
        predicted_field=predicted_field,
        include_extension=include_extension,
    )
    return output, raw, predicted_field


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


def build_datasets(config: TrainConfig, directory: Path):
    records = training_records(config.data_dir, config.source_manifest)
    train, validation = make_subject_split(records, config.val_fraction, config.seed)
    split = dict(
        seed=config.seed,
        validation_fraction=config.val_fraction,
        train=train,
        validation=validation,
    )
    path = directory / "split.json"
    if path.is_file() and json.loads(path.read_text()) != split:
        raise ValueError("Saved source split differs from the requested data")
    write_json(path, split)
    representatives = {r["subject"]: r for r in reversed(validation)}
    validation = [representatives[s] for s in sorted(representatives)]
    if config.max_train_volumes:
        original = [r for r in train if not r["subject"].startswith("synthetic:")]
        synthetic = [r for r in train if r["subject"].startswith("synthetic:")]
        limit = min(config.max_train_volumes, len(train))
        if original and synthetic and limit > 1:
            count = min(
                len(synthetic), max(1, round(limit * config.synthetic_fraction))
            )
            train = original[: limit - count] + synthetic[:count]
        else:
            train = train[:limit]
    if config.max_val_subjects:
        validation = validation[: config.max_val_subjects]
    training_set = TrainingVolumes(
        train,
        config.cache_dir,
        seed=config.seed,
        chi_scale_mode="rms_mixture",
        crop_probability=config.crop_probability,
        crop_resize=config.crop_resize,
    )
    # Fixed held-out views follow training's susceptibility distribution.
    validation_set = TrainingVolumes(
        validation,
        config.cache_dir,
        validation=True,
        views=config.val_views,
        seed=config.seed + 100000,
        chi_scale_mode="rms_mixture",
    )
    shape = tuple(training_set[0]["original_shape"].tolist())
    return training_set, validation_set, shape


def _loader(dataset, config: TrainConfig, sampler=None, epoch: int = 0):
    options = dict(
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        generator=torch.Generator().manual_seed(config.seed + epoch),
        worker_init_fn=_worker_init,
    )
    if config.num_workers:
        options["prefetch_factor"] = 2
    if getattr(dataset, "crop_probability", 0):
        options["collate_fn"] = collate_shapes
    return DataLoader(dataset, **options)


def backward_batch(
    model,
    batch,
    synthesizer,
    device,
    weights,
    config,
    group_samples,
    figure_title=None,
    *,
    gradient_diagnostics=False,
    background_progress=1.0,
    filter_only=False,
):
    """Accumulate equally weighted samples across their model-input grids."""
    totals = defaultdict(float)
    figure = None
    samples = cropped = 0
    for shaped_batch in shape_groups(batch):
        data = synthesizer(
            shaped_batch, **background_options(config, background_progress)
        )
        count = data["chi"].shape[0]
        fraction = count / group_samples
        output, raw, predicted_field = _predict_training_batch(
            model, data, device, config.bf16, filter_only, bool(weights.extension)
        )
        terms = {
            name: value.mean()
            for name, value in normalize_loss_terms(
                raw,
                data["local"],
                data["chi"],
                data["mask"],
                normalization_config(config),
                input_field=data["phase"],
            ).items()
        }
        loss = sum(getattr(weights, name) * value for name, value in terms.items())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        if gradient_diagnostics:
            totals.update(loss_gradient_metrics(model, terms, weights))
            totals["gradient_diagnostic_samples"] = count
            gradient_diagnostics = False
        (loss * fraction).backward()
        for name, value in dict(loss=loss, **terms).items():
            totals[name] += float(value.detach()) * fraction
        for name, value in raw.items():
            totals["raw_" + name] += float(value.detach().mean()) * fraction
        for name, value in quality_totals(
            output, data, predicted_field=predicted_field
        ).items():
            totals[name] += value
        if figure_title is not None and figure is None:
            figure = (filtering_figure if filter_only else reconstruction_figure)(
                data, output, figure_title
            )
        samples += count
        if "crop_applied" in shaped_batch:
            cropped += int(shaped_batch["crop_applied"].sum())
        del data, output, loss, terms, raw, value, predicted_field
    return totals, figure, samples, cropped


@torch.no_grad()
def validate(
    model,
    loader,
    synthesizer,
    device,
    weights,
    bf16: bool,
    *,
    normalization=LossNormalization(),
    synthesis_options=None,
    filter_only=False,
):
    model.eval()
    subjects = defaultdict(list)
    first = None
    quality = defaultdict(float)
    for batch in loader:
        data = synthesizer(
            batch, **(synthesis_options or dict(background_mode="mixture"))
        )
        output, raw, predicted_field = _predict_training_batch(
            model, data, device, bf16, filter_only, bool(weights.extension)
        )
        per_sample = {
            name: values.flatten()
            for name, values in normalize_loss_terms(
                raw,
                data["local"],
                data["chi"],
                data["mask"],
                normalization,
                input_field=data["phase"],
            ).items()
        }
        per_sample["loss"] = sum(
            getattr(weights, name) * values for name, values in per_sample.items()
        )
        per_sample.update(
            {"raw_" + name: value.flatten() for name, value in raw.items()}
        )
        for name, value in quality_totals(
            output, data, predicted_field=predicted_field
        ).items():
            quality[name] += value
        errors = dict(
            local_nrmse=masked_nrmse(
                output if filter_only else output.local_field,
                data["local"],
                data["mask"],
            )
        )
        if not filter_only:
            errors.update(
                chi_nrmse=masked_nrmse(output.chi, data["chi"], data["mask"]),
                tkd_nrmse=masked_nrmse(output.tkd, data["chi"], data["mask"]),
            )
        for index, subject in enumerate(batch["subject"]):
            subjects[subject].append(
                dict(
                    **{name: values[index] for name, values in errors.items()},
                    **{
                        name: float(values[index])
                        for name, values in per_sample.items()
                    },
                )
            )
        if first is None:
            first = (filtering_figure if filter_only else reconstruction_figure)(
                data, output, "Held-out validation (physical units)"
            )
    result = {}
    for name in next(iter(subjects.values()))[0]:
        means = [
            float(np.mean([row[name] for row in rows if row[name] is not None]))
            for rows in subjects.values()
            if any(row[name] is not None for row in rows)
        ]
        result[name] = float(np.mean(means)) if means else None
    result.update(
        {
            key: value
            for key, value in summarize_quality(quality).items()
            if key.startswith("bucket_") or key == "physics_nrmse"
        }
    )
    return result, first


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return dict(
        torch=torch.get_rng_state(),
        python=random.getstate(),
        numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_training(config: TrainConfig) -> Path:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    config.validate()
    validation_profile = (
        "reference" if config.validation_synthesis is not None else "matched"
    )
    if config.cuda_allocator_config:
        os.environ["PYTORCH_ALLOC_CONF"] = config.cuda_allocator_config
    device = require_device(config.device)
    configure_memory_budget(device, config.cuda_memory_gb)
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    resume = (
        torch.load(config.resume, map_location="cpu", weights_only=True, mmap=True)
        if config.resume
        else None
    )
    if resume:
        if resume.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint version")
        if resume.get("loss_version", 1) != LOSS_VERSION:
            raise ValueError(
                "Checkpoint uses an older loss. Use --init-checkpoint to start "
                "a new run with the current objective."
            )
        # Scheduling, sampling, and model changes cannot silently alter a resume.
        runtime = {
            "resume",
            "init_checkpoint",
            "run_dir",
            "device",
            "cuda_memory_gb",
            "threads",
            "num_workers",
            "comet",
            "comet_env_file",
            "comet_project",
            "max_steps",
            "evaluate_external",
            "evaluation_data_dir",
            "cosmos_simulation",
            "sim2_pad",
            "log_every",
            "save_every_steps",
            "gradient_every",
        }
        current = json.loads(json.dumps(asdict(config)))
        previous = checkpoint_training_config(json.loads(json.dumps(resume["config"])))
        for key in (
            "crop_probability",
            "external_eval_every",
            "cuda_allocator_config",
            "mse_input_ratio_cap",
            "snr_min",
            "snr_max",
            "validation_synthesis",
        ):
            previous.setdefault(key, getattr(TrainConfig(), key))
        changed = [
            key
            for key in current
            if key not in runtime and current[key] != previous[key]
        ]
        if changed:
            raise ValueError(f"Resume configuration changed: {changed}")
        directory = Path(config.resume).resolve().parent
    else:
        directory = (
            Path(config.run_dir)
            if config.run_dir
            else RUNS_DIR / datetime.now().strftime("dnqsm_%Y%m%d_%H%M%S_%f")
        )
        directory.mkdir(parents=True, exist_ok=False)
    config.run_dir = str(directory.resolve())
    training_set, validation_set, shape = build_datasets(config, directory)
    stress_set = TrainingVolumes(
        validation_set.records,
        config.cache_dir,
        validation=True,
        views=config.val_views,
        seed=config.seed + 100000,
        chi_scale_mode="legacy",
    )
    write_json(directory / "config.json", asdict(config))
    write_json(
        directory / "status.json",
        dict(
            status="running",
            device=str(device),
            **phase_metadata(config, resume["epoch"] if resume else 0),
        ),
    )
    model = DNQSM(model_config(config)).to(device)
    if config.init_checkpoint:
        initialized = load_trained_model(config.init_checkpoint, "cpu")
        if any(
            getattr(initialized.config, name) != getattr(config, name)
            for name in (
                "filter_input_version",
                "laplacian_input",
                "residual_filter",
                "full_grid_field",
            )
        ):
            raise ValueError(
                "Initialization checkpoint has a different filtering architecture. "
                "Start a fresh run for masked Laplacian-only filtering, or match "
                "the checkpoint's filter-input-version, input channels, and masking."
            )
        model.load_state_dict(initialized.state_dict())
        del initialized
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    weights = training_weights(config)
    synthesizer = ShapeFieldSynthesizer(device)
    sampler = BalancedSourceSampler(
        training_set.records, config.synthetic_fraction, config.seed
    )
    batches_per_epoch = math.ceil(len(training_set) / config.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / config.accumulation_steps)
    pretrain_steps = updates_per_epoch * config.filter_pretrain_epochs
    physics_warmup_steps = config.physics_warmup_epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: phased_learning_rate_multiplier(config, s, updates_per_epoch),
    )
    epoch, next_batch, step, best = 0, 0, 0, float("inf")
    epoch_samples = epoch_cropped = 0
    epoch_clipped = 0
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        epoch, next_batch, step, best = (
            resume[k] for k in ("epoch", "next_batch", "step", "best")
        )
        _restore_rng(resume["rng"])
        epoch_samples = resume.get("epoch_samples", 0)
        epoch_cropped = resume.get("epoch_cropped", 0)
        epoch_clipped = resume.get("epoch_clipped", 0)
    is_resuming = resume is not None
    del resume
    logger = None
    total_vram = (
        torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else 0
    )
    memory_monitor = ProcessMemoryMonitor(
        device,
        (config.cuda_memory_gb * 1024**3 if config.cuda_memory_gb else total_vram)
        / 1e9,
    )
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, _request_stop)

    def save():
        payload = dict(
            version=CHECKPOINT_VERSION,
            loss_version=LOSS_VERSION,
            model_config=asdict(model.config),
            model=model.state_dict(),
            config=asdict(config),
            step=step,
            epoch=epoch,
            next_batch=next_batch,
            best=best,
            epoch_samples=epoch_samples,
            epoch_cropped=epoch_cropped,
            epoch_clipped=epoch_clipped,
            physics_grid_version=2,
            validation_profile=validation_profile,
            **phase_metadata(config, epoch),
        )
        atomic_save(payload, directory / "model_last.pt")
        atomic_save(
            dict(
                payload,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=_rng_state(),
            ),
            directory / "training_last.pt",
        )

    try:
        logger = RunLogger(directory, config, resume=is_resuming)
        cases = None
        logger.log(
            "started",
            step,
            epoch,
            train_volumes=len(training_set),
            validation_subjects=len(validation_set.records),
            validation_views=config.val_views,
            updates_per_epoch=updates_per_epoch,
            shape=list(shape),
            cuda_memory_gb=config.cuda_memory_gb,
            parameters=sum(p.numel() for p in model.parameters()),
            batch_size=config.batch_size,
            gpu_total_memory_gib=total_vram / 1024**3,
            crop_probability=config.crop_probability,
            crop_resize=config.crop_resize,
            external_eval_every=config.external_eval_every,
            physics_grid_version=2,
            validation_profile=validation_profile,
            normalize_loss=config.normalize_loss,
            full_grid_field=config.full_grid_field,
            laplacian_input=config.laplacian_input,
            filter_input_version=config.filter_input_version,
            filter_pretrain_epochs=config.filter_pretrain_epochs,
            end_to_end_epochs=config.epochs,
            **phase_metadata(config, epoch),
        )
        while epoch < config.filter_pretrain_epochs + config.epochs:
            phase = phase_metadata(config, epoch)
            filter_only = epoch < config.filter_pretrain_epochs
            model.null_network.requires_grad_(not filter_only)
            if config.max_steps and step >= config.max_steps:
                save()
                write_json(
                    directory / "status.json",
                    dict(status="bounded", step=step, epoch=epoch, **phase),
                )
                return directory
            training_set.set_epoch(epoch)
            sampler.set_epoch(epoch)
            indices = list(sampler)[next_batch * config.batch_size :]
            loader = _loader(training_set, config, sampler=indices, epoch=epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            group_start = next_batch
            group_terms = defaultdict(float)
            started = time.perf_counter()
            train_figure = None
            background_progress = (
                min(1.0, (epoch + 1) / config.background_curriculum_epochs)
                if config.background_curriculum_epochs
                else 1.0
            )
            for batch in loader:
                group_end = min(
                    group_start + config.accumulation_steps, batches_per_epoch
                )
                # Weight the final partial batch by its actual sample count.
                group_samples = (
                    min(group_end * config.batch_size, len(training_set))
                    - group_start * config.batch_size
                )
                terms, figure, sample_count, cropped_count = backward_batch(
                    model,
                    batch,
                    synthesizer,
                    device,
                    training_weights(
                        config, max(0, step - pretrain_steps), physics_warmup_steps
                    ),
                    config,
                    group_samples,
                    f"Training epoch {epoch + 1}" if train_figure is None else None,
                    gradient_diagnostics=bool(
                        config.gradient_every
                        and group_start == next_batch
                        and (step == 0 or (step + 1) % config.gradient_every == 0)
                    ),
                    background_progress=background_progress,
                    filter_only=filter_only,
                )
                for name, value in terms.items():
                    group_terms[name] += value
                epoch_samples += sample_count
                epoch_cropped += cropped_count
                next_batch += 1
                if figure is not None:
                    train_figure = figure
                if next_batch != group_end:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip, error_if_nonfinite=True
                )
                clipped = int(float(norm) > config.grad_clip)
                epoch_clipped += clipped
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if (
                    step % config.log_every == 0
                    or "gradient_diagnostic_samples" in group_terms
                ):
                    memory = (
                        {}
                        if device.type != "cuda"
                        else dict(
                            peak_allocated_gb=torch.cuda.max_memory_allocated(device)
                            / 1024**3,
                            peak_reserved_gb=torch.cuda.max_memory_reserved(device)
                            / 1024**3,
                            peak_reserved_percent=100
                            * torch.cuda.max_memory_reserved(device)
                            / total_vram,
                            **memory_monitor.metrics(),
                        )
                    )
                    logger.log(
                        "train",
                        step,
                        epoch + 1,
                        **summarize_quality(group_terms),
                        learning_rate=lr,
                        gradient_norm=float(norm),
                        gradient_clipped=clipped,
                        gradient_clip_threshold=config.grad_clip,
                        active_physics_weight=training_weights(
                            config,
                            max(0, step - pretrain_steps - 1),
                            physics_warmup_steps,
                        ).physics
                        if not filter_only
                        else 0.0,
                        **phase,
                        background_curriculum_progress=background_progress,
                        seconds=time.perf_counter() - started,
                        **memory,
                    )
                group_terms.clear()
                started = time.perf_counter()
                group_start = next_batch
                bounded = bool(config.max_steps and step >= config.max_steps)
                if (
                    step == 1
                    or step % config.save_every_steps == 0
                    or STOP_REQUESTED
                    or bounded
                ):
                    save()
                if STOP_REQUESTED or bounded:
                    if train_figure is not None:
                        logger.figure(
                            train_figure,
                            "train",
                            epoch + 1,
                            epoch + 1,
                        )
                    write_json(
                        directory / "status.json",
                        dict(
                            status="interrupted" if STOP_REQUESTED else "bounded",
                            step=step,
                            epoch=epoch,
                            next_batch=next_batch,
                            **phase,
                        ),
                    )
                    return directory
            # All epoch updates are durable before either validation suite or
            # external evaluation. Resume may repeat metrics, never updates.
            save()
            if train_figure is not None:
                logger.figure(train_figure, "train", epoch + 1, epoch + 1)
            logger.log(
                "augmentation",
                step,
                epoch + 1,
                samples=epoch_samples,
                cropped_samples=epoch_cropped,
                crop_fraction=epoch_cropped / max(1, epoch_samples),
                **phase,
            )
            logger.log(
                "optimization",
                step,
                epoch + 1,
                comet_step=epoch + 1,
                clipped_updates=epoch_clipped,
                clipping_fraction=epoch_clipped / updates_per_epoch,
                background_curriculum_progress=background_progress,
                **phase,
            )
            metrics, figure = validate(
                model,
                _loader(validation_set, config),
                synthesizer,
                device,
                weights,
                config.bf16,
                normalization=normalization_config(config),
                filter_only=filter_only,
                synthesis_options=(
                    config.validation_synthesis
                    if config.validation_synthesis is not None
                    else background_options(config)
                ),
            )
            logger.log(
                "validation", step, epoch + 1, comet_step=epoch + 1, **phase, **metrics
            )
            if figure is not None:
                logger.figure(figure, "validation", epoch + 1, epoch + 1)
            stress_metrics, stress_figure = validate(
                model,
                _loader(stress_set, config),
                synthesizer,
                device,
                weights,
                config.bf16,
                normalization=normalization_config(config),
                filter_only=filter_only,
                synthesis_options=dict(background_mode="legacy"),
            )
            logger.log(
                "validation_stress",
                step,
                epoch + 1,
                comet_step=epoch + 1,
                **phase,
                **stress_metrics,
            )
            if stress_figure is not None:
                logger.figure(stress_figure, "validation_stress", epoch + 1, epoch + 1)
            if not filter_only and metrics["chi_nrmse"] is None:
                raise ValueError(
                    "All validation susceptibility references have zero energy"
                )
            if not filter_only and metrics["chi_nrmse"] < best:
                best = metrics["chi_nrmse"]
                atomic_save(
                    dict(
                        version=CHECKPOINT_VERSION,
                        loss_version=LOSS_VERSION,
                        physics_grid_version=2,
                        model_config=asdict(model.config),
                        model=model.state_dict(),
                        step=step,
                        epoch=epoch + 1,
                        best=best,
                        validation_profile=validation_profile,
                        **phase,
                    ),
                    directory / "model_best.pt",
                )
            if (
                not filter_only
                and config.external_eval_every
                and phase["phase_epoch"] % config.external_eval_every == 0
            ):
                if cases is None:
                    cases = evaluation.load_cases(
                        data_dir=config.evaluation_data_dir or None,
                        cosmos_simulation=config.cosmos_simulation,
                        sim2_pad=config.sim2_pad,
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                evaluation.evaluate_training_epoch(
                    model,
                    cases,
                    directory,
                    config.device,
                    logger,
                    step,
                    epoch + 1,
                    config.bf16,
                )
            epoch += 1
            next_batch = 0
            epoch_samples = epoch_cropped = 0
            epoch_clipped = 0
            save()
            write_json(
                directory / "status.json",
                dict(
                    status="running",
                    step=step,
                    epoch=epoch,
                    best_chi_nrmse=best if math.isfinite(best) else None,
                    **phase_metadata(config, epoch),
                ),
            )
            if STOP_REQUESTED:
                write_json(
                    directory / "status.json",
                    dict(
                        status="interrupted",
                        step=step,
                        epoch=epoch,
                        next_batch=0,
                        **phase_metadata(config, epoch),
                    ),
                )
                return directory
        write_json(
            directory / "status.json",
            dict(
                status="trained",
                step=step,
                epoch=epoch,
                best_chi_nrmse=best,
                **phase_metadata(config, epoch - 1),
            ),
        )
        if config.evaluate_external:
            # Evaluation loads the best checkpoint into a separate model.
            # Drop training weights and Adam moments before allocating it.
            del model, optimizer, scheduler, synthesizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if cases is None:
                cases = evaluation.load_cases(
                    data_dir=config.evaluation_data_dir or None,
                    cosmos_simulation=config.cosmos_simulation,
                    sim2_pad=config.sim2_pad,
                )
            evaluation.evaluate_checkpoint(
                directory / "model_best.pt",
                cases,
                directory / "evaluation" / "best",
                config.device,
                logger=logger,
                step=step,
                epoch=epoch,
                bf16=config.bf16,
                name_prefix="best",
            )
        write_json(
            directory / "status.json",
            dict(
                status="complete",
                step=step,
                epoch=epoch,
                best_chi_nrmse=best,
                external_evaluation=config.evaluate_external,
                **phase_metadata(config, epoch - 1),
            ),
        )
        return directory
    except BaseException as error:
        write_json(
            directory / "status.json",
            dict(
                status="failed",
                step=step,
                epoch=epoch,
                error=str(error),
                **phase_metadata(config, epoch),
            ),
        )
        raise
    finally:
        if logger:
            logger.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="JSON training configuration; CLI flags override it"
    )
    defaults = TrainConfig()
    integer_fields = {
        f.name for f in fields(defaults) if type(getattr(defaults, f.name)) is int
    }
    float_fields = {
        f.name for f in fields(defaults) if type(getattr(defaults, f.name)) is float
    }
    for field in fields(defaults):
        name, value = field.name, getattr(defaults, field.name)
        kwargs = dict(default=argparse.SUPPRESS)
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        elif name == "features":
            kwargs.update(type=int, nargs="+")
        elif name == "validation_synthesis":
            kwargs["type"] = json.loads
        else:
            kwargs["type"] = (
                int
                if name in integer_fields
                else float
                if name in float_fields
                else str
            )
        parser.add_argument("--" + name.replace("_", "-"), **kwargs)
    overrides = vars(parser.parse_args(argv))
    values = asdict(defaults)
    config_file = overrides.pop("config", None)
    if config_file:
        if overrides.get("resume"):
            parser.error(
                "Use --config for a new experiment or --resume for exact continuation"
            )
        configured = json.loads(config_file.read_text())
        if not isinstance(configured, dict) or configured.keys() - values.keys():
            parser.error("Training JSON must contain only TrainConfig field names")
        if configured.get("resume"):
            parser.error("Use --resume for exact continuation, without a JSON profile")
        values.update(configured)
    if overrides.get("resume"):
        saved = torch.load(
            overrides["resume"], weights_only=True, map_location="cpu", mmap=True
        )
        values.update(checkpoint_training_config(saved["config"]))
        values["init_checkpoint"] = None
    values.update(overrides)
    values["features"] = tuple(values["features"])
    result = TrainConfig(**values)
    result.validate()
    return result


def main(argv: list[str] | None = None) -> None:
    directory = run_training(parse_args(argv))
    print(f"Run artifacts: {directory}")


if __name__ == "__main__":
    main()
