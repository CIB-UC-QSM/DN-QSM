"""Overfit one fixed COSMOS simulation before starting a full experiment."""

import argparse
import json
import math
import signal
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import torch

from .evaluation import evaluate_checkpoint, load_cosmos_case
from .logging import RunLogger, reconstruction_figure, write_json
from .losses import LOSS_VERSION, LossWeights, reconstruction_loss
from .memory import ProcessMemoryMonitor
from .model import DNQSM, ModelConfig
from .paths import RUNS_DIR
from .physics import apply_dipole, masked_nrmse
from .runtime import (
    CHECKPOINT_VERSION,
    atomic_save,
    autocast_context,
    configure_memory_budget,
    require_device,
)


@dataclass
class OverfitConfig:
    cosmos_simulation: str | None = None
    run_dir: str = ""
    resume: str | None = None
    device: str = "cuda:0"
    cuda_memory_gb: float = 27.0
    max_vram_gb: float = 30.0
    steps: int = 250
    learning_rate: float = 1e-3
    min_lr_factor: float = 0.05
    grad_clip: float = 1.0
    features: tuple[int, ...] = (54, 108, 216, 432, 864)
    threshold: float = 0.1
    checkpoint_blocks: bool = True
    residual_filter: bool = True
    bf16: bool = True
    seed: int = 0
    threads: int = 4
    log_every: int = 10
    figure_every: int = 50
    stop_on_pass: bool = True
    target_nrmse: float = 10.0
    target_loss_ratio: float = 0.2
    lambda_forward: float = 1.0
    lambda_local: float = 1.0
    lambda_chi: float = 1.0
    lambda_physics: float = 1.0
    comet: bool = False
    comet_env_file: str | None = None
    comet_project: str = "dn-qsm"

    def validate(self) -> None:
        positive = (
            self.steps,
            self.learning_rate,
            self.grad_clip,
            self.threads,
            self.log_every,
            self.figure_every,
            self.target_nrmse,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError(
                "Step counts, learning rate, clipping, and target must be positive"
            )
        if not 0 < self.target_loss_ratio < 1 or not 0 < self.min_lr_factor <= 1:
            raise ValueError("Loss-ratio target and LR floor must be in (0,1]")
        if (
            not math.isfinite(self.cuda_memory_gb)
            or self.cuda_memory_gb < 0
            or 0 < self.cuda_memory_gb <= 1
        ):
            raise ValueError("GPU budget must be zero or greater than 1 GiB")
        if not math.isfinite(self.max_vram_gb) or self.max_vram_gb <= 0:
            raise ValueError("Process VRAM ceiling must be finite and positive")
        if (
            not self.cuda_memory_gb
            or self.cuda_memory_gb * 1024**3 >= self.max_vram_gb * 1e9
        ):
            raise ValueError(
                "Choose a nonzero allocator budget in GiB below the process ceiling in decimal GB"
            )
        ModelConfig(
            tuple(self.features),
            self.threshold,
            checkpoint_blocks=self.checkpoint_blocks,
            residual_filter=self.residual_filter,
        )
        LossWeights(
            self.lambda_forward, self.lambda_local, self.lambda_chi, self.lambda_physics
        )


def passes_sanity(metrics: dict, initial_loss: float, config: OverfitConfig) -> bool:
    errors = [metrics[name] for name in ("chi_nrmse", "local_nrmse", "physics_nrmse")]
    return (
        all(
            v is not None and math.isfinite(v) and v <= config.target_nrmse
            for v in errors
        )
        and metrics["loss"] <= initial_loss * config.target_loss_ratio
    )


def run_overfit(config: OverfitConfig) -> Path:
    """Fresh initialization, fixed noisy total field, no augmentation or resynthesis.

    This is deliberate training on a benchmark case and is not a held-out score.
    Its weights must never initialize the subsequent full-data experiment.
    """
    config.validate()
    device = require_device(config.device)
    configure_memory_budget(device, config.cuda_memory_gb)
    with ProcessMemoryMonitor(device, config.max_vram_gb) as monitor:
        return _run_overfit(config, device, monitor)


def _run_overfit(
    config: OverfitConfig, device: torch.device, monitor: ProcessMemoryMonitor
) -> Path:
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    case = load_cosmos_case(config.cosmos_simulation)
    config.cosmos_simulation = case.metadata["simulation"]
    resumed = (
        torch.load(config.resume, weights_only=True, map_location="cpu", mmap=True)
        if config.resume
        else None
    )
    start_step = int(resumed["step"]) if resumed else 0
    if resumed:
        if (
            resumed.get("loss_version") != LOSS_VERSION
            or resumed.get("version") != CHECKPOINT_VERSION
        ):
            raise ValueError("Incompatible overfit checkpoint")
        if (
            tuple(resumed["model_config"]["features"]) != tuple(config.features)
            or resumed["model_config"]["threshold"] != config.threshold
        ):
            raise ValueError("Resume must preserve model capacity and TKD threshold")
        if (
            Path(resumed["config"]["cosmos_simulation"]).resolve()
            != Path(config.cosmos_simulation).resolve()
        ):
            raise ValueError("Resume must use the same fixed COSMOS simulation")
        if start_step >= config.steps:
            raise ValueError("The requested update budget is already exhausted")
    directory = (
        Path(config.resume).resolve().parent
        if resumed
        else (
            Path(config.run_dir)
            if config.run_dir
            else RUNS_DIR / datetime.now().strftime("cosmos_overfit_%Y%m%d_%H%M%S_%f")
        )
    )
    directory.mkdir(parents=True, exist_ok=bool(resumed))
    config.run_dir = str(directory.resolve())
    previous_summary = (
        json.loads((directory / "summary.json").read_text()) if resumed else None
    )
    if resumed:
        previous_memory = previous_summary.get("memory", previous_summary["final"])
        monitor.record_peak(round(previous_memory.get("peak_process_vram_gb", 0) * 1e9))
        write_json(
            directory / f"config_before_resume_{start_step:05d}.json", resumed["config"]
        )
    write_json(directory / "config.json", asdict(config))
    if not resumed:
        write_json(
            directory / "case.json",
            dict(
                case.metadata,
                training_case=True,
                augmentation=False,
                fresh_initialization=True,
                loss_version=LOSS_VERSION,
            ),
        )
    data = {
        name: value.to(device)
        for name, value in dict(
            phase=case.phase,
            mask=case.mask,
            dipole=case.dipole,
            chi=case.chi,
            local=case.local,
        ).items()
    }
    model = DNQSM(
        ModelConfig(
            tuple(config.features),
            config.threshold,
            checkpoint_blocks=config.checkpoint_blocks,
            residual_filter=config.residual_filter,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.steps,
        eta_min=config.learning_rate * config.min_lr_factor,
    )
    phases = (
        previous_summary.get(
            "training_phases",
            [
                dict(
                    start_step=0,
                    learning_rate=resumed["config"]["learning_rate"],
                    min_lr_factor=resumed["config"]["min_lr_factor"],
                    residual_filter=resumed["model_config"].get(
                        "residual_filter", False
                    ),
                )
            ],
        )
        if resumed
        else [
            dict(
                start_step=0,
                learning_rate=config.learning_rate,
                min_lr_factor=config.min_lr_factor,
                residual_filter=config.residual_filter,
            )
        ]
    )
    for phase in phases:
        phase.setdefault("residual_filter", False)
        phase.setdefault("min_lr_factor", 0.05)
    if resumed:
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        if (
            config.learning_rate == resumed["config"]["learning_rate"]
            and config.steps == resumed["config"]["steps"]
            and config.min_lr_factor == resumed["config"]["min_lr_factor"]
        ):
            scheduler.load_state_dict(resumed["scheduler"])
        else:
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate
                group["initial_lr"] = config.learning_rate
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.steps - start_step,
                eta_min=config.learning_rate * config.min_lr_factor,
            )
        if (
            config.learning_rate != resumed["config"]["learning_rate"]
            or config.min_lr_factor != resumed["config"]["min_lr_factor"]
            or config.residual_filter
            != resumed["model_config"].get("residual_filter", False)
        ):
            phases.append(
                dict(
                    start_step=start_step,
                    learning_rate=config.learning_rate,
                    min_lr_factor=config.min_lr_factor,
                    residual_filter=config.residual_filter,
                )
            )
    weights = LossWeights(
        config.lambda_forward,
        config.lambda_local,
        config.lambda_chi,
        config.lambda_physics,
    )
    stopped = False
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    for sig in handlers:
        signal.signal(sig, stop)
    logger = None
    step, best_error, best_step, best_metrics = 0, float("inf"), 0, None
    initial_loss = None
    initial_metrics = None
    passed = False
    best_passed = False
    started = time.perf_counter()
    if resumed:
        initial_metrics = previous_summary["initial"]
        initial_loss = initial_metrics["loss"]
        best_metrics = previous_summary["best"]
        best_error = best_metrics["chi_nrmse"]
        best_step = previous_summary["best_step"]
        best_passed = previous_summary["passed"]
        started -= previous_summary["final"]["seconds"]

    @torch.no_grad()
    def measure():
        model.eval()
        with autocast_context(device, config.bf16):
            prediction = model(data["phase"], data["mask"], data["dipole"])
        loss, terms = reconstruction_loss(
            prediction,
            data["local"],
            data["chi"],
            data["mask"],
            data["dipole"],
            weights,
        )
        metrics = {
            "loss": float(loss),
            **{name: float(value) for name, value in terms.items()},
        }
        for name, prediction_value, reference in (
            ("chi", prediction.chi, data["chi"]),
            ("tkd", prediction.tkd, data["chi"]),
            ("local", prediction.local_field, data["local"]),
            ("physics", apply_dipole(prediction.chi, data["dipole"]), data["local"]),
        ):
            metrics[name + "_nrmse"] = masked_nrmse(
                prediction_value, reference, data["mask"]
            )[0]
        metrics["seconds"] = time.perf_counter() - started
        if device.type == "cuda":
            metrics["peak_allocated_gb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            metrics["peak_reserved_gb"] = (
                torch.cuda.max_memory_reserved(device) / 1024**3
            )
        metrics.update(monitor.metrics())
        return prediction, metrics

    def save_model(name: str):
        atomic_save(
            dict(
                version=CHECKPOINT_VERSION,
                loss_version=LOSS_VERSION,
                model_config=asdict(model.config),
                model=model.state_dict(),
                step=step,
                training_case="cosmos_overfit",
            ),
            directory / name,
        )

    try:
        logger = RunLogger(directory, config, resume=bool(resumed))
        write_json(
            directory / "status.json", dict(status="running", steps=config.steps)
        )
        for step in range(start_step, config.steps + 1):
            if step > start_step:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, config.bf16):
                    output = model(data["phase"], data["mask"], data["dipole"])
                loss, _ = reconstruction_loss(
                    output,
                    data["local"],
                    data["chi"],
                    data["mask"],
                    data["dipole"],
                    weights,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite single-case training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                monitor.check()
                del output, loss
            if (
                step != start_step
                and step % config.log_every
                and step != config.steps
                and not stopped
            ):
                continue
            prediction, metrics = measure()
            if not all(
                math.isfinite(value) for value in metrics.values() if value is not None
            ):
                raise FloatingPointError("Nonfinite overfitting metrics")
            if initial_loss is None:
                initial_loss = metrics["loss"]
                initial_metrics = dict(metrics)
            passed = passes_sanity(metrics, initial_loss, config)
            logger.log(
                "cosmos_overfit",
                step,
                0,
                **metrics,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
            if metrics["chi_nrmse"] is not None and (
                (passed and not best_passed)
                or (passed == best_passed and metrics["chi_nrmse"] < best_error)
            ):
                best_passed = passed
                best_error, best_step, best_metrics = (
                    metrics["chi_nrmse"],
                    step,
                    dict(metrics),
                )
                save_model("model_best.pt")
            if (
                step == 0
                or step % config.figure_every == 0
                or step == config.steps
                or passed
                or stopped
            ):
                logger.figure(
                    reconstruction_figure(
                        data, prediction, f"COSMOS single-case overfit — update {step}"
                    ),
                    "cosmos_overfit",
                    step,
                    0,
                )
            del prediction
            if (passed and config.stop_on_pass) or stopped:
                break
        save_model("model_last.pt")
        atomic_save(
            dict(
                version=CHECKPOINT_VERSION,
                loss_version=LOSS_VERSION,
                model_config=asdict(model.config),
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                config=asdict(config),
                step=step,
            ),
            directory / "overfit_last.pt",
        )
        summary = dict(
            passed=best_passed,
            status="interrupted"
            if stopped
            else "passed"
            if best_passed
            else "failed_sanity",
            steps=step,
            best_step=best_step,
            initial=initial_metrics,
            best=best_metrics,
            target_nrmse=config.target_nrmse,
            target_loss_ratio=config.target_loss_ratio,
            final=metrics,
            final_passed=passed,
            parameter_count=sum(p.numel() for p in model.parameters()),
            training_phases=phases,
            loss_ratio=metrics["loss"] / max(initial_loss, 1e-30),
            simulation=config.cosmos_simulation,
            interpretation="Training fit on one COSMOS simulation; not a generalization benchmark",
            loss_version=LOSS_VERSION,
        )
        write_json(directory / "summary.json", summary)
        write_json(
            directory / "status.json",
            {key: summary[key] for key in ("status", "passed", "steps", "best_step")},
        )
        # Reuse the ordinary evaluation/export path for the best observed fit.
        evaluate_checkpoint(
            directory / "model_best.pt",
            [case],
            directory / "evaluation",
            str(device),
            bf16=config.bf16,
        )
        summary["memory"] = monitor.metrics()
        write_json(directory / "summary.json", summary)
        return directory
    except BaseException as error:
        write_json(
            directory / "status.json",
            dict(status="failed", step=step, error=str(error)),
        )
        raise
    finally:
        if logger:
            logger.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = OverfitConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        kwargs = dict(default=argparse.SUPPRESS)
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        elif field.name == "features":
            kwargs.update(type=int, nargs="+")
        else:
            kwargs["type"] = type(value) if value is not None else str
        parser.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    overrides = vars(parser.parse_args(argv))
    values = asdict(defaults)
    if overrides.get("resume"):
        checkpoint = torch.load(
            overrides["resume"], weights_only=True, map_location="cpu", mmap=True
        )
        values.update(checkpoint["config"])
        del checkpoint
    values.update(overrides)
    values["features"] = tuple(values["features"])
    directory = run_overfit(OverfitConfig(**values))
    print(f"Single-case artifacts: {directory}")
    summary = json.loads((directory / "summary.json").read_text())
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
