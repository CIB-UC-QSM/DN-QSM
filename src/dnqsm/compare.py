"""Compare a trained DN-QSM baseline suite on COSMOS and sim2."""

import argparse
import json
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path

import torch

from .baselines import INVERSES, DirectSusceptibility, LocalFieldBaseline
from .evaluation import _export_volumes, load_cases
from .experiment_metrics import reconstruction_metrics
from .inference import SlidingWindowConfig
from .logging import _pyplot, comparison_figure, nrmse_label, write_json
from .model import LocalFieldEstimator, ModelConfig
from .runtime import autocast_context, load_trained_model, require_device
from .train_comparison import DIRECT, DNQSM_ARM, MANIFEST_VERSION


def load_comparison(manifest_path, device):
    """Yield named models with the selected comparison weights."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest["version"] != MANIFEST_VERSION:
        raise ValueError("Unsupported comparison manifest version")
    stages = manifest["stages"]
    model_config = dict(manifest["model_config"])
    model_config["features"] = tuple(model_config["features"])
    config = ModelConfig(**model_config)

    def load(module, stage):
        checkpoint = path.parent / stages[stage]["checkpoint"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if (
            saved.get("comparison_version") != MANIFEST_VERSION
            or saved.get("stage") != stage
        ):
            raise ValueError(
                f"Unexpected comparison checkpoint for {stage}: {checkpoint}"
            )
        module.load_state_dict(saved["model"])

    direct = DirectSusceptibility(config)
    load(direct, DIRECT)
    yield DIRECT, direct.to(device).eval()
    del direct

    estimator = LocalFieldEstimator(config)
    load(estimator, "local_field")
    estimator.to(device).eval()
    for inverse in INVERSES:
        baseline = LocalFieldBaseline(estimator, inverse, **manifest["inverses"])
        yield f"model2_{inverse}", baseline.eval()
        baseline = LocalFieldBaseline(
            estimator, inverse, correction=True, **manifest["inverses"]
        )
        stage = f"model3_{inverse}"
        load(baseline.residual, stage)
        yield stage, baseline.to(device).eval()
        del baseline
    del estimator
    yield (
        DNQSM_ARM,
        load_trained_model(path.parent / stages[DNQSM_ARM]["checkpoint"], device),
    )


def _evaluate_output(
    case, output, phase, roi, dipole, chi, local, directory, arm, threshold
):
    metrics = reconstruction_metrics(
        output, chi, roi, dipole, local, hfen=False, threshold=threshold
    )[0]
    metrics["initial_chi_nrmse"] = metrics.pop("tkd_nrmse")
    rows = [
        ("Total field (ppm)", phase * roi, False),
        ("Reference susceptibility (ppm)", chi * roi, True),
        (nrmse_label(output.chi, chi, roi), output.chi * roi, True),
        ("Susceptibility error (ppm)", (output.chi - chi) * roi, True),
    ]
    if output.local_field is not None:
        rows.insert(1, ("Estimated local field (ppm)", output.local_field * roi, False))
    if output.initial_chi is not None:
        label = nrmse_label(output.initial_chi, chi, roi)
        rows.insert(
            -2, (f"Initial susceptibility, {label}", output.initial_chi * roi, True)
        )
    directory.mkdir(parents=True)
    figure = comparison_figure(rows, f"{arm} — {case.name.upper()}")
    try:
        figure.savefig(directory / "comparison.png", dpi=100, bbox_inches="tight")
    finally:
        _pyplot().close(figure)
    _export_volumes(case, directory, phase, roi, chi, output, legacy=False)
    write_json(directory / "metrics.json", metrics)
    return metrics


def _difference(metrics, before, after):
    before_value = metrics[before]["chi_nrmse"]
    after_value = metrics[after]["chi_nrmse"]
    return (
        None
        if before_value is None or after_value is None
        else after_value - before_value
    )


@torch.inference_mode()
def compare(
    manifest_path,
    cases,
    output,
    *,
    device="cpu",
    window_size=None,
    bf16=True,
    experiment=None,
):
    """Evaluate each arm on identical inputs and report paired case differences."""
    device = require_device(device)
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    inference = SlidingWindowConfig(**manifest["inference_config"])
    if window_size is not None:
        inference = replace(inference, window_size=window_size)
    if inference.window_size and inference.window_size < 2 ** (
        len(manifest["model_config"]["features"]) + 1
    ):
        raise ValueError("Inference window is too small for the model's U-Net depth")
    if not cases or len({case.name for case in cases}) != len(cases):
        raise ValueError("Provide at least one case and use unique case names")
    for case in cases:
        case.validate()
    output.mkdir(parents=True, exist_ok=False)
    threshold = manifest["model_config"]["threshold"]
    summary = dict(
        manifest=str(manifest_path.resolve()),
        inference=asdict(inference),
        device=str(device),
        bf16=bool(bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()),
        band_threshold=threshold,
        cases={case.name: dict(metadata=case.metadata, arms={}) for case in cases},
    )
    if experiment is not None:
        summary["comet_experiment_key"] = experiment.get_key()
        experiment.log_parameters(
            {
                "inference": summary["inference"],
                "device": str(device),
                "bf16": summary["bf16"],
                "band_threshold": threshold,
            },
            nested_support=True,
        )
    for arm, model in load_comparison(manifest_path, device):
        for case in cases:
            phase, roi, dipole, chi = (
                value.to(device)
                for value in (case.phase, case.mask, case.dipole, case.chi)
            )
            local = None if case.local is None else case.local.to(device)
            with autocast_context(device, bf16):
                prediction = (
                    model.infer(phase, roi, dipole, **asdict(inference))
                    if arm == DNQSM_ARM
                    else model.infer(phase, roi, dipole, inference=inference)
                )
            directory = output / case.name / arm
            metrics = _evaluate_output(
                case,
                prediction,
                phase,
                roi,
                dipole,
                chi,
                local,
                directory,
                arm,
                threshold,
            )
            summary["cases"][case.name]["arms"][arm] = metrics
            if experiment is not None:
                experiment.log_metrics(
                    {
                        f"{case.name}/{arm}/{name}": value
                        for name, value in metrics.items()
                        if isinstance(value, (float, int)) and isfinite(value)
                    }
                )
                experiment.log_image(
                    str(directory / "comparison.png"),
                    name=f"{case.name}-{arm}-central-slices.png",
                    metadata={"dataset": case.name, "arm": arm},
                )
            del prediction, phase, roi, dipole, chi, local
        del model

    for case_name, record in summary["cases"].items():
        metrics = record["arms"]
        record["chi_nrmse_deltas_pp"] = {
            f"model3_minus_model2_{inverse}": _difference(
                metrics, f"model2_{inverse}", f"model3_{inverse}"
            )
            for inverse in INVERSES
        }
        record["chi_nrmse_deltas_pp"]["model4_minus_model3_tikhonov"] = _difference(
            metrics, "model3_tikhonov", DNQSM_ARM
        )
        write_json(output / case_name / "summary.json", record)
        if experiment is not None:
            experiment.log_metrics(
                {
                    f"{case_name}/chi_nrmse_deltas_pp/{name}": value
                    for name, value in record["chi_nrmse_deltas_pp"].items()
                    if value is not None and isfinite(value)
                }
            )
    write_json(output / "summary.json", summary)
    if experiment is not None:
        experiment.log_asset(str(output / "summary.json"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=("cosmos", "sim2"), default=("cosmos", "sim2")
    )
    parser.add_argument("--cosmos-simulation", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--sim2-pad", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    device = require_device(args.device)
    torch.set_num_threads(args.threads)
    cases = load_cases(
        tuple(args.datasets),
        cosmos_simulation=args.cosmos_simulation,
        data_dir=args.data_dir,
        sim2_pad=args.sim2_pad,
    )
    compare(
        args.manifest,
        cases,
        args.output_dir,
        device=device,
        window_size=args.window_size,
        bf16=args.bf16,
    )
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
