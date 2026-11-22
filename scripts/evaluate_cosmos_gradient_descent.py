"""Compare TKD with 30-step truncated-kernel descent on a COSMOS-selected model."""

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import torch

from dnqsm.evaluation import load_cosmos_case
from dnqsm.logging import _pyplot, comparison_figure, nrmse_label, write_json
from dnqsm.model import Reconstruction
from dnqsm.physics import (
    SPATIAL,
    apply_dipole,
    apply_multiplier,
    cutoff_kernels,
    masked_mean,
    masked_nrmse,
    masked_rms,
    rfft_kernel,
    truncated_gradient_descent,
)
from dnqsm.runtime import load_trained_model


def metrics(output, case, truncated):
    forward = apply_dipole(output.chi, case.dipole)
    return dict(
        initializer_chi_nrmse=masked_nrmse(output.tkd, case.chi, case.mask)[0],
        final_chi_nrmse=masked_nrmse(output.chi, case.chi, case.mask)[0],
        local_nrmse=masked_nrmse(output.local_field, case.local, case.mask)[0],
        physics_nrmse=masked_nrmse(forward, case.local, case.mask)[0],
        local_forward_residual_rms_ppm=float(
            masked_mean((forward - output.local_field).square(), case.mask).sqrt()
        ),
        truncated_objective_mse=float(
            (apply_multiplier(output.tkd, truncated) - output.local_field)
            .square()
            .mean()
        ),
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cosmos-simulation", type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "summary.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    case = load_cosmos_case(args.cosmos_simulation)
    selected_model = selected_output = None
    selection = []
    for path in args.checkpoints:
        model = load_trained_model(path, "cpu")
        if model.config.cutoff_mode != "hard":
            raise ValueError(
                "The truncated-gradient-descent comparison requires hard-cutoff checkpoints"
            )
        if model.inference_config.window_size:
            raise ValueError("This experiment requires a full-volume checkpoint")
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        start = time.perf_counter()
        output = model.infer(case.phase, case.mask, case.dipole)
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        row = dict(
            checkpoint=str(path.resolve()),
            epoch=saved.get("epoch"),
            step=saved.get("step"),
            cosmos_chi_nrmse=masked_nrmse(output.chi, case.chi, case.mask)[0],
            seconds=time.perf_counter() - start,
        )
        if not selection or row["cosmos_chi_nrmse"] < min(
            r["cosmos_chi_nrmse"] for r in selection
        ):
            selected_model, selected_output = model, output
        selection.append(row)
        write_json(directory / "selection.json", dict(candidates=selection))
        print(json.dumps(dict(event="candidate", **row)), flush=True)
        del model, output, saved
    winner = min(selection, key=lambda row: row["cosmos_chi_nrmse"])
    model, baseline = selected_model, selected_output
    assert not torch.is_autocast_enabled("cpu")
    local = baseline.local_field
    inverse, projection = cutoff_kernels(case.dipole, model.config.threshold)
    truncated = rfft_kernel(case.dipole, local.shape[-3:]) * (
        1 - rfft_kernel(projection, local.shape[-3:])
    )
    step_size = (
        args.step_size
        if args.step_size is not None
        else 1 / float(truncated.square().max())
    )
    torch.testing.assert_close(baseline.tkd, apply_multiplier(local, inverse))
    start = time.perf_counter()
    initial = truncated_gradient_descent(
        local, case.dipole, model.config.threshold, args.iterations, args.step_size
    )
    initializer_seconds = time.perf_counter() - start
    scale = masked_rms(initial, case.mask, model.config.scale_floor)
    start = time.perf_counter()
    residual = model.null_network(initial / scale).float() * scale
    correction = apply_multiplier(residual, projection)
    result = Reconstruction(
        local, initial, correction, initial + correction, baseline.harmonic_field
    )
    null_network_seconds = time.perf_counter() - start
    values = dict(
        tkd=metrics(baseline, case, truncated),
        gradient_descent=metrics(result, case, truncated),
    )
    for record in values.values():
        assert all(math.isfinite(value) for value in record.values())
    relative_support = {}
    rprojection = rfft_kernel(projection, local.shape[-3:])
    for name, volume, forbidden in (
        ("gd_initializer_null_band", initial, rprojection),
        ("gd_correction_retained_band", correction, 1 - rprojection),
    ):
        spectrum = torch.fft.rfftn(volume, dim=SPATIAL)
        ratio = float(
            (spectrum * forbidden).abs().max() / spectrum.abs().max().clamp_min(1e-12)
        )
        assert ratio < 2e-6
        relative_support[name] = ratio
    torch.testing.assert_close(result.chi, initial + correction)
    assert initial.dtype == correction.dtype == result.chi.dtype == torch.float32
    retained = truncated != 0
    response = 1 - (1 - step_size * truncated.double().square()).pow(args.iterations)
    metadata = dict(
        checkpoint=winner,
        checkpoint_sha256=hashlib.sha256(
            Path(winner["checkpoint"]).read_bytes()
        ).hexdigest(),
        cosmos=case.metadata,
        model_config=asdict(model.config),
        device="cpu",
        dtype="float32",
        autocast=False,
        iterations=args.iterations,
        step_size=step_size,
        threshold=model.config.threshold,
        truncation="D_tau = D where abs(D) > threshold, zero elsewhere",
        objective="0.5 * ||D_tau*x - estimated_local_field||_2^2 over the full periodic grid",
        update="x[k+1] = x[k] - step_size * D_tau * (D_tau*x[k] - estimated_local_field)",
        initialization="zeros",
        implementation="complex64 Fourier-domain gradient updates; FP32 real fields",
        null_network="same trained weights, scale normalization and near-null projection; no retraining",
        minimum_retained_response_fraction=float(response[retained].min()),
        initializer_seconds=initializer_seconds,
        null_network_seconds=null_network_seconds,
        learning_updates=0,
        projection_verification=relative_support,
    )
    rows = [("Ground truth susceptibility", case.chi * case.mask, True)]
    for name, volume in (
        ("TKD", baseline.tkd),
        ("TKD + null network", baseline.chi),
        (f"GD {args.iterations}", initial),
        (f"GD {args.iterations} + null network", result.chi),
        ("TKD + null error", baseline.chi - case.chi),
        (f"GD {args.iterations} + null error", result.chi - case.chi),
    ):
        rows.append(
            (
                f"{name}: {nrmse_label(volume, case.chi, case.mask)}"
                if "error" not in name
                else name,
                volume * case.mask,
                True,
            )
        )
    figure = comparison_figure(
        rows,
        f"COSMOS — truncated-kernel gradient descent ({args.iterations} steps), FP32",
    )
    figure.savefig(directory / "comparison.png", dpi=120, bbox_inches="tight")
    _pyplot().close(figure)
    for name, volume in (
        ("chi_reference", case.chi),
        ("local_reference", case.local),
        ("estimated_local_field", local),
        ("tkd", baseline.tkd),
        ("tkd_null", baseline.chi),
        ("gd", initial),
        ("gd_null", result.chi),
        ("mask", case.mask),
    ):
        array = case.restore_volume(volume * case.mask)
        nib.save(nib.Nifti1Image(array, case.affine), directory / f"{name}.nii.gz")
    write_json(
        directory / "selection.json", dict(candidates=selection, selected=winner)
    )
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "summary.json", values)
    shutil.copy2(__file__, directory / "evaluate_cosmos_gradient_descent.py")
    print(
        json.dumps(
            dict(event="complete", selected=winner, results=values, metadata=metadata)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
