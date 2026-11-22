"""Compare saved and physical normalization on external cases without training."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch

from dnqsm.evaluation import load_cases
from dnqsm.model import DNQSM, ResidualInstanceNorm3d
from dnqsm.physics import masked_nrmse, masked_rms
from dnqsm.runtime import autocast_context, configure_memory_budget, load_trained_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-memory-gb", type=float, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    configure_memory_budget(device, args.cuda_memory_gb)
    directory = args.directory or Path("artifacts/diagnostics") / (
        "physical_normalization_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    directory.mkdir(parents=True, exist_ok=False)
    source = load_trained_model(args.checkpoint, device)
    config = replace(
        source.config,
        field_normalization="laplacian_rms",
        normalization_layer="residual_instance",
        conditional_laplacian_threshold=0,
        laplacian_extension_mode="quadratic",
        laplacian_extension_iters=6,
    )
    candidate = DNQSM(config).to(device).eval()
    candidate.initialize_from(source)
    candidate.inference_config = source.inference_config
    for name, value in source.state_dict().items():
        assert torch.equal(candidate.state_dict()[name], value), name
    gates = [m for m in candidate.modules() if isinstance(m, ResidualInstanceNorm3d)]
    assert gates and all(float(m.residual_gain) == 0 for m in gates)
    report = dict(
        source_checkpoint=str(args.checkpoint.resolve()),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        model_config=asdict(config),
        device=str(device),
        learning_updates=0,
        transferred_weights_verified=True,
        normalization_residuals=len(gates),
        normalization_residuals_initial_value=0,
        parameters=sum(p.numel() for p in candidate.parameters()),
        candidate_checkpoint=str((directory / "model_candidate.pt").resolve()),
        rows=[],
    )

    def save():
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    for case in load_cases():
        phase, mask, dipole, chi = (
            t.to(device) for t in (case.phase, case.mask, case.dipole, case.chi)
        )
        base = None
        for name, model, gain, offset in (
            ("saved_preprocessing", source, 1, 0),
            ("physical_normalization", candidate, 1, 0),
            ("constant_offset", candidate, 1, 0.2),
            ("double_amplitude", candidate, 2, 0),
        ):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with autocast_context(device):
                output = model.infer(gain * phase + offset * mask, mask, dipole)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            target = gain * chi
            row = dict(
                dataset=case.name,
                experiment=name,
                input_gain=gain,
                added_constant_ppm=offset,
                chi_nrmse=masked_nrmse(output.chi, target, mask)[0],
                tkd_nrmse=masked_nrmse(output.tkd, target, mask)[0],
                chi_rms_over_reference=float(
                    masked_rms(output.chi, mask) / masked_rms(target, mask)
                ),
                seconds=time.perf_counter() - started,
            )
            if name == "physical_normalization":
                base = output.chi.clone()
            if base is not None and name in ("constant_offset", "double_amplitude"):
                row["equivariance_error_percent"] = masked_nrmse(
                    output.chi, gain * base, mask
                )[0]
            report["rows"].append(row)
            save()
            print(json.dumps(row), flush=True)
        del phase, mask, dipole, chi, output, base
    torch.save(
        dict(
            version=1,
            model_config=asdict(config),
            inference_config=asdict(candidate.inference_config),
            model={name: t.cpu() for name, t in candidate.state_dict().items()},
            source_checkpoint=str(args.checkpoint.resolve()),
            learning_updates=0,
        ),
        directory / "model_candidate.pt",
    )
    report["status"] = "passed"
    report["interpretation"] = (
        "Residual normalization paths are zero-initialized and require adaptation. "
        "The measured pretrained change is physical scaling and centered bypass; "
        "this audit does not establish gains from learned residual paths. "
        "Candidate is not promoted to the default trained checkpoint."
    )
    save()
    print(
        json.dumps(dict(status="passed", directory=str(directory.resolve()))),
        flush=True,
    )


if __name__ == "__main__":
    main()
