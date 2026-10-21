"""Summarize completed trials and evaluate external cases without selecting on them."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from normalization_ablation import ExperimentModel

from dnqsm.evaluation import load_cosmos_case, load_sim2_case
from dnqsm.model import ModelConfig
from dnqsm.physics import masked_nrmse
from dnqsm.runtime import (
    CHECKPOINT_VERSION,
    atomic_save,
    autocast_context,
    load_trained_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--export-conditional", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    config = json.loads(
        (Path(manifest["checkpoint"]).parent / "config.json").read_text()
    )
    torch.set_num_threads(4)
    rows = []
    cases = None
    source = None
    if args.external or args.export_conditional:
        if args.external:
            cases = [load_cosmos_case(config["cosmos_simulation"]), load_sim2_case()]
        source = torch.load(
            manifest["checkpoint"], weights_only=True, map_location="cpu"
        )
    for path in sorted(args.directory.glob("*/result.json")):
        result = json.loads(path.read_text())
        if (
            args.export_conditional
            and result["variant"] == "conditional_interior_scale"
        ):
            saved = torch.load(
                path.parent / "model.pt", weights_only=True, map_location="cpu"
            )
            experimental = (
                ExperimentModel(
                    ModelConfig(**saved["model_config"]),
                    source["model"],
                    saved["variant"],
                )
                .cuda()
                .eval()
            )
            experimental.load_state_dict(saved["model"])
            production_config = dict(
                saved["model_config"],
                laplacian_fp32=True,
                conditional_laplacian_threshold=32.0,
                filter_output_skip=False,
            )
            export = path.parent / "model_production.pt"
            atomic_save(
                dict(
                    version=CHECKPOINT_VERSION,
                    model_config=production_config,
                    model=saved["model"],
                    step=saved["steps"],
                    source_step=manifest["checkpoint_step"],
                    source_checkpoint=manifest["checkpoint"],
                    adaptation_seed=result["seed"],
                    validation_metrics=result["after"],
                ),
                export,
            )
            production = load_trained_model(export, "cuda:0")
            field = (
                torch.randn(
                    2, 1, 24, 24, 24, generator=torch.Generator().manual_seed(719)
                )
                * 0.01
            )
            field[1] += 100
            field = field.cuda()
            mask = torch.zeros_like(field)
            mask[..., 2:-2, 2:-2, 2:-2] = 1
            with torch.no_grad(), autocast_context(field.device):
                expected = experimental.predict_local_field(field, mask)
                actual = production.predict_local_field(field, mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            (path.parent / "production_parity.json").write_text(
                json.dumps(
                    dict(
                        export=str(export),
                        max_absolute_error=float((actual - expected).abs().max()),
                        cases="ordinary and strong-offset fields; CUDA BF16 CNN and FP32 derivative",
                    ),
                    indent=2,
                )
                + "\n"
            )
            del production, experimental
            torch.cuda.empty_cache()
        reference_path = (
            args.directory / f"baseline_seed{result['seed']}" / "after.json"
        )
        paired = None
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())["samples"]
            current = json.loads((path.parent / "after.json").read_text())["samples"]
            lookup = {(r["subject"], r["seed"]): r for r in reference}
            subjects = defaultdict(list)
            for row in current:
                ref = lookup[(row["subject"], row["seed"])]
                subjects[row["subject"]].append(row["chi_nrmse"] - ref["chi_nrmse"])
            differences = torch.tensor([sum(v) / len(v) for v in subjects.values()])
            indices = torch.randint(
                len(differences),
                (10000, len(differences)),
                generator=torch.Generator().manual_seed(0),
            )
            bounds = torch.quantile(
                differences[indices].mean(1), torch.tensor([0.025, 0.975])
            )
            paired = dict(
                mean_delta=float(differences.mean()),
                lower=float(bounds[0]),
                upper=float(bounds[1]),
                subjects=len(subjects),
                method="Descriptive paired subject bootstrap, 10000 draws; no multiple-comparison correction",
            )
        if args.external and not (path.parent / "external.json").exists():
            saved = torch.load(
                path.parent / "model.pt", weights_only=True, map_location="cpu"
            )
            model = ExperimentModel(
                ModelConfig(**saved["model_config"]), source["model"], saved["variant"]
            )
            model.load_state_dict(saved["model"])
            model = model.cuda().eval()
            scores = {}
            with torch.no_grad():
                for case in cases:
                    field, mask, dipole, target = [
                        v.cuda() for v in (case.phase, case.mask, case.dipole, case.chi)
                    ]
                    with autocast_context(field.device):
                        output = model(field, mask, dipole)
                    scores[case.name] = dict(
                        chi_nrmse=masked_nrmse(output.chi, target, mask)[0],
                        tkd_nrmse=masked_nrmse(output.tkd, target, mask)[0],
                    )
                    if case.local is not None:
                        scores[case.name]["local_nrmse"] = masked_nrmse(
                            output.local_field, case.local.cuda(), mask
                        )[0]
            (path.parent / "external.json").write_text(
                json.dumps(scores, indent=2) + "\n"
            )
            del model
            torch.cuda.empty_cache()
        external = path.parent / "external.json"
        rows.append(
            dict(
                variant=result["variant"],
                seed=result["seed"],
                **{
                    k: result["after"][k]
                    for k in (
                        "chi_nrmse",
                        "local_nrmse",
                        "bucket_clean_local_nrmse",
                        "bucket_typical_local_nrmse",
                        "bucket_strong_local_nrmse",
                    )
                },
                clipping_fraction=result["clipping_fraction"],
                paired_chi_delta=paired,
                external=json.loads(external.read_text())
                if external.exists()
                else None,
            )
        )
    (args.directory / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("variant seed chi local clean typical strong clipped COSMOS sim2")
    for r in sorted(rows, key=lambda r: r["chi_nrmse"]):
        ext = r["external"] or {}
        print(
            r["variant"],
            r["seed"],
            *[
                round(r[k], 3)
                for k in (
                    "chi_nrmse",
                    "local_nrmse",
                    "bucket_clean_local_nrmse",
                    "bucket_typical_local_nrmse",
                    "bucket_strong_local_nrmse",
                    "clipping_fraction",
                )
            ],
            *[
                round(ext[name]["chi_nrmse"], 3) if name in ext else "-"
                for name in ("cosmos", "sim2")
            ],
        )


if __name__ == "__main__":
    main()
