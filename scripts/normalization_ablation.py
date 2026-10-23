"""Matched, bounded adaptation experiments; never changes an existing training run.

All variants start from the same checkpoint with fresh AdamW state and the same
per-seed samples. Experimental modules live here until evidence supports promotion.
"""

import argparse
import copy
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dnqsm.data import ShapeFieldSynthesizer, TrainingVolumes, collate_shapes
from dnqsm.diagnostics import quality_totals, summarize_quality
from dnqsm.evaluation import load_cosmos_case
from dnqsm.losses import LossNormalization, LossWeights, reconstruction_loss
from dnqsm.model import DNQSM, ModelConfig
from dnqsm.physics import (
    apply_dipole,
    apply_multiplier,
    cutoff_kernels,
    masked_mean,
    masked_nrmse,
    masked_rms,
)
from dnqsm.runtime import atomic_save, autocast_context

VARIANTS = (
    "baseline",
    "fp32",
    "filter_edge_none",
    "null_edge_none",
    "filter_masked",
    "filter_group",
    "filter_residual_none",
    "mask_channels",
    "interior_scale",
    "interior_scale_masks",
    "physics_huber",
    "conditional_interior_scale",
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def interior_mask(mask):
    # A valid 3x3x3 stencil has no exterior voxels, including at grid edges.
    with torch.autocast(mask.device.type, enabled=False):
        return (F.avg_pool3d(mask.float(), 3, 1, 1) > 1 - 1e-6).float()


class MaskedInstanceNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())
        self.eps = original.eps
        self.mask = None

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(x.device.type, enabled=False):
            mask = F.interpolate(self.mask.float(), size=x.shape[-3:], mode="nearest")
            count = mask.sum((-3, -2, -1), keepdim=True).clamp_min(1)
            mean = (x.float() * mask).sum((-3, -2, -1), keepdim=True) / count
            var = ((x.float() - mean).square() * mask).sum(
                (-3, -2, -1), keepdim=True
            ) / count
            value = (x.float() - mean) * torch.rsqrt(var + self.eps)
            value = (
                value * self.weight[None, :, None, None, None]
                + self.bias[None, :, None, None, None]
            )
        return value.to(dtype)


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        cin, cout = block[0].in_channels, block[0].out_channels
        self.skip = nn.Conv3d(cin, cout, 1, bias=False)
        nn.init.dirac_(self.skip.weight)

    def forward(self, x):
        return self.skip(x) + 0.1 * self.block(x)


class ExperimentModel(DNQSM):
    def __init__(self, config, state, variant):
        # These recorded ablations predate the raw-field output skip.
        super().__init__(
            replace(
                config,
                filter_output_skip=False,
                filter_laplacian_residual=False,
                filter_refinement_features=0,
            )
        )
        self.load_state_dict(state)
        self.variant = variant
        branch = self.null_network if variant == "null_edge_none" else self.filter
        if variant.endswith("edge_none"):
            for block in (branch.downs[0], branch.up_blocks[-1]):
                block[1] = nn.Identity()
                block[4] = nn.Identity()
        if variant in ("filter_masked", "filter_group", "filter_residual_none"):
            for block in [*branch.downs, branch.bottleneck, *branch.up_blocks]:
                for index in (1, 4):
                    old = block[index]
                    if variant == "filter_masked":
                        block[index] = MaskedInstanceNorm(old)
                    elif variant == "filter_group":
                        channels = old.num_features
                        groups = max(g for g in range(1, 9) if channels % g == 0)
                        block[index] = nn.GroupNorm(groups, channels, eps=old.eps)
                        block[index].load_state_dict(old.state_dict())
                    else:
                        block[index] = nn.Identity()
            if variant == "filter_residual_none":
                branch.downs = nn.ModuleList(ResidualBlock(b) for b in branch.downs)
                branch.bottleneck = ResidualBlock(branch.bottleneck)
                branch.up_blocks = nn.ModuleList(
                    ResidualBlock(b) for b in branch.up_blocks
                )
        if variant in ("mask_channels", "interior_scale_masks"):
            old = self.filter.downs[0][0]
            channels = 3 if variant == "mask_channels" else 4
            conv = nn.Conv3d(channels, old.out_channels, 3, padding=1)
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, :1].copy_(old.weight)
                conv.bias.copy_(old.bias)
            self.filter.downs[0][0] = conv

    def predict_local_field(self, total_field, mask):
        # Baseline explicitly retains the historical autocast implementation,
        # independent of any subsequently promoted production changes.
        field = total_field.float() * mask
        total_scale = masked_rms(field, mask, self.config.scale_floor)
        with torch.autocast(field.device.type, enabled=False):
            valid = interior_mask(mask)
        if self.variant == "baseline":
            lap = F.conv3d(field / total_scale, self.laplacian_kernel, padding=1)
        else:
            with torch.autocast(field.device.type, enabled=False):
                lap = F.conv3d(
                    field / total_scale, self.laplacian_kernel.float(), padding=1
                )
        scale = total_scale
        inputs = lap
        if (
            self.variant.startswith("interior_scale")
            or self.variant == "conditional_interior_scale"
        ):
            with torch.autocast(field.device.type, enabled=False):
                scale = masked_rms(
                    lap.float() * total_scale, valid, self.config.scale_floor
                )
                has_interior = valid.sum((-3, -2, -1), keepdim=True) > 0
                scale = torch.where(has_interior, scale, total_scale)
                inputs = valid * lap.float() * total_scale / scale
                if self.variant == "conditional_interior_scale":
                    # Fixed before evaluation: use interior conditioning only
                    # when total RMS is at least 32 times interior derivative RMS.
                    selected = total_scale / scale >= 32.0
                    inputs = torch.where(selected, inputs, lap.float())
                    scale = torch.where(selected, scale, total_scale)
            if self.variant == "interior_scale_masks":
                # Preserve boundary information on a separately scaled channel.
                inputs = torch.cat((inputs, lap.float() * (1 - valid), mask, valid), 1)
        elif self.variant == "mask_channels":
            inputs = torch.cat((lap.float(), mask, valid), 1)
        for module in self.filter.modules():
            if isinstance(module, MaskedInstanceNorm):
                module.mask = mask
        return self.filter(inputs).float() * scale * mask


def score(model, batches, synthesizer, options, device):
    model.eval()
    totals = defaultdict(float)
    records = []
    # Synthesis caches kernels that later participate in training autograd.
    with torch.no_grad():
        for batch in batches:
            data = synthesizer(batch, **options)
            with autocast_context(device):
                pred = model(data["phase"], data["mask"], data["dipole"])
            for k, v in quality_totals(pred, data).items():
                totals[k] += v
            mask = data["mask"]
            valid = interior_mask(mask)
            for i, subject in enumerate(batch["subject"]):
                m = mask[i : i + 1]
                target, output = data["local"][i : i + 1], pred.local_field[i : i + 1]
                centered = target - masked_mean(target, m)[..., None, None, None]
                gain = masked_mean(output * centered, m) / masked_mean(
                    centered.square(), m
                ).clamp_min(1e-12)
                row = dict(
                    subject=subject,
                    seed=int(batch["seed"][i]),
                    chi_nrmse=masked_nrmse(
                        pred.chi[i : i + 1], data["chi"][i : i + 1], m
                    )[0],
                    local_nrmse=masked_nrmse(output, target, m)[0],
                    contrast_gain=float(gain.item()),
                    mean_bias=float(masked_mean(output - target, m).item()),
                )
                for label, region in (
                    ("interior", valid[i : i + 1]),
                    ("boundary", m - valid[i : i + 1]),
                ):
                    row[label + "_local_nrmse"] = (
                        masked_nrmse(output, target, region)[0]
                        if bool(region.any())
                        else None
                    )
                records.append(row)
    return dict(metrics=summarize_quality(totals), samples=records)


def oracle(case, device):
    chi, mask, d = (x.to(device) for x in (case.chi, case.mask, case.dipole))
    full = apply_dipole(chi * mask, d)
    results = {}
    for threshold in (0.05, 0.1, 0.2):
        inverse, projection = cutoff_kernels(d, threshold)
        for name, field in (("full", full), ("masked", full * mask)):
            initial = apply_multiplier(field, inverse)
            # Best full-grid L2 correction within the allowed Fourier subspace.
            ideal = initial + apply_multiplier(chi * mask - initial, projection)
            results[f"{name}_{threshold}"] = dict(
                tkd_roi_nrmse=masked_nrmse(initial, chi, mask)[0],
                oracle_roi_nrmse=masked_nrmse(ideal, chi, mask)[0],
                oracle_full_grid_nrmse=masked_nrmse(
                    ideal, chi * mask, torch.ones_like(mask)
                )[0],
            )

    # Solve the masked normal equations with a tiny ridge using conjugate gradients.
    # This is an oracle diagnostic, not an inference algorithm using ground truth.
    def op(x):
        return apply_dipole(apply_dipole(x, d) * mask, d) + 1e-6 * x

    rhs = apply_dipole(full * mask, d)
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    p = r.clone()
    rr = (r * r).sum()
    for _ in range(60):
        ap = op(p)
        alpha = rr / (p * ap).sum().clamp_min(1e-25)
        x += alpha * p
        r -= alpha * ap
        new = (r * r).sum()
        p = r + (new / rr.clamp_min(1e-25)) * p
        rr = new
    results["masked_cg_60"] = dict(
        chi_nrmse=masked_nrmse(x, chi, mask)[0],
        local_nrmse=masked_nrmse(apply_dipole(x, d), full, mask)[0],
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Matched fresh initialization instead of checkpoint adaptation",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-volume experiments")
    config = json.loads((args.run / "config.json").read_text())
    split = json.loads((args.run / "split.json").read_text())
    checkpoint = args.run / "model_last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_config = ModelConfig(
        **{
            **saved["model_config"],
            "checkpoint_blocks": False,
            "filter_output_skip": saved["model_config"].get(
                "filter_output_skip", False
            ),
            "filter_laplacian_residual": saved["model_config"].get(
                "filter_laplacian_residual", False
            ),
            "filter_refinement_features": saved["model_config"].get(
                "filter_refinement_features", 0
            ),
            "activation": saved["model_config"].get("activation", "celu"),
        }
    )
    if (
        model_config.filter_output_skip
        or model_config.filter_laplacian_residual
        or model_config.filter_refinement_features
    ):
        raise ValueError(
            "These ablations require a checkpoint without an output skip, Laplacian residual, or refinement block"
        )
    if model_config.filter_input_version != 2:
        raise ValueError("Experiments require the masked Laplacian-only architecture")
    initial_state = copy.deepcopy(saved["model"])
    training = split["train"]
    representatives = {r["subject"]: r for r in reversed(split["validation"])}
    heldout = [representatives[s] for s in sorted(representatives)]
    assert not {r["subject"] for r in training} & set(representatives)
    validation = TrainingVolumes(
        heldout,
        config["cache_dir"],
        validation=True,
        views=3,
        seed=config["seed"] + 100000,
        chi_scale_mode="rms_mixture",
    )
    val_batches = list(
        DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    )
    val_options = dict(config["validation_synthesis"])
    train_options = dict(
        background_mode=config["background_profile"],
        curriculum_progress=1.0,
        strong_probability=config["strong_probability"],
        clean_probability=config["clean_probability"],
        weak_probability=config["weak_probability"],
        strong_min_ratio=config["strong_min_ratio"],
        snr_min=config["snr_min"],
        snr_max=config["snr_max"],
    )
    synthesizer = ShapeFieldSynthesizer(device)
    normalization = LossNormalization(mse_input_ratio_cap=config["mse_input_ratio_cap"])
    weights = LossWeights(local=1.5, chi=1, physics=1, extension=0, local_gradient=0)
    manifest = dict(
        checkpoint=str(checkpoint),
        checkpoint_step=saved["step"],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        model_config=asdict(model_config),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seeds=args.seeds,
        variants=args.variants,
        train_options=train_options,
        validation_options=val_options,
        validation_subjects=sorted(representatives),
        torch_version=torch.__version__,
        device=torch.cuda.get_device_name(),
        selection="final held-out chi NRMSE with bucket checks; external cases reporting only",
        initialization="fresh" if args.fresh else "checkpoint",
        limitation="Bounded trials, not convergence comparisons",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        keys = (
            "checkpoint_sha256",
            "steps",
            "batch_size",
            "learning_rate",
            "seeds",
            "variants",
            "train_options",
            "validation_options",
            "validation_subjects",
        )
        changed = [key for key in keys if previous[key] != manifest[key]]
        if previous.get("initialization", "checkpoint") != manifest["initialization"]:
            changed.append("initialization")
        if changed:
            raise ValueError(f"Existing experiment protocol changed: {changed}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write(args.output / f"continuation_{stamp}.json", manifest)
    else:
        write(manifest_path, manifest)
    if args.diagnostics:
        case = load_cosmos_case(config["cosmos_simulation"])
        write(args.output / "oracle.json", oracle(case, device))
        kernel = initial_state["laplacian_kernel"].to(device)
        constant = torch.ones(1, 1, 24, 24, 24, device=device)
        with autocast_context(device):
            low = F.conv3d(constant, kernel, padding=1)
        with torch.autocast("cuda", enabled=False):
            high = F.conv3d(constant, kernel, padding=1)
        write(
            args.output / "operator.json",
            dict(
                bf16_constant_response=float(low[..., 2:-2, 2:-2, 2:-2].float().mean()),
                fp32_constant_response=float(high[..., 2:-2, 2:-2, 2:-2].mean()),
            ),
        )
        print("diagnostics complete", flush=True)
    for seed in args.seeds:
        if args.fresh:
            torch.manual_seed(seed)
            initial_state = copy.deepcopy(DNQSM(model_config).state_dict())
        generator = torch.Generator().manual_seed(seed)
        pools = [
            [
                i
                for i, r in enumerate(training)
                if r["subject"].startswith("synthetic:") == synthetic
            ]
            for synthetic in (False, True)
        ]
        indices = []
        for i in range(args.steps * args.batch_size):
            pool = pools[int(float(torch.rand((), generator=generator)) < 0.25)]
            index = pool[int(torch.randint(len(pool), (), generator=generator))]
            indices.append(index + (i + 1) * len(training))
        for variant in args.variants:
            directory = args.output / f"{variant}_seed{seed}"
            directory.mkdir(exist_ok=True)
            if (directory / "result.json").exists() and (
                directory / "model.pt"
            ).exists():
                continue
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = ExperimentModel(model_config, initial_state, variant).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=1e-4,
                foreach=False,
            )
            train_set = TrainingVolumes(
                training,
                config["cache_dir"],
                seed=seed + 1000000,
                chi_scale_mode="rms_mixture",
                crop_probability=0.2,
                crop_resize=True,
            )
            loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                sampler=indices,
                num_workers=2,
                collate_fn=collate_shapes,
                pin_memory=True,
            )
            start = time.monotonic()
            before = score(model, val_batches, synthesizer, val_options, device)
            write(directory / "before.json", before)
            print(
                json.dumps(
                    dict(
                        variant=variant,
                        seed=seed,
                        stage="before",
                        metrics=before["metrics"],
                    )
                ),
                flush=True,
            )
            clipped = 0
            with (directory / "train.jsonl").open("w") as log:
                for step, groups in enumerate(loader, 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    count = sum(b["chi"].shape[0] for b in groups)
                    step_loss = 0.0
                    for batch in groups:
                        data = synthesizer(batch, **train_options)
                        with autocast_context(device):
                            pred = model(data["phase"], data["mask"], data["dipole"])
                        loss, terms = reconstruction_loss(
                            pred,
                            data["local"],
                            data["chi"],
                            data["mask"],
                            data["dipole"],
                            weights,
                            normalization=normalization,
                            input_field=data["phase"],
                        )
                        if variant == "physics_huber":
                            scale = masked_rms(
                                data["local"], data["mask"], normalization.field_floor
                            )
                            residual = (
                                apply_dipole(pred.chi, data["dipole"]) - data["local"]
                            ) / scale
                            robust = masked_mean(
                                F.huber_loss(
                                    residual,
                                    torch.zeros_like(residual),
                                    reduction="none",
                                    delta=1.0,
                                ),
                                data["mask"],
                            ).mean()
                            loss = loss - terms["physics"] + robust
                        fraction = batch["chi"].shape[0] / count
                        (loss * fraction).backward()
                        step_loss += float(loss.detach()) * fraction
                    norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 10, error_if_nonfinite=True
                        )
                    )
                    clipped += norm > 10
                    optimizer.step()
                    row = dict(
                        step=step,
                        loss=step_loss,
                        gradient_norm=norm,
                        seconds=time.monotonic() - start,
                    )
                    log.write(json.dumps(row) + "\n")
                    if step % 20 == 0:
                        log.flush()
                        print(
                            json.dumps(dict(variant=variant, seed=seed, **row)),
                            flush=True,
                        )
                    del pred, data, loss, terms
            after = score(model, val_batches, synthesizer, val_options, device)
            result = dict(
                variant=variant,
                seed=seed,
                before=before["metrics"],
                after=after["metrics"],
                clipping_fraction=clipped / args.steps,
                seconds=time.monotonic() - start,
            )
            write(directory / "after.json", after)
            atomic_save(
                dict(
                    model=model.state_dict(),
                    model_config=asdict(model_config),
                    variant=variant,
                    seed=seed,
                    steps=args.steps,
                    source_checkpoint=str(checkpoint),
                ),
                directory / "model.pt",
            )
            write(directory / "result.json", result)
            print(json.dumps(result), flush=True)
            del model, optimizer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
