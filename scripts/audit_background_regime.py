"""Data-only audit of the realized background regime; no model weights or training.

The filter's accuracy is a near-deterministic function of the background/local
amplitude ratio, and its only input channel is a Laplacian, so this reports both
the amplitude ratio and the Laplacian-channel contamination the network actually
sees. External reference values are printed alongside so a configured mixture can
be checked against the regime it is meant to cover before any GPU time is spent.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dnqsm.data import (
    BalancedSourceSampler,
    ShapeFieldSynthesizer,
    TrainingVolumes,
    make_subject_split,
    training_records,
)
from dnqsm.evaluation import load_cosmos_case, load_sim2_case
from dnqsm.physics import apply_dipole
from dnqsm.training import background_options, parse_args

# The same fixed stencil the filter uses, as a non-trainable reference.
LAPLACIAN = torch.tensor(
    [
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
        [
            [3 / 26, 3 / 13, 3 / 26],
            [3 / 13, -44 / 13, 3 / 13],
            [3 / 26, 3 / 13, 3 / 26],
        ],
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
    ]
)[None, None]


def interior(mask):
    """Voxels whose complete 3x3x3 stencil lies inside the ROI."""
    return F.avg_pool3d(mask.float(), 3, stride=1, padding=1) > 1 - 1e-6


def roi_rms(volume, mask):
    selected = volume[mask > 0]
    return float(selected.pow(2).mean().sqrt()) if selected.numel() else 0.0


def ratios(background, local, mask):
    """Amplitude ratio, and the same ratio in the Laplacian the filter receives."""
    valid = interior(mask)

    def laplacian_rms(volume):
        response = F.conv3d(volume * mask, LAPLACIAN, padding=1)[valid]
        return float(response.pow(2).mean().sqrt()) if response.numel() else 0.0

    local_laplacian = laplacian_rms(local)
    return (
        roi_rms(background, mask) / max(roi_rms(local, mask), 1e-30),
        laplacian_rms(background) / max(local_laplacian, 1e-30),
    )


def external_reference():
    rows = {}
    for loader in (load_cosmos_case, load_sim2_case):
        case = loader()
        local = apply_dipole(case.chi * case.mask, case.dipole) * case.mask
        amplitude, laplacian = ratios(
            (case.phase - local) * case.mask, local, case.mask
        )
        rows[case.name] = dict(bg_local=amplitude, laplacian_contamination=laplacian)
    return rows


def audit(config, samples, epoch):
    records = training_records(config.data_dir, config.source_manifest)
    train, _ = make_subject_split(records, config.val_fraction, config.seed)
    dataset = TrainingVolumes(
        train,
        config.cache_dir,
        seed=config.seed,
        chi_scale_mode="rms_mixture",
        crop_probability=config.crop_probability,
        crop_resize=config.crop_resize,
        chi_augmentation=config.chi_augmentation,
    )
    dataset.set_epoch(epoch)
    sampler = BalancedSourceSampler(
        dataset.records, config.synthetic_fraction, config.seed
    )
    sampler.set_epoch(epoch)
    order = list(sampler)[: samples or None]
    synthesizer = ShapeFieldSynthesizer("cpu")
    options = background_options(config)

    amplitudes, contaminations, clean = [], [], 0
    for index in order:
        sample = dataset[index]
        batch = {
            key: (
                sample[key][None]
                if torch.is_tensor(sample[key])
                else torch.tensor([sample[key]])
            )
            for key in ("chi", "mask", "dilated", "seed")
        }
        data = synthesizer(batch, **options)
        amplitude, contamination = ratios(
            data["background"], data["local"], data["mask"]
        )
        if amplitude < 1e-6:
            clean += 1
            continue
        amplitudes.append(amplitude)
        contaminations.append(contamination)

    total = len(order)
    quantiles = [10, 25, 50, 75, 90]
    return dict(
        config=dict(
            clean_probability=config.clean_probability,
            weak_probability=config.weak_probability,
            strong_probability=config.strong_probability,
            strong_min_ratio=config.strong_min_ratio,
            background_profile=config.background_profile,
        ),
        sampled=total,
        epoch=epoch,
        clean_fraction=clean / total,
        fraction_bg_local_ge_10=sum(v >= 10 for v in amplitudes) / total,
        fraction_bg_local_ge_33=sum(v >= 33 for v in amplitudes) / total,
        bg_local_quantiles={
            f"p{q}": float(np.percentile(amplitudes, q)) for q in quantiles
        },
        laplacian_contamination_quantiles={
            f"p{q}": float(np.percentile(contaminations, q)) for q in quantiles
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64, help="0 audits the epoch")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-external", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)

    config = parse_args(["--config", str(args.config), "--device", "cpu"])
    result = audit(config, args.samples, args.epoch)
    if not args.no_external:
        result["external_reference"] = external_reference()
    print(json.dumps(result, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
