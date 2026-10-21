"""Descriptive paired-study metrics, with full-grid Fourier band accounting."""

import numpy as np
import torch
from scipy.ndimage import gaussian_laplace

from .physics import SPATIAL, masked_mean, masked_nrmse, rfft_kernel


@torch.no_grad()
def reconstruction_metrics(
    output, chi, mask, dipole, local=None, *, hfen=True, threshold=0.2
):
    target = chi.float() * mask
    predicted = output.chi.float()
    error = predicted - target
    mean_target = masked_mean(target, mask)
    mean_predicted = masked_mean(predicted, mask)
    variance = masked_mean(target.square(), mask) - mean_target.square()
    covariance = masked_mean(target * predicted, mask) - mean_target * mean_predicted
    slope = covariance / variance.clamp_min(1e-20)
    values = {
        "chi_nrmse": masked_nrmse(predicted, target, mask),
        "tkd_nrmse": masked_nrmse(output.tkd, target, mask),
        "local_nrmse": masked_nrmse(output.local_field, local, mask)
        if local is not None
        else [None] * len(chi),
        "bias_ppm": masked_mean(error, mask).flatten().tolist(),
        "slope": slope.flatten().tolist(),
        "intercept_ppm": (mean_predicted - slope * mean_target).flatten().tolist(),
    }
    for name, selected in {
        "positive": target > 0,
        "negative": target < 0,
        "high_abs": target.abs() > 0.1,
    }.items():
        region = selected.float() * mask
        energy = (target.square() * region).sum(dim=SPATIAL)
        squared_error = (error.square() * region).sum(dim=SPATIAL)
        values[name + "_chi_nrmse"] = [
            100 * float((e / t).sqrt()) if float(t) > 0 else None
            for e, t in zip(squared_error.flatten(), energy.flatten(), strict=True)
        ]
    # RFFT Parseval weights count conjugate frequencies once each. Band
    # projection is defined on the full grid, before any ROI restriction.
    error_fft = torch.fft.rfftn(error, dim=SPATIAL)
    target_fft = torch.fft.rfftn(target, dim=SPATIAL)
    weights = torch.full((target.shape[-1] // 2 + 1,), 2.0, device=target.device)
    weights[0] = 1
    if target.shape[-1] % 2 == 0:
        weights[-1] = 1
    d = rfft_kernel(dipole, target.shape[-3:])
    for name, selected in {
        "retained": d.abs() > threshold,
        "near_null": d.abs() <= threshold,
    }.items():
        numerator = (error_fft.abs().square() * selected * weights).sum(dim=SPATIAL)
        denominator = (target_fft.abs().square() * selected * weights).sum(dim=SPATIAL)
        values[name + "_nrmse"] = [
            100 * float((n / t).sqrt()) if float(t) > 0 else None
            for n, t in zip(numerator.flatten(), denominator.flatten(), strict=True)
        ]
    if hfen:
        hf_errors = []
        for p, t, m in zip(
            predicted.cpu().numpy(),
            target.cpu().numpy(),
            mask.bool().cpu().numpy(),
            strict=True,
        ):
            # Exclude the channel axis from filtering. Sigma is one voxel on
            # the evaluation grid; masks/padding therefore remain case-specific.
            filtered_target = gaussian_laplace(t[0], sigma=1.0)[m[0]]
            filtered_error = gaussian_laplace(p[0] - t[0], sigma=1.0)[m[0]]
            denominator = float(np.sum(filtered_target.astype(np.float64) ** 2))
            hf_errors.append(
                100
                * float(
                    np.sqrt(
                        np.sum(filtered_error.astype(np.float64) ** 2) / denominator
                    )
                )
                if denominator > 0
                else None
            )
        values["hfen_percent"] = hf_errors
    return [{key: value[i] for key, value in values.items()} for i in range(len(chi))]


def subject_average(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["subject"], []).append(row)
    return {
        subject: {
            key: float(np.mean([row[key] for row in views if row.get(key) is not None]))
            if any(row.get(key) is not None for row in views)
            else None
            for key in views[0]
            if key not in ("subject", "profile", "seed")
        }
        for subject, views in groups.items()
    }


def paired_summary(joint, frozen, *, samples=10000, seed=20261020):
    subjects = sorted(set(joint) & set(frozen))
    if set(joint) != set(frozen) or not subjects:
        raise ValueError(
            "Paired summaries require exactly the same nonempty subject set"
        )
    differences = np.array(
        [joint[s]["chi_nrmse"] - frozen[s]["chi_nrmse"] for s in subjects]
    )
    rng = np.random.default_rng(seed)
    bootstrap = differences[
        rng.integers(len(subjects), size=(samples, len(subjects)))
    ].mean(axis=1)
    return dict(
        subjects=len(subjects),
        joint_chi_nrmse=float(np.mean([joint[s]["chi_nrmse"] for s in subjects])),
        frozen_chi_nrmse=float(np.mean([frozen[s]["chi_nrmse"] for s in subjects])),
        delta_joint_minus_frozen_pp=float(differences.mean()),
        paired_subject_bootstrap_95ci=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        per_subject_delta=dict(zip(subjects, differences.tolist(), strict=True)),
    )
