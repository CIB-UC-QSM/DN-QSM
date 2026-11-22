"""Physical-unit supervision against clean local fields and susceptibility."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import Reconstruction
from .physics import (
    apply_dipole,
    apply_multiplier,
    cutoff_kernels,
    masked_mean,
    spatial_laplacian,
    spherical_erosion,
    validate_mask,
)

LOSS_VERSION = 8


@dataclass(frozen=True)
class LossNormalization:
    chi_floor: float = 1e-4
    field_floor: float = 1e-4
    mse_input_ratio_cap: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.chi_floor, self.field_floor)
        ):
            raise ValueError("Loss RMS floors must be finite and positive")
        if not math.isfinite(self.mse_input_ratio_cap) or self.mse_input_ratio_cap < 0:
            raise ValueError("MSE input-ratio cap must be finite and nonnegative")


@dataclass(frozen=True)
class LossWeights:
    local: float = 1.5
    chi: float = 1.0
    physics: float = 1.0
    extension: float = 1.0
    local_gradient: float = 1.0
    local_tkd: float = 0.0
    harmonic_laplacian: float = 0.0
    field_decomposition: float = 0.0

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(v) or v < 0
                for v in (
                    self.local,
                    self.chi,
                    self.physics,
                    self.extension,
                    self.local_gradient,
                    self.local_tkd,
                    self.harmonic_laplacian,
                    self.field_decomposition,
                )
            )
            or self.local == 0
            or self.chi == 0
        ):
            raise ValueError(
                "Local/chi weights must be positive; other loss weights nonnegative"
            )


def local_field_gradient_loss(
    local_field: Tensor, local_target: Tensor, mask: Tensor
) -> Tensor:
    """Per-sample L1 gradient mismatch over valid ROI neighbor pairs.

    Differences use unit voxel steps, with no padding or periodic wraparound.
    Pool all valid edges across the three spatial axes before averaging.
    """
    validate_mask(mask, local_field)
    if local_target.shape != mask.shape:
        raise ValueError("Target and prediction shapes must agree")
    error = local_field.float() - local_target.detach().float()
    penalties = 0
    counts = 0
    for axis in (-3, -2, -1):
        length = error.shape[axis] - 1
        valid = (
            mask.narrow(axis, 0, length).float() * mask.narrow(axis, 1, length).float()
        )
        penalties = penalties + (error.diff(dim=axis).abs() * valid).sum(
            dim=(-3, -2, -1)
        )
        counts = counts + valid.sum(dim=(-3, -2, -1))
    return penalties / counts.clamp_min(1)


def local_tkd_loss(
    local_field: Tensor,
    chi_target: Tensor,
    mask: Tensor,
    dipole: Tensor,
    threshold: float,
    cutoff_mode: str = "hard",
) -> Tensor:
    """Per-sample L1 between inverted predicted field and true susceptibility.

    The inverse weights the filter error by the conditioning of the operator
    that consumes it. Its gain is bounded by 1/threshold in hard mode and
    1/(2*threshold) in soft mode. A voxelwise field penalty charges the same
    price for every spatial frequency.
    """
    validate_mask(mask, local_field)
    if chi_target.shape != mask.shape:
        raise ValueError("Target and prediction shapes must agree")
    inverse, _ = cutoff_kernels(dipole, threshold, cutoff_mode)
    tkd = apply_multiplier(local_field.float(), inverse)
    return masked_mean((tkd - chi_target.detach().float()).abs(), mask)


def harmonic_laplacian_loss(harmonic_field: Tensor, mask: Tensor) -> Tensor:
    """Per-sample Delta(mask*h) MSE inside a radius-3 spherical erosion of mask.

    Mask before differentiating with the original ROI, then select the eroded
    ROI for the reduction; remasking with the eroded ROI would create new edges.
    Empty eroded samples contribute a finite, differentiable zero.
    """
    validate_mask(mask, harmonic_field)
    interior = spherical_erosion(mask, radius=3)
    with torch.autocast(harmonic_field.device.type, enabled=False):
        laplacian = spatial_laplacian(harmonic_field.float() * mask.float())
        return masked_mean(laplacian.square(), interior)


def harmonic_field_loss_terms(
    local_field: Tensor,
    harmonic_field: Tensor,
    mask: Tensor,
    *,
    total_field: Tensor | None = None,
    include_harmonic: bool = True,
    include_decomposition: bool = True,
) -> dict[str, Tensor]:
    """Penalize Delta(mask*h) and tie both branches to the measured total field.

    The stencil is applied after masking with the original ROI, with zero
    padding. Average its square only inside a radius-3 spherical erosion.
    The field-sum penalty still uses the original ROI and prevents h=0.
    """
    validate_mask(mask, harmonic_field)
    if local_field.shape != mask.shape:
        raise ValueError("Local/harmonic fields and mask shapes must agree")
    terms = {}
    with torch.autocast(harmonic_field.device.type, enabled=False):
        harmonic = harmonic_field.float() * mask.float()
        if include_harmonic:
            terms["harmonic_laplacian"] = harmonic_laplacian_loss(harmonic, mask)
        if include_decomposition:
            if total_field is None or total_field.shape != mask.shape:
                raise ValueError("Field decomposition requires the aligned total field")
            terms["field_decomposition"] = masked_mean(
                (local_field.float() + harmonic - total_field.detach().float()).abs(),
                mask,
            )
    return terms


def local_field_loss_terms(
    local_field: Tensor,
    local_target: Tensor,
    mask: Tensor,
    *,
    include_gradient: bool = True,
    local_full_target: Tensor | None = None,
    chi_target: Tensor | None = None,
    dipole: Tensor | None = None,
    threshold: float | None = None,
    cutoff_mode: str = "hard",
    harmonic_field: Tensor | None = None,
    total_field: Tensor | None = None,
    include_harmonic: bool = True,
    include_decomposition: bool = True,
) -> dict[str, Tensor]:
    """ROI supervision shared by independent filtering and joint training.

    Exterior and TKD supervision are opt-in: pass the full-grid field target
    for `extension`, and all three of the susceptibility target, dipole, and
    threshold for `local_tkd`. Both reach the filter on filtering-only batches,
    which are the only batches that update it under the alternating schedule.
    """
    validate_mask(mask, local_field)
    if local_target.shape != mask.shape:
        raise ValueError("Target and prediction shapes must agree")
    terms = {
        "local": masked_mean(
            (local_field.float() - local_target.detach().float()).abs(), mask
        )
    }
    if include_gradient:
        terms["local_gradient"] = local_field_gradient_loss(
            local_field, local_target, mask
        )
    if chi_target is not None:
        if dipole is None or threshold is None:
            raise ValueError("TKD supervision needs both a dipole and a threshold")
        terms["local_tkd"] = local_tkd_loss(
            local_field, chi_target, mask, dipole, threshold, cutoff_mode
        )
    if local_full_target is not None:
        if local_full_target.shape != mask.shape:
            raise ValueError("Full-grid local target and prediction shapes must agree")
        terms["extension"] = masked_mean(
            (local_field.float() - local_full_target.detach().float()).abs(),
            1 - mask.float(),
        )
    if harmonic_field is not None and (include_harmonic or include_decomposition):
        terms.update(
            harmonic_field_loss_terms(
                local_field,
                harmonic_field,
                mask,
                total_field=total_field,
                include_harmonic=include_harmonic,
                include_decomposition=include_decomposition,
            )
        )
    return terms


def per_sample_loss_terms(
    prediction: Reconstruction,
    local_target: Tensor,
    chi_target: Tensor,
    mask: Tensor,
    dipole: Tensor,
    *,
    local_full_target: Tensor | None = None,
    predicted_field: Tensor | None = None,
    include_extension: bool = True,
    include_gradient: bool = True,
    include_tkd: bool = False,
    input_field: Tensor | None = None,
    include_harmonic: bool = True,
    include_decomposition: bool = True,
) -> dict[str, Tensor]:
    """Return raw [N,1] penalties; targets never receive gradients."""
    validate_mask(mask, prediction.chi)
    if local_target.shape != mask.shape or chi_target.shape != mask.shape:
        raise ValueError("Target and prediction shapes must agree")
    local_target, chi_target = (
        local_target.detach().float(),
        chi_target.detach().float(),
    )
    if include_extension and local_full_target is None:
        local_full_target = apply_dipole(chi_target * mask, dipole).detach()
    field = (
        apply_dipole(prediction.chi, dipole)
        if predicted_field is None
        else predicted_field
    )
    if field.shape != mask.shape:
        raise ValueError("Predicted forward field and mask shapes must agree")
    terms = {
        # Ground-truth physics supervision is independent of the learned filter:
        # a jointly biased chi/filter pair cannot make this penalty vanish.
        "physics": masked_mean((field - local_target.float()).square(), mask),
        **local_field_loss_terms(
            prediction.local_field,
            local_target,
            mask,
            include_gradient=include_gradient,
            local_full_target=local_full_target if include_extension else None,
            harmonic_field=prediction.harmonic_field,
            total_field=input_field,
            include_harmonic=include_harmonic,
            include_decomposition=include_decomposition,
        ),
        "chi": masked_mean((prediction.chi.float() - chi_target.float()).abs(), mask),
    }
    if include_tkd:
        # The reconstruction already inverted this field; reuse it rather than
        # repeating the transform pair that local_tkd_loss would perform.
        terms["local_tkd"] = masked_mean(
            (prediction.tkd.float() - chi_target.float()).abs(), mask
        )
    return terms


def normalize_loss_terms(
    terms: dict[str, Tensor],
    local_target: Tensor,
    chi_target: Tensor,
    mask: Tensor,
    normalization: LossNormalization | None,
    *,
    input_field: Tensor | None = None,
) -> dict[str, Tensor]:
    if normalization is None:
        return terms
    field_energy = masked_mean(local_target.detach().float().square(), mask).clamp_min(
        normalization.field_floor**2
    )
    chi_scale = (
        masked_mean(chi_target.detach().float().square(), mask)
        .clamp_min(normalization.chi_floor**2)
        .sqrt()
    )
    mse_energy = field_energy
    if normalization.mse_input_ratio_cap:
        if input_field is None or input_field.shape != mask.shape:
            raise ValueError(
                "Capped MSE normalization requires the aligned input field"
            )
        # Detached input/target scales cannot be changed by a prediction to
        # reduce its own loss. Keep the L1 denominators target-relative.
        input_energy = masked_mean(input_field.detach().float().square(), mask)
        mse_energy = field_energy.maximum(
            input_energy / normalization.mse_input_ratio_cap**2
        )
    scales = dict(
        chi=chi_scale,
        local=field_energy.sqrt(),
        local_gradient=field_energy.sqrt(),
        local_tkd=chi_scale,
        extension=field_energy.sqrt(),
        physics=mse_energy,
        harmonic_laplacian=field_energy,
        field_decomposition=field_energy.sqrt(),
    )
    return {name: value / scales[name] for name, value in terms.items()}


def reconstruction_loss(
    prediction: Reconstruction,
    local_target: Tensor,
    chi_target: Tensor,
    mask: Tensor,
    dipole: Tensor,
    weights: LossWeights | None = None,
    *,
    normalization: LossNormalization | None = LossNormalization(),
    local_full_target: Tensor | None = None,
    input_field: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = weights or LossWeights()
    raw = per_sample_loss_terms(
        prediction,
        local_target,
        chi_target,
        mask,
        dipole,
        local_full_target=local_full_target,
        include_extension=bool(weights.extension),
        include_gradient=bool(weights.local_gradient),
        include_tkd=bool(weights.local_tkd),
        input_field=input_field,
        include_harmonic=bool(weights.harmonic_laplacian),
        include_decomposition=bool(weights.field_decomposition),
    )
    terms = {
        name: value.mean()
        for name, value in normalize_loss_terms(
            raw, local_target, chi_target, mask, normalization, input_field=input_field
        ).items()
    }
    total = sum(
        getattr(weights, name) * value
        for name, value in terms.items()
        if getattr(weights, name)
    )
    return total, terms
