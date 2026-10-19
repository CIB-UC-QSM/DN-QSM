"""Physical-unit supervision against clean local fields and susceptibility."""

import math
from dataclasses import dataclass

from torch import Tensor

from .model import Reconstruction
from .physics import apply_dipole, masked_mean, validate_mask

LOSS_VERSION = 4


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

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(v) or v < 0
                for v in (
                    self.local,
                    self.chi,
                    self.physics,
                    self.extension,
                )
            )
            or self.local == 0
            or self.chi == 0
        ):
            raise ValueError(
                "Local/chi weights must be positive; physics/extension weights nonnegative"
            )


def local_field_loss_terms(
    local_field: Tensor, local_target: Tensor, mask: Tensor
) -> dict[str, Tensor]:
    """ROI supervision shared by independent filtering and joint training."""
    validate_mask(mask, local_field)
    if local_target.shape != mask.shape:
        raise ValueError("Target and prediction shapes must agree")
    return {
        "local": masked_mean(
            (local_field.float() - local_target.detach().float()).abs(), mask
        )
    }


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
    if include_extension and local_full_target.shape != mask.shape:
        raise ValueError("Full-grid local target and prediction shapes must agree")
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
        **local_field_loss_terms(prediction.local_field, local_target, mask),
        "chi": masked_mean((prediction.chi.float() - chi_target.float()).abs(), mask),
    }
    if include_extension:
        terms["extension"] = masked_mean(
            (prediction.local_field.float() - local_full_target.detach().float()).abs(),
            1 - mask.float(),
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
        extension=field_energy.sqrt(),
        physics=mse_energy,
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
    )
    terms = {
        name: value.mean()
        for name, value in normalize_loss_terms(
            raw, local_target, chi_target, mask, normalization, input_field=input_field
        ).items()
    }
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms
