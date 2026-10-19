"""Physical-unit supervision and consistency with predicted and clean fields."""

import math
from dataclasses import dataclass

from torch import Tensor

from .model import Reconstruction
from .physics import apply_dipole, masked_mean, validate_mask

LOSS_VERSION = 3


@dataclass(frozen=True)
class LossNormalization:
    chi_floor: float = 1e-4
    field_floor: float = 1e-4

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(v) or v <= 0 for v in (self.chi_floor, self.field_floor)
        ):
            raise ValueError("Loss RMS floors must be finite and positive")


@dataclass(frozen=True)
class LossWeights:
    forward: float = 1.0
    local: float = 1.5
    chi: float = 1.0
    physics: float = 1.0
    extension: float = 1.0

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(v) or v < 0
                for v in (
                    self.forward,
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
                "Local/chi weights must be positive; forward/physics weights nonnegative"
            )


def per_sample_loss_terms(
    prediction: Reconstruction,
    local_target: Tensor,
    chi_target: Tensor,
    mask: Tensor,
    dipole: Tensor,
    *,
    local_full_target: Tensor | None = None,
    predicted_field: Tensor | None = None,
) -> dict[str, Tensor]:
    """Return raw [N,1] penalties; targets never receive gradients."""
    validate_mask(mask, prediction.chi)
    if local_target.shape != mask.shape or chi_target.shape != mask.shape:
        raise ValueError("Target and prediction shapes must agree")
    local_target, chi_target = (
        local_target.detach().float(),
        chi_target.detach().float(),
    )
    if local_full_target is None:
        local_full_target = apply_dipole(chi_target * mask, dipole).detach()
    if local_full_target.shape != mask.shape:
        raise ValueError("Full-grid local target and prediction shapes must agree")
    field = (
        apply_dipole(prediction.chi, dipole)
        if predicted_field is None
        else predicted_field
    )
    if field.shape != mask.shape:
        raise ValueError("Predicted forward field and mask shapes must agree")
    return {
        "forward": masked_mean(
            (field - prediction.local_field.float()).square(),
            mask,
        ),
        # Ground-truth physics supervision is independent of the learned filter:
        # a jointly biased chi/filter pair cannot make this penalty vanish.
        "physics": masked_mean((field - local_target.float()).square(), mask),
        "local": masked_mean(
            (prediction.local_field.float() - local_target.float()).abs(), mask
        ),
        "chi": masked_mean((prediction.chi.float() - chi_target.float()).abs(), mask),
        "extension": masked_mean(
            (prediction.local_field.float() - local_full_target.detach().float()).abs(),
            1 - mask.float(),
        ),
    }


def normalize_loss_terms(
    terms: dict[str, Tensor],
    local_target: Tensor,
    chi_target: Tensor,
    mask: Tensor,
    normalization: LossNormalization | None,
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
    scales = dict(
        chi=chi_scale,
        local=field_energy.sqrt(),
        extension=field_energy.sqrt(),
        forward=field_energy,
        physics=field_energy,
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
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = weights or LossWeights()
    raw = per_sample_loss_terms(
        prediction,
        local_target,
        chi_target,
        mask,
        dipole,
        local_full_target=local_full_target,
    )
    terms = {
        name: value.mean()
        for name, value in normalize_loss_terms(
            raw, local_target, chi_target, mask, normalization
        ).items()
    }
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms
