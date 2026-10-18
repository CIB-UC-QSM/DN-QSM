"""Physical-unit supervision and consistency with predicted and clean fields."""

import math
from dataclasses import dataclass

from torch import Tensor

from .model import Reconstruction
from .physics import apply_dipole, masked_mean, validate_mask

LOSS_VERSION = 2


@dataclass(frozen=True)
class LossWeights:
    forward: float = 1.0
    local: float = 1.0
    chi: float = 1.0
    physics: float = 1.0

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(v) or v < 0
                for v in (self.forward, self.local, self.chi, self.physics)
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
) -> dict[str, Tensor]:
    """Return [N,1] penalties shared by training and validation."""
    validate_mask(mask, prediction.chi)
    if local_target.shape != mask.shape or chi_target.shape != mask.shape:
        raise ValueError("Target and prediction shapes must agree")
    field = apply_dipole(prediction.chi, dipole)
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
    }


def reconstruction_loss(
    prediction: Reconstruction,
    local_target: Tensor,
    chi_target: Tensor,
    mask: Tensor,
    dipole: Tensor,
    weights: LossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = weights or LossWeights()
    terms = {
        name: value.mean()
        for name, value in per_sample_loss_terms(
            prediction, local_target, chi_target, mask, dipole
        ).items()
    }
    total = sum(getattr(weights, name) * value for name, value in terms.items())
    return total, terms
