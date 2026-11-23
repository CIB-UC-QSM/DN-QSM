"""Trainable total-field reconstruction baselines."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .inference import (
    SlidingWindowConfig,
    gaussian_sliding_window,
    sliding_window_fields,
    sliding_window_residual,
)
from .model import LocalFieldEstimator, ModelConfig, susceptibility_unet
from .physics import (
    apply_multiplier,
    clipped_tkd,
    cutoff_kernels,
    linear_gradient_descent,
    masked_mean,
    masked_rms,
    validate_mask,
)

INVERSES = ("tkd", "tikhonov", "gd")


@dataclass
class BaselineReconstruction:
    chi: Tensor
    local_field: Tensor | None = None
    initial_chi: Tensor | None = None
    correction: Tensor | None = None


class DirectSusceptibility(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.network = susceptibility_unet(config)

    def _predict(self, total_field, mask):
        centered = (
            total_field.float() - masked_mean(total_field, mask)[..., None, None, None]
        ) * mask
        scale = masked_rms(centered, mask, self.config.scale_floor)
        return self.network(centered / scale).float() * scale * mask

    def forward(self, total_field, mask, dipole=None):
        validate_mask(mask, total_field)
        return BaselineReconstruction(self._predict(total_field, mask))

    @torch.inference_mode()
    def infer(self, total_field, mask, dipole=None, *, inference=SlidingWindowConfig()):
        if self.training:
            raise ValueError("Call model.eval() before inference")
        if not inference.window_size:
            return self(total_field, mask)
        validate_mask(mask, total_field)

        def predict(windows):
            field, roi = windows[:, :1], windows[:, 1:]
            valid = roi.flatten(1).any(1)
            prediction = torch.zeros_like(field)
            if bool(valid.any()):
                prediction[valid] = self._predict(field[valid], roi[valid])
            return prediction

        inputs = torch.cat((total_field.float() * mask, mask.float()), dim=1)
        return BaselineReconstruction(
            gaussian_sliding_window(inputs, predict, inference)
        )


class LocalFieldBaseline(nn.Module):
    def __init__(
        self,
        estimator: LocalFieldEstimator,
        inverse: str,
        *,
        correction=False,
        tkd_threshold=0.1,
        gd_iterations=30,
    ):
        super().__init__()
        if inverse not in INVERSES:
            raise ValueError(f"Unsupported dipole inverse {inverse!r}")
        self.estimator = estimator
        self.inverse = inverse
        self.tkd_threshold = tkd_threshold
        self.gd_iterations = gd_iterations
        self.residual = susceptibility_unet(estimator.config) if correction else None

    def _initial(self, local, dipole):
        if self.inverse == "tkd":
            return clipped_tkd(local, dipole, self.tkd_threshold)
        if self.inverse == "gd":
            return linear_gradient_descent(local, dipole, self.gd_iterations)
        inverse, _ = cutoff_kernels(dipole, self.estimator.config.threshold, "soft")
        return apply_multiplier(local, inverse)

    def _reconstruct(self, local, mask, dipole, inference):
        initial = self._initial(local, dipole)
        if self.residual is None:
            return BaselineReconstruction(initial, local, initial)
        floor = self.estimator.config.scale_floor
        if inference.window_size:
            residual = sliding_window_residual(
                self.residual, initial, mask, floor, inference
            )
        else:
            scale = masked_rms(initial, mask, floor)
            residual = self.residual(initial / scale).float() * scale
        return BaselineReconstruction(initial + residual, local, initial, residual)

    def forward(self, total_field, mask, dipole):
        local = self.estimator.predict_local_field(total_field, mask)
        return self._reconstruct(
            local, mask, dipole, SlidingWindowConfig(window_size=0)
        )

    @torch.inference_mode()
    def infer(self, total_field, mask, dipole, *, inference=SlidingWindowConfig()):
        if self.training:
            raise ValueError("Call model.eval() before inference")
        if not inference.window_size:
            return self(total_field, mask, dipole)
        local = sliding_window_fields(
            self.estimator, total_field, mask, inference
        ).local_field
        return self._reconstruct(local, mask, dipole, inference)
