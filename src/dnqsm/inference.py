"""Gaussian sliding-window CNN inference with full-volume Fourier physics."""

import math
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F

from .physics import apply_multiplier, cutoff_kernels, masked_rms, validate_mask


@dataclass(frozen=True)
class SlidingWindowConfig:
    window_size: int = 64
    overlap: float = 0.5
    sigma_scale: float = 0.125
    window_batch_size: int = 1

    def __post_init__(self):
        if (
            type(self.window_size) is not int
            or self.window_size < 0
            or self.window_size == 1
        ):
            raise ValueError(
                "Window size must be zero (full volume) or an integer >= 2"
            )
        if not math.isfinite(self.overlap) or not 0 <= self.overlap < 1:
            raise ValueError("Window overlap must be finite and in [0, 1)")
        if not math.isfinite(self.sigma_scale) or self.sigma_scale <= 0:
            raise ValueError("Gaussian sigma scale must be finite and positive")
        if type(self.window_batch_size) is not int or self.window_batch_size < 1:
            raise ValueError("Window batch size must be a positive integer")


def gaussian_window(size, sigma_scale=0.125, *, device="cpu"):
    """FP32, strictly positive weights; sigma is a fraction of window width."""
    SlidingWindowConfig(size, sigma_scale=sigma_scale)
    if not size:
        raise ValueError("A Gaussian window needs a positive size")
    coordinate = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    sigma = max(size * sigma_scale, torch.finfo(torch.float32).eps)
    axis = torch.exp(-0.5 * (coordinate / sigma).square())
    weights = axis[:, None, None] * axis[None, :, None] * axis[None, None, :]
    return (
        weights / weights.max().clamp_min(torch.finfo(torch.float32).tiny)
    ).clamp_min(1e-6)[None, None]


def _starts(length, size, stride):
    positions = list(range(0, length - size + 1, stride))
    if positions[-1] != length - size:
        positions.append(length - size)
    return positions


@torch.inference_mode()
def gaussian_sliding_window(inputs, predictor, config=SlidingWindowConfig()):
    """Blend same-resolution predictor outputs, retaining the original geometry.

    Large axes receive a half-window halo so their outer voxels also occur
    near a window center. Small axes are padded only to the window width.
    Tensor inputs and accumulators remain on the caller's device; only a
    bounded number of CNN windows are evaluated together.
    """
    size = config.window_size
    if not size:
        raise ValueError("Sliding-window inference requires a positive window size")
    if inputs.ndim != 5 or min(inputs.shape) < 1:
        raise ValueError("Sliding-window inputs must have shape [N,C,D,H,W]")
    shape = inputs.shape[-3:]
    margins = [
        ((size - n) // 2, size - n - (size - n) // 2)
        if n <= size
        else (size // 2, size // 2)
        for n in shape
    ]
    padded = F.pad(inputs, tuple(v for pair in reversed(margins) for v in pair))
    stride = max(1, round(size * (1 - config.overlap)))
    positions = list(product(*(_starts(n, size, stride) for n in padded.shape[-3:])))
    weight = gaussian_window(size, config.sigma_scale, device=inputs.device)
    weights = torch.zeros(
        (1, 1, *padded.shape[-3:]), device=inputs.device, dtype=torch.float32
    )
    output = None
    for sample in range(inputs.shape[0]):
        for offset in range(0, len(positions), config.window_batch_size):
            group = positions[offset : offset + config.window_batch_size]
            windows = torch.cat(
                [
                    padded[
                        sample : sample + 1, :, d : d + size, h : h + size, w : w + size
                    ]
                    for d, h, w in group
                ]
            )
            predictions = predictor(windows).float()
            if (
                predictions.ndim != 5
                or predictions.shape[0] != len(group)
                or tuple(predictions.shape[-3:]) != (size,) * 3
            ):
                raise ValueError(
                    "Window predictor must retain batch and spatial dimensions"
                )
            if output is None:
                output = torch.zeros(
                    (inputs.shape[0], predictions.shape[1], *padded.shape[-3:]),
                    device=inputs.device,
                    dtype=torch.float32,
                )
            if predictions.shape[1] != output.shape[1]:
                raise ValueError("Window predictor changed its output channel count")
            for index, (d, h, w) in enumerate(group):
                region = (
                    ...,
                    slice(d, d + size),
                    slice(h, h + size),
                    slice(w, w + size),
                )
                output[sample : sample + 1][region] += (
                    predictions[index : index + 1] * weight
                )
                if sample == 0:
                    weights[region] += weight
    output /= weights.clamp_min(1e-6)
    crop = tuple(
        slice(left, left + n) for (left, _), n in zip(margins, shape, strict=True)
    )
    return output[(..., *crop)].contiguous()


@torch.inference_mode()
def sliding_window_fields(model, total_field, mask, config=SlidingWindowConfig()):
    from .model import FilteredFields

    validate_mask(mask, total_field)
    if model.training:
        raise ValueError("Call model.eval() before sliding-window inference")
    if config.window_size < 2 ** (len(model.config.features) + 1):
        raise ValueError("Inference window is too small for the model's U-Net depth")
    channels = 2 if model.config.filter_input_version == 3 else 1

    def predict(windows):
        phase, roi = windows[:, :1], windows[:, 1:]
        valid = roi.flatten(1).any(1)
        result = torch.zeros(
            (len(windows), channels, *phase.shape[-3:]),
            device=phase.device,
            dtype=torch.float32,
        )
        # Empty windows have no measurable field. They still contribute zero
        # with their Gaussian weight, including outside the original ROI.
        if bool(valid.any()):
            fields = model.predict_fields(phase[valid], roi[valid])
            result[valid, :1] = fields.local_field.float()
            if fields.harmonic_field is not None:
                result[valid, 1:] = fields.harmonic_field.float()
        return result

    output = gaussian_sliding_window(
        torch.cat((total_field.float() * mask, mask.float()), dim=1), predict, config
    )
    local = output[:, :1]
    if not model.config.full_grid_field:
        local = local * mask
    harmonic = output[:, 1:] * mask if channels == 2 else None
    return FilteredFields(local, harmonic)


@torch.inference_mode()
def sliding_window_reconstruction(
    model, total_field, mask, dipole, config=SlidingWindowConfig()
):
    from .model import Reconstruction

    fields = sliding_window_fields(model, total_field, mask, config)
    inverse, projection = cutoff_kernels(
        dipole, model.config.threshold, model.config.cutoff_mode
    )
    initial = apply_multiplier(fields.local_field, inverse)

    def predict(windows):
        chi, roi = windows[:, :1], windows[:, 1:]
        # In exterior-only windows use all voxels for scale. Null-network
        # predictions remain unmasked before the global Fourier projection.
        roi = torch.where(
            roi.flatten(1).any(1)[:, None, None, None, None], roi, torch.ones_like(roi)
        )
        scale = masked_rms(chi, roi, model.config.scale_floor)
        return model.null_network(chi / scale).float() * scale

    residual = gaussian_sliding_window(
        torch.cat((initial, mask.float()), dim=1), predict, config
    )
    correction = apply_multiplier(residual, projection)
    return Reconstruction(
        fields.local_field,
        initial,
        correction,
        initial + correction,
        fields.harmonic_field,
    )
