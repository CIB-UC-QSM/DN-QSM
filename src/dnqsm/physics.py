"""FP32 periodic dipole physics, with unshifted FFT conventions from wh_net."""

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

SPATIAL = (-3, -2, -1)


def continuous_dipole_kernel(
    shape: Sequence[int],
    voxel_size: Sequence[float] = (1, 1, 1),
    b0_dir: Sequence[float] = (0, 0, 1),
) -> np.ndarray:
    """Continuous dipole on native voxel frequencies, including odd grids.

    This agrees with wh_net on even grids. Using fftfreq also preserves voxel
    geometry on odd rectangular crops rather than normalizing each axis by its
    largest sampled frequency.
    """
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("Dipole grids need three axes of length >= 2")
    if len(voxel_size) != 3 or any(not math.isfinite(v) or v <= 0 for v in voxel_size):
        raise ValueError("Voxel sizes must be finite and positive")
    direction = np.asarray(b0_dir, dtype=np.float64)
    if (
        direction.shape != (3,)
        or not np.isfinite(direction).all()
        or not np.isclose(np.linalg.norm(direction), 1)
    ):
        raise ValueError("B0 direction must be a finite unit vector")
    axes = [2 * np.fft.fftfreq(n, d=v) for n, v in zip(shape, voxel_size, strict=True)]
    grids = np.meshgrid(*axes, indexing="ij")
    squared = sum(k**2 for k in grids)
    parallel = sum(k * d for k, d in zip(grids, direction, strict=True))
    kernel = 1 / 3 - parallel**2 / (squared + np.finfo(np.float64).eps)
    kernel[0, 0, 0] = 0
    return kernel


def dipole_tensor(shape: Sequence[int], device: str | torch.device = "cpu") -> Tensor:
    """Return a [1,1,D,H,W//2+1] real-FFT dipole multiplier."""
    full = continuous_dipole_kernel(shape)
    return torch.as_tensor(
        full[..., : shape[-1] // 2 + 1].copy(), dtype=torch.float32, device=device
    )[None, None]


def rfft_kernel(dipole: Tensor, shape: Sequence[int]) -> Tensor:
    expected = (*shape[:-1], shape[-1] // 2 + 1)
    if tuple(dipole.shape[-3:]) == tuple(shape):
        dipole = dipole[..., : expected[-1]]
    elif tuple(dipole.shape[-3:]) != expected:
        raise ValueError("Dipole and volume spatial shapes do not match")
    return dipole.float()


def apply_multiplier(volume: Tensor, multiplier: Tensor) -> Tensor:
    """Spatial FFTs only; safe inside an outer convolution autocast context."""
    with torch.autocast(volume.device.type, enabled=False):
        shape = volume.shape[-3:]
        return torch.fft.irfftn(
            torch.fft.rfftn(volume.float(), dim=SPATIAL)
            * rfft_kernel(multiplier, shape),
            s=shape,
            dim=SPATIAL,
        )


def apply_dipole(chi: Tensor, dipole: Tensor) -> Tensor:
    return apply_multiplier(chi, dipole)


def cutoff_kernels(dipole: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    """Hard TKD inverse and complementary orthogonal near-null projection."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("TKD threshold must be finite and positive")
    d = dipole.float()
    retained = d.abs() > threshold
    inverse = torch.zeros_like(d)
    inverse[retained] = d[retained].reciprocal()
    return inverse, (~retained).float()


def validate_mask(mask: Tensor, volume: Tensor | None = None) -> None:
    if (
        mask.ndim != 5
        or mask.shape[1] != 1
        or (volume is not None and volume.shape != mask.shape)
    ):
        raise ValueError("Volume and mask must have identical [N,1,D,H,W] shapes")
    if not bool(torch.isfinite(mask).all()) or not bool(
        ((mask == 0) | (mask == 1)).all()
    ):
        raise ValueError("Mask must be finite and binary")
    if bool((mask.sum(dim=SPATIAL) == 0).any()):
        raise ValueError("Each sample needs a nonempty mask")


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Per-sample means; caller validates nonempty binary masks."""
    return (values.float() * mask).sum(dim=SPATIAL) / mask.sum(dim=SPATIAL).clamp_min(1)


def masked_rms(values: Tensor, mask: Tensor, floor: float = 1e-6) -> Tensor:
    # Clamp before sqrt to keep the derivative finite for zero inputs.
    return (
        masked_mean(values.float().square(), mask)
        .clamp_min(floor**2)
        .sqrt()[..., None, None, None]
    )


@torch.no_grad()
def masked_nrmse(
    prediction: Tensor, reference: Tensor, mask: Tensor
) -> list[float | None]:
    """Per-sample percentage; a zero-energy reference has undefined NRMSE."""
    validate_mask(mask, prediction)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")
    numerator = masked_mean(
        (prediction.float() - reference.float()).square(), mask
    ).sqrt()
    denominator = masked_mean(reference.float().square(), mask).sqrt()
    return [
        100 * float(n / d) if float(d) > 0 else None
        for n, d in zip(numerator.flatten(), denominator.flatten(), strict=True)
    ]
