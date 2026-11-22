"""FP32 periodic dipole physics, with unshifted FFT conventions from wh_net."""

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

SPATIAL = (-3, -2, -1)


def laplacian_stencil(device=None) -> Tensor:
    """Fixed isotropic 3x3x3 stencil, in unit voxel coordinates."""
    return torch.tensor(
        [
            [
                [1 / 13, 3 / 26, 1 / 13],
                [3 / 26, 3 / 13, 3 / 26],
                [1 / 13, 3 / 26, 1 / 13],
            ],
            [
                [3 / 26, 3 / 13, 3 / 26],
                [3 / 13, -44 / 13, 3 / 13],
                [3 / 26, 3 / 13, 3 / 26],
            ],
            [
                [1 / 13, 3 / 26, 1 / 13],
                [3 / 26, 3 / 13, 3 / 26],
                [1 / 13, 3 / 26, 1 / 13],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )[None, None]


def spatial_laplacian(volume: Tensor) -> Tensor:
    """Zero-padded, shape-preserving FP32 Laplacian, including under autocast."""
    with torch.autocast(volume.device.type, enabled=False):
        return F.conv3d(volume.float(), laplacian_stencil(volume.device), padding=1)


def frequency_laplacian_kernel(shape: Sequence[int], device="cpu") -> Tensor:
    """Unshifted rFFT multiplier sum_j |1-exp(2*pi*i*k_j/N_j)|^2.

    This is the positive semidefinite, periodic negative discrete Laplacian
    on a unit-voxel grid, rather than the continuous |2*pi*f|^2 operator.
    """
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError("Frequency Laplacian needs three positive spatial dimensions")
    axes = [
        torch.fft.fftfreq(shape[0], device=device, dtype=torch.float32),
        torch.fft.fftfreq(shape[1], device=device, dtype=torch.float32),
        torch.fft.rfftfreq(shape[2], device=device, dtype=torch.float32),
    ]
    # |1-exp(i*theta)|^2 = 4*sin(theta/2)^2; no complex exponentials needed.
    terms = [4 * torch.sin(torch.pi * axis).square() for axis in axes]
    return (
        terms[0][:, None, None] + terms[1][None, :, None] + terms[2][None, None, :]
    )[None, None]


def frequency_laplacian(volume: Tensor) -> Tensor:
    """FFT-based periodic difference operator matching the supplied MATLAB code.

    Only spatial axes are transformed. Preserve shape (including odd axes),
    stay FP32 under autocast, and retain gradients with respect to the input.
    """
    if volume.ndim < 3:
        raise ValueError("Frequency Laplacian needs at least three spatial dimensions")
    with torch.autocast(volume.device.type, enabled=False):
        multiplier = frequency_laplacian_kernel(volume.shape[-3:], volume.device)
        # Drop only broadcast batch/channel axes for bare 3D or 4D inputs.
        multiplier = multiplier.reshape(
            *([1] * (volume.ndim - 3)), *multiplier.shape[-3:]
        )
        return torch.fft.irfftn(
            torch.fft.rfftn(volume.float(), dim=SPATIAL) * multiplier,
            s=volume.shape[-3:],
            dim=SPATIAL,
        )


def smooth_harmonic_extension(
    field: Tensor, mask: Tensor, num_iters: int = 6
) -> Tensor:
    """Extend an N1DHW field outside its ROI, preserving its interior exactly.

    Initialize the first exterior shell with valid-neighbor averages, then
    perform six-neighbor Jacobi relaxation only outside the mask. This is a
    finite relaxation, not a converged harmonic solve. Work in mean-centered
    FP32 with zero padding and retain autograd through all field operations.
    Restore the mean everywhere afterward: the far field approaches the ROI
    mean, and the FFT Laplacian is unchanged by that global constant.
    ``num_iters=0`` still initializes the first exterior shell.
    """
    validate_mask(mask, field)
    if type(num_iters) is not int or num_iters < 0:
        raise ValueError("Extension iterations must be a nonnegative integer")
    with torch.autocast(field.device.type, enabled=False):
        field = field.float()
        inside = mask.bool()
        mean = torch.where(inside, field, 0).sum(dim=SPATIAL, keepdim=True)
        mean = mean / mask.float().sum(dim=SPATIAL, keepdim=True).clamp_min(1)
        u = torch.where(inside, field - mean, 0)
        neighbors = field.new_zeros((1, 1, 3, 3, 3))
        neighbors[0, 0, 0, 1, 1] = neighbors[0, 0, 2, 1, 1] = 1
        neighbors[0, 0, 1, 0, 1] = neighbors[0, 0, 1, 2, 1] = 1
        neighbors[0, 0, 1, 1, 0] = neighbors[0, 0, 1, 1, 2] = 1
        count = F.conv3d(mask.float(), neighbors, padding=1)
        exterior = F.conv3d(u, neighbors, padding=1) / count.clamp_min(1)
        u = torch.where(inside, u, exterior * (count > 0))
        for _ in range(num_iters):
            u = torch.where(inside, u, F.conv3d(u, neighbors / 6, padding=1))
        # Avoid subtraction/addition roundoff inside the ROI.
        return torch.where(inside, field, u + mean)


def quadratic_exterior_extension(field: Tensor, mask: Tensor) -> Tensor:
    """Add a one-voxel ghost shell using interior-only quadratic extrapolation.

    For an exterior point, three consecutive interior samples on a ray give
    ``u_ghost = 3*u1 - 3*u2 + u3``. Prefer axial rays; use diagonal rays only
    when no axial ray supports that degree. If three samples are unavailable,
    fall back to linear extrapolation, then the nearest sample. Average valid
    rays of the same degree, never mixing lower-order guesses into a quadratic.

    Interior samples are copied exactly; no erosion, taper, or interior blur.
    The returned N1(D+2)(H+2)(W+2) field includes ghosts beyond volume faces,
    preventing periodic FFT wraparound from entering the original ROI. Crop
    one voxel on each side AFTER differentiation. Work in FP32 with autograd.
    """
    validate_mask(mask, field)
    with torch.autocast(field.device.type, enabled=False):
        field = field.float()
        inside = mask.bool()
        mean = torch.where(inside, field, 0).sum(dim=SPATIAL, keepdim=True)
        mean = mean / mask.float().sum(dim=SPATIAL, keepdim=True).clamp_min(1)
        centered = torch.where(inside, field - mean, 0)
        source = F.pad(centered, (4,) * 6)
        support = F.pad(inside, (4,) * 6)
        padded_inside = F.pad(inside, (1,) * 6)
        shape = padded_inside.shape[-3:]

        def shifted(volume, direction, step):
            slices = tuple(
                slice(3 + step * offset, 3 + step * offset + length)
                for offset, length in zip(direction, shape, strict=True)
            )
            return volume[(..., *slices)]

        totals = [
            torch.zeros_like(padded_inside, dtype=torch.float32) for _ in range(3)
        ]
        counts = [
            torch.zeros_like(padded_inside, dtype=torch.float32) for _ in range(3)
        ]
        axes = [
            (d, h, w)
            for d in (-1, 0, 1)
            for h in (-1, 0, 1)
            for w in (-1, 0, 1)
            if abs(d) + abs(h) + abs(w) == 1
        ]
        diagonals = [
            (d, h, w)
            for d in (-1, 0, 1)
            for h in (-1, 0, 1)
            for w in (-1, 0, 1)
            if abs(d) + abs(h) + abs(w) > 1
        ]
        # These masks depend only on the ROI, never on measured field values.
        allowed = [~padded_inside] * 3
        for directions in (axes, diagonals):
            for direction in directions:
                m1, m2, m3 = [shifted(support, direction, step) for step in (1, 2, 3)]
                v1, v2, v3 = [shifted(source, direction, step) for step in (1, 2, 3)]
                valid = (m1, m1 & m2, m1 & m2 & m3)
                predictions = (v1, 2 * v1 - v2, 3 * v1 - 3 * v2 + v3)
                for degree in range(3):
                    use = valid[degree] & allowed[degree]
                    totals[degree] = totals[degree] + torch.where(
                        use, predictions[degree], 0
                    )
                    counts[degree] = counts[degree] + use
            # Diagonal fits supplement unsupported degrees, preserving axial fits.
            allowed = [(count == 0) & ~padded_inside for count in counts]
        exterior = totals[0] / counts[0].clamp_min(1)
        for degree in (1, 2):
            exterior = torch.where(
                counts[degree] > 0,
                totals[degree] / counts[degree].clamp_min(1),
                exterior,
            )
        return torch.where(padded_inside, F.pad(field, (1,) * 6), exterior + mean)


@torch.no_grad()
def spherical_erosion(mask: Tensor, radius: int = 3) -> Tensor:
    """Binary erosion by integer offsets satisfying d*d+h*h+w*w <= radius**2.

    Radius is in voxels, and positions outside the volume are background.
    Boolean shifts avoid a large dense 7x7x7 convolution workspace on the GPU.
    An eroded sample may be empty even though the original ROI is nonempty.
    """
    validate_mask(mask)
    if type(radius) is not int or radius < 0:
        raise ValueError("Erosion radius must be a nonnegative integer")
    padded = F.pad(mask.bool(), (radius,) * 6, value=False)
    eroded = torch.ones_like(mask, dtype=torch.bool)
    depth, height, width = mask.shape[-3:]
    for d in range(-radius, radius + 1):
        for h in range(-radius, radius + 1):
            for w in range(-radius, radius + 1):
                if d * d + h * h + w * w <= radius * radius:
                    eroded &= padded[
                        ...,
                        radius + d : radius + d + depth,
                        radius + h : radius + h + height,
                        radius + w : radius + w + width,
                    ]
    return eroded


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


def isotropic_dipole_tensors(shape: Sequence[int], voxel_sizes: Tensor) -> Tensor:
    """Per-sample rFFT dipoles for positive isotropic N-by-3 voxel sizes in mm.

    B0 is along the third axis, matching the simulation convention. A common
    isotropic spacing cancels in the frequency ratio; retain the same small
    FP64 denominator epsilon as continuous_dipole_kernel. No cache keyed by
    random floating-point spacings is created.
    """
    if len(shape) != 3 or any(n < 2 for n in shape):
        raise ValueError("Dipole grids need three axes of length >= 2")
    if (
        voxel_sizes.ndim != 2
        or voxel_sizes.shape[1] != 3
        or voxel_sizes.shape[0] < 1
        or not bool(torch.isfinite(voxel_sizes).all())
        or not bool((voxel_sizes > 0).all())
        or not bool((voxel_sizes == voxel_sizes[:, :1]).all())
    ):
        raise ValueError("Voxel sizes must be positive, finite, isotropic [N,3] values")
    axes = [
        2 * torch.fft.fftfreq(n, device=voxel_sizes.device, dtype=torch.float64)
        for n in shape
    ]
    squared = (
        axes[0].square()[:, None, None]
        + axes[1].square()[None, :, None]
        + axes[2][: shape[2] // 2 + 1].square()[None, None, :]
    )
    spacing = voxel_sizes[:, 0].double().reshape(-1, 1, 1, 1, 1)
    kernel = 1 / 3 - axes[2][: shape[2] // 2 + 1].square()[
        None, None, None, None, :
    ] / (squared[None, None] + torch.finfo(torch.float64).eps * spacing.square())
    kernel[..., 0, 0, 0] = 0
    return kernel.float()


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


def cutoff_kernels(
    dipole: Tensor, threshold: float, mode: str = "hard"
) -> tuple[Tensor, Tensor]:
    """Regularized inverse B and complementary correction weight I-BD.

    Soft mode uses B=D/(D^2+tau^2), Q=tau^2/(D^2+tau^2). Q is a smooth
    spectral weight, not an orthogonal projection. The default hard mode
    preserves the historical TKD inverse and binary near-null projection.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("TKD threshold must be finite and positive")
    if mode not in ("hard", "soft"):
        raise ValueError("Cutoff mode must be hard or soft")
    with torch.autocast(dipole.device.type, enabled=False):
        d = dipole.float()
        if mode == "soft":
            tau2 = threshold**2
            denominator = d.square() + tau2
            return d / denominator, tau2 / denominator
        retained = d.abs() > threshold
        inverse = torch.zeros_like(d)
        inverse[retained] = d[retained].reciprocal()
        return inverse, (~retained).float()


def truncated_gradient_descent(
    field: Tensor,
    dipole: Tensor,
    threshold: float,
    iterations: int = 30,
    step_size: float | None = None,
) -> Tensor:
    """Zero-start full-grid least squares with a hard-truncated dipole.

    Solve min_x 0.5*||D_tau*x-field||^2 with D_tau=D*1[abs(D)>threshold].
    Each update is x <- x - eta*D_tau*(D_tau*x-field). The periodic operator
    is diagonal in Fourier space, so all updates can be made there without
    repeatedly transforming the volumes. No spatial masking or preconditioning
    is applied. The default eta=1/max(D_tau^2) is stable without oscillation.
    Real fields and multipliers use FP32, Fourier iterates use complex64.
    """
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 0
    ):
        raise ValueError("iterations must be a nonnegative integer")
    if step_size is not None and (not math.isfinite(step_size) or step_size <= 0):
        raise ValueError("step_size must be positive and finite")
    with torch.autocast(field.device.type, enabled=False):
        kernel = rfft_kernel(dipole, field.shape[-3:])
        if not bool(torch.isfinite(kernel).all()):
            raise ValueError("Dipole must be finite")
        _, projection = cutoff_kernels(kernel, threshold)
        truncated = kernel * (1 - projection)
        bound = float(truncated.square().amax())
        if bound == 0:
            return field.float() * 0
        maximum_step = 1 / bound
        if step_size is not None and step_size > maximum_step * (1 + 1e-7):
            raise ValueError("step_size exceeds 1/max(D_tau^2)")
        eta = maximum_step if step_size is None else step_size
        spectrum = torch.fft.rfftn(field.float(), dim=SPATIAL)
        estimate = torch.zeros_like(spectrum)
        for _ in range(iterations):
            estimate = estimate - eta * truncated * (truncated * estimate - spectrum)
        return torch.fft.irfftn(estimate, s=field.shape[-3:], dim=SPATIAL)


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
