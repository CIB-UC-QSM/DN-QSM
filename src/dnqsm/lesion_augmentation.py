"""Seeded focal susceptibility lesions in physical ppm, after tissue scaling."""

import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

LESION_SEED = 67867967
DEPHASING_SEED = 86028121
# 2022 CODATA proton gamma/(2*pi), MHz/T:
# https://physics.nist.gov/cgi-bin/cuu/Value?gammapbar
PROTON_MHZ_PER_T = 42.577478461


@dataclass(frozen=True)
class LesionAugmentationConfig:
    version: int = 1
    probability: float = 0.0
    count: tuple[int, int] = (1, 3)
    sphere_probability: float = 0.5
    sphere_radius_voxels: tuple[float, float] = (2.0, 4.0)
    ellipsoid_semiaxes_voxels: tuple[float, float] = (2.0, 6.0)
    ellipsoid_transition_voxels: tuple[float, float] = (1.0, 2.0)
    susceptibility_abs_ppm: tuple[float, float] = (0.25, 0.6)
    positive_probability: float = 0.5
    original_only: bool = True
    max_placement_attempts: int = 64
    dephasing_probability: float = 0.0
    dephasing_signal_loss_probability: float = 0.5
    dephasing_relative_magnitude: tuple[float, float] = (0.01, 0.1)
    dephasing_spike_radians: tuple[float, float] = (math.pi / 2, math.pi)
    dephasing_b0_tesla: float = 3.0
    dephasing_echo_time_ms: float = 20.0

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported lesion augmentation version")
        if type(self.original_only) is not bool:
            raise ValueError("Lesion original_only must be boolean")
        for name in (
            "probability",
            "sphere_probability",
            "positive_probability",
            "dephasing_probability",
            "dephasing_signal_loss_probability",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Invalid lesion {name}")
        for field in fields(self):
            if not isinstance(field.default, tuple):
                continue
            value = getattr(self, field.name)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in value
                )
                or not 0 < value[0] <= value[1]
            ):
                raise ValueError(f"Invalid lesion range {field.name}")
            object.__setattr__(self, field.name, tuple(value))
        if any(type(v) is not int for v in self.count) or self.count[1] > 32:
            raise ValueError("Lesion count must be an integer range in [1,32]")
        if np.float32(self.susceptibility_abs_ppm[0]) <= np.float32(0.2):
            raise ValueError(
                "Lesion core susceptibility must have abs(chi) > 0.2 ppm in FP32"
            )
        if (
            type(self.max_placement_attempts) is not int
            or self.max_placement_attempts < 1
        ):
            raise ValueError("Lesion placement attempts must be a positive integer")
        if self.dephasing_relative_magnitude[1] > 1:
            raise ValueError("Dephasing relative signal magnitude must be in (0,1]")
        if self.dephasing_spike_radians[1] > math.pi:
            raise ValueError("Dephasing phase-spike bounds must be in (0,pi]")
        for name in ("dephasing_b0_tesla", "dephasing_echo_time_ms"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid lesion {name}")
        if (
            not math.isfinite(self.phase_radians_per_ppm)
            or self.phase_radians_per_ppm <= 0
        ):
            raise ValueError(
                "Dephasing phase-to-ppm conversion must be finite and positive"
            )

    @property
    def enabled(self):
        return self.probability > 0

    @property
    def phase_radians_per_ppm(self):
        # MHz/T times ppm cancels 1e6; echo time is converted from ms to s.
        return (
            2
            * math.pi
            * PROTON_MHZ_PER_T
            * self.dephasing_b0_tesla
            * self.dephasing_echo_time_ms
            * 1e-3
        )

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown lesion augmentation configuration fields")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def _opacity(principal, semiaxes, transition):
    """Constant core and a compact C2 transition between nested ellipsoids."""
    axes = np.asarray(semiaxes).reshape(3, *([1] * (principal.ndim - 1)))
    radius = np.sqrt(np.sum((principal / axes) ** 2, axis=0))
    core = radius <= 1
    if not transition:
        return core.astype(np.float32), core
    outer_radius = np.sqrt(np.sum((principal / (axes + transition)) ** 2, axis=0))
    t = np.clip((radius - 1) / np.maximum(radius - outer_radius, 1e-12), 0, 1)
    # 1 - smootherstep(t), factored to avoid cancellation near the outer edge.
    opacity = (1 - t) ** 3 * (1 + 3 * t + 6 * t * t)
    return opacity.astype(np.float32), core


def _dephasing_event(core, start, config, seed, lesion_index):
    """Independent stream: changing this policy never changes lesion geometry."""
    if not config.dephasing_probability:
        return None
    rng = np.random.default_rng(seed + DEPHASING_SEED + lesion_index * 1000003)
    if rng.random() >= config.dephasing_probability:
        return None
    candidates = np.argwhere(core)
    voxel = start + candidates[int(rng.integers(len(candidates)))]
    if rng.random() < config.dephasing_signal_loss_probability:
        kind = "signal_loss"
        magnitude = float(rng.uniform(*config.dephasing_relative_magnitude))
        angle = float(rng.uniform(-math.pi, math.pi))
    else:
        kind = "phase_spike"
        magnitude = 1.0
        angle = float(rng.uniform(*config.dephasing_spike_radians))
        angle *= 1 if rng.random() < 0.5 else -1
    # Relative complex signal referenced to the otherwise measured voxel.
    real, imag = magnitude * math.cos(angle), magnitude * math.sin(angle)
    phase_error = math.atan2(imag, real)
    return dict(
        voxel=voxel.tolist(),
        kind=kind,
        relative_magnitude=magnitude,
        relative_signal_real=real,
        relative_signal_imag=imag,
        phase_error_radians=phase_error,
        field_error_ppm=phase_error / config.phase_radians_per_ppm,
        b0_tesla=config.dephasing_b0_tesla,
        echo_time_ms=config.dephasing_echo_time_ms,
    )


@torch.no_grad()
def apply_lesion_dephasing(phase, chi, mask, encoded_events):
    """Corrupt only selected measured voxels, after background and noise.

    The relative complex-signal angle is converted to an unwrapped-field
    error in ppm, preserving the surrounding field's original phase branch.
    Return the original tensor and no artifact allocation when disabled.
    """
    if encoded_events is None:
        return phase, None
    if len(encoded_events) != len(phase):
        raise ValueError("Dephasing metadata must match the batch size")
    events = [json.loads(value) for value in encoded_events]
    if not any(events):
        return phase, None
    corrupted = phase.clone()
    artifact = torch.zeros_like(phase)
    for sample, points in enumerate(events):
        for point in points:
            voxel = point["voxel"]
            if len(voxel) != 3 or any(
                type(v) is not int or not 0 <= v < n
                for v, n in zip(voxel, phase.shape[-3:], strict=True)
            ):
                raise ValueError("Invalid dephasing voxel")
            index = (sample, 0, *voxel)
            if not bool(mask[index]) or not bool(chi[index].abs() > 0.2):
                raise ValueError(
                    "Dephasing voxel must lie inside an extreme-susceptibility lesion core"
                )
            error = point["field_error_ppm"]
            if not math.isfinite(error):
                raise ValueError("Dephasing field error must be finite")
            corrupted[index] += error
            artifact[index] += error
    return corrupted, artifact


def augment_lesions(chi, mask, config, seed, subject):
    """Insert nonoverlapping, fully intracranial spheres/rotated ellipsoids.

    Call after resizing, remapping, and final tissue RMS scaling, before field
    simulation. No later global rescaling is allowed to dilute lesion cores.
    Source tensors are never modified. A failed placement is reported, rather
    than clipping a lesion at the brain boundary or inventing a new mask.
    """
    config = LesionAugmentationConfig.from_dict(config)
    metadata = dict(
        requested_count=0,
        count=0,
        spheres=0,
        ellipsoids=0,
        skipped=0,
        parameters=[],
        dephasing_events=[],
    )
    if not config.enabled or (
        config.original_only and subject.startswith("synthetic:")
    ):
        return chi, metadata
    rng = np.random.default_rng(seed + LESION_SEED)
    if rng.random() >= config.probability:
        return chi, metadata
    if chi.device.type != "cpu" or mask.device.type != "cpu":
        raise ValueError("Lesion augmentation runs in the CPU dataset loader")
    if chi.ndim != 4 or chi.shape[0] != 1 or chi.shape != mask.shape:
        raise ValueError("Lesion image and mask must have shape [1,D,H,W]")
    if (
        not bool(torch.isfinite(chi).all())
        or not bool(((mask == 0) | (mask == 1)).all())
        or not bool(mask.any())
    ):
        raise ValueError(
            "Lesions require finite susceptibility and a nonempty binary mask"
        )
    roi = mask[0].numpy().astype(bool, copy=False)
    foreground = np.flatnonzero(roi)
    occupied = np.zeros_like(roi)
    count = int(rng.integers(config.count[0], config.count[1] + 1))
    metadata["requested_count"] = count
    result = chi
    for _ in range(count):
        sphere = rng.random() < config.sphere_probability
        if sphere:
            axes = np.repeat(rng.uniform(*config.sphere_radius_voxels), 3)
            transition = 0.0
            rotation = np.eye(3)
        else:
            axes = rng.uniform(*config.ellipsoid_semiaxes_voxels, size=3)
            transition = float(rng.uniform(*config.ellipsoid_transition_voxels))
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
        amplitude = float(rng.uniform(*config.susceptibility_abs_ppm))
        value = amplitude if rng.random() < config.positive_probability else -amplitude
        half_width = np.ceil(
            np.sqrt(np.sum((rotation * (axes + transition)[None, :]) ** 2, axis=1))
        ).astype(int)
        for _ in range(config.max_placement_attempts):
            center = np.array(np.unravel_index(int(rng.choice(foreground)), roi.shape))
            start, stop = center - half_width, center + half_width + 1
            if np.any(start < 0) or np.any(stop > roi.shape):
                continue
            region = tuple(
                slice(int(a), int(b)) for a, b in zip(start, stop, strict=True)
            )
            coordinates = np.stack(
                np.meshgrid(
                    *(
                        np.arange(a, b) - c
                        for a, b, c in zip(start, stop, center, strict=True)
                    ),
                    indexing="ij",
                )
            )
            principal = np.einsum("ij,j...->i...", rotation.T, coordinates)
            opacity, core = _opacity(principal, axes, transition)
            support = opacity > 0
            if not np.all(roi[region][support]) or np.any(occupied[region][support]):
                continue
            if result is chi:
                result = chi.clone()
            alpha = torch.from_numpy(opacity)
            target = result[(0, *region)]
            target.copy_(target * (1 - alpha) + value * alpha)
            occupied[region] |= support
            metadata["count"] += 1
            metadata["spheres" if sphere else "ellipsoids"] += 1
            lesion = dict(
                kind="sphere" if sphere else "ellipsoid",
                center_voxels=center.tolist(),
                semiaxes_voxels=axes.tolist(),
                transition_voxels=transition,
                rotation=rotation.tolist(),
                susceptibility_ppm=value,
                core_voxels=int(core.sum()),
                support_voxels=int(support.sum()),
            )
            event = _dephasing_event(core, start, config, seed, metadata["count"])
            if event is not None:
                lesion["dephasing"] = event
                metadata["dephasing_events"].append(event)
            metadata["parameters"].append(lesion)
            break
        else:
            metadata["skipped"] += 1
    return result, metadata
