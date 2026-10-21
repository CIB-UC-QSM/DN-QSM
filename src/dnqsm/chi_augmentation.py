"""Seeded susceptibility shape families; no validation images enter this module."""

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr

PROFILES = ("baseline", "bimodal", "central_tail")
CHI_AUGMENTATION_SEED = 53107147


@dataclass(frozen=True)
class ChiAugmentationConfig:
    version: int = 1
    final_grid_rms: bool = False
    bimodal_probability: float = 0.0
    central_probability: float = 0.0
    original_only: bool = True
    cdf_samples: int = 32768
    quantile_knots: int = 257
    quantile_tail: float = 0.001
    min_slope: float = 0.05
    max_slope: float = 8.0
    min_span_ppm: float = 1e-6
    bimodal_negative_mean: tuple[float, float] = (-0.045, -0.025)
    bimodal_positive_mean: tuple[float, float] = (0.008, 0.025)
    bimodal_std: tuple[float, float] = (0.003, 0.012)
    bimodal_negative_weight: tuple[float, float] = (0.30, 0.50)
    bimodal_tail_weight: tuple[float, float] = (0.005, 0.04)
    bimodal_tail_mean: tuple[float, float] = (0.06, 0.16)
    bimodal_tail_std: tuple[float, float] = (0.02, 0.05)
    bimodal_alpha: tuple[float, float] = (0.7, 1.0)
    central_mean: tuple[float, float] = (-0.004, 0.003)
    central_std: tuple[float, float] = (0.014, 0.026)
    central_positive_weight: tuple[float, float] = (0.01, 0.05)
    central_positive_mean: tuple[float, float] = (0.07, 0.16)
    central_positive_std: tuple[float, float] = (0.02, 0.06)
    central_negative_weight: tuple[float, float] = (0.0, 0.03)
    central_negative_mean: tuple[float, float] = (-0.12, -0.05)
    central_negative_std: tuple[float, float] = (0.015, 0.04)
    central_alpha: tuple[float, float] = (0.2, 0.6)

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported chi augmentation version")
        if (
            type(self.final_grid_rms) is not bool
            or type(self.original_only) is not bool
        ):
            raise ValueError("Chi augmentation switches must be booleans")
        for name in (
            "bimodal_probability",
            "central_probability",
            "quantile_tail",
            "min_slope",
            "max_slope",
            "min_span_ppm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid chi augmentation {name}")
        if (
            min(self.bimodal_probability, self.central_probability) < 0
            or self.bimodal_probability + self.central_probability > 1
        ):
            raise ValueError("Chi profile probabilities must sum to at most one")
        if (
            self.bimodal_probability or self.central_probability
        ) and not self.final_grid_rms:
            raise ValueError("Chi shape profiles require final_grid_rms")
        if not (
            0 < self.quantile_tail < 0.05
            and 0 < self.min_slope <= self.max_slope
            and self.min_span_ppm > 0
        ):
            raise ValueError("Invalid chi quantile, slope, or span bounds")
        for name, low, high in (
            ("cdf_samples", 64, 1_000_000),
            ("quantile_knots", 8, 4097),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid chi augmentation {name}")
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
                or value[0] > value[1]
            ):
                raise ValueError(f"Invalid chi augmentation range {field.name}")
            if "std" in field.name and value[0] <= 0:
                raise ValueError("Chi component standard deviations must be positive")
            if ("weight" in field.name or "alpha" in field.name) and not 0 <= value[
                0
            ] <= value[1] <= 1:
                raise ValueError("Chi weights and alpha must be in [0,1]")
            object.__setattr__(self, field.name, tuple(value))
        if (
            self.bimodal_negative_weight[1] + self.bimodal_tail_weight[1] >= 1
            or self.central_positive_weight[1] + self.central_negative_weight[1] >= 1
        ):
            raise ValueError("Chi component weights must leave positive core mass")

    @property
    def enabled(self):
        return self.final_grid_rms

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("Unknown chi augmentation configuration fields")
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def draw_target(config, profile, rng):
    """Return a random Gaussian mixture, not a fitted external histogram."""

    def draw(name):
        return float(rng.uniform(*getattr(config, name)))

    if profile == "bimodal":
        negative, tail = draw("bimodal_negative_weight"), draw("bimodal_tail_weight")
        weights = [negative, 1 - negative - tail, tail]
        means = [
            draw("bimodal_negative_mean"),
            draw("bimodal_positive_mean"),
            draw("bimodal_tail_mean"),
        ]
        stds = [draw("bimodal_std"), draw("bimodal_std"), draw("bimodal_tail_std")]
        alpha = draw("bimodal_alpha")
    elif profile == "central_tail":
        positive, negative = (
            draw("central_positive_weight"),
            draw("central_negative_weight"),
        )
        weights = [1 - positive - negative, positive, negative]
        means = [
            draw("central_mean"),
            draw("central_positive_mean"),
            draw("central_negative_mean"),
        ]
        stds = [
            draw("central_std"),
            draw("central_positive_std"),
            draw("central_negative_std"),
        ]
        alpha = draw("central_alpha")
    else:
        raise ValueError(f"Unknown chi shape profile: {profile}")
    return dict(weights=weights, means_ppm=means, stds_ppm=stds, alpha=alpha)


def target_quantiles(probabilities, target):
    means, stds, weights = (
        np.asarray(target[key], dtype=np.float64)
        for key in ("means_ppm", "stds_ppm", "weights")
    )
    grid = np.linspace(np.min(means - 9 * stds), np.max(means + 9 * stds), 8193)
    cdf = (ndtr((grid[:, None] - means) / stds) * weights).sum(axis=1)
    return np.interp(probabilities, cdf, grid)


def monotone_transport(values, probabilities, source_quantiles, target, config):
    """Bound knot secants, interpolate monotonically, extend tails without clamps."""
    x, indices = np.unique(source_quantiles, return_index=True)
    if len(x) < 8 or x[-1] - x[0] < config.min_span_ppm:
        return None, {"fallback_reason": "degenerate_quantiles"}
    y = target_quantiles(probabilities, target)[indices]
    dx = np.diff(x)
    secants = np.diff(y) / dx
    bounded = np.clip(secants, config.min_slope, config.max_slope)
    adjusted = np.r_[0.0, np.cumsum(bounded * dx)]
    middle = len(x) // 2
    adjusted += y[middle] - adjusted[middle]
    interpolator = PchipInterpolator(x, adjusted, extrapolate=False)
    output = interpolator(np.clip(values, x[0], x[-1]))
    # Unit-slope tail extensions preserve source-tail differences and are
    # continuous at the robust fit bounds. No susceptibility values are clipped.
    below, above = values < x[0], values > x[-1]
    output[below] = adjusted[0] + (values[below] - x[0])
    output[above] = adjusted[-1] + (values[above] - x[-1])
    alpha = target["alpha"]
    output = (1 - alpha) * values + alpha * output
    return output.astype(np.float32), dict(
        limited_interval_fraction=float(np.mean(bounded != secants)),
        source_fit_bounds_ppm=[float(x[0]), float(x[-1])],
        mapped_fit_bounds_ppm=[float(adjusted[0]), float(adjusted[-1])],
        source_unique_knots=len(x),
    )


def augment_chi(chi, mask, config, seed, subject, forced_profile=None):
    """CPU data augmentation; independent RNG preserves all existing seed streams."""
    config = ChiAugmentationConfig.from_dict(config)
    if forced_profile is not None and forced_profile not in PROFILES:
        raise ValueError("Invalid forced chi profile")
    metadata = dict(
        profile="baseline", requested_profile="baseline", fallback=False, parameters={}
    )
    if not config.enabled or (
        config.original_only and subject.startswith("synthetic:")
    ):
        return chi, metadata
    rng = np.random.default_rng(seed + CHI_AUGMENTATION_SEED)
    selector = rng.random()
    requested = forced_profile or (
        "bimodal"
        if selector < config.bimodal_probability
        else "central_tail"
        if selector < config.bimodal_probability + config.central_probability
        else "baseline"
    )
    metadata["requested_profile"] = requested
    if requested == "baseline":
        return chi, metadata
    if chi.device.type != "cpu":
        raise ValueError("Chi shape augmentation runs in the CPU dataset loader")
    selected = chi[mask.bool()].numpy()
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("Chi augmentation needs finite masked values")
    sample = selected[
        rng.integers(len(selected), size=min(config.cdf_samples, len(selected)))
    ]
    probabilities = np.linspace(
        config.quantile_tail, 1 - config.quantile_tail, config.quantile_knots
    )
    quantiles = np.quantile(sample, probabilities)
    target = draw_target(config, requested, rng)
    mapped, details = monotone_transport(
        selected, probabilities, quantiles, target, config
    )
    metadata["parameters"] = dict(target, **details)
    if mapped is None:
        metadata["fallback"] = True
        return chi, metadata
    result = torch.zeros_like(chi, dtype=torch.float32)
    result[mask.bool()] = torch.from_numpy(mapped)
    metadata["profile"] = requested
    return result, metadata
