"""U-Net -> TKD -> regularized null space network (arXiv:1806.06137 §3.2)."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .physics import apply_multiplier, cutoff_kernels, masked_rms, validate_mask


@dataclass(frozen=True)
class ModelConfig:
    features: tuple[int, ...] = (16, 32, 64)
    threshold: float = 0.1
    scale_floor: float = 1e-6
    checkpoint_blocks: bool = False
    residual_filter: bool = False

    def __post_init__(self) -> None:
        if not self.features or any(v < 1 for v in self.features):
            raise ValueError("U-Net features must be positive")
        if (
            not math.isfinite(self.threshold)
            or self.threshold <= 0
            or not math.isfinite(self.scale_floor)
            or self.scale_floor <= 0
        ):
            raise ValueError("Threshold and scaling floor must be finite and positive")


@dataclass
class Reconstruction:
    local_field: Tensor
    tkd: Tensor
    correction: Tensor
    chi: Tensor


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.CELU(),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.CELU(),
    )


class UNet3D(nn.Module):
    def __init__(self, features: tuple[int, ...], checkpoint_blocks: bool = False):
        super().__init__()
        self.depth = len(features)
        self.checkpoint_blocks = checkpoint_blocks
        self.downs = nn.ModuleList()
        previous = 1
        for width in features:
            self.downs.append(double_conv(previous, width))
            previous = width
        self.bottleneck = double_conv(previous, previous * 2)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        previous *= 2
        for width in reversed(features):
            self.up_convs.append(nn.Conv3d(previous, width, 3, padding=1))
            self.up_blocks.append(double_conv(width * 2, width))
            previous = width
        self.final = nn.Conv3d(features[0], 1, 1)
        nn.init.normal_(self.final.weight, std=0.01)
        nn.init.zeros_(self.final.bias)

    def _block(self, block: nn.Module, x: Tensor) -> Tensor:
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: Tensor) -> Tensor:
        # InstanceNorm needs >1 voxel at the bottleneck. Padding is internal to
        # the CNN; FFT grids and the externally visible geometry never change.
        minimum = 2 ** (self.depth + 1)
        padding = tuple(
            v for n in reversed(x.shape[-3:]) for v in (0, max(0, minimum - n))
        )
        shape = x.shape[-3:]
        if any(padding):
            x = F.pad(x, padding)
        skips = []
        for block in self.downs:
            x = self._block(block, x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self._block(self.bottleneck, x)
        for up, block, skip in zip(
            self.up_convs, self.up_blocks, reversed(skips), strict=True
        ):
            x = F.celu(
                up(
                    F.interpolate(
                        x, size=skip.shape[-3:], mode="trilinear", align_corners=False
                    )
                )
            )
            x = self._block(block, torch.cat((skip, x), dim=1))
        return self.final(x)[..., : shape[0], : shape[1], : shape[2]]


class DNQSM(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.filter = UNet3D(self.config.features, self.config.checkpoint_blocks)
        self.null_network = UNet3D(self.config.features, self.config.checkpoint_blocks)

    def forward(
        self, total_field: Tensor, mask: Tensor, dipole: Tensor
    ) -> Reconstruction:
        validate_mask(mask, total_field)
        field = total_field.float() * mask
        scale = masked_rms(field, mask, self.config.scale_floor)
        filtered = self.filter(field / scale).float() * scale
        local = (field + filtered if self.config.residual_filter else filtered) * mask
        inverse, projection = cutoff_kernels(dipole, self.config.threshold)
        initial = apply_multiplier(local, inverse)
        chi_scale = masked_rms(initial, mask, self.config.scale_floor)
        residual = self.null_network(initial / chi_scale).float() * chi_scale
        correction = apply_multiplier(residual, projection)
        # Spatial masking here would destroy the projection's Fourier support.
        return Reconstruction(local, initial, correction, initial + correction)
