"""U-Net -> TKD -> regularized null space network (arXiv:1806.06137 §3.2)."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .physics import apply_multiplier, cutoff_kernels, masked_rms, validate_mask

DEFAULT_FEATURES = (23, 47, 90, 180, 360, 720)
LEAKY_RELU_NEGATIVE_SLOPE = 0.01


@dataclass(frozen=True)
class ModelConfig:
    features: tuple[int, ...] = DEFAULT_FEATURES
    threshold: float = 0.2
    scale_floor: float = 1e-6
    checkpoint_blocks: bool = False
    residual_filter: bool = False
    full_grid_field: bool = False
    laplacian_input: bool = True
    filter_input_version: int = 2
    laplacian_fp32: bool = True
    conditional_laplacian_threshold: float = 32.0
    filter_output_skip: bool = True
    filter_laplacian_residual: bool = False
    filter_refinement_features: int = 0
    activation: str = "leaky_relu"
    leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE

    def __post_init__(self) -> None:
        if self.activation not in ("leaky_relu", "celu"):
            raise ValueError("Activation must be leaky_relu or celu")
        if not math.isfinite(self.leaky_relu_slope) or self.leaky_relu_slope < 0:
            raise ValueError("LeakyReLU slope must be finite and nonnegative")
        if (
            type(self.filter_refinement_features) is not int
            or self.filter_refinement_features < 0
        ):
            raise ValueError("Filter refinement features must be a nonnegative integer")
        if self.filter_laplacian_residual and (
            self.filter_output_skip
            or not self.laplacian_input
            or self.filter_input_version != 2
            or self.residual_filter
            or self.full_grid_field
        ):
            raise ValueError(
                "Laplacian residual requires version-2 Laplacian input without "
                "output concatenation, total-field residual, or full-grid output"
            )
        if (
            not math.isfinite(self.conditional_laplacian_threshold)
            or self.conditional_laplacian_threshold < 0
        ):
            raise ValueError(
                "Conditional Laplacian threshold must be finite and nonnegative"
            )
        if self.conditional_laplacian_threshold and (
            not self.laplacian_fp32 or self.filter_input_version != 2
        ):
            raise ValueError(
                "Conditional Laplacian scaling requires FP32 version-2 filtering"
            )
        if self.filter_input_version not in (1, 2):
            raise ValueError("Unsupported filter input version")
        # Output masking is independent of input preprocessing: the filter
        # still receives the Laplacian of the masked field, and full_grid_field
        # only decides whether its output is truncated before TKD. Coupling the
        # two forced unmasking to also adopt the version-1 two-channel input,
        # which measured 8.4 points of susceptibility NRMSE worse on the fixed
        # COSMOS fit while unmasking itself was 4.5 points better.
        if self.filter_input_version == 2 and (
            not self.laplacian_input or self.residual_filter
        ):
            raise ValueError(
                "Laplacian-only filtering requires laplacian_input=True and "
                "residual_filter=False; filter_input_version=1 selects the "
                "historical architecture"
            )
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


def activation_layer(
    name: str = "leaky_relu", leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE
) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=leaky_relu_slope)
    if name == "celu":
        return nn.CELU()
    raise ValueError("Activation must be leaky_relu or celu")


def double_conv(
    in_channels: int,
    out_channels: int,
    activation: str = "leaky_relu",
    leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm3d(out_channels, affine=True),
        activation_layer(activation, leaky_relu_slope),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm3d(out_channels, affine=True),
        activation_layer(activation, leaky_relu_slope),
    )


class PhaseRefinementBlock(nn.Module):
    """Single-channel ResNet refinement with a signed identity output."""

    def __init__(
        self,
        features: int,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(1, features, 3, padding=1)
        self.norm = nn.InstanceNorm3d(features, affine=True)
        self.activation = activation_layer(activation, leaky_relu_slope)
        self.conv2 = nn.Conv3d(features, 1, 3, padding=1)
        self.zero_output()

    def zero_output(self) -> None:
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(self.activation(self.norm(self.conv1(x))))
        return x.float() + residual.float()


class UNet3D(nn.Module):
    def __init__(
        self,
        features: tuple[int, ...],
        checkpoint_blocks: bool = False,
        in_channels: int = 1,
        output_skip_channels: int = 0,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
    ):
        super().__init__()
        self.depth = len(features)
        self.checkpoint_blocks = checkpoint_blocks
        self.output_skip_channels = output_skip_channels
        self.up_activation = activation_layer(activation, leaky_relu_slope)
        self.downs = nn.ModuleList()
        previous = in_channels
        for width in features:
            self.downs.append(
                double_conv(previous, width, activation, leaky_relu_slope)
            )
            previous = width
        self.bottleneck = double_conv(
            previous, previous * 2, activation, leaky_relu_slope
        )
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        previous *= 2
        for width in reversed(features):
            self.up_convs.append(nn.Conv3d(previous, width, 3, padding=1))
            self.up_blocks.append(
                double_conv(width * 2, width, activation, leaky_relu_slope)
            )
            previous = width
        self.final = nn.Conv3d(features[0] + output_skip_channels, 1, 1)
        nn.init.normal_(self.final.weight, std=0.01)
        nn.init.zeros_(self.final.bias)
        if output_skip_channels:
            with torch.no_grad():
                self.final.weight[:, features[0] :].zero_()

    def _block(self, block: nn.Module, x: Tensor) -> Tensor:
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: Tensor, output_skip: Tensor | None = None) -> Tensor:
        if self.output_skip_channels:
            expected = (x.shape[0], self.output_skip_channels, *x.shape[-3:])
            if output_skip is None or tuple(output_skip.shape) != expected:
                raise ValueError(
                    "Output skip must match the input batch and spatial shape"
                )
        elif output_skip is not None:
            raise ValueError("This U-Net has no output skip channels")
        # InstanceNorm needs >1 voxel at the bottleneck. Padding is internal to
        # the CNN; FFT grids and the externally visible geometry never change.
        minimum = 2 ** (self.depth + 1)
        padding = tuple(
            v for n in reversed(x.shape[-3:]) for v in (0, max(0, minimum - n))
        )
        shape = x.shape[-3:]
        if any(padding):
            x = F.pad(x, padding)
            if output_skip is not None:
                output_skip = F.pad(output_skip, padding)
        skips = []
        for block in self.downs:
            x = self._block(block, x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self._block(self.bottleneck, x)
        for up, block, skip in zip(
            self.up_convs, self.up_blocks, reversed(skips), strict=True
        ):
            x = self.up_activation(
                up(
                    F.interpolate(
                        x, size=skip.shape[-3:], mode="trilinear", align_corners=False
                    )
                )
            )
            x = self._block(block, torch.cat((skip, x), dim=1))
        if output_skip is not None:
            x = torch.cat((x, output_skip.to(dtype=x.dtype)), dim=1)
        return self.final(x)[..., : shape[0], : shape[1], : shape[2]]


class DNQSM(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        # A buffer follows the model's device and checkpoint state without
        # becoming a trainable parameter. None preserves legacy checkpoints.
        self.register_buffer(
            "laplacian_kernel",
            torch.tensor(
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
            )[None, None]
            if self.config.laplacian_input
            else None,
        )
        self.filter = UNet3D(
            self.config.features,
            self.config.checkpoint_blocks,
            in_channels=2
            if self.config.filter_input_version == 1 and self.config.laplacian_input
            else 1,
            output_skip_channels=int(self.config.filter_output_skip),
            activation=self.config.activation,
            leaky_relu_slope=self.config.leaky_relu_slope,
        )
        self.null_network = UNet3D(
            self.config.features,
            self.config.checkpoint_blocks,
            activation=self.config.activation,
            leaky_relu_slope=self.config.leaky_relu_slope,
        )
        # Register under filter so phase/chi freezing and gradient diagnostics
        # include the refinement. Build last to preserve both U-Nets' RNG draws.
        self.filter.refinement = (
            PhaseRefinementBlock(
                self.config.filter_refinement_features,
                self.config.activation,
                self.config.leaky_relu_slope,
            )
            if self.config.filter_refinement_features
            else nn.Identity()
        )

    def initialize_from(self, source: "DNQSM") -> None:
        """Warm-start compatible weights, adding an identity refinement if needed."""
        if tuple(source.config.features) != tuple(self.config.features):
            raise ValueError(
                "Initialization checkpoint has different U-Net widths or depth; "
                "start a fresh run for the deeper architecture"
            )
        if any(
            getattr(source.config, name) != getattr(self.config, name)
            for name in (
                "filter_input_version",
                "laplacian_input",
                "residual_filter",
                "full_grid_field",
                "filter_laplacian_residual",
            )
        ):
            raise ValueError(
                "Initialization checkpoint has a different filtering architecture"
            )
        state = source.state_dict()
        if (
            source.config.filter_refinement_features
            != self.config.filter_refinement_features
        ):
            if source.config.filter_refinement_features:
                raise ValueError(
                    "Initialization cannot remove or resize a trained phase refinement block"
                )
            self.filter.refinement.zero_output()
            state.update(
                {
                    "filter.refinement." + name: value
                    for name, value in self.filter.refinement.state_dict().items()
                }
            )
        if self.config.filter_output_skip and not source.config.filter_output_skip:
            weight = state["filter.final.weight"]
            state["filter.final.weight"] = torch.cat(
                (weight, torch.zeros_like(weight[:, :1])), dim=1
            )
        elif source.config.filter_output_skip and not self.config.filter_output_skip:
            raise ValueError(
                "Initialization cannot discard a trained filter output skip"
            )
        self.load_state_dict(state)

    def predict_local_field(self, total_field: Tensor, mask: Tensor) -> Tensor:
        """Predict the clean field, masking both sides of the learned filter."""
        validate_mask(mask, total_field)
        field = total_field.float() * mask
        scale = masked_rms(field, mask, self.config.scale_floor)
        normalized = field / scale
        filter_input = normalized
        if self.laplacian_kernel is not None:
            if self.config.laplacian_fp32:
                # Autocast rounds the fractional stencil to a nonzero sum in
                # BF16, leaking constant backgrounds into this derivative.
                with torch.autocast(total_field.device.type, enabled=False):
                    laplacian = F.conv3d(
                        normalized.float(), self.laplacian_kernel.float(), padding=1
                    )
            else:
                # Preserve the exact preprocessing of historical checkpoints.
                laplacian = F.conv3d(normalized, self.laplacian_kernel, padding=1)
            filter_input = (
                torch.cat((normalized, laplacian), dim=1)
                if self.config.filter_input_version == 1
                else laplacian
            )
            if self.config.conditional_laplacian_threshold:
                with torch.autocast(total_field.device.type, enabled=False):
                    valid = (
                        F.avg_pool3d(mask.float(), 3, stride=1, padding=1) > 1 - 1e-6
                    ).float()
                    interior_scale = masked_rms(
                        laplacian.float() * scale, valid, self.config.scale_floor
                    )
                    selected = (
                        scale / interior_scale
                        >= self.config.conditional_laplacian_threshold
                    ) & (valid.sum(dim=(-3, -2, -1), keepdim=True) > 0)
                    interior_input = valid * laplacian.float() * scale / interior_scale
                    filter_input = torch.where(
                        selected, interior_input, laplacian.float()
                    )
                    scale = torch.where(selected, interior_scale, scale)
        raw = (
            self.filter(filter_input, field / scale)
            if self.config.filter_output_skip
            else self.filter(filter_input)
        ).float()
        # Add in the same normalized units before restoring physical scale.
        # Conditional scaling changes both the CNN input and this bypass.
        if self.config.filter_laplacian_residual:
            raw = raw + filter_input.float()
        if self.config.filter_refinement_features:
            raw = self.filter._block(self.filter.refinement, raw).float()
        filtered = raw * scale
        local = field + filtered if self.config.residual_filter else filtered
        if not self.config.full_grid_field:
            local = local * mask
        return local

    def forward(
        self, total_field: Tensor, mask: Tensor, dipole: Tensor
    ) -> Reconstruction:
        local = self.predict_local_field(total_field, mask)
        inverse, projection = cutoff_kernels(dipole, self.config.threshold)
        initial = apply_multiplier(local, inverse)
        chi_scale = masked_rms(initial, mask, self.config.scale_floor)
        residual = self.null_network(initial / chi_scale).float() * chi_scale
        correction = apply_multiplier(residual, projection)
        # Spatial masking here would destroy the projection's Fourier support.
        return Reconstruction(local, initial, correction, initial + correction)
