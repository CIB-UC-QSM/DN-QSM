"""U-Net -> regularized inverse -> spectral residual network."""

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .inference import (
    SlidingWindowConfig,
    sliding_window_fields,
    sliding_window_reconstruction,
)
from .physics import (
    apply_multiplier,
    cutoff_kernels,
    frequency_laplacian,
    laplacian_stencil,
    masked_rms,
    quadratic_exterior_extension,
    smooth_harmonic_extension,
    validate_mask,
)

# Four encoder/decoder levels shared by the field and null-space U-Nets.
DEFAULT_FEATURES = (32, 64, 128, 256)
LEAKY_RELU_NEGATIVE_SLOPE = 0.01


@dataclass(frozen=True)
class ModelConfig:
    features: tuple[int, ...] = DEFAULT_FEATURES
    threshold: float = 0.1
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
    downsampling: str = "strided_conv"
    bias: bool = False
    laplacian_operator: str = "frequency"
    laplacian_extension_iters: int = 6
    laplacian_extension_mode: str = "quadratic"
    field_normalization: str = "legacy"
    normalization_layer: str = "instance"
    cutoff_mode: str = "soft"

    def __post_init__(self) -> None:
        if self.cutoff_mode not in ("hard", "soft"):
            raise ValueError("Cutoff mode must be hard or soft")
        if self.field_normalization not in ("legacy", "laplacian_rms"):
            raise ValueError("Field normalization must be legacy or laplacian_rms")
        if self.normalization_layer not in ("instance", "residual_instance"):
            raise ValueError(
                "Normalization layer must be instance or residual_instance"
            )
        if self.field_normalization == "laplacian_rms" and (
            not self.laplacian_input
            or self.filter_input_version != 2
            or not self.laplacian_fp32
        ):
            raise ValueError(
                "Laplacian RMS scaling requires FP32 version-2 Laplacian input"
            )
        if self.laplacian_extension_mode not in ("quadratic", "harmonic"):
            raise ValueError("Laplacian extension mode must be quadratic or harmonic")
        if (
            type(self.laplacian_extension_iters) is not int
            or self.laplacian_extension_iters < 0
        ):
            raise ValueError("Extension iterations must be a nonnegative integer")
        if self.laplacian_operator not in ("frequency", "stencil"):
            raise ValueError("Laplacian operator must be frequency or stencil")
        if (
            self.laplacian_input
            and self.laplacian_operator == "frequency"
            and not self.laplacian_fp32
        ):
            raise ValueError("Frequency Laplacian requires FP32 preprocessing")
        if self.downsampling not in ("strided_conv", "max_pool"):
            raise ValueError("Downsampling must be strided_conv or max_pool")
        if type(self.bias) is not bool:
            raise ValueError("Layer bias must be a boolean")
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
        if self.filter_input_version not in (1, 2, 3):
            raise ValueError("Unsupported filter input version")
        if self.filter_input_version == 3 and (
            self.laplacian_input
            or self.conditional_laplacian_threshold
            or self.filter_laplacian_residual
            or self.residual_filter
        ):
            raise ValueError(
                "Dual-branch filtering requires masked total-field input without "
                "Laplacian preprocessing, conditional scaling, or residual filtering"
            )
        # Output masking is independent of input preprocessing: the filter
        # receives the configured masked/extended input Laplacian; full_grid_field
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
    harmonic_field: Tensor | None = None

    @property
    def initial_chi(self) -> Tensor:
        return self.tkd


@dataclass
class FilteredFields:
    local_field: Tensor
    harmonic_field: Tensor | None = None


def activation_layer(
    name: str = "leaky_relu", leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE
) -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=leaky_relu_slope)
    if name == "celu":
        return nn.CELU()
    raise ValueError("Activation must be leaky_relu or celu")


class ResidualInstanceNorm3d(nn.InstanceNorm3d):
    """InstanceNorm with a learnable amplitude path on half the channels.

    Zero initial gain exactly preserves pretrained InstanceNorm behavior.
    The residual path exposes unnormalized feature amplitude during adaptation;
    tanh bounds its coefficient without killing its gradient at initialization.
    No batch statistics or running averages, and no new additive offsets.
    """

    def __init__(self, channels: int):
        super().__init__(channels, affine=True)
        self.residual_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        normalized = super().forward(x)
        split = self.num_features // 2
        raw = x[:, split:] * self.weight[split:].to(x.dtype)[None, :, None, None, None]
        if self.bias is not None:
            raw = raw + self.bias[split:].to(x.dtype)[None, :, None, None, None]
        amplitude = normalized[:, split:] + self.residual_gain.tanh().to(x.dtype) * (
            raw - normalized[:, split:]
        )
        return torch.cat((normalized[:, :split], amplitude), dim=1)


def instance_norm(
    channels: int, bias: bool, kind: str = "instance"
) -> nn.InstanceNorm3d:
    """Keep learned channel scales, optionally omitting normalization offsets."""
    norm = (
        ResidualInstanceNorm3d(channels)
        if kind == "residual_instance"
        else nn.InstanceNorm3d(channels, affine=True)
    )
    if not bias:
        norm.register_parameter("bias", None)
    return norm


def downsample_layer(channels: int, mode: str, bias: bool) -> nn.Module:
    if mode == "strided_conv":
        return nn.Conv3d(channels, channels, kernel_size=2, stride=2, bias=bias)
    if mode == "max_pool":
        return nn.MaxPool3d(2)
    raise ValueError("Downsampling must be strided_conv or max_pool")


def double_conv(
    in_channels: int,
    out_channels: int,
    activation: str = "leaky_relu",
    leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
    bias: bool = False,
    normalization: str = "instance",
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=bias),
        instance_norm(out_channels, bias, normalization),
        activation_layer(activation, leaky_relu_slope),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=bias),
        instance_norm(out_channels, bias, normalization),
        activation_layer(activation, leaky_relu_slope),
    )


class PhaseRefinementBlock(nn.Module):
    """Single-channel ResNet refinement with a signed identity output."""

    def __init__(
        self,
        features: int,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
        bias: bool = False,
        normalization: str = "instance",
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(1, features, 3, padding=1, bias=bias)
        self.norm = instance_norm(features, bias, normalization)
        self.activation = activation_layer(activation, leaky_relu_slope)
        self.conv2 = nn.Conv3d(features, 1, 3, padding=1, bias=bias)
        self.zero_output()

    def zero_output(self) -> None:
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv2(self.activation(self.norm(self.conv1(x))))
        return x.float() + residual.float()


class CheckpointedNetwork(nn.Module):
    def _block(self, block: nn.Module, x: Tensor) -> Tensor:
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)


class UNet3D(CheckpointedNetwork):
    def __init__(
        self,
        features: tuple[int, ...],
        checkpoint_blocks: bool = False,
        in_channels: int = 1,
        output_skip_channels: int = 0,
        activation: str = "leaky_relu",
        leaky_relu_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
        downsampling: str = "strided_conv",
        bias: bool = False,
        normalization: str = "instance",
    ):
        super().__init__()
        self.depth = len(features)
        self.checkpoint_blocks = checkpoint_blocks
        self.output_skip_channels = output_skip_channels
        self.up_activation = activation_layer(activation, leaky_relu_slope)
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        previous = in_channels
        for width in features:
            self.downs.append(
                double_conv(
                    previous, width, activation, leaky_relu_slope, bias, normalization
                )
            )
            self.downsamples.append(downsample_layer(width, downsampling, bias))
            previous = width
        self.bottleneck = double_conv(
            previous, previous * 2, activation, leaky_relu_slope, bias, normalization
        )
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        previous *= 2
        for width in reversed(features):
            self.up_convs.append(nn.Conv3d(previous, width, 3, padding=1, bias=bias))
            self.up_blocks.append(
                double_conv(
                    width * 2, width, activation, leaky_relu_slope, bias, normalization
                )
            )
            previous = width
        self.final = nn.Conv3d(features[0] + output_skip_channels, 1, 1, bias=bias)
        nn.init.normal_(self.final.weight, std=0.01)
        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)
        if output_skip_channels:
            with torch.no_grad():
                self.final.weight[:, features[0] :].zero_()

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
        for block, downsample in zip(self.downs, self.downsamples, strict=True):
            x = self._block(block, x)
            skips.append(x)
            x = self._block(downsample, x)
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


class FieldDecoder3D(CheckpointedNetwork):
    """One independent decoder from a shared bottleneck and encoder skips."""

    def __init__(
        self,
        features: tuple[int, ...],
        checkpoint_blocks: bool,
        output_skip_channels: int,
        activation: str,
        slope: float,
        bias: bool,
        normalization: str = "instance",
    ):
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.up_activation = activation_layer(activation, slope)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        previous = features[-1] * 2
        for width in reversed(features):
            self.up_convs.append(nn.Conv3d(previous, width, 3, padding=1, bias=bias))
            self.up_blocks.append(
                double_conv(width * 2, width, activation, slope, bias, normalization)
            )
            previous = width
        self.final = nn.Conv3d(features[0] + output_skip_channels, 1, 1, bias=bias)
        nn.init.normal_(self.final.weight, std=0.01)
        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)
        if output_skip_channels:
            with torch.no_grad():
                self.final.weight[:, features[0] :].zero_()

    def forward(
        self, x: Tensor, skips: list[Tensor], output_skip: Tensor | None = None
    ) -> Tensor:
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
        return self.final(x)


class DualFieldUNet3D(CheckpointedNetwork):
    """Share the encoder and bottleneck; decode local and harmonic fields separately."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.depth = len(config.features)
        self.checkpoint_blocks = config.checkpoint_blocks
        self.output_skip_channels = int(config.filter_output_skip)
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        previous = 1
        for width in config.features:
            self.downs.append(
                double_conv(
                    previous,
                    width,
                    config.activation,
                    config.leaky_relu_slope,
                    config.bias,
                    config.normalization_layer,
                )
            )
            self.downsamples.append(
                downsample_layer(width, config.downsampling, config.bias)
            )
            previous = width
        self.bottleneck = double_conv(
            previous,
            previous * 2,
            config.activation,
            config.leaky_relu_slope,
            config.bias,
            config.normalization_layer,
        )
        args = (
            config.features,
            config.checkpoint_blocks,
            self.output_skip_channels,
            config.activation,
            config.leaky_relu_slope,
            config.bias,
            config.normalization_layer,
        )
        self.local_decoder = FieldDecoder3D(*args)
        self.harmonic_decoder = FieldDecoder3D(*args)

    def forward(
        self, x: Tensor, output_skip: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if self.output_skip_channels:
            if output_skip is None or output_skip.shape != x.shape:
                raise ValueError(
                    "Output skip must match the input batch and spatial shape"
                )
        elif output_skip is not None:
            raise ValueError("This U-Net has no output skip channels")
        shape = x.shape[-3:]
        minimum = 2 ** (self.depth + 1)
        padding = tuple(v for n in reversed(shape) for v in (0, max(0, minimum - n)))
        if any(padding):
            x = F.pad(x, padding)
            if output_skip is not None:
                output_skip = F.pad(output_skip, padding)
        skips = []
        for block, downsample in zip(self.downs, self.downsamples, strict=True):
            x = self._block(block, x)
            skips.append(x)
            x = self._block(downsample, x)
        x = self._block(self.bottleneck, x)
        local = self.local_decoder(x, skips, output_skip)
        harmonic = self.harmonic_decoder(x, skips, output_skip)
        crop = (..., slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))
        return local[crop], harmonic[crop]


def susceptibility_unet(config: ModelConfig) -> UNet3D:
    return UNet3D(
        config.features,
        config.checkpoint_blocks,
        activation=config.activation,
        leaky_relu_slope=config.leaky_relu_slope,
        downsampling=config.downsampling,
        bias=config.bias,
        normalization=config.normalization_layer,
    )


class LocalFieldEstimator(nn.Module):
    def __init__(self, config: ModelConfig | None = None, *, _defer_refinement=False):
        super().__init__()
        self.config = config or ModelConfig()
        # Runtime inference policy, saved separately from architecture/weights.
        # Historical models retain full-volume inference until explicitly set.
        self.inference_config = SlidingWindowConfig(window_size=0)
        # A buffer follows the model's device and checkpoint state without
        # becoming a trainable parameter. None preserves legacy checkpoints.
        self.register_buffer(
            "laplacian_kernel",
            laplacian_stencil()
            if self.config.laplacian_input
            and self.config.laplacian_operator == "stencil"
            else None,
        )
        self.filter = (
            DualFieldUNet3D(self.config)
            if self.config.filter_input_version == 3
            else UNet3D(
                self.config.features,
                self.config.checkpoint_blocks,
                in_channels=2
                if self.config.filter_input_version == 1 and self.config.laplacian_input
                else 1,
                output_skip_channels=int(self.config.filter_output_skip),
                activation=self.config.activation,
                leaky_relu_slope=self.config.leaky_relu_slope,
                downsampling=self.config.downsampling,
                bias=self.config.bias,
                normalization=self.config.normalization_layer,
            )
        )
        if not _defer_refinement:
            self._build_refinement()

    def _build_refinement(self) -> None:
        # Register under filter so phase/chi freezing and gradient diagnostics
        # include the refinement. Build last to preserve both U-Nets' RNG draws.
        self.filter.refinement = (
            PhaseRefinementBlock(
                self.config.filter_refinement_features,
                self.config.activation,
                self.config.leaky_relu_slope,
                self.config.bias,
                self.config.normalization_layer,
            )
            if self.config.filter_refinement_features
            else nn.Identity()
        )

    def predict_fields(self, total_field: Tensor, mask: Tensor) -> FilteredFields:
        """Predict both fields in ppm from the masked, RMS-normalized total field."""
        if self.config.filter_input_version != 3:
            return FilteredFields(self.predict_local_field(total_field, mask))
        validate_mask(mask, total_field)
        field = total_field.float() * mask
        scale = masked_rms(field, mask, self.config.scale_floor)
        normalized = field / scale
        local, harmonic = (
            self.filter(normalized, normalized)
            if self.config.filter_output_skip
            else self.filter(normalized)
        )
        local = local.float()
        if self.config.filter_refinement_features:
            local = self.filter._block(self.filter.refinement, local).float()
        local = local * scale
        if not self.config.full_grid_field:
            local = local * mask
        return FilteredFields(local, harmonic.float() * scale * mask)

    def _input_laplacian(self, normalized: Tensor, mask: Tensor) -> Tensor:
        """Apply the recorded input operator and boundary policy."""
        padded_extension = (
            self.config.laplacian_extension_iters > 0
            and self.config.laplacian_extension_mode == "quadratic"
        )
        if padded_extension:
            operator_input = quadratic_exterior_extension(normalized, mask)
        elif self.config.laplacian_extension_iters:
            operator_input = smooth_harmonic_extension(
                normalized, mask, self.config.laplacian_extension_iters
            )
        else:
            operator_input = normalized
        if self.config.laplacian_operator == "frequency":
            laplacian = frequency_laplacian(operator_input)
        elif self.config.laplacian_fp32:
            # Autocast rounds the fractional stencil to a nonzero sum in
            # BF16, leaking constant backgrounds into this derivative.
            with torch.autocast(normalized.device.type, enabled=False):
                laplacian = F.conv3d(
                    operator_input.float(), self.laplacian_kernel.float(), padding=1
                )
        else:
            # Preserve the exact preprocessing of historical checkpoints.
            laplacian = F.conv3d(operator_input, self.laplacian_kernel, padding=1)
        if self.config.laplacian_extension_iters:
            if padded_extension:
                laplacian = laplacian[..., 1:-1, 1:-1, 1:-1]
            laplacian = laplacian * mask
        return laplacian

    def _filter_inputs(
        self, field: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.field_normalization == "laplacian_rms":
            with torch.autocast(field.device.type, enabled=False):
                # Remove the arbitrary field reference before FFTs and bypass.
                mean = field.sum(dim=(-3, -2, -1), keepdim=True) / mask.float().sum(
                    dim=(-3, -2, -1), keepdim=True
                ).clamp_min(1)
                centered = (field - mean) * mask
                laplacian = self._input_laplacian(centered, mask) * mask
                valid = (F.avg_pool3d(mask.float(), 3, 1, 1) > 1 - 1e-6).float()
                # Thin ROIs still use derivative units, never background amplitude.
                scale_mask = torch.where(
                    valid.sum(dim=(-3, -2, -1), keepdim=True) > 0, valid, mask.float()
                )
                scale = masked_rms(laplacian, scale_mask, self.config.scale_floor)
                return laplacian / scale, centered / scale, scale
        # Preserve the arithmetic and bypass of historical checkpoints exactly.
        scale = masked_rms(field, mask, self.config.scale_floor)
        normalized = field / scale
        filter_input = normalized
        if self.config.laplacian_input:
            laplacian = self._input_laplacian(normalized, mask)
            filter_input = (
                torch.cat((normalized, laplacian), dim=1)
                if self.config.filter_input_version == 1
                else laplacian
            )
            if self.config.conditional_laplacian_threshold:
                with torch.autocast(field.device.type, enabled=False):
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
                    input_mask = (
                        mask if self.config.laplacian_extension_iters else valid
                    )
                    interior_input = (
                        input_mask * laplacian.float() * scale / interior_scale
                    )
                    filter_input = torch.where(
                        selected, interior_input, laplacian.float()
                    )
                    scale = torch.where(selected, interior_scale, scale)
        return filter_input, field / scale, scale

    def predict_local_field(self, total_field: Tensor, mask: Tensor) -> Tensor:
        """Predict the clean field with the saved input scaling and output policy."""
        if self.config.filter_input_version == 3:
            return self.predict_fields(total_field, mask).local_field
        validate_mask(mask, total_field)
        field = total_field.float() * mask
        filter_input, bypass, scale = self._filter_inputs(field, mask)
        raw = (
            self.filter(filter_input, bypass)
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

    @torch.inference_mode()
    def predict_fields_inference(
        self, total_field: Tensor, mask: Tensor
    ) -> FilteredFields:
        if self.training:
            raise ValueError("Call model.eval() before inference")
        if self.inference_config.window_size:
            return sliding_window_fields(self, total_field, mask, self.inference_config)
        return self.predict_fields(total_field, mask)


class DNQSM(LocalFieldEstimator):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config, _defer_refinement=True)
        self.null_network = susceptibility_unet(self.config)
        self._build_refinement()

    def initialize_from(self, source: "DNQSM") -> None:
        """Warm-start compatible weights, adding an identity refinement if needed."""
        if (source.config.downsampling, source.config.bias) != (
            self.config.downsampling,
            self.config.bias,
        ):
            raise ValueError(
                "Initialization checkpoint has different downsampling or layer biases; start a fresh run"
            )
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
        if source.config.normalization_layer != self.config.normalization_layer:
            if self.config.normalization_layer != "residual_instance":
                raise ValueError(
                    "Initialization cannot discard learned normalization residuals"
                )
            # The added zero gains preserve the transferred feature transform.
            state.update(
                {
                    name: value
                    for name, value in self.state_dict().items()
                    if name.endswith("residual_gain")
                }
            )
        # A new run may explicitly reuse learned weights with a new input
        # operator; its deterministic stencil buffer belongs to the target.
        if source.config.laplacian_operator != self.config.laplacian_operator:
            if self.laplacian_kernel is None:
                state.pop("laplacian_kernel", None)
            else:
                state["laplacian_kernel"] = self.laplacian_kernel
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
            keys = (
                [
                    "filter.local_decoder.final.weight",
                    "filter.harmonic_decoder.final.weight",
                ]
                if self.config.filter_input_version == 3
                else ["filter.final.weight"]
            )
            for key in keys:
                weight = state[key]
                state[key] = torch.cat((weight, torch.zeros_like(weight[:, :1])), dim=1)
        elif source.config.filter_output_skip and not self.config.filter_output_skip:
            raise ValueError(
                "Initialization cannot discard a trained filter output skip"
            )
        self.load_state_dict(state)

    @torch.inference_mode()
    def infer(
        self,
        total_field: Tensor,
        mask: Tensor,
        dipole: Tensor,
        *,
        window_size: int | None = None,
        overlap: float | None = None,
        sigma_scale: float | None = None,
        window_batch_size: int | None = None,
    ) -> Reconstruction:
        """Reconstruct using the saved Gaussian-window policy or explicit overrides."""
        if self.training:
            raise ValueError("Call model.eval() before inference")
        overrides = dict(
            window_size=window_size,
            overlap=overlap,
            sigma_scale=sigma_scale,
            window_batch_size=window_batch_size,
        )
        config = replace(
            self.inference_config,
            **{k: v for k, v in overrides.items() if v is not None},
        )
        if not config.window_size:
            return self(total_field, mask, dipole)
        return sliding_window_reconstruction(self, total_field, mask, dipole, config)

    def forward(
        self, total_field: Tensor, mask: Tensor, dipole: Tensor
    ) -> Reconstruction:
        fields = self.predict_fields(total_field, mask)
        local = fields.local_field
        inverse, projection = cutoff_kernels(
            dipole, self.config.threshold, self.config.cutoff_mode
        )
        initial = apply_multiplier(local, inverse)
        chi_scale = masked_rms(initial, mask, self.config.scale_floor)
        residual = self.null_network(initial / chi_scale).float() * chi_scale
        correction = apply_multiplier(residual, projection)
        # Preserve the spectral correction weighting without a spatial remask.
        return Reconstruction(
            local, initial, correction, initial + correction, fields.harmonic_field
        )
