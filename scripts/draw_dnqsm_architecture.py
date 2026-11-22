"""Draw the architecture of the evaluated soft-cutoff DN-QSM checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dnqsm-article-mpl")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn

from create_cosmos_composite_figures import latex_style, write_json
from dnqsm.model import ResidualInstanceNorm3d, UNet3D
from dnqsm.runtime import load_trained_model


INK = "#263343"
MUTED = "#637184"
BLUE = "#356C9B"
PURPLE = "#756294"
AMBER = "#A97532"
GREEN = "#397866"
BLUE_BG = "#EFF5FB"
PURPLE_BG = "#F5F0F9"
AMBER_BG = "#FCF5EA"
GREEN_BG = "#EFF8F3"


def checkpoint_spec(path: Path) -> dict:
    model = load_trained_model(path, "cpu")
    config = model.config
    assert config.features == (32, 64, 128, 256)
    assert config.filter_input_version == 2 and config.filter_output_skip
    assert config.field_normalization == "laplacian_rms"
    assert config.normalization_layer == "residual_instance"
    assert config.laplacian_operator == "frequency"
    assert config.laplacian_extension_mode == "quadratic"
    assert config.cutoff_mode == "soft" and config.threshold == .1
    assert not config.residual_filter and config.full_grid_field
    assert config.filter_refinement_features == 0
    assert isinstance(model.filter, UNet3D) and isinstance(model.null_network, UNet3D)
    assert not ({id(p) for p in model.filter.parameters()} & {id(p) for p in model.null_network.parameters()})
    assert all(layer.bias is None for layer in model.modules() if isinstance(layer, nn.Conv3d))
    assert all(layer.bias is None for layer in model.modules() if isinstance(layer, ResidualInstanceNorm3d))
    for network in (model.filter, model.null_network):
        assert len(network.downs) == len(network.downsamples) == len(network.up_blocks) == 4
        assert all(d.stride == d.kernel_size == (2, 2, 2) for d in network.downsamples)
        assert network.bottleneck[0].out_channels == 512
        assert network.final.kernel_size == (1, 1, 1) and network.final.out_channels == 1
    assert model.filter.final.in_channels == 33
    assert model.null_network.final.in_channels == 32
    with path.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(
        checkpoint=str(path.resolve()), checkpoint_sha256=checkpoint_hash,
        model_config=asdict(config), parameter_count=sum(p.numel() for p in model.parameters()),
        field_unet_parameters=sum(p.numel() for p in model.filter.parameters()),
        correction_unet_parameters=sum(p.numel() for p in model.null_network.parameters()),
        residual_instance_norm_layers=sum(isinstance(x, ResidualInstanceNorm3d) for x in model.modules()),
        encoder_channels=[32, 64, 128, 256], bottleneck_channels=512,
        decoder_channels=[256, 128, 64, 32], independent_unet_weights=True,
        field_head_input_channels=33, correction_head_input_channels=32,
        architecture_checks_passed=True,
    )


class Diagram:
    def __init__(self):
        latex_style()
        self.fig = plt.figure(figsize=(12.8, 8.6))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 16)
        self.ax.set_ylim(0, 10.75)
        self.ax.axis("off")
        self.fitted = []

    def text(self, x, y, value, size=14, *, color=INK, align="center", max_width=None, **kwargs):
        artist = self.ax.text(x, y, value, fontsize=size, color=color,
                              ha=align, va="center", **kwargs)
        if max_width is not None:
            self.fitted.append((artist, max_width))
        return artist

    def card(self, x, y, w, h, edge, face, *, radius=.10):
        patch = patches.FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=1.2, edgecolor=edge, facecolor=face, zorder=1)
        self.ax.add_patch(patch)
        return patch

    def arrow(self, points, *, color=INK, lw=1.35, dashed=False, zorder=2):
        # Orthogonal polylines remain explicit; only the last segment has a tip.
        if len(points) > 2:
            self.ax.plot(*np.array(points[:-1]).T, color=color, lw=lw,
                          ls=(0, (3, 2.5)) if dashed else "-", zorder=zorder)
        self.ax.add_patch(patches.FancyArrowPatch(points[-2], points[-1],
            arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color,
            linestyle=(0, (3, 2.5)) if dashed else "-", shrinkA=0, shrinkB=0, zorder=zorder))

    def dot(self, x, y):
        self.ax.add_patch(patches.Circle((x, y), .035, color=INK, zorder=5))

    def mini_unet(self, x, y, width, color):
        height_scale = .75
        positions = [(x+width*t, y+u*height_scale) for t, u in
                     ((0, .19), (.13, .04), (.26, -.11), (.39, -.26), (.5, -.34),
                      (.61, -.26), (.74, -.11), (.87, .04), (1, .19))]
        for i in range(4):
            self.ax.plot([positions[i][0], positions[8-i][0]],
                          [positions[i][1], positions[8-i][1]], color=color, alpha=.30,
                          lw=.65, zorder=2)
        self.ax.plot(*np.array(positions).T, color=color, lw=1.0, zorder=2)
        for i, (px, py) in enumerate(positions):
            h = (.21-.014*min(i, 8-i))*height_scale
            self.ax.add_patch(patches.Rectangle((px-.055, py-h/2), .11, h,
                facecolor=color, edgecolor="white", linewidth=.4, zorder=3))

    def cube(self, x, y, width, height, color, channels):
        depth = .075
        left, bottom = x-width/2, y-height/2
        self.ax.add_patch(patches.Polygon([
            (left, bottom+height), (left+depth, bottom+height+depth),
            (left+width+depth, bottom+height+depth), (left+width, bottom+height)],
            facecolor=color, edgecolor="white", alpha=.5, linewidth=.7, zorder=3))
        self.ax.add_patch(patches.Polygon([
            (left+width, bottom), (left+width+depth, bottom+depth),
            (left+width+depth, bottom+height+depth), (left+width, bottom+height)],
            facecolor=color, edgecolor="white", alpha=.7, linewidth=.7, zorder=3))
        self.ax.add_patch(patches.Rectangle((left, bottom), width, height,
            facecolor=color, edgecolor="white", linewidth=.7, zorder=4))
        self.text(x, y, str(channels), 13, color="white", zorder=5)

    def fit(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for artist, width in self.fitted:
            available = self.ax.transData.transform((width, 0))[0]-self.ax.transData.transform((0, 0))[0]
            actual = artist.get_window_extent(renderer).width
            if actual > available:
                artist.set_fontsize(artist.get_fontsize()*available/actual*.98)
        self.fig.canvas.draw()
        for artist, width in self.fitted:
            available = self.ax.transData.transform((width, 0))[0]-self.ax.transData.transform((0, 0))[0]
            assert artist.get_window_extent(self.fig.canvas.get_renderer()).width <= available+1


def draw_pipeline(d: Diagram):
    d.text(.5, 10.42, r"\textbf{DN-QSM architecture}", 22, align="left")
    for x, edge, face in ((10.0, BLUE, BLUE_BG), (10.23, PURPLE, PURPLE_BG)):
        d.card(x, 10.30, .18, .18, edge, face, radius=.025)
    d.text(10.55, 10.40, "Learned", 12.5, align="left")
    d.card(12.1, 10.30, .22, .18, AMBER, AMBER_BG, radius=.025)
    d.text(12.45, 10.40, "Fixed physics", 12.5, align="left")
    d.card(14.38, 10.30, .22, .18, GREEN, GREEN_BG, radius=.025)
    d.text(14.73, 10.40, "Output", 12.5, align="left")

    d.text(.5, 9.92, r"\textbf{(a) Local-field estimation}", 17, align="left")
    d.card(.5, 8.02, 1.6, 1.55, MUTED, "#F6F7F9")
    d.text(1.3, 9.23, "Measured inputs", 14, max_width=1.4)
    d.text(1.3, 8.82, r"$b_{\mathrm{tot}},\ M$", 21)
    d.text(1.3, 8.36, "Total field / mask", 12, max_width=1.4)

    d.card(2.65, 8.02, 3.5, 1.55, AMBER, AMBER_BG)
    d.text(4.4, 9.23, "Boundary-aware input", 15, max_width=3.2)
    d.text(4.4, 8.88, r"Center $\to$ quadratic extension", 12.5, max_width=3.15)
    d.text(4.4, 8.55, "FFT Laplacian + RMS scaling", 12.5, max_width=3.15)
    d.text(4.4, 8.22, r"$\ell=M\mathcal{L}E(c)/s_b$", 14.5, max_width=3.15)

    d.card(7.0, 8.02, 3.3, 1.55, BLUE, BLUE_BG)
    d.text(8.65, 9.23, r"Field U-Net $G_\theta$", 16, max_width=3.0)
    d.mini_unet(7.55, 8.75, 2.20, BLUE)
    d.text(8.65, 8.22, r"$G_\theta(\ell,\widetilde b)$", 15)

    d.card(11.15, 8.02, 4.1, 1.55, GREEN, GREEN_BG)
    d.text(13.20, 9.23, "Estimated local field", 16, max_width=3.8)
    d.text(13.20, 8.79, r"$\widehat b_{\mathrm{loc}}=s_b\,G_\theta(\ell,\widetilde b)$", 18, max_width=3.8)
    d.text(13.20, 8.28, "Restore physical scale; retain full grid", 12.3, max_width=3.8)
    d.arrow([(2.1, 8.8), (2.65, 8.8)])
    d.arrow([(6.15, 8.8), (7.0, 8.8)])
    d.arrow([(10.3, 8.8), (11.15, 8.8)])
    d.arrow([(5.45, 9.57), (5.45, 9.76), (9.57, 9.76), (9.57, 9.57)],
            color=BLUE, dashed=True, lw=1.2)
    d.text(7.75, 9.88, r"$\widetilde b=c/s_b$: concatenate at output head", 12.5, max_width=5.15)
    # Forward the local field to the second stage without crossing labels.
    d.arrow([(13.2, 8.02), (13.2, 7.56), (.25, 7.56), (.25, 5.95), (.6, 5.95)], lw=1.25)
    d.text(8.1, 7.70, r"$\widehat b_{\mathrm{loc}}$", 14,
           bbox=dict(facecolor="white", edgecolor="none", pad=1))

    d.text(.5, 7.12, r"\textbf{(b) Susceptibility reconstruction}", 17, align="left")
    d.card(.6, 5.22, 2.4, 1.45, AMBER, AMBER_BG)
    d.text(1.8, 6.35, "Soft-TKD", 16)
    d.text(1.8, 5.91, r"$K_\tau=\dfrac{D}{D^2+\tau^2}$", 17, max_width=2.1)
    d.text(1.8, 5.43, r"FFT $\to K_\tau\to$ inverse FFT", 11.8, max_width=2.12)

    d.card(4.05, 5.22, 3.3, 1.45, PURPLE, PURPLE_BG)
    d.text(5.7, 6.35, r"Correction U-Net $R_\phi$", 15.5, max_width=3.02)
    d.mini_unet(4.6, 5.96, 2.2, PURPLE)
    d.text(5.7, 5.43, r"$r=s_\chi R_\phi(\chi_0/s_\chi)$", 14, max_width=3.0)

    d.card(8.20, 5.22, 3.05, 1.45, AMBER, AMBER_BG)
    d.text(9.725, 6.35, "Soft near-null weighting", 14.4, max_width=2.8)
    d.text(9.725, 5.91, r"$P_\tau=\dfrac{\tau^2}{D^2+\tau^2}$", 17, max_width=2.8)
    d.text(9.725, 5.43, r"$\Delta\chi=\mathcal{F}^{-1}[P_\tau\mathcal{F}(r)]$", 13, max_width=2.8)

    d.ax.add_patch(patches.Circle((12, 5.95), .25, facecolor="white", edgecolor=GREEN, linewidth=1.5, zorder=4))
    d.text(12, 5.95, "+", 22, zorder=5, color=GREEN)
    d.card(13.05, 5.22, 2.45, 1.45, GREEN, GREEN_BG)
    d.text(14.275, 6.35, "Final DN-QSM", 16, max_width=2.2)
    d.text(14.275, 5.91, r"$\widehat\chi=\chi_0+\Delta\chi$", 17, max_width=2.2)
    d.text(14.275, 5.43, "Susceptibility map", 12.5, max_width=2.2)

    d.arrow([(3, 5.95), (4.05, 5.95)])
    d.dot(3.5, 5.95)
    d.text(3.5, 6.19, r"$\chi_0$", 15)
    d.arrow([(7.35, 5.95), (8.2, 5.95)])
    d.text(7.77, 6.18, r"$r$", 15)
    d.arrow([(11.25, 5.95), (11.75, 5.95)])
    d.text(11.49, 6.20, r"$\Delta\chi$", 14)
    d.arrow([(12.25, 5.95), (13.05, 5.95)])
    d.arrow([(3.5, 5.95), (3.5, 4.78), (12, 4.78), (12, 5.7)], color=GREEN, lw=1.4)
    d.text(7.7, 4.86, r"Soft-TKD identity path: $\chi_0$", 13.2,
           bbox=dict(facecolor="white", edgecolor="none", pad=1))
    d.text(.6, 4.32, r"$c=M(b_{\mathrm{tot}}-\mu_M)$; $s_b$: masked Laplacian RMS; $s_\chi=\mathrm{RMS}_M(\chi_0)$; $D$: dipole kernel; $\tau=0.1$.",
           12.8, align="left", max_width=14.8)


def draw_backbone(d: Diagram):
    d.ax.plot([.5, 15.5], [4.03, 4.03], color="#D9E1E9", linewidth=.8)
    d.text(.5, 3.75, r"\textbf{(c) 3D U-Net backbone} \quad Independent parameters for $G_\theta$ and $R_\phi$", 16, align="left", max_width=15)
    xs = [.95, 2.30, 3.65, 5.00, 6.35, 7.70, 9.05, 10.40, 11.75]
    ys = [2.78, 2.28, 1.78, 1.28, .83, 1.28, 1.78, 2.28, 2.78]
    heights = [.95, .85, .75, .65, .55, .65, .75, .85, .95]
    channels = [32, 64, 128, 256, 512, 256, 128, 64, 32]
    # Encoder-to-decoder skip features meet the decoder stream by concatenation.
    for level in range(4):
        target = 8-level
        destination = (xs[target]-.66, ys[target])
        d.arrow([(xs[level]+.36, ys[level]), destination], color="#9CAFC0", lw=.9, zorder=1)
        d.ax.add_patch(patches.Circle(destination, .12, facecolor="white", edgecolor="#91A3B3", linewidth=.9, zorder=5))
        d.text(*destination, r"$C$", 9.3, color=MUTED, zorder=6)
    for index in range(8):
        start = (xs[index]+.4, ys[index]-.15 if index<4 else ys[index]+.15)
        if index<4:
            target = (xs[index+1]-.40, ys[index+1]+.16)
        else:
            target = (xs[index+1]-.66, ys[index+1]-.12)
        d.arrow([start, target], color=BLUE if index<4 else GREEN, lw=1.25)
        if index>=4:
            d.arrow([(xs[index+1]-.54, ys[index+1]), (xs[index+1]-.36, ys[index+1])], color=GREEN, lw=.9)
    for index, (x, y, height, count) in enumerate(zip(xs, ys, heights, channels, strict=True)):
        color = BLUE if index<4 else PURPLE if index==4 else "#438881"
        d.cube(x, y, .72, height, color, count)
        label = rf"$E_{{{index}}}$" if index<4 else "Bottleneck" if index==4 else rf"$D_{{{8-index}}}$"
        d.text(x, y-height/2-.22, label, 11.5, max_width=1.25)
    d.arrow([(12.15, 2.78), (12.62, 2.78)], lw=1.1)
    d.cube(12.86, 2.78, .44, .68, BLUE, 1)
    d.text(12.86, 2.14, "Head", 11)

    d.card(13.45, .47, 2.15, 2.76, "#C6D2DD", "#F7F9FC", radius=.08)
    d.text(14.525, 2.99, "Building blocks", 13.5, max_width=1.95)
    for y, label in (
        (2.64, r"$[3^3\mathrm{Conv},\mathrm{RIN},\mathrm{LReLU}]\times2$"),
        (2.29, r"Down: $2^3$ Conv, stride 2"),
        (1.94, r"Up: trilinear $+\,3^3$ Conv"),
        (1.59, r"Linear $1^3$ output head"),
        (1.24, r"LeakyReLU slope $0.01$"),
        (.89, "Bias-free layers"),
        (.62, "RIN: IN + amplitude path"),
    ):
        d.text(14.525, y, label, 11.6, max_width=1.98)
    d.text(.5, .15, r"$C$: channel concatenation. The field U-Net additionally concatenates $\widetilde b$ before its output head.",
           12.0, align="left", max_width=12.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/evaluations/graz_softcut01_best18_fp32_20261027_130316/model_snapshot.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/dnqsm_architecture_best18_20261027"))
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    spec = checkpoint_spec(args.checkpoint)
    d = Diagram()
    draw_pipeline(d)
    draw_backbone(d)
    d.fit()
    stem = args.output/"dnqsm_architecture"
    for extension in ("png", "pdf", "svg"):
        d.fig.savefig(stem.with_suffix(f".{extension}"), dpi=args.dpi, facecolor="white")
    plt.close(d.fig)
    with Image.open(stem.with_suffix(".png")) as image:
        preview = image.convert("RGB")
        preview.thumbnail((2200, 1800), Image.Resampling.LANCZOS)
        preview.save(args.output/"dnqsm_architecture_preview.png")
    write_json(args.output/"architecture_spec.json", spec)
    write_json(args.output/"figure_settings.json", dict(
        dpi=args.dpi, formats=["PNG", "PDF", "SVG"], typography="Actual LaTeX, Latin Modern",
        depicts="Inference architecture of the COSMOS-selected epoch 18 checkpoint",
        positive_discrete_laplacian="Lambda(k)=4*sum_i sin(pi*k_i/N_i)^2",
        field_centering="c=M*(b_total-masked_mean(b_total))",
        field_scale="Masked RMS of boundary-extended Laplacian over valid interior voxels; epsilon floor 1e-6",
        field_skip="c/s_b concatenated with last decoder features before the learned linear head",
        soft_inverse="K_tau=D/(D^2+tau^2)", correction_weight="P_tau=tau^2/(D^2+tau^2)",
        residual_scale="s_chi=masked_rms(chi_0), floor=1e-6",
        final_equation="chi_final=chi_0+IFFT[P_tau*FFT(s_chi*R_phi(chi_0/s_chi))]",
        mask_policy="Mask input and Laplacian; retain full-grid predicted field and susceptibility during inference",
        rin="InstanceNorm on all channels, with a learned interpolation toward raw affine-scaled features on half the channels",
        correction_is_strict_null_projection=False,
        training_losses_shown=False,
    ))
    shutil.copy2(__file__, args.output/Path(__file__).name)
    (args.output/"caption.md").write_text(
        "**DN-QSM architecture.** (a) The measured total field is masked and centered. A quadratic boundary extension precedes the positive FFT-based discrete Laplacian. "
        "The Laplacian is divided by its masked interior RMS before the field-estimation U-Net. The centered field, normalized by the same RMS, is concatenated "
        "with the final decoder features before the linear output head. Restoring the scale yields the estimated full-grid local field. "
        "(b) A soft-TKD inverse, K_tau=D/(D²+tau²), produces the initial susceptibility chi_0. An independent U-Net processes its RMS-normalized form, "
        "and its rescaled output is spectrally weighted by P_tau=tau²/(D²+tau²). The resulting correction is added to chi_0. Both filters use tau=0.1. "
        "The weight P_tau emphasizes the dipole near-null region and is not an orthogonal null-space projection. "
        "(c) Both U-Nets use four encoder stages (32, 64, 128, 256 channels), a 512-channel bottleneck, and mirrored decoders with concatenated skip features. "
        "Each double-convolution block contains two 3×3×3 convolutions, residual instance normalization and LeakyReLU (slope 0.01). "
        "Downsampling uses 2×2×2 stride-2 convolutions; upsampling uses trilinear interpolation to the matching skip resolution, a 3×3×3 convolution and LeakyReLU. "
        "Output heads are linear 1×1×1 convolutions. Convolution biases and normalization offsets are omitted; normalization scales and residual gains remain trainable. "
        "Residual instance normalization adds a learned amplitude path on half of the channels. Networks have independent parameters; only their topology is shared.\n"
    )
    print(json.dumps(dict(output=str(args.output.resolve()), parameter_count=spec["parameter_count"], architecture_checks_passed=True)), flush=True)


if __name__ == "__main__":
    main()
