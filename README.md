# DN-QSM

DN-QSM jointly removes background fields and noise, performs a soft-cutoff inverse,
and corrects susceptibility artifacts with a deep null space network. It follows
the maintained dataloader and COSMOS/sim2 preprocessing in the sibling `wh_net`
repository, without importing that package at runtime.

The best trained **full-volume reference** checkpoint is retained at:
`artifacts/models/current.pt` links to `model_best.pt` from
`dnqsm_original105_joint800_20261022_202817` (epoch 180, update 1,800).
`load_trained_model()` and `dnqsm-evaluate` select it by default.
New training defaults to
[configs/full_volume_strided.json](configs/full_volume_strided.json): full volumes,
learned stride-2 downsampling, and no layer biases. It uses only the original
105 samples (95 training / 10 held-out subjects), with patch extraction and
sliding-window inference disabled. It initializes fresh weights because the
new downsampling layers and removed offsets change the architecture. No
training has been launched for this revision.
The exact trained baseline remains available through
[configs/full_volume_best.json](configs/full_volume_best.json), with max pooling
and its original biases. Loading an existing checkpoint preserves that
checkpoint's architecture; it does not convert its weights automatically.

| Checkpoint | Held-out χ NRMSE | COSMOS χ NRMSE | sim2 χ NRMSE |
|---|---:|---:|---:|
| Restored full-volume best | 39.15% | 47.23% | 61.37% |
| Completed 64³ patch run | 109.48% | 99.31% | 99.16% |

The restored scores were reproduced on the GPU without weight updates. The
selection uses held-out susceptibility NRMSE; scalar losses from different
objectives are not used to rank the runs. Verification details are saved in
`artifacts/models/current.json`. Existing experimental runs and explicitly
selected dual-field/patch profiles remain available.

## Setup

Use Python 3.12 or later:

```sh
uv sync
uv run dnqsm-train --help
uv run dnqsm-evaluate --help
```

The dependencies include PyTorch, NumPy, SciPy, Matplotlib, NiBabel, and Comet.
Install a CUDA-enabled PyTorch build appropriate for your training host. CPU is
supported when explicitly selected with `--device cpu`; CUDA requests never
silently fall back to CPU.

For verification in the current workspace, `.venv` reuses installed dependencies
from `../wh_net/.venv` through an ignored `.pth` file. The package itself is
installed locally in editable mode. A fresh `uv sync` installs independent
dependencies and does not require the reference environment.

## Architecture and loss

Volumes use `[N,1,D,H,W]`. Inputs, local fields, and susceptibility use physical
ppm units and the reference's unshifted continuous dipole kernel convention.
The phase filter and null-space U-Net each use four levels with widths
`[32,64,128,256]` and a 512-channel bottleneck. Each network has one
decoder. Each encoder stage now downsamples with a learned **2×2×2 convolution,
stride 2×2×2**, retaining the stage's channel count. There are four such layers
per network, replacing max pooling. Each network contains 27 convolutions;
together they contain **53,157,413 parameters**, including 36 normalization
residual gains (53,157,377 parameters with historical InstanceNorm).
All convolutions, including downsampling, decoder, output-head, and optional
refinement layers, use `bias=false`. Normalization retains its learned scale
(gamma) and has no learned offset (beta). The active profile uses the FP32
frequency-domain difference operator below as input (version 2), always-on
physical derivative scaling, centered raw-field output concatenation, full-grid local-field output,
and soft-cutoff threshold 0.1. It has no harmonic decoder or harmonic/decomposition
losses. Training and inference operate on full volumes. Batch size is **10**.
The previous max-pooling/bias checkpoint has 410,626,887 parameters and 33
convolutions per network. See the [historical capacity measurements](docs/depth_capacity.md).

The input operator matches `ifftn(EE2 .* fftn(u))` from the MATLAB example:

```text
u = (total_field - masked_mean(total_field, mask)) * mask
u_ext = quadratic_exterior_extension(u, mask)  # includes a one-voxel halo
EE2(k) = Σj |1 - exp(2πi kj/Nj)|² = 4 Σj sin²(π kj/Nj)
L = crop_halo(real(IFFT(EE2 * FFT(u_ext)))) * mask
scale = max(masked_RMS(L, valid_interior), scale_floor)
encoder_input = L / scale
predicted_local = scale * filter(encoder_input, u / scale)
```

This is the positive-semidefinite **negative discrete Laplacian** with periodic
boundaries and unit voxel steps: `6*u - Σj (roll(u,+1,j) + roll(u,-1,j))`.
It preserves the MATLAB sign and uses no FFT shift. Real FFTs operate only
over the three spatial axes on the extended grid,
including odd and rectangular volumes. The operator and multiplier stay FP32
under convolution autocast, and gradients propagate through the FFTs.
Before differentiation, missing neighbors are extrapolated from three
consecutive interior voxels: `u_ghost = 3*u1 - 3*u2 + u3`. This continuation
preserves quadratic curvature as well as constant and linear fields, whereas
the previous exterior harmonic relaxation could change the normal derivative
at the ROI edge. Polynomial extrapolation is also the principle behind
[polynomial edge handling in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html).
Axial rays take priority; diagonal rays fill unsupported directions. Where
three interior samples are unavailable, the method falls back to two-sample
linear extrapolation, then a nearest-sample value. It averages supported rays
of the same degree without mixing lower-order estimates into a quadratic.
The added one-voxel halo also covers all six volume faces, preventing FFT
wraparound from connecting opposite brain edges. Crop only the halo after
differentiation: every original ROI voxel is retained. The field inside the ROI
is copied exactly, with no taper or smoothing. The operation remains FP32 and
differentiable. The finite difference interior is unchanged up to FFT roundoff.
Boundary derivatives are estimates from interior samples; unobserved exterior
structure cannot be recovered exactly for arbitrary fields.
Scale estimation uses the conservative 3×3×3-valid ROI interior, while the full
masked derivative, including every border voxel, reaches the encoder. If that
interior is empty, scale is measured over the original ROI; it never falls back to
total-field amplitude in this mode. Centering the raw-field bypass prevents the
arbitrary field reference from leaking into the prediction.

The active profile records `field_normalization="laplacian_rms"` and
`normalization_layer="residual_instance"`. Physical scaling has no threshold
switch and remains FP32 under convolution autocast. A constant background does
not change its physical derivative scale; positive input gain is restored at the
output. This replaces the previous dependence on total-field RMS, which caused
large amplitude errors across weak- and strong-background inputs.

`ResidualInstanceNorm3d` adds an amplitude path on half the channels:
`z = InstanceNorm(x)` and `y_right = z_right + tanh(g) * (affine(x_right) - z_right)`;
the other half remains `z_left`. Each layer starts with `g=0`, exactly preserving
transferred InstanceNorm behavior, and its new gain receives gradients immediately.
This is inspired by [half instance normalization for image restoration](https://arxiv.org/abs/2105.06086),
but uses a learned residual rather than a fixed unnormalized half. It does not use
batch statistics or running averages. All additive offsets remain absent in the
active bias-free model. The 36 gains need adaptation; inference with zero gains
does not demonstrate an improvement from learning those paths.

Generic Python constructors and historical profiles/checkpoints retain
`field_normalization="legacy"` and `normalization_layer="instance"` when these
fields are absent. The active training profile explicitly enables both new modes.
Exact resume rejects changed modes. An explicit new run with `--init-checkpoint`
can copy compatible convolutions and affine scales into zero-initialized residual
normalization; initialization cannot discard learned normalization residuals.
The raw-input experimental dual-field architecture continues to use total-field
scaling; `laplacian_rms` requires FP32 version-2 Laplacian input.

`scripts/audit_physical_normalization.py --checkpoint <path> --device cuda:0`
compares saved preprocessing with the new policy, tests constant offsets and
positive amplitude scaling, and exports a candidate without changing default
trained-model selection or performing learning updates. With the four-level
epoch-101 checkpoint, COSMOS changes from 50.56% to 46.26% chi NRMSE and sim2 from
65.04% to 65.33%. Thus this restores sim2's prior baseline rather than establishing
a new best sim2 result. Historical checkpoints are loaded with their saved policy.

`laplacian_extension_mode="quadratic"` is saved in the active profile and new
model metadata. The existing `laplacian_extension_iters` switch still disables
extension at zero. Positive values enable quadratic continuation, which uses a
fixed three-sample fit without Jacobi iterations. For comparisons,
`--laplacian-extension-mode harmonic --laplacian-extension-iters 6` selects the
previous six-step exterior relaxation. Historical checkpoints and JSON profiles
without a mode retain `"harmonic"`; those also missing the iteration setting
retain zero (no extension). Exact resume preserves the saved mode. Use a new
adaptation run with `--init-checkpoint --laplacian-extension-mode quadratic`
and a positive iteration setting to change preprocessing with compatible learned
weights. Training remains stopped after this code change.

`laplacian_operator="frequency"` is saved in the active profile and new model
metadata. Checkpoints and historical JSON profiles without this field retain
`"stencil"`, the original zero-padded isotropic 3×3×3 convolution. Exact resume
preserves the saved operator. To change it while reusing compatible learned
weights, explicitly start a new run with `--init-checkpoint` and
`--laplacian-operator frequency`; the deterministic stencil buffer is not
transferred into the FFT model. This change does not restart training.

The experimental version-3 phase filter shares an encoder and splits into
independent local/harmonic decoders. Its five-level 64³ profile has
126,003,527 parameters. It remains an explicit opt-in through
`configs/dual_field_100epochs.json`; it is no longer the active model.
The large overfit profile below uses five wider levels. New models use
InstanceNorm/LeakyReLU double-convolution blocks and trilinear upsampling.
Every encoder, bottleneck, decoder and optional refinement activation is
LeakyReLU with negative slope **0.01**, including the activations after decoder
upsampling convolutions. Output heads remain linear to preserve signed fields.
`--leaky-relu-slope` overrides the slope for a new training or overfit run; it
is saved with the model and preserved by exact resume. Missing slope metadata
defaults to 0.01 for older checkpoints.
The activation is recorded in checkpoints; historical checkpoints without this
metadata retain CELU for inference and exact resume. To adapt existing weights
to LeakyReLU, start a new run with `--init-checkpoint` and the new defaults.
Inputs are normalized by
masked RMS, and outputs are rescaled to ppm.

1. The main model accepts `total_field`, the ROI `mask`, and the `dipole` kernel.
   For experimental version-3 models, `field = total_field * mask`, `scale = RMS(field, mask)`, and
   the encoder receives `field / scale` directly. There is no input Laplacian
   or conditional Laplacian scaling. The shared encoder and bottleneck feed
   two independent decoders, each with its own upsampling blocks, skip
   connections from the shared encoder, and signed linear output head. One
   predicts the clean local field; the other predicts the harmonic field.
   Both outputs are restored to ppm using `scale`. The harmonic estimate is
   masked; the local estimate is masked unless `--full-grid-field` is enabled.
   By default, each output head also receives the normalized masked total
   field through `filter_output_skip=true`.
   Version-3 checkpoints record `filter_input_version=3`, `laplacian_input=false`,
   and `conditional_laplacian_threshold=0`. `model.predict_fields(total, mask)`
   returns both fields; `model.predict_local_field(total, mask)` retains the
   local-only interface. The full reconstruction exposes `harmonic_field`.
   Historical version-1 and version-2 checkpoints retain their single decoder
   and original preprocessing. Their weights cannot initialize the new filter
   silently; switching to that architecture requires a fresh training run.
2. The inverse computes `chi0 = F^-1(B_tau * F(local))`. New models use
   `cutoff_mode="soft"` and `tau=0.1`, with the smooth multiplier
   `B_tau = D / (D^2 + tau^2)`. Its maximum inverse gain is `1/(2*tau)`
   (5 at the active threshold), and DC maps to zero.
3. A second U-Net predicts a susceptibility residual from `chi0`. The output is
   `chi = chi0 + F^-1(Q_tau * F(N(chi0)))`, with complementary smooth weight
   `Q_tau = 1 - B_tau*D = tau^2 / (D^2 + tau^2)`. The network input and
   output retain the existing RMS normalization and restoration.

This is the regularized residual structure `(I-B_tau A)N` from
[Schwab, Antholzer, and Haltmeier, Section 3.2](https://arxiv.org/html/1806.06137#S3.SS2),
with a regularized inverse. In soft mode the correction weight is not an
orthogonal projection: it is strongest near the dipole zeros and decreases
smoothly away from them. Corrections can affect all spatial frequencies.
`--cutoff-mode soft --threshold 0.1` selects this policy for a new run.
Training, full-volume inference, sliding-window inference and the filtering-only
inverse-supervision loss use the same saved mode and threshold. The historical
`tkd` output/metric name is retained for the initial susceptibility estimate.

`cutoff_mode="hard"` retains the original inverse `1/D` for `abs(D)>tau`,
zero otherwise, and its complementary binary near-null projection. Checkpoints
and historical JSON profiles without `cutoff_mode` load as hard, preserving
their original thresholds and outputs. Exact resume preserves the saved mode;
switching an existing model to soft is an explicit warm start using
`--init-checkpoint` for a new run. Both U-Nets' learned weights are compatible.

The predicted local field is zero outside the ROI before inversion. This changes
the TKD frequency content compared with the historical full-grid architecture.
Susceptibility and projected correction remain unmasked on the FFT grid to
preserve the correction's Fourier support. Use masked exports for ROI analysis.

The ROI penalties are dimensionless and normalized independently per sample.
`s_chi` and `s_b` are detached ground-truth susceptibility and clean-local-field
masked RMS, clamped by `--chi-loss-floor` and `--field-loss-floor` (both `1e-4`
ppm). Neither the total field nor a prediction sets these scales:

```text
loss = lambda_local   * masked_MAE(local, local_gt)  / s_b
     + lambda_local_gradient * ROI_edge_MAE(grad(local), grad(local_gt)) / s_b
     + lambda_local_tkd * masked_MAE(TKD(local), chi_gt) / s_chi
     + lambda_chi     * masked_MAE(chi, chi_gt)      / s_chi
     + lambda_physics * masked_MSE(A(chi), local_gt) / s_b**2
     + lambda_harmonic_laplacian * masked_mean(Laplacian(mask * harmonic)**2, eroded_mask) / s_b**2
     + lambda_field_decomposition * masked_MAE(local + harmonic, total_field) / s_b
```

Library defaults are `local=1.5`, `local_gradient=chi=physics=1.0`, `extension=0`,
and `local_tkd=0`. The active profile sets `extension=local_tkd=1` and disables
the harmonic and decomposition terms. The opt-in dual-branch profile uses
`harmonic_laplacian=100.0` and `field_decomposition=1.0`. The harmonic term applies the fixed
3x3x3 stencil **after masking** the estimate, in FP32 with zero padding. The
squared response is averaged inside `eroded_mask`, the original ROI eroded by
a Euclidean sphere of radius **3 voxels** (`d*d+h*h+w*w <= 9`). Positions outside
the volume count as background. The Laplacian uses the original masked field;
the eroded mask is applied only to the reduction, avoiding new mask-boundary
responses. An empty eroded ROI contributes zero for that sample. The stencil
uses unit voxel steps and its target is zero. The field-sum term uses the
original ROI and ties both
branches to the measured total field so the harmonic branch cannot minimize
its Laplacian penalty by predicting zero independently of the local estimate.
Both scales are detached from the ground-truth clean local field. Use
`--lambda-harmonic-laplacian` and `--lambda-field-decomposition` to tune or
independently disable these terms. Zero skips the corresponding computation.
They are absent for historical models without a harmonic branch.
The gradient term compares forward differences along all three spatial axes
using unit voxel steps. It averages over neighbor pairs with both voxels inside
the ROI, pooling valid edges across axes per sample. It uses no padding or
wraparound and excludes artificial mask-boundary edges. Samples with no valid
pairs contribute zero. Its normalization uses the detached clean-field RMS,
and computation stays FP32. `--lambda-local-gradient` controls its weight;
zero disables both the term and its computation.
`--lambda-local-tkd` penalizes the configured inversion of the **predicted**
local field against ground-truth susceptibility. Its inverse gain is bounded
by `1/(2*tau)` for soft cutoff and `1/tau` for hard cutoff, so a voxelwise field penalty charges
the same price for every spatial frequency while the inversion does not. This
term weights the filter's error by the conditioning of the operator that
consumes it. The full pipeline reuses the reconstruction's own `tkd` output
rather than repeating the transform pair; filtering-only batches compute it
with `local_tkd_loss`. Its normalization uses the detached susceptibility RMS.
Zero disables both the term and its computation, which is the default.

Phase-only updates use local, local-gradient, local-TKD, exterior, harmonic,
and field-sum terms; these are the only batches that update the filter under the alternating
schedule, so every filter objective has to be computed there. Chi-only updates
use chi and physics terms with coefficient 1.0 each. Validation reports the
combined objective shown above. Forward-field error remains a diagnostic metric.
Exterior supervision is disabled
and its computation is skipped. Historical runs can retain their existing
exterior objective. Reductions average voxels per sample, then
samples. `--no-normalize-loss` provides a raw-loss ablation. The physics term is masked MSE of
`A(chi)` against the **ground-truth clean local field**, divided by `s_b**2`.
The penalty comparing `A(chi)` with the estimated local field has been removed;
its weight and training diagnostics are no longer exposed.
Forward physics and losses stay FP32 even when eligible GPU convolutions use BF16.
The CNN and its output head use BF16 under CUDA autocast; the harmonic-field Laplacian
regularizer uses FP32. Historical Laplacian-input models retain their saved precision. Physics-weight warmup is disabled by default.
`--physics-warmup-epochs` optionally ramps its weight; validation uses
the final coefficients.

```python
from dnqsm.training import load_trained_model

model = load_trained_model("artifacts/runs/<run>/model_best.pt", device="cuda:0")
result = model.infer(total_field, mask, dipole)
# result.local_field, result.tkd, result.correction, result.chi
```

## Data and training

The matched joint/frozen-phase study is available through
`python -m dnqsm.staged_experiment --study <study-directory>`. It consumes a saved
phase-pretraining boundary and persisted subject split, initializes Deep Null
once per paired seed, and trains joint and frozen-phase branches on identical
draws. The frozen branch excludes the phase model from autograd and AdamW;
both branches use per-network clipping and matched update-based schedules.
The runner saves resumable checkpoints, per-view and subject metrics, a paired
bootstrap comparison, external results, and learning curves. It rejects a
concurrent runner for the same study. See the
[study design](docs/plans/joint_vs_frozen_phase_experiment.md) for its budgets,
initialization choices, and limitations.

By default, training reads `../wh_net/data/training/joint/` without modifying its
source files. Override `--data-dir`, or set `DNQSM_REFERENCE_ROOT` before launching
to change the reference root. Other datasets contain `image_XXXX.pt` and
`mask_XXXX.pt` pairs with shape `[1,D,H,W]`, plus `subjects.json` mapping image
filenames to source subject IDs. Use `--source-manifest` for a manifest elsewhere.

The experimental `configs/dual_field_100epochs.json` profile uses only the 105 original
QSM volumes (`image_0000.pt`–`image_0104.pt`, with matching `qsm:` identities),
split into 95 training and 10 held-out subjects. `original_samples_only=true`
requires all 105 originals and excludes stored augmentations and other cohorts.
Source files and caches remain unchanged.

With `patch_size=64`, each training view first augments susceptibility on the
full source grid, simulates its clean local field and background, adds noise,
and then extracts one random **64×64×64** patch. The identical crop applies to
the total field, local field, harmonic/background field, noise, susceptibility,
and mask. An independent deterministic per-view RNG ensures a nonempty ROI;
patches change each epoch and exact resume reproduces their positions. No
resizing or patch-wise noise/background renormalization follows simulation.
The old pre-simulation crop augmentation is disabled in this profile.
`--patch-size 0` preserves full-volume training for historical experiments.

TKD and null-space projection use a 64³ dipole kernel during patch training.
The physics loss retains the full simulation grid: insert predicted
susceptibility error at the patch location, forward-project on that grid, and
crop the field error. Ground truth outside the patch supplies fixed context
only for the loss, never as a network input. This avoids falsely penalizing
the true susceptibility for fields generated by tissue outside the patch.
The harmonic penalty still uses weight 100 and the radius-3 spherical erosion
of the patch mask, excluding artificial patch edges. Validation and external
evaluation retain full-volume inputs and their original kernels. CNN predictions
use the Gaussian sliding-window procedure below; TKD boundary conditions still
differ from the 64³ training grid.

### Gaussian sliding-window inference

Patch-trained runs use **64³ windows, 50% overlap (32-voxel stride), and Gaussian
sigma = 0.125 × 64 = 8 voxels**. Predictions are multiplied by the Gaussian,
summed over overlapping windows, and divided by the sum of weights in FP32.
The Gaussian has strictly positive weights. A half-window padding halo on
large axes provides centered predictions at outer boundaries; smaller axes
are padded to one window. Outputs are cropped back to the original geometry.

First, slide the phase filter and blend its local and harmonic outputs in ppm.
Compute TKD on the original full-volume dipole grid. Then slide the null U-Net,
blend its unprojected susceptibility residual, and apply the null-space
projection once on the full grid. This retains the final correction's Fourier
support and avoids introducing an independent FFT boundary at every window.
Window-specific RMS normalization matches the patch input convention.
Empty-mask phase windows contribute zero; null-network windows outside the ROI
use their full-window RMS and remain unmasked before projection.

`model.infer(...)` uses the policy saved with a checkpoint. Held-out
validation, external evaluation, and exports use the same saved inference
settings; training updates themselves still call the ordinary patch `forward`.
Historical checkpoints without a patch policy retain full-volume inference.
The original `model(...)` forward remains available for training and explicit
full-grid comparisons. To override inference settings:

```python
model = load_trained_model("artifacts/runs/<run>/model_best.pt", device="cuda:0")
result = model.infer(
    total_field, mask, dipole,
    window_size=64, overlap=0.5, sigma_scale=0.125, window_batch_size=1,
)
```

`dnqsm-evaluate` also accepts `--window-size 64 --window-overlap 0.5
--gaussian-sigma-scale 0.125 --window-batch-size 1`; `--window-size 0` selects a
full-volume comparison. Evaluation metadata records the actual policy.
Training profiles save `inference_overlap`, `inference_sigma_scale`, and
`inference_batch_size`; window size follows `patch_size`. Increasing window
batch size trades GPU memory for speed. Gaussian blending reduces artificial
window seams; it does not guarantee artifact-free reconstructions.

The loader groups all augmentations of a source subject before the 90/10 split.
Synthetic subjects (`synthetic:` IDs) enter training only. Each epoch draws 75%
original and 25% synthetic samples, when both cohorts exist. Training uses the
reference's physical-unit susceptibility RMS mixture, flips/permutations,
exterior-source backgrounds, and reliability-dependent noise with clean-field
RMS/noise RMS SNR of 40–100. The default background mixture contains 15% clean
(no background), 10% weak backgrounds (local/background RMS ratio 3–100),
25% strong backgrounds (ratio 0.001–0.03), and 50% typical backgrounds (0.03–3).
Over five epochs, `--background-curriculum-epochs` ramps the strong fraction
and lowers its minimum ratio. Unused strong probability initially becomes typical
background probability. Fractions and `--strong-min-ratio` are configurable;
`--background-profile ratio` and `legacy` provide ablations.

For historical Laplacian-input filters, the mixture's **amplitude** fractions
do not determine what the encoder sees: synthesized exterior sources are far
smoother than the field they are meant to stand in for. The new raw-field
encoder receives the amplitude regime directly. Audit
the realized regime with `scripts/audit_background_regime.py --config <json>`,
which reports the background/local amplitude ratio and the ratio of the two
fields' interior Laplacians, alongside both external cases for reference. The
Laplacian ratio diagnoses the input SNR of the historical derivative encoder.

At fixed sample
seeds, RNG streams are independent of batch construction; strongly amplified
backgrounds can expose small FP32 FFT rounding differences across batch sizes.
The clean local target is recomputed from susceptibility; no sparse impulses
are applied. Held-out subjects have three fixed views in each of two suites:
`validation` uses the training susceptibility/background distribution at its
fully developed mixture. The active profile selects its best checkpoint using
COSMOS susceptibility NRMSE; historical validation-selection profiles use
`validation` instead. `validation_stress`
retains the legacy susceptibility scales and unrescaled strong backgrounds.
Both remain subject-disjoint from training, uncropped, and fixed across epochs.

The active profile draws one voxel size uniformly from **[0.5, 1.0] mm per
simulated training view** and repeats it on all three axes. Configure the bounds
with `--voxel-size-min` and `--voxel-size-max`. An independent seed stream keeps
the draw reproducible across epochs, workers, batch grouping and exact resume,
without consuming susceptibility, lesion, background or noise RNG streams.
Generated data include `voxel_size`, an `[N,3]` FP64 tensor in mm; scalar logs
include the sample-weighted mean `voxel_size_mm`. Local and background simulation,
the model inversion, and physics losses share the matching per-sample FP32 rFFT
dipoles. Patch mode retains the full-grid source dipoles and the same voxel sizes
when constructing patch dipoles. Kernels are not cached by random floating-point
spacing, so the shape cache remains bounded.

This setting changes **kernel spacing and metadata only**: it does not resample
susceptibility or masks, change lesion geometry, or simulate a different acquisition
resolution. Uniform isotropic spacing cancels in the ideal dipole frequency ratio;
therefore the generated fields are effectively unchanged apart from floating-point
roundoff. The active reference-validation settings retain fixed unit spacing;
SIM2 evaluation uses fixed `[0.65, 0.65, 0.65]` mm kernel spacing, and COSMOS uses
its saved simulation kernel. To randomize a custom validation suite, put the two voxel-size
bounds explicitly in its `validation_synthesis` object; seeded views remain fixed.
Historical profiles and checkpoints without bounds use `[1,1,1]` exactly. Exact
resume preserves the saved bounds; the running experiment is not changed by this
source-code update. Use a new run to enable the policy with existing weights.

### Susceptibility shape augmentation

The optional `--chi-augmentation` JSON policy adds smooth monotone intensity
remapping on the final crop/resize grid. Version 1 provides randomized `bimodal`
and `central_tail` susceptibility families. The transform retains voxel locations
and intensity ordering, bounds the quantile-knot secants, and extends extreme
tails continuously without clamping susceptibility. Nearly constant maps fall
back to the existing shape. Original source files and caches remain unchanged.

With the policy enabled, the existing target-RMS law (95% log-uniform
0.018–0.040 ppm; 5% 0.040–0.080 ppm) is applied after remapping and resizing.
The field synthesizer computes new clean local fields from these final maps,
then generates background and noise. Shape remapping uses an independent RNG
stream, preserving the existing flips, crops, target-RMS draws, and field
synthesis seeds. The default policy is off;
old checkpoints retain their original sampling behavior.

Four five-epoch screening configurations share the latest interrupted checkpoint,
model settings, source split, optimizer settings, and update budget:

| Config | Training susceptibility policy |
|---|---|
| `configs/chi_augmentation_A0.json` | Existing pipeline |
| `configs/chi_augmentation_A1.json` | Final-grid RMS only |
| `configs/chi_augmentation_A2.json` | A1 plus bimodal remapping on one third of original draws |
| `configs/chi_augmentation_A3.json` | A1 plus equal baseline/bimodal/central-tail profiles on original draws |

Synthetic maps retain their existing shapes. With 75/25 original/synthetic
sampling, A3 yields approximately 50% existing shapes, 25% bimodal, and 25%
central-tail draws. Configurations serialize the full parameter ranges; use
`--config configs/chi_augmentation_A3.json` to opt in. The screening configs
disable Comet and external evaluation; they do not start training when loaded.
Update their `init_checkpoint` path if using another checkout or source run.

`--chi-shape-validation` adds a separate `validation_shape` suite with three fixed
views per held-out subject: baseline, bimodal, and central-tail. Its profile
ranges are fixed independently of the training policy, and its aggregate and
per-profile NRMSE metrics never select the best checkpoint. Existing `validation`
and `validation_stress` remain unchanged. Training logs and resumed checkpoints
record realized profile and fallback counts; sample metadata records proposed
mixture parameters, remapping bounds, target RMS, and achieved RMS.

Run a paired data-only comparison before training:

```bash
.venv/bin/python scripts/analyze_domain_histograms.py --chi-policy-audit \
  --epochs 53 54 55 --max-per-epoch 192 --workers 4 \
  --output artifacts/diagnostics/chi_augmentation_audit
```

Omit `--max-per-epoch` to audit complete epochs. Use
`--policy-config A3=configs/chi_augmentation_A3.json` to audit custom ranges.
Outputs include shared-bin physical and normalized histograms, per-profile and
cohort summaries, per-image distances, spatial gradients, radial spectra, tail
energy, and paired susceptibility/field examples. Sim2's clean local field is a
forward-model proxy. These audits measure coverage; reconstruction improvements
require held-out training comparisons. See the
[implementation plan](docs/plans/chi_distribution_augmentation.md) for the
experimental controls and interpretation of development-informed benchmarks.

### Focal lesion augmentation

The active profile enables `lesion_augmentation` for subsequent training runs.
On 50% of training views, place 1–3 nonoverlapping lesions entirely inside the
brain mask. Half the shape draws are spheres with radius 2–4 voxels; the other
half are randomly rotated ellipsoids with independently drawn semiaxes of
2–6 voxels. Core susceptibility is sampled uniformly in absolute value from
**0.25–0.6 ppm**, with positive and negative signs equally likely. Every core
voxel therefore satisfies **abs(chi) > 0.2 ppm**.

Ellipsoids have a 1–2-voxel transition between their core and an outer ellipsoid.
A quintic smootherstep blends the lesion value into the susceptibility already
present at each tissue voxel. It has zero first and second derivatives at both
ends. Core values remain constant and extreme; shell values approach ordinary
tissue continuously and need not exceed 0.2 ppm. Spheres use sharp boundaries.
The transition width and all shape, amplitude, count, and probability ranges
are configurable in the versioned `lesion_augmentation` JSON object.

Lesions are inserted **after** flips, crop/resize, susceptibility histogram
remapping, and final tissue RMS scaling, and **before** clean-field, background,
and noise simulation. No subsequent susceptibility RMS rescaling dilutes their
absolute core values. `chi_rms_before_lesions` records the tissue scale;
`chi_rms_final` includes the lesions. Clean local-field targets are recomputed
from the augmented susceptibility, rather than adding impulses to field images.

An independent per-view/epoch seed makes lesion geometry reproducible across
workers and exact resume without consuming existing augmentation RNG streams.
Metadata records centers, semiaxes, rotations, transition widths, values, and
skipped placements. Per-batch and per-epoch logs report lesion counts by shape.
Placements that cannot fit wholly inside the mask without overlap are skipped
and counted. Masks, original files, caches, held-out validation (including the
shape suite), and external cases are not modified by the lesion policy.

Omitting `lesion_augmentation` or setting its `probability` to zero preserves
the old data path. Existing runs keep their recorded lesion/dephasing policy;
changing it requires a new run with `--init-checkpoint` to reuse weights,
rather than altering an exact resume. Updating the augmentation does not stop
or restart an existing run.

### Lesion dephasing artifacts

For subsequent training, **30% of successfully generated lesions** receive one
event at a uniformly selected voxel inside their extreme-susceptibility core.
The default is an equal mixture of two modes (15% of lesions each on average):

- **Signal loss:** relative complex-signal magnitude is 0.01–0.10 and its phase
  error is uniformly random in `[-pi, pi]`.
- **Phase spike:** magnitude stays one and a randomly signed phase error of
  `pi/2`–`pi` radians is applied.

This is a single-voxel complex-signal augmentation. The relative signal
`m * exp(i * phase_error)` supplies the corrupted phase; magnitude is retained
in sample metadata. The network continues to consume the field map and ROI
mask. The phase error is converted to the input's physical ppm units using

```text
field_error_ppm = phase_error_radians /
                 (2*pi*42.577478461*B0_tesla*TE_seconds)
```

The gyromagnetic ratio in MHz/T is from
[NIST CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?gammapbar).
**3 T and TE = 20 ms** are configurable simulation defaults, not parameters
inferred from the dataset. A relative phase error preserves the original
unwrapped-field branch; the rest of the volume is not wrapped or rescaled.

Apply the artifact **after** clean-field, background, and noise simulation.
Only the measured input voxel changes: susceptibility, clean local fields,
background, Gaussian noise, and the brain mask remain identical. The optional
`dephasing` tensor stores the added field error, so the decomposition is
`phase = local + background + noise + dephasing`. Post-simulation patches crop
the artifact with the same coordinates as the other fields.

The policy fields are `dephasing_probability`,
`dephasing_signal_loss_probability`, `dephasing_relative_magnitude`,
`dephasing_spike_radians`, `dephasing_b0_tesla`, and `dephasing_echo_time_ms`.
An independent RNG stream keeps lesion geometry, susceptibility, and existing
noise draws unchanged when dephasing is toggled. Metadata records the chosen
voxel, mode, complex signal, angle, and ppm error; training logs count events
by mode. Validation and external cases remain unchanged. Historical policies
without these fields default to dephasing off, including exact resume.

### Spatial crop augmentation

`--crop-probability 0.2` additionally selects 20% of training samples in
expectation for a tight foreground-mask bounding-box crop. Selection is
deterministic per sampled view and epoch. Cropping follows the existing flips
and axis permutations. Selected crops are then interpolated to **160×160×160**
by default (`--crop-resize`). Susceptibility uses trilinear interpolation with
`align_corners=False`; masks use nearest-neighbor interpolation and remain
binary. Susceptibility is masked again and dilation is recomputed on the output
grid. The tight crop's bounds and original dimensions remain in sample metadata.
Non-cubic crops are scaled independently along each axis to fill the cube.
Unselected samples, validation, external evaluation, and source files are unchanged.

Cropping and interpolation happen **before** field synthesis: clean fields,
backgrounds, noise, TKD kernels, and forward-field metrics all use the same 160³ grid.
With the standard 160³ source data, cropped and uncropped samples stack together
in one `[N,1,160,160,160]` tensor and share one model forward pass.
`--no-crop-resize` retains the previous native-crop behavior, grouping different
shapes and accumulating gradients weighted by sample count. Kernels use a bounded
cache and voxel-frequency coordinates, including odd rectangular grids.

```sh
uv run dnqsm-train --device cuda:0 --comet \
  --cuda-memory-gb 12 \
  --comet-env-file ../wh_net/.env \
  --cosmos-simulation ../wh_net/artifacts/runs/wh_tv_cosmos_20260914_000637_662208/simulation.pt
```

The active strided-convolution profile starts with fresh weights for **100 joint epochs**,
no filtering-only pretraining, and AdamW at `3e-5`. An explicit
`--epochs` overrides that budget; running the command starts training.
The lower-level `TrainConfig` defaults retain **10 filtering-only epochs** followed
by **50 alternating epochs** for explicit experiments. `--epochs` counts epochs after pretraining;
`--filter-pretrain-epochs 0` disables pretraining. Pretraining calls only the
filter, using local-field L1, gradient matching, harmonic regularization, and
field-sum losses. The susceptibility network is bypassed. Optional local-TKD
supervision also inverts the local prediction during filtering-only training.

After pretraining, the first batch updates only the phase filter with the same
local-field objective. The next batch freezes the phase filter and updates only
the chi correction network, using susceptibility and physics losses on the
filter's predicted local field. This phase/chi sequence continues across epoch
boundaries, including epochs with an odd number of batches. Frozen networks are
in evaluation mode and receive no gradients; their weights and Adam moments stay unchanged.
Validation evaluates the full reconstruction after every alternating epoch.

Both stages use batch size 10, AdamW at `1e-4`, weight decay `1e-4`, three warmup
epochs, cosine decay to `5e-6`, and gradient clipping at 10. The default
`--accumulation-steps 1` gives one optimizer update per batch; alternating
training requires this setting. `--training-schedule joint` retains the joint
schedule and supports accumulation over multiple batches. Filter weights and
Adam moments carry across the pretraining boundary. The warmup/cosine schedule
restarts at that boundary, and optional physics-weight warmup starts there.
Global sampling epochs, background curriculum, and update counts continue
across the transition. CUDA convolutions use BF16 when supported.
Activation checkpointing is enabled. The new profile
`configs/deep_alternating_batch10.json` uses the measured six-level architecture
and the latest susceptibility augmentation with 10 pretraining plus 100
alternating epochs. Changing depth requires fresh initialization; shallower
checkpoints retain their recorded architecture and cannot be exactly resumed
into the deeper model. Check GPU occupancy before a full run.
`--cuda-memory-gb 12` sets an 11 GiB PyTorch allocator ceiling and reserves
1 GiB for CUDA context/library allocations. Zero leaves the allocator unlimited.
The initial experiment used the 12 GiB budget to share the GPU. The current
experiment uses a budget derived from the available device capacity.

To size nominal batches for the twice-baseline profile on an otherwise idle GPU:

```sh
uv run python -m dnqsm.batch_size \
  --directory artifacts/capacity/double_batch_search --target-fraction 0.95 \
  --features 23 47 90 --crop-probability 0.2 --crop-resize \
  --expandable-segments
```

This measures residual field filtering, BF16, and activation checkpointing on
actual 160³ training volumes. `--config <json>` accepts a saved training config.
Each candidate runs in an isolated process and includes synthesis, three
uncropped forward/backward/Adam cycles with zero learning rate, crop/resize
augmentation, gradient telemetry, partial training/matched/stress validation
batches, visualizations, and both external
evaluation cases. The search brackets the limit, checks the
adjacent larger integer, and writes the chosen batch and all measured peaks to
`summary.json`. It reserves 1 GiB outside the allocator within the requested
fraction of CUDA-reported capacity. Integer batch sizes may leave some headroom;
the search does not allocate dummy tensors to inflate utilization.

Use `selected_batch_size` and `selected.cuda_memory_gib` from that summary,
and preserve its allocator setting in the training process:

```sh
uv run dnqsm-train --features 23 47 90 --filter-pretrain-epochs 10 \
  --batch-size <selected_batch_size> --accumulation-steps 1 --epochs 200 \
  --crop-probability 0.2 --crop-resize --external-eval-every 1 \
  --cuda-allocator-config expandable_segments:True \
  --cuda-memory-gb <selected.cuda_memory_gib> \
  --comet --comet-env-file ../wh_net/.env \
  --cosmos-simulation ../wh_net/artifacts/runs/wh_tv_cosmos_20260914_000637_662208/simulation.pt
```

Training uses per-tensor AdamW updates (`foreach=False`), matching the probe.
Memory logging includes allocated/reserved GiB, reserved percentage, and sampled
process VRAM including CUDA libraries. Completed graphs are released before the
next forward pass; final external evaluation releases training weights and Adam
state before loading the best checkpoint.
The allocator configuration is checkpointed and restored on resume. The 95%
target applies to a full-size batch; interpolated crops now use that same size.

Each new run creates `artifacts/runs/dnqsm_<timestamp>/` with configuration,
subject split, JSONL metrics, central-slice images, and these checkpoints:

- `model_best.pt`: model with the lowest COSMOS susceptibility NRMSE after pretraining in the active profile.
- `best_checkpoint.json`: selection metric, NRMSE, epoch and step of those exact weights.
- `model_last.pt`: most recent inference weights.
- `training_last.pt`: model, optimizer, scheduler, RNG state, phase, and resume cursor.

SIGINT/SIGTERM requests a checkpoint after the current accumulated optimizer
update. Training also checkpoints after the first update, every 50 updates,
and at epoch boundaries.
`--checkpoint-metric cosmos --external-eval-every 1` selects the final
susceptibility reconstruction's COSMOS `chi_nrmse` after every joint/alternating
epoch. Improvements atomically replace `model_best.pt`; ties retain the earlier
best. Missing or nonfinite COSMOS scores fail without falling back to held-out
validation. Subject validation and SIM2 remain logged but do not select this
checkpoint. The selected metric and running best are stored in resume checkpoints.
`--checkpoint-metric validation` retains the historical held-out selection rule;
checkpoints and profiles without the setting retain that behavior. Exact resume
preserves the selection rule and best score.
Resume preserves sampling, scheduling, architecture, and loss configuration:

```sh
uv run dnqsm-train --resume artifacts/runs/<run>/training_last.pt --max-steps 0
```

To append training after a run has reached its target, use
`--resume <run>/training_last.pt --additional-epochs 50`. This keeps the latest
weights, optimizer, RNG, epoch cursor, COSMOS-best score and original learning-rate
schedule. A completed cosine schedule stays at its floor. `epochs` remains the
original schedule duration; the saved `additional_epochs` is cumulative, and the
training target is their sum (plus pretraining). A later resume without the flag
keeps that target; explicitly supplying the flag again appends another block.
An unfinished target cannot be extended through this option.

Exact resume preserves interpolation, loss floors/weights and warmup, curriculum,
input/masking architecture, pretraining duration, training schedule, TKD threshold,
and validation definition.
It works within either phase and at the transition. Checkpoints record loss
version **8**, which evaluates harmonic-field supervision inside a radius-3
spherically eroded ROI. Earlier
objectives require a new run with `--init-checkpoint`; their optimizer and
scheduler cannot silently resume with a different objective.
Historical checkpoints retain their original channels, masking, and residual
behavior for inference; missing pretraining metadata means
zero pretraining epochs. Checkpoints without `training_schedule` retain joint
training, and saved TKD thresholds are preserved. Alternating resumes recover
the next branch from the saved epoch and batch cursor. To reuse weights with a matching architecture, start
a separate experiment with `--init-checkpoint <run>/model_last.pt`;
this resets the optimizer, scheduler, and training cursor and preserves the old
run's history. Do not initialize a full run from COSMOS-overfitted weights.
Compatible filters without concatenation can initialize the concatenation head
by appending a zero raw-field coefficient. A different Laplacian-residual policy
requires a fresh run. Historical inference and exact resume
keep the original head and prediction rule (`filter_output_skip=false` and
`filter_laplacian_residual=false` when the respective flags are absent).
Checkpoints without `filter_refinement_features` retain their original output
without refinement. A new run with `--init-checkpoint` can append the identity
refinement to a compatible model; an existing trained block cannot be silently
removed or resized. Exact resume preserves its saved width.

`configs/dual_field_100epochs.json` explicitly selects the experimental architecture for 100 joint
epochs with no filtering-only pretraining, retaining the full-grid local-field
output, TKD supervision, and background distribution of the preceding full-grid
profile. Earlier named profiles explicitly retain their historical filtering
architecture; loading one overrides the new model defaults. No training is
started by loading a profile. Harmonic estimates are included in evaluation
plots, NumPy outputs, padded predictions, and NIfTI exports.

Portable corrective profiles are in `configs/corrective_200epochs.json` and
`configs/corrective_ablation.json`. `--config` loads JSON for a new experiment;
explicit CLI flags override it. Exact resume uses `--resume` without `--config`.
Historical chi-augmentation A0–A3 and `pretrain20_e2e100` profiles explicitly
retain joint training and their recorded threshold of 0.1; explicit JSON
settings override the new defaults.
The 200-epoch profile uses 10 pretraining plus 200 alternating epochs, the 6.42M model
and BF16, and `3e-5` learning rate,
20% crop/resize, and per-epoch COSMOS/sim2 evaluation. The ablation profile bounds
training to 128 records and validation to eight subjects for 10 pretraining
plus five alternating epochs.
These profiles do not start or modify an existing run. Before a new long run,
repeat the fixed-case overfit gate and compare short ablations across seeds.
Re-profile VRAM using the updated architecture and sampled gradient diagnostics.

`configs/background_regime_100epochs.json` re-centres training on the background
regime the external benchmarks occupy: 5% clean, 5% weak, 50% typical, and 40%
strong backgrounds with `--strong-min-ratio 0.005`. Audited over a sampler
epoch that yields a median background/local RMS ratio of 35 and places 64% of
samples at or above 10, against 5.1 and 14% for the preceding profile; COSMOS
is at 22. Its `validation_synthesis` object matches, so checkpoint selection
targets the same regime, which makes its `validation` series incomparable with
earlier runs. `validation_stress` is unchanged and remains the cross-run anchor.
`configs/background_regime_ablation.json` is the bounded matched arm: 400
filtering-only updates from a supplied `init_checkpoint`, sharing that
validation suite so two mixtures differ only in what they train on.

`configs/fullgrid_100epochs.json` retains the historical single-decoder
architecture for that mixture: `full_grid_field` with `filter_input_version=2`,
`lambda_extension=1`, `threshold=0.05`, and `lambda_local_tkd=1`. It keeps the
Laplacian-only encoder and conditional scaling at 32. The corresponding new
dual-decoder profile is `configs/dual_field_100epochs.json`. Masking the
predicted field before TKD applies a full-grid operator to a truncated field;
on COSMOS that costs about 15 percentage points of susceptibility NRMSE that
training cannot recover, measured against the exact clean field.

`configs/balanced_local_total_100epochs.json` targets both noisy local and total
fields: 50% have no background, 10% weak, 35% typical, and 5% strong backgrounds
after a 20-epoch curriculum. Training noise spans clean-local RMS SNR 5–100 via
`--snr-min`/`--snr-max`; their defaults remain 40–100. Its `validation_synthesis`
object preserves the preceding experiment's background mixture and SNR 40–100,
recorded as the `reference` validation profile. Stress and external benchmarks
remain unchanged. An explicit validation object can also be supplied as JSON
with `--validation-synthesis`.

The profile enables `--mse-input-ratio-cap 10`: physics MSE uses
`max(RMS(local_gt), RMS(input)/10)^2` as their denominator, with the existing
floor. Local and chi L1 normalization and their coefficients stay the
same. Scales are detached; conventional NRMSE remains target-relative. A cap
of zero (the default) preserves historical normalization. New settings require
a new experiment; exact resume rejects changed sampling or normalization.

The generic legacy constructors enable conditional scaling with
`--conditional-laplacian-threshold 32`; the active profile instead uses always-on
physical derivative scaling and sets the unused threshold to zero. The historical profile
`configs/conditional_laplacian_100epochs.json` records this setting explicitly
alongside FP32 preprocessing. For each
sample, it measures physical Laplacian RMS only where all 27 stencil voxels lie
inside the ROI. If total-field RMS is at least 32 times that interior RMS, the
filter receives the Laplacian divided by interior RMS and rescales its output by
that same interior RMS. Models with exterior extension retain the entire ROI;
historical models without it keep only the valid interior in this branch.
Other samples retain total-RMS preprocessing.
Samples without a valid stencil also retain total-RMS preprocessing. This changes
neither U-Net channels nor the projected correction. Zero disables the option.
The threshold is checkpointed; historical models default to zero, and exact
resume rejects a changed value. Existing weights require a new adaptation run.

In two matched 80-update adaptation trials, conditional scaling reduced mean
held-out susceptibility NRMSE from 64.75% (FP32 only) to 58.08%, with clean local
NRMSE 13.10% versus 13.00%. These are short trials on 16 held-out subjects with
three fixed views each; they do not establish converged or clinical performance.
The external COSMOS advantage did not repeat across seeds. Use
`--conditional-laplacian-threshold 0` for a comparison with total-RMS scaling;
this is also required when selecting a historical version-1 architecture.
InstanceNorm and physics loss remain the defaults. The experiment
runner is `scripts/normalization_ablation.py`; complete local evidence is in
`artifacts/diagnostics/normalization_ablation_20261019/`.

```sh
uv run python -m dnqsm.batch_size --config configs/corrective_200epochs.json \
  --directory artifacts/capacity/corrective --expandable-segments
uv run dnqsm-train --config configs/corrective_ablation.json \
  --batch-size <selected_batch_size> --cuda-memory-gb <selected.cuda_memory_gib>
uv run dnqsm-train --config configs/corrective_200epochs.json \
  --batch-size <selected_batch_size> --cuda-memory-gb <selected.cuda_memory_gib> \
  --comet-env-file ../wh_net/.env
```

`--max-steps` is a total update limit, including resumed updates; zero removes
the limit. `--max-train-volumes` and `--max-val-subjects` bound diagnostics and
are recorded in the run configuration. A bounded run does not replace a full
experiment. For an explicit CPU smoke test:

```sh
uv run dnqsm-train --device cpu --max-steps 1 --accumulation-steps 1 \
  --max-train-volumes 2 --max-val-subjects 1 --num-workers 0 \
  --features 2 4 --no-comet --no-evaluate-external
```

Local logging is always enabled. `--comet` additionally logs to project
`dn-qsm` (override `--comet-project` or `COMET_PROJECT_NAME`). Export
`COMET_API_KEY`, put it in ignored `.env`, or specify `--comet-env-file`.
Credentials are never saved to artifacts. Resume reuses the experiment key in
`comet.json`. Tests mock Comet and require neither credentials nor network.

All susceptibility panels, including errors and TKD outputs, use `gray` and
`[-0.1,0.1]` ppm. Figures show the three central planes and masked NRMSE in
percent where a reference exists. Zero-energy references display `NRMSE=N/A`.

Joint training logs normalized and `raw_*` penalties, susceptibility/local/physics
NRMSE, background-bucket counts/errors, active physics weight, curriculum
progress, clipping indicator, and epoch clipping fractions as native scalars.
`--gradient-every 100` samples per-term gradient norms, weighted norms, and
pairwise cosines separately for the filter and null network on the first shape
group of the selected update. It preserves `.grad` and RNG state; zero disables
the extra backward computations. Undefined bucket NRMSE values are omitted.
Validation bucket metrics are sample-weighted; its overall NRMSE retains equal
subject weighting. `validation` and `validation_stress` have distinct stable
image series indexed by epoch, alongside COSMOS/sim2 series.

## Single-COSMOS overfitting check

Keep full training stopped while checking whether the complete model can fit
one fixed simulated total-field input and its clean local-field/susceptibility
targets, using the masked total-field dual-branch filter and normalized objective:

```sh
uv run dnqsm-overfit-cosmos --device cuda:0 --cuda-memory-gb 27 --max-vram-gb 30 \
  --steps 250 --features 54 108 216 432 864 --no-stop-on-pass \
  --cosmos-simulation ../wh_net/artifacts/runs/wh_tv_cosmos_20260914_000637_662208/simulation.pt \
  --comet --comet-env-file ../wh_net/.env
```

This initializes a fresh model and reuses the exact saved noisy simulation on
every update, with no augmentation or new noise. The large overfit default is
250 updates with five encoder levels `[54,108,216,432,864]`, a 1,728-channel
bottleneck, and a single-decoder phase filter. The exact parameter count is
recorded in each run; historical checkpoints retain max pooling and biases.
The former `[16,32,64]` model has 3,212,515 parameters. The sequential
U-Net/TKD/projected-correction architecture uses the normalized local-field,
gradient, susceptibility, and physics terms. Harmonic and field-sum penalties
belong to the opt-in dual-field profile.
This step-based diagnostic trains the complete model from its first update;
independent pretraining belongs to the full training loop.

AdamW uses `1e-3`, zero weight decay, a memory-efficient per-tensor update
(`foreach=False`), and cosine decay to 5% of the initial rate. BF16 convolutions,
FP32 physics/losses, and activation checkpointing control memory. The active loss
weights can be overridden independently. Use `--features 16 32 64 --steps 2000`
for the former small-model experiment settings.
Overfit physics loss is enabled by default; `--physics-warmup-steps` optionally
ramps its weight.
Its fit gate always measures the final fixed objective, so changing warmup
weights cannot manufacture a passing loss reduction. `--gradient-every 50`
records per-objective gradient norms/cosines without updating optimizer state.

`--cuda-memory-gb` specifies a **GiB budget**, including a 1 GiB reserve outside
the allocator. The 27 GiB setting therefore limits the PyTorch allocator to
26 GiB. `--max-vram-gb 30` separately enforces a **decimal 30 GB process limit**:
`nvidia-smi` samples the process every 250 ms, including CUDA library allocations,
and observed peaks are saved as `peak_process_vram_gb`. A monitor failure or an
observed process allocation at/above the ceiling aborts the run. Memory-only,
zero-learning-rate probes selected this profile at 28.43 GB observed peak; the
next tested width `[56,112,224,448,896]` exceeded the allocator budget.

By default the check stops early when susceptibility, filtered-local-field, and
ground-truth forward-field NRMSE are **each at most 10%**, and total loss is at
most 20% of its initial value. `--target-nrmse`, `--target-loss-ratio`, and
`--steps` configure this explicit gate. Failure to reach it is reported as
`failed_sanity`, rather than success merely because updates completed.
The command exits with code 2 when its gate is unmet, allowing automation to
stop before starting a full experiment.
`--no-stop-on-pass` uses all configured updates and retains the best qualifying
checkpoint if a pass occurs before the final update.
After an interruption, `--resume <run>/overfit_last.pt --steps 250` continues
the same weights, optimizer moments, fixed case, and global update counter.
An explicit `--learning-rate` change starts a new cosine phase over the remaining
updates and is recorded in `training_phases`; the original configuration and
initial metrics are preserved. Resuming an already exhausted budget is rejected.
`summary.json` records initial, best, and final errors, loss reduction, and the
pass decision. Figures, checkpoints, Comet curves, and standard evaluation
exports are also saved. These are training-fit results, not held-out COSMOS
benchmark scores. The command never starts full-data training automatically.

## COSMOS and sim2 evaluation

A completed full run evaluates its best checkpoint on both datasets by default.
`--no-evaluate-external` disables that final step. Run evaluation independently:

Omitting `--checkpoint` selects the restored `artifacts/models/current.pt`.

```sh
uv run dnqsm-evaluate --checkpoint artifacts/runs/<run>/model_best.pt \
  --datasets cosmos sim2 --device cuda:0 \
  --cosmos-simulation ../wh_net/artifacts/runs/wh_tv_cosmos_20260914_000637_662208/simulation.pt \
  --data-dir ../wh_net/data/evaluation
```

COSMOS loads the completed SNR=100/background=10 simulation and its adjacent
`config.json`. It restores physical units once with the stored factor (the
existing benchmark uses `0.2125`), preserves the supplied dipole, and removes
the recorded padding (12 voxels in that benchmark). If no path is supplied,
the newest matching completed reference simulation is selected and recorded.
Its clean local reference comes from ground-truth susceptibility, not noisy
total-field subtraction. COSMOS exports use an identity affine because its
MAT/simulation inputs provide no anatomical affine.

sim2 reads `Sim2.mat:phase`, `mask_final.mat:mask_final`, and
`volumes/Sim2ChiGT.nii.gz`. It follows the reference's mask bounding-box crop
and centered cube padding (`--sim2-pad 20`). Its dipole kernel uses fixed
`[0.65, 0.65, 0.65]` mm spacing, recorded as `voxel_size_mm` in evaluation metadata.
TKD, the null-space projection, and forward metrics use this same kernel.
This changes kernel spacing only, without resampling the supplied volumes.
Inference uses ppm directly,
without the old solver factor `0.264`. DN-QSM does not consume the old solver's
fidelity-weight map. Cropping follows the reference's index alignment even
though the supplied phase array is larger than the mask. NIfTI outputs restore
the mask's original shape and susceptibility reference affine.

sim2's supplied phase is **not** the dipole forward field of its reference
susceptibility: they differ by roughly 51% NRMSE, masking the reference changes
nothing (0.02% of its susceptibility energy lies outside the mask), and the
residual is rough rather than a smooth background. Every sim2 case therefore
records `forward_consistency_nrmse` in its metadata so the benchmark's own
inconsistency travels with the score it bounds. Treat sim2 as an
out-of-model-class consistency check; it does not select checkpoints.

Evaluation reports final and TKD susceptibility NRMSE, consistency MSE against
the predicted pre-TKD field (`forward_mse`), and inference time. COSMOS also
reports clean local-field NRMSE and the forward field's MSE/NRMSE against that
ground-truth field (`physics_mse`/`physics_nrmse`); sim2 has no such reference.
Ground truth never enters model inference. The active profile uses the COSMOS
susceptibility reference to rank checkpoints by NRMSE; this does not add COSMOS
samples to optimizer updates.
`--external-eval-every 1` evaluates the **current weights** on both fixed cases
after every post-pretraining epoch, retaining the optimizer and restoring model mode.
Pretraining validation reports local-field losses/NRMSE and field-only figures;
best-model selection and external reconstruction evaluation begin after pretraining. Logs and status include `phase` and `phase_epoch`. Comet's
native scalar metrics `external_cosmos_chi_nrmse` and
`external_sim2_chi_nrmse` use epoch as their step axis, rendering interactive
charts in the Metrics tab. TKD and available local/physics NRMSE are also scalar
series. Undefined NRMSE values are omitted rather than replaced with zero.
Static NRMSE curve images are no longer generated or uploaded.

Case comparisons use Comet's image API with stable names `external_cosmos` and
`external_sim2`, `step=epoch`, and `overwrite=False`. The Images tab can track
each case through successive epochs. Training and held-out validation images
similarly use stable `train` and `validation` names. Local comparisons remain in
their epoch directories; `evaluation/history.json` retains numerical history.
A boundary checkpoint allows an interrupted evaluation to resume without
repeating optimizer updates.
Per-epoch evaluation writes metrics and images only under
`evaluation/epochs/epoch_XXXX/`, avoiding 200 copies of full volume exports.
Each case writes metrics and preprocessing metadata, a central-slice comparison,
cropped `volumes.npz`, full-grid `padded_predictions.pt`, and masked NIfTI
susceptibility, TKD, local-field, and mask outputs during standalone evaluation
or final best-model evaluation under `<run>/evaluation/best/`. Final best-model
Comet metrics use the separate `best_cosmos`/`best_sim2` prefixes.

## Checks

```sh
uv run python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
```

Tests cover FFT physics, cutoff boundaries and projection invariance, gradient
flow, deterministic subject splitting and synthesis, clean targets and SNR,
exact resume behavior with mixed native shapes, 160³ crop interpolation and
binary masks, single-group batching, accumulation, native Comet scalar series
and image progression across epochs,
unchanged optimizer state during evaluation, figure limits, physical-unit
evaluation, crop selection and geometry, and padding/affine restoration.
Filtering tests cover raw masked input, shared encoding and independent decoders,
harmonic and field-sum gradients, historical Laplacian preprocessing, frozen-network pretraining, restarted learning rates, and
exact resume within either stage and at their transition, alternating branch
isolation, and unchanged weights/Adam moments on frozen batches.
Corrective-objective tests cover per-sample amplitude invariance, detached
normalization scales, empty-exterior handling, full-grid inversion, historical
checkpoint inference, unchanged BF16 output precision, mixture/curriculum
sampling, gradient-diagnostic invariance, matched/stress scalar logging,
checkpoint selection, and recovery from interrupted stress validation.

Shared device, precision, memory-budget and checkpoint utilities live in
`dnqsm.runtime`, allowing evaluation and overfit tools to run independently of
the training module. Checkpoint loading uses memory mapping; existing imports
of these utilities from `dnqsm.training` remain supported. Synthesis retains
reliability-dependent noise without allocating an unused fidelity-weight output.
