# DN-QSM

DN-QSM jointly removes background fields and noise, performs hard-cutoff TKD,
and corrects susceptibility artifacts with a deep null space network. It follows
the maintained dataloader and COSMOS/sim2 preprocessing in the sibling `wh_net`
repository, without importing that package at runtime.

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
The initial small baseline uses widths `[16,32,64]` and a 128-channel bottleneck.
The current twice-baseline profile uses `[23,47,90]` and a 180-channel bottleneck:
**6,422,027 parameters**, versus 3,212,946 in the baseline (1.999×), both with
the Laplacian input channel enabled.
The large overfit profile below uses five wider levels. All use InstanceNorm/CELU
double-convolution blocks and trilinear upsampling. Inputs are normalized by
masked RMS, and outputs are rescaled to ppm.

1. The phase-filtering U-Net receives two channels: the masked total field and
   its convolution with the fixed 3×3×3 Laplacian kernel. The kernel has center
   coefficient `-44/13`, six face neighbors `3/13`, twelve edge neighbors `3/26`,
   and eight corner neighbors `1/13`. It is a non-trainable checkpoint buffer;
   convolution uses stride 1 and zero padding 1 to preserve the volume shape.
   Both channels share the input's masked RMS scale: for `u = field/scale`,
   `filter_input = cat([u, Laplacian(u)], dim=1)`. The U-Net produces a
   **full-grid** clean local field `local`, including its exterior extension.
   Residual filtering computes
   `local = field + scale * UNet(filter_input)`; `--no-residual-filter` selects
   direct output. `--no-full-grid-field --lambda-extension 0` provides a masked
   field ablation. `--no-laplacian-input` selects the previous one-channel
   architecture. Historical checkpoints retain their original input channels
   and field behavior when evaluated or exactly resumed. Training the updated
   architecture requires a fresh run; old one-channel weights are not silently
   converted.
2. TKD computes `chi0 = F^-1(B_tau * F(local))`, where `B_tau=1/D` for
   `abs(D)>tau` and zero otherwise. The default threshold is `tau=0.1`.
3. A second U-Net predicts a susceptibility residual from `chi0`. The output is
   `chi = chi0 + F^-1(1[abs(D)<=tau] * F(N(chi0)))`.

This is the regularized residual structure `(I-B_tau A)N` from
[Schwab, Antholzer, and Haltmeier, Section 3.2](https://arxiv.org/html/1806.06137#S3.SS2),
with a hard-cutoff inverse. The correction lies in the null space of the
**truncated** dipole operator; near-null corrections can change the original
dipole field. A fixed threshold does not establish the paper's asymptotic
convergence claims for this trained pipeline.

Local field, susceptibility, and correction remain unmasked on the FFT grid.
Zeroing the local field exterior would change retained-frequency information
before inversion; the projected correction cannot change those coefficients.
Use the mask for ROI metrics and visualization; masked exports are separate.

The four ROI penalties are dimensionless and normalized independently per sample.
`s_chi` and `s_b` are detached ground-truth susceptibility and clean-local-field
masked RMS, clamped by `--chi-loss-floor` and `--field-loss-floor` (both `1e-4`
ppm). Neither the total field nor a prediction sets these scales:

```text
loss = lambda_forward * masked_MSE(A(chi), local)    / s_b**2
     + lambda_physics * masked_MSE(A(chi), local_gt) / s_b**2
     + lambda_local   * masked_MAE(local, local_gt)  / s_b
     + lambda_chi     * masked_MAE(chi, chi_gt)      / s_chi
     + lambda_extension * exterior_MAE(local, local_full_gt) / s_b
```

Default weights are `local=1.5` and `chi=physics=forward=extension=1.0`.
The predicted-local/clean-local penalty therefore has a coefficient 50% higher
than every other normalized penalty. The fifth term
supervises the exterior using the full simulated field `A(mask*chi_gt)`;
an empty exterior contributes zero. Reductions average voxels per sample, then
samples. `--no-normalize-loss` provides a raw-loss ablation. `lambda_forward` compares with
the predicted **pre-TKD field**, with its gradient intact. `lambda_physics`
compares directly with the **ground-truth clean local field**, independently
penalizing a susceptibility/filter pair that agrees on the wrong field.
Physics and losses stay FP32 even when eligible GPU convolutions use BF16.
The filtering precision policy is unchanged: the CNN and its output head still
use BF16 under CUDA autocast. Physics-weight warmup is disabled by default so
the 1.5:1 coefficient ratio holds from the first update. Optional
`--physics-warmup-epochs` ramps only the physics term; validation always uses
the final coefficients.

```python
from dnqsm.training import load_trained_model

model = load_trained_model("artifacts/runs/<run>/model_best.pt", device="cuda:0")
result = model(total_field, mask, dipole)
# result.local_field, result.tkd, result.correction, result.chi
```

## Data and training

By default, training reads `../wh_net/data/training/joint/` without modifying its
source files. Override `--data-dir`, or set `DNQSM_REFERENCE_ROOT` before launching
to change the reference root. Other datasets contain `image_XXXX.pt` and
`mask_XXXX.pt` pairs with shape `[1,D,H,W]`, plus `subjects.json` mapping image
filenames to source subject IDs. Use `--source-manifest` for a manifest elsewhere.

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
`--background-profile ratio` and `legacy` provide ablations. At fixed sample
seeds, RNG streams are independent of batch construction; strongly amplified
backgrounds can expose small FP32 FFT rounding differences across batch sizes.
The clean local target is recomputed from susceptibility; no sparse impulses
are applied. Held-out subjects have three fixed views in each of two suites:
`validation` uses the training susceptibility/background distribution at its
fully developed mixture and selects the best checkpoint. `validation_stress`
retains the legacy susceptibility scales and unrescaled strong backgrounds.
Both remain subject-disjoint from training, uncropped, and fixed across epochs.

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
backgrounds, noise, TKD kernels, and forward losses all use the same 160³ grid.
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

The default run lasts 50 epochs, with batch size 1, accumulation over five
batches, AdamW at `1e-4`, weight decay `1e-4`, three warmup epochs, cosine decay
to `5e-6`, and gradient clipping at 10 for the normalized objective. CUDA
convolutions use BF16 when supported.
Activation checkpointing is enabled. Check GPU occupancy before a full run.
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
uv run dnqsm-train --features 23 47 90 --residual-filter \
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

- `model_best.pt`: model selected by matched held-out subject susceptibility NRMSE.
- `model_last.pt`: most recent inference weights.
- `training_last.pt`: model, optimizer, scheduler, RNG state, and resume cursor.

SIGINT/SIGTERM requests a checkpoint after the current accumulated optimizer
update. Training also checkpoints after the first update, every 50 updates,
and at epoch boundaries.
Resume preserves sampling, scheduling, architecture, and loss configuration:

```sh
uv run dnqsm-train --resume artifacts/runs/<run>/training_last.pt --max-steps 0
```

Exact resume preserves interpolation, loss floors/weights and warmup, curriculum,
full-grid field behavior, and validation definition. Checkpoints record loss
version **3**. Earlier objectives cannot silently resume with the normalized
objective and exterior supervision. For inference, historical checkpoints
continue using their original masked-field behavior. To reuse weights with
matching input channels, start a separate experiment with
`--init-checkpoint <old-run>/model_last.pt` (add `--no-laplacian-input` for
historical one-channel checkpoints);
this resets the optimizer, scheduler, and training cursor and preserves the old
run's history. Do not initialize a full run from COSMOS-overfitted weights.

Portable corrective profiles are in `configs/corrective_200epochs.json` and
`configs/corrective_ablation.json`. `--config` loads JSON for a new experiment;
explicit CLI flags override it. Exact resume uses `--resume` without `--config`.
The 200-epoch profile retains the 6.42M model and BF16, uses `3e-5` learning rate,
20% crop/resize, and per-epoch COSMOS/sim2 evaluation. The ablation profile bounds
training to 128 records and validation to eight subjects for five epochs.
These profiles do not start or modify an existing run. Before a new long run,
repeat the fixed-case overfit gate and compare short ablations across seeds.
Re-profile VRAM: exterior targets and sampled gradient diagnostics are now included.

`configs/balanced_local_total_100epochs.json` targets both noisy local and total
fields: 50% have no background, 10% weak, 35% typical, and 5% strong backgrounds
after a 20-epoch curriculum. Training noise spans clean-local RMS SNR 5–100 via
`--snr-min`/`--snr-max`; their defaults remain 40–100. Its `validation_synthesis`
object preserves the preceding experiment's background mixture and SNR 40–100,
recorded as the `reference` validation profile. Stress and external benchmarks
remain unchanged. An explicit validation object can also be supplied as JSON
with `--validation-synthesis`.

The profile enables `--mse-input-ratio-cap 10`: forward and physics MSE use
`max(RMS(local_gt), RMS(input)/10)^2` as their denominator, with the existing
floor. Local, chi, and exterior L1 normalization and all coefficients stay the
same. Scales are detached; conventional NRMSE remains target-relative. A cap
of zero (the default) preserves historical normalization. New settings require
a new experiment; exact resume rejects changed sampling or normalization.

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

Training logs normalized and `raw_*` penalties, susceptibility/local/forward
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
targets, using the same normalized objective and full-grid field supervision:

```sh
uv run dnqsm-overfit-cosmos --device cuda:0 --cuda-memory-gb 27 --max-vram-gb 30 \
  --steps 250 --features 54 108 216 432 864 --residual-filter --no-stop-on-pass \
  --cosmos-simulation ../wh_net/artifacts/runs/wh_tv_cosmos_20260914_000637_662208/simulation.pt \
  --comet --comet-env-file ../wh_net/.env
```

This initializes a fresh model and reuses the exact saved noisy simulation on
every update, with no augmentation or new noise. The large overfit default is
250 updates with five encoder levels `[54,108,216,432,864]`, a 1,728-channel
bottleneck, and **590,872,214 trainable parameters** across the two U-Nets.
The former `[16,32,64]` model has 3,212,514 parameters. The sequential
U-Net/TKD/projected-correction architecture uses the normalized ROI terms and
full-grid exterior supervision;
full-training defaults remain separately configurable.

AdamW uses `1e-3`, zero weight decay, a memory-efficient per-tensor update
(`foreach=False`), and cosine decay to 5% of the initial rate. BF16 convolutions,
FP32 physics/losses, and activation checkpointing control memory. All five loss
weights can be overridden independently. Use `--features 16 32 64 --steps 2000`
for the former small-model experiment settings.
Overfit physics-weight warmup is also disabled by default; optional
`--physics-warmup-steps` enables a ramp.
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
and centered cube padding (`--sim2-pad 20`). Inference uses ppm directly,
without the old solver factor `0.264`. DN-QSM does not consume the old solver's
fidelity-weight map. Cropping follows the reference's index alignment even
though the supplied phase array is larger than the mask. NIfTI outputs restore
the mask's original shape and susceptibility reference affine.

Evaluation reports final and TKD susceptibility NRMSE, consistency MSE against
the predicted pre-TKD field (`forward_mse`), and inference time. COSMOS also
reports clean local-field NRMSE and the forward field's MSE/NRMSE against that
ground-truth field (`physics_mse`/`physics_nrmse`); sim2 has no such reference.
Ground truth never enters inference or checkpoint selection in full training.
`--external-eval-every 1` evaluates the **current weights** on both fixed cases
after every epoch, retaining the optimizer and restoring model mode. Comet's
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
