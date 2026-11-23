# DN-QSM

DN-QSM reconstructs magnetic susceptibility from total-field MRI. A 3D U-Net
estimates the local field, a dipole inverse produces an initial susceptibility
map, and a second 3D U-Net estimates a correction that is weighted toward the
dipole's near-null frequencies.

## Installation

Use Python 3.12 or later. From the repository root, install the package and its
dependencies with `uv`.

```sh
uv sync
uv run dnqsm-evaluate --help
uv run python -m unittest discover -s tests -v
```

The checkout requires an external version-1 DN-QSM checkpoint for learned
inference. The default path is `artifacts/models/current.pt`. Pass
`--checkpoint` to use a checkpoint stored elsewhere.

## Model and metrics

The package exports `DNQSM`, `ModelConfig`, and `Reconstruction`. Construct
a model with `DNQSM(ModelConfig(...))` or load saved weights with
`dnqsm.runtime.load_trained_model(checkpoint, device="cpu")`. The loader
restores the saved architecture and inference policy and supplies historical
defaults for fields missing from older checkpoints.

Both `model(total_field, mask, dipole)` and
`model.infer(total_field, mask, dipole)` return a `Reconstruction` with
`local_field`, `tkd` (the initial dipole inverse), `correction`, `chi`, and
the optional `harmonic_field`. Provide a total field in ppm and a binary mask
as tensors of shape `[N, 1, D, H, W]`. The physics module provides dipole kernels
on the matching unshifted FFT grid. Use `model.eval()` before calling
`infer`. That method uses the checkpoint's full-volume or Gaussian
sliding-window policy. Set `window_size=0` to force full-volume inference or a
positive size to run the CNNs on windows. The dipole inverse and correction
weighting operate on the full Fourier grid in both modes.

- `dnqsm.physics.masked_nrmse` returns a percentage per sample and `None` for
  a reference with zero energy inside the mask.
- `dnqsm.experiment_metrics.reconstruction_metrics` returns a dictionary per
  sample with masked NRMSE, bias and linear fit, optional HFEN, and full-grid
  Fourier NRMSE for retained and near-null bands.
- `dnqsm.experiment_metrics.harmonic_laplacian_mse` masks the harmonic field,
  applies the spatial Laplacian in FP32, and computes per-sample MSE inside a
  radius-three spherical erosion of the mask. Empty eroded regions return zero.
- `subject_average` averages views by subject. `paired_summary` reports
  between-run susceptibility NRMSE differences and a paired bootstrap interval
  for matched subjects. Both helpers live in `dnqsm.experiment_metrics`.

## COSMOS and sim2 evaluation

Run both benchmarks on CPU with explicit paths to the inputs and output
directory. The CLI defaults to `cuda:0` when `--device` is omitted.

```sh
uv run dnqsm-evaluate \
  --device cpu \
  --checkpoint /path/to/dnqsm/model_best.pt \
  --datasets cosmos sim2 \
  --cosmos-simulation /path/to/cosmos/run/simulation.pt \
  --data-dir /path/to/sim2/evaluation \
  --output-dir ./artifacts/evaluation
```

- **COSMOS** expects `simulation.pt` next to `config.json` with SNR 100 and
  background amplitude 10. The loader applies the stored scaling to recover
  ppm and computes the clean local-field reference by forward-projecting the
  masked susceptibility on the padded grid.
- **sim2** expects `Sim2.mat`, `mask_final.mat`, and
  `volumes/Sim2ChiGT.nii.gz` within `--data-dir`. The loader crops to the
  mask bounding box, pads centrally to a cube with an additional 20 voxels on
  each side by default, and uses an isotropic 0.65 mm dipole. Metadata includes
  `forward_consistency_nrmse` between the supplied phase and the dipole
  projection of the supplied susceptibility within the mask.

The default reference root is the sibling `wh_net` directory. Set
`DNQSM_REFERENCE_ROOT` to override it. Without explicit paths, sim2 is read from
`data/evaluation` under that root. COSMOS discovery picks the most recently
modified `simulation.pt` under `artifacts/runs/wh_tv_cosmos_*` with matching
configuration and a completed `status.json`.

For Python workflows, `dnqsm.evaluation.load_cases` loads either or both
benchmarks. Pass those cases and an output directory to `evaluate_checkpoint`
with a checkpoint path, or to `evaluate_model` with a model instance.

Evaluation writes a root `metrics.json` and a directory for each case with
metrics, metadata, and a comparison figure. The cropped arrays are in
`volumes.npz`, and the padded predictions are in `padded_predictions.pt`.
Masked NIfTI exports remove padding and restore the original geometry, including
the source affine for sim2.

## Research archive

Research notes, experiment plans, abstract drafts, and figure sources are in
[docs/](docs/). Use Git commit `358ef51` to consult the earlier synthesis and
training workflows, experiment configurations, and scripts referenced there.

## AHEAD QSM patch generation

Generate training and validation patches from the paired AHEAD tensors.

```sh
uv run dnqsm-generate-patches \
  --source-dir /WorkSpace/D/DN-QSM/data/ahead-original105 \
  --output-dir /WorkSpace/D/DN-QSM/artifacts/ahead-qsm-patches
```

The full dataset uses about 196 GiB for the patch arrays. The generator stages
the output and publishes it as a new directory when complete. Use `--workers`
to control CPU parallelism and `--seed` to reproduce the split and background
sources.

The generator applies the brain mask and keeps all 64³ crops on a six-voxel
stride with at most 20% exactly zero susceptibility voxels. Training includes the
original, positive-only scaling by four, and the sign inversion of each.
Validation uses the original susceptibility. Fields are simulated on the padded
full source volume before cropping. The manifest records the source split, crop
coordinates, and the four exterior susceptibility maps' parameters per subject.
The `qc/volumes` and `qc/patches` directories show the three tensor-axis center
slices in grayscale with a fixed display range of −0.1 to +0.1. Use
`uv run dnqsm-generate-patches --refresh-qc` to refresh the PNGs for the
existing dataset.

```python
from dnqsm.patch_data import make_patch_dataloader

loader = make_patch_dataloader(
    "/WorkSpace/D/DN-QSM/artifacts/ahead-qsm-patches",
    "train", batch_size=2, num_workers=2, seed=2026,
)
batch = next(iter(loader))
dipole = loader.dataset.dipole()
# With a DNQSM model on the same device:
# prediction = model(batch["total_field"], batch["mask"], dipole)
```

Every read selects one of the subject's four saved background fields with
replacement, including reads from the `validation` split. The sample reports
the selected `background_option`. Fields and susceptibility are in ppm,
the mask is binary, and batched volumes have shape `[N, 1, 64, 64, 64]`.
The patch-grid dipole is for DN-QSM inference and analytic initialization.
The stored local-field target includes contributions from susceptibility
outside the crop on the full simulation grid.

## Train and compare baselines

Train the direct total-field U-Net, the shared local-field estimator, and
the four correction networks on the materialized AHEAD patches.

```sh
uv run dnqsm-train-comparison \
  --patch-dir /WorkSpace/D/DN-QSM/artifacts/ahead-qsm-patches \
  --output-dir artifacts/baseline-training

uv run dnqsm-compare \
  --manifest artifacts/baseline-training/comparison.json \
  --cosmos-simulation /path/to/cosmos/run/simulation.pt \
  --data-dir /path/to/sim2/evaluation \
  --output-dir artifacts/baseline-evaluation
```

Model 1 directly predicts susceptibility. Model 2 applies clipped TKD,
Tikhonov, or full-dipole linear gradient descent after local-field estimation.
Model 3 learns a separate unrestricted correction for each inverse. Model 4
uses Tikhonov and applies DN-QSM's spectral weighting to the learned correction.
Models 2–4 share frozen local-field weights. Use the Tikhonov arms to compare
inversion alone, unrestricted correction, and spectrally weighted correction.

Training defaults to 10,000 updates per stage, with validation every 1,000
updates on fixed backgrounds. AHEAD validation determines the saved checkpoints.
Use `--device cpu --features 2 --steps 1 --batch-size 1` for a small test dataset.

Each training stage has a Comet experiment. A shared tag links the six stages,
and the local manifests record the experiment keys. Comet records relative MSE
at each optimizer update and subject-averaged NRMSE at each validation, including
the final update. Each validation also logs three central views of the same fixed
patch with total field, reference, and prediction on a common ppm scale.

Run `uv run comet login` or set `COMET_API_KEY` to upload automatically. To load
the key from a local file, set `COMET_API_KEY` in the Git-ignored `.env` and
start training with `uv run --env-file .env dnqsm-train-comparison` followed by
the usual arguments. The default project is `dn-qsm`. Set `COMET_WORKSPACE` or
`COMET_PROJECT_NAME` to choose the destination. Without credentials, or with
`COMET_START_ONLINE=0`, training saves archives under `<output-dir>/comet/`.
After logging in, upload an archive with `uv run comet upload <path-to-archive.zip>`.

Evaluation defaults to Gaussian 64³ CNN windows and full-grid inversion.
The evaluation `summary.json` reports each arm's scores and differences in
susceptibility NRMSE in percentage points. Negative differences mean lower error
for the later arm. Each case also includes per-arm figures, padded predictions,
and reconstructed maps in the source geometry.
