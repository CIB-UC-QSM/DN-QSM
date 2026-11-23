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
