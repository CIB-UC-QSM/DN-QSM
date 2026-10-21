# Susceptibility augmentation plan

Status: implemented as an opt-in data policy, with A0–A3 screening configs, a
separate fixed shape-validation suite, and a paired data-only audit. The default
remains the existing pipeline. Full model screening and performance confirmation
have not been run.

The initial [implementation audit](../../artifacts/diagnostics/chi_augmentation_20261020/report.md)
used 192 paired draws from each of three sampler epochs (576 per policy; 2,304
augmented volumes across A0–A3). There were no shape-remapping fallbacks. The
RMS-normalized chi JS distance to sim2 was 0.309 for A0 and 0.226 for the bimodal
profile; to COSMOS it was 0.122 for A0 and 0.076 for the central-tail profile.
These are profile-specific coverage comparisons. The full A3 mixture does not
match either external histogram as closely as its corresponding profile, and
these results do not establish improved reconstruction performance.

## Evidence and objective

The [histogram analysis](../../artifacts/diagnostics/domain_histograms_20261020/report.md) used all 1,236 training records and a full realization of sampler epoch 55 from the latest run.

| Quantity | Current augmented training | COSMOS | sim2 |
|---|---:|---:|---:|
| Image-equal susceptibility RMS, ppm | 0.03030 | 0.02800 | 0.02824 |
| Image-equal clean-local RMS, ppm | 0.00815 | 0.00846 | 0.00864 |
| Negative susceptibility fraction | 51.4% | 54.4% | 42.0% |
| Physical susceptibility JS distance to training | — | 0.121 | 0.326 |
| Centered/std susceptibility JS distance to training | — | 0.116 | 0.303 |

The main missing coverage is susceptibility histogram shape. sim2 has prominent modes near -0.0353 and +0.0163 ppm; its discrepancy survives centering and scaling. COSMOS has a smoother central peak with asymmetric tails. Both external RMS values are already within the training distribution. The objective is to represent both shapes through a mixture of training examples, rather than force the pooled training histogram to equal either external histogram.

Clean local fields in this comparison are forward modeled. sim2 has no independent clean-local reference in the current evaluation loader. These measurements do not establish that histogram shift causes the model's external errors.

## 1. Keep amplitude coverage and original examples

Keep the current target-RMS law: 95% log-uniform 0.018–0.040 ppm and 5% log-uniform 0.040–0.080 ppm. Keep the 75% anatomical/original and 25% geometric/synthetic sampler for the first experiments.

Proposed initial profile proportions are hypotheses for ablation, not measured optima:

| Training draw | Overall probability | Modification |
|---|---:|---|
| Existing geometric/synthetic map | 25% | Existing shape, current amplitude augmentation |
| Existing anatomical/original map | 25% | Existing shape, current amplitude augmentation |
| Anatomical/original map with bimodal contrast | 25% | Random smooth intensity remapping into a two-mode family |
| Anatomical/original map with central peak and variable tails | 25% | Mild random smooth intensity remapping into a single-dominant-mode family |

Implementation: leave synthetic maps in their existing shape branch; give original maps three equally probable profiles. This keeps approximately half of all draws at their existing susceptibility shape. Apply shape augmentation to original maps initially because monotone remapping of a few geometric intensity plateaus cannot reproduce realistic tissue distributions. It also avoids modifying the generator in the sibling reference repository.

## 2. Add randomized, smooth monotone susceptibility remapping

The paired training inputs currently provide susceptibility and binary ROI masks, without anatomical tissue labels. Use intensity-distribution remapping first; do not interpret susceptibility sign or histogram clusters as GM/WM segmentation.

For each selected original map, estimate its masked CDF F from a fixed-size random voxel sample. Draw a target distribution G from a parameterized family and construct a smooth increasing approximation to:

`T(x) = inverse_CDF_G(CDF_F(x))`

Blend it with the original map:

`chi_shape = mask * ((1 - alpha) * chi + alpha * T(chi))`

Use quantile knots or a lookup table, consolidate duplicate source knots, maintain positive slopes, and use continuous affine tail extensions rather than clipping. Limit excessive slope amplification; measure the resulting spatial gradients before accepting parameter ranges. Handle almost constant maps with an explicit fallback to the baseline profile and log the fallback rate.

This preserves voxel locations and intensity ordering. It changes regional contrasts and apparent intensity fractions, but does not create new anatomy or guarantee realistic tissue identity. Anatomical tissue-proportion changes require independent segmentations and should be a later experiment.

### Initial target-family proposals

These ranges are starting proposals informed by the observed shapes. They are not published biological tissue limits and must be checked on the final augmented maps.

| Profile | Initial parameters before final RMS scaling | Blend alpha |
|---|---|---:|
| Bimodal | Negative mode center -0.045 to -0.025 ppm; positive mode +0.008 to +0.025 ppm; within-mode std 0.003–0.012 ppm; nominal negative-component weight 0.30–0.50; optional positive shoulder/tail mass 0.5–4% | 0.7–1.0 |
| Central peak with variable tails | Dominant central component near zero, std 0.014–0.026 ppm; positive shoulder mass 1–5% with center +0.07 to +0.16 ppm; weaker negative shoulder mass 0–3% with center -0.12 to -0.05 ppm | 0.2–0.6 |

Normalize component weights to sum to one. Randomize locations, widths, weights, and blend strength jointly. Include overlapping as well as well-separated modes. Target-family means follow the sampled mixture; do not add a separate mandatory zero-mean operation.

Final RMS scaling changes physical mode positions. Consequently, nominal component parameters cannot be claimed as final peak bounds, and component weight is not identical to the measured fraction of negative voxels. Audit both after every processing step. Retune the broad family ranges on development data if they fail to span the desired shapes.

Do not perform exact histogram matching to either external volume, load their voxel values in the augmentation code, or independently alter chi and its field labels.

## 3. Apply transformations on the final spatial grid and regenerate fields

Proposed processing order for the new policy:

1. Load read-only source chi and mask.
2. Apply existing flips/permutations and crop/resize.
3. Apply the selected shape profile inside the final mask.
4. Sample the existing target RMS and scale the final masked chi once to that value.
5. Compute the clean local field from this final chi with the actual training dipole on the final grid: `local = mask * D(chi * mask)`.
6. Generate backgrounds and noise using the existing synthesis configuration, with noise scaled from the new clean local-field RMS.

Current RMS scaling precedes interpolation, so cropped/resized maps can lose RMS afterward. Moving the RMS operation to the final grid is a separate change and needs its own control. Keep the old code path exactly available when the new policy is off.

Mean changes introduced by a shape profile are part of a new synthetic susceptibility source, not a harmless change of output reference. Recomputing its field is necessary under a finite mask. Do not globally clamp susceptibility values or add independent per-voxel impulses.

Keep the architecture, Laplacian settings, losses, optimizer, background/noise profile, and crop probability fixed during the chi-policy comparison.

## 4. Inspect spatial statistics and rare sources

Before training, compare physical and normalized chi/local histograms, per-image quantiles, negative fractions, tail energy, spatial-gradient distributions, and radial power spectra. Examine representative slices and their simulated fields.

Also compare `RMS(local) / RMS(chi)`. This ratio is insensitive to a global chi multiplier and captures differences missed by susceptibility RMS alone. Current augmented median ratios are approximately 0.296 for qsm, 0.276 for separate, and 0.193 for geometric/synthetic maps, versus 0.302 for COSMOS and 0.306 for sim2. Changing chi scale alone cannot fix an incorrect ratio; contrast arrangement and spatial structure matter. This ratio is diagnostic, not a per-image normalization target.

sim2 contains 37 voxels above |chi|=0.5 ppm, contributing about 4.97% of its chi squared energy, with a minimum near -2.034 ppm. Audit the source of these values before designing an extreme-source branch. Preserve and report existing tails; do not force every training map to contain such extremes.

If a separate rare-source experiment is justified, add compact, connected, smoothly bounded positive/negative inclusions inside suitable masks, then synthesize their full dipole fields. Vary their sizes and amplitudes using a separately specified development distribution. Do not combine this with the first histogram-shape experiment or normalize away the intended inclusion amplitude without recording its final value.

## 5. Run a small sequence of controlled experiments

First do a data-only audit using at least three fixed sampler epochs, keeping image-equal weighting and reporting every profile and source cohort separately. Report per-image and profile-mixture distances as well as pooled distances: an average of two distinct target shapes need not resemble either individual target.

Then use matched short training screens:

| Variant | Change relative to the previous control |
|---|---|
| A0 | Exact current pipeline |
| A1 | Final-grid RMS scaling only |
| A2 | A1 plus bimodal profiles on one third of original draws; 75% of all draws retain existing shapes |
| A3 | A1 plus both shape families; proposed overall 50/25/25 existing/bimodal/central-tail mixture |

Use the same initialization checkpoint, subject split, source draws, model settings, optimizer settings, and optimizer-update budget for all variants. Use independent deterministic augmentation streams so new shape draws do not perturb flips, crop choices, source sampling, or background/noise seeds. Record any data-policy curriculum explicitly rather than change it mid-comparison.

Select candidates on fixed held-out development subjects, using both their existing fixed views and a separately defined shape-stress suite. Keep checkpoint selection on the fixed development objective rather than repeatedly select on external benchmarks. Evaluate chi/local NRMSE, bias, regression slope, HFEN or edge error, and tail/region errors where valid references exist. Assess tradeoffs for both external shapes instead of selecting by average histogram distance alone.

Repeat only the best candidate and its control with at least three seeds before a full training run. If helpful after this comparison, test the winning policy at 25% versus 10% synthetic sampling; do not change the source mixture at the same time as introducing chi remapping.

### Decision criteria

- Data invariants pass: finite FP32 outputs, zero outside masks, valid masks, repeatable augmentations, and local-field targets consistent with the final chi and grid.
- Final chi RMS follows the configured law; crop/resize does not silently shift its scale.
- Both observed shape types have meaningful support in the generated profile families; their negative fractions, quantiles, and normalized shape descriptors are represented across multiple source subjects, rather than by a single unusually close generated image.
- Spatial-gradient/tail diagnostics and visual inspection show no remapping-induced quantization, isolated impulses, or systematic disappearance of existing high-contrast structures.
- Held-out reconstruction improves beyond seed variability with no material deterioration in baseline development cases; define the acceptable regression margin from control-run variability before choosing a winner.
- An improved histogram score by itself is insufficient for promotion.

Since the policy ranges are informed by sim2/COSMOS susceptibility labels, these datasets become development-informed benchmarks. An untouched dataset is required for an unbiased final claim of external generalization. Their data do not enter training as source images.

## 6. Implementation scope

- `src/dnqsm/data.py`: versioned chi-policy configuration, deterministic shape remapping, final-grid RMS option, metadata, and baseline fallback. Preserve read-only source files and unchanged cache semantics.
- `src/dnqsm/training.py`: validate and serialize profile probabilities/ranges, pass them to training only, and leave existing fixed validation views unchanged. Define the optional stress suite separately.
- `scripts/analyze_domain_histograms.py`: add candidate-policy and profile-specific comparisons, shared histogram bins for direct policy comparisons, spatial diagnostics, and final rather than nominal augmentation parameters.
- `tests/test_data.py` and a focused chi-augmentation test module: test disabled-policy compatibility, determinism across workers/epochs, final-grid RMS, ordering/continuity of remapping, constant-map fallback, mask invariants, and paired-field consistency. Use existing crop/balanced-training tests to check integration.
- `configs/`: explicit A0–A3 configurations for reproducible comparisons. Choose a default only after the development criteria pass.

## Research context

The proposed profile families and parameter ranges above are hypotheses derived from this repository's diagnostics. They are not implementations validated by the following papers.

- [QSMnet+](https://arxiv.org/abs/1909.07716) demonstrates that susceptibility-range augmentation can improve quantitative behavior outside the original training range. Here, global RMS coverage already exists, so shape is the primary next variable.
- [Generalization of deep learning-based QSM by expanding the diversity of spatial gradient in training data](https://cds.ismrm.org/protected/21MProceedings/PDFfiles/3977.html), an ISMRM conference study, reports sensitivity to spatial-gradient differences and improvement with gradient-diverse training. This motivates checking spatial statistics alongside intensity histograms; it does not validate the specific remapping design.
