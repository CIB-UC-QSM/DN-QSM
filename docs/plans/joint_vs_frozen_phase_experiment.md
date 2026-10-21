# Small experiment: joint training versus a frozen phase model

Status: implemented and executing. The long run was stopped cleanly at step
2,840. The strict shared-pretraining design is being run under
`artifacts/experiments/joint_vs_frozen_phase_20261020_162750/`:
20 shared phase-refinement epochs followed by paired 390-update joint/frozen
branches. Deep Null is freshly initialized once per paired seed. A difference
of at least 1 percentage point in the first pair triggers the two additional
predeclared paired seeds. The runner is `dnqsm.staged_experiment`; it writes
`summary.json`, `report.md`, and `learning_curves.png` automatically.

Validation: 101 regression tests passed, and a tiny mixed-precision GPU fixture
completed both branches with identical data draws and zero frozen-phase probe
drift. The fixture is an implementation check, not a scientific result.

## Question and outcomes

Does allowing susceptibility and physics supervision to update the pretrained
phase/local-field network worsen held-out susceptibility reconstruction,
compared with training Deep Null on the same network's frozen predictions?

Measure local-field accuracy separately from final susceptibility accuracy. A
worse intermediate prediction need not mean a worse final reconstruction.
Prespecify final susceptibility NRMSE as the primary outcome; local-field NRMSE
is an important secondary outcome.

## Existing observations, not a controlled comparison

Source: `artifacts/runs/dnqsm_chi_A3_pretrain20_e2e100_20261020_114740/metrics.jsonl`.

| Checkpoint position | Local validation NRMSE | Susceptibility validation NRMSE |
|---|---:|---:|
| End of 20 local-field pretraining epochs | 31.20% | Not recorded in the pretraining validation pass |
| After 1 joint epoch | 31.99% | 49.75% |
| After 2 joint epochs | 32.07% | 49.41% |
| After 15 joint epochs | 30.16% | 46.91% |

There is an initial local-field regression followed by recovery. These values
do not show that joint training is worse than frozen-phase training; no frozen
control was run. Twelve sampled joint-training filter-gradient diagnostics had
positive pairwise cosine similarities for local/chi, physics/chi, and
physics/local objectives. Consequently gradient conflict is a hypothesis, not
an observed explanation in those samples.

## Minimum paired experiment

Create one immutable common checkpoint C containing the phase weights F0 and
the initial Deep Null weights N0. Evaluate C before either arm trains.

| Arm | Phase network | Deep Null | Backpropagation to phase network |
|---|---|---|---|
| A: joint | Trainable | Trainable | Local, susceptibility, and physics objectives |
| B: sequential/frozen | Fixed at F0 | Trainable | None |

Both arms start with exactly the same F0 and N0. In B, Deep Null consumes the
frozen model's **predicted** local fields, not ground-truth local fields.
Training B on clean ground-truth fields would introduce a different input
distribution and answer a different question.

Choose the starting checkpoint according to the question:

- **Cheapest continuation pilot:** freeze a copy of the current checkpoint,
  including both networks, and fork A/B from it. This tests whether continuing
  joint training is better or worse than freezing the phase network **now**.
  It cannot establish which strategy would have been better immediately after
  the original 20 pretraining epochs.
- **Strict two-stage comparison:** train/refine the phase network for 20 epochs
  once, save `phase_pretrain20.pt`, then fork A/B. Initialize Deep Null once per
  paired seed and copy that same initialization into both arms. A fresh Deep
  Null initialization tests learning the second stage; a previously trained
  initialization tests adaptation and must be labeled accordingly.

The current run did not retain a distinct epoch-20 checkpoint: its latest files
have been overwritten by joint training. Its pretraining filter cannot be
recovered from scalar logs or from today's joint-trained weights. A strict
comparison therefore requires reconstructing the shared pretraining stage from
the original initialization/configuration, or saving the boundary in a future
run. Reproduction need not be bitwise identical on CUDA; the essential control
is for both new arms to share the same newly saved boundary checkpoint.

## Small budget and fixed controls

Run one A/B pair for **390 optimizer updates each**, approximately five complete
epochs of the current 1,236-record dataset at batch size 16. Evaluate at updates
0, 78, 156, 234, 312, and 390. Use the final fixed-budget endpoint for the primary
comparison; record the learning curves to distinguish transient regressions
from persistent differences. A short pilot cannot prove the asymptotic winner.

Keep the following matched:

- Architecture, TKD threshold 0.1, Laplacian precision/scaling, and filter output
  skip; A3 susceptibility augmentation and 75/25 original/synthetic sampling.
- The exact persisted subject split, training draw order, augmentation seeds,
  crop choices, synthesized backgrounds/noise, and validation views.
- Batch size 16, accumulation 1, AdamW learning rate 3e-5, weight decay 1e-4,
  precision, and normalization/loss weights (local 1.5, chi 1, physics 1).
- Fresh AdamW states for both arms at the fork, including identical Deep Null
  state initialization. Do not retain historical optimizer moments in only A.
- A common short schedule: 20-update linear warmup to 3e-5, then constant rate
  through update 390. The existing three-epoch warmup would consume most of this
  five-epoch pilot. This deliberate pilot schedule differs from the long run.
- Clip gradient norms **per network** at 10 in both arms. With global clipping,
  A's filter gradients can change the effective Deep Null update magnitude
  while B has no filter gradients; that would complicate interpretation.
- Hold background curriculum progress at 1 rather than restart it in one arm.
  Use the same data epoch offsets in both arms (20 onward for the strict design;
  a declared common offset for the continuation pilot).

At the observed roughly 7.6 seconds per joint update, A's 390 updates take about
50 minutes before validation and other overhead. Measure B's actual timing;
do not equate faster updates with a different update budget. The strict design
adds the cost of the shared 20-epoch phase stage; the continuation pilot does not.

If the pilot shows a meaningful difference, repeat with **three paired seeds**.
Keep the subject split fixed: the current global `TrainConfig.seed` also changes
subject splitting, so changing that flag alone is not a valid seed replicate.
Use a dedicated study runner that loads the persisted split and varies declared
optimization/data seeds within it, identically for each A/B pair. With a common
pretrained checkpoint, seed replicates should vary the training draws/augmentations
or the shared Deep Null initialization rather than repeat an otherwise
deterministic identical trajectory.

## Metrics and interpretation

Evaluate all held-out development subjects, averaging each subject's fixed views
before aggregating subjects. Keep ordinary validation and shape-stress results
separate. Report COSMOS and sim2 individually at the same evaluation checkpoints,
without selecting endpoints or hyperparameters using their scores.

Collect:

1. **Primary:** final susceptibility NRMSE per subject and its paired A-minus-B
   difference, in percentage points. Positive differences mean joint is worse.
2. Local-field NRMSE, TKD-only susceptibility NRMSE, and final-minus-TKD improvement.
3. Susceptibility bias and regression slope; an edge-sensitive error such as
   HFEN; errors within positive/negative susceptibility and high-|chi| regions.
4. Full-grid Fourier errors split into retained `|D| > 0.1` and near-null
   `|D| <= 0.1` bands. Apply these operators on the actual padded FFT grid before
   optionally reporting errors in the ROI.
5. Phase-output drift from F0 on the same inputs, phase parameter drift, weighted
   filter-gradient norms/cosines, and clipping frequency. Verify B's phase
   weights and predictions remain unchanged within the declared numerical tolerance.

In this implementation Deep Null's correction is projected into `|D| <= 0.1`.
It cannot change the final susceptibility's retained-frequency component on the
full FFT grid. Therefore phase/TKD errors in the retained band are a useful
mechanistic check; the spatial ROI mask does not preserve that orthogonality.
sim2 has no independent clean-local reference in the loader; use its chi metrics,
and label any susceptibility-derived local reference as a forward-model proxy.

Prespecify **1 percentage point** in mean chi NRMSE as a candidate practical
effect threshold before running the pilot. It is an experimental choice, not
a universal significance threshold. Report paired subject-level bootstrap
intervals for each seed and the effects across paired seeds. Do not treat
voxels, repeated views, or all subject/seed pairs as independent samples.

| Observation | Conclusion supported |
|---|---|
| A has worse chi NRMSE than B at the matched endpoint, consistently across seeds | Evidence that joint training is detrimental under these settings and this budget |
| A has worse local-field NRMSE but better chi NRMSE | Intermediate physical accuracy trades off against better final reconstruction |
| A degrades on sim2/COSMOS while development chi improves | Possible external-domain generalization cost; not a general failure of joint training |
| A initially regresses but recovers | Short-lived optimization transient; assess the prespecified endpoint |
| Differences are small or seed dependent | Inconclusive; do not declare freezing superior |

If the methods have different convergence rates, extend both arms by the same
budget before drawing a long-run conclusion. A superior frozen arm would not
prove that all end-to-end methods are worse: phase learning rate, objective
weights, and optimization schedule remain possible explanations.

## Optional follow-up only if needed

- **Decoupled arm:** train phase with local-field supervision only, detach its
  predictions before TKD/Deep Null, and train Deep Null with chi/physics losses.
  This separates downstream-loss feedback from the effect of a moving phase model.
- **Lower phase learning rate:** repeat joint training with phase LR 3e-6 and
  Deep Null LR 3e-5 to test whether the problem is update magnitude.
- **Crossed evaluation:** evaluate F0 with A's final Deep Null, and A's final
  phase model with B's Deep Null. This is a diagnostic of coadaptation; the
  crossed models have different input distributions from their training and
  must not replace the trained-pair primary comparison.

## Implementation needed before running

A dedicated small-experiment runner should save C and both endpoints, freeze
`model.filter.requires_grad_(False)` and call `model.filter.eval()` after
`model.train()` in B, detach its local output before the second stage, and
optimize only Deep Null parameters. Keep the frozen forward pass outside
autograd. A zero local-loss weight alone does not freeze phase, because chi and
physics gradients still pass through TKD. Excluding its parameters only from
the optimizer while retaining gradients would also confound global clipping.

Keep data draws, optimizer resets, per-network clipping, and update-based
scheduling explicit in the study manifest. Do not change the ongoing training
job or use its evolving weights as one arm of an independently restarted pair.

## Research context

[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
establishes that interference between objectives can be detrimental under
particular optimization conditions. It motivates diagnostics, not a diagnosis
of this run. [NeXtQSM](https://arxiv.org/abs/2107.07752) demonstrates a joint QSM
pipeline with a different inversion architecture, so joint training should not
be assumed inferior in general. Neither paper determines the result for this
Deep Null implementation; that is the purpose of the paired experiment.
