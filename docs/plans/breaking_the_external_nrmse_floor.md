# Breaking the external susceptibility NRMSE floor

Status: Stages 0, A and B implemented and measured; Stage C gates complete and
attributed. The recommended configuration is `configs/fullgrid_100epochs.json`.
A from-scratch training run, the only test of generalization, has not been run.

Seven full runs spanning three architectures, two loss versions and three TKD
thresholds all stopped between 48% and 54% COSMOS susceptibility NRMSE. The
measured diagnosis is in
`artifacts/diagnostics/plateau_rootcause_20261021/report.md`. This note records
the experimental controls for the changes it motivated, so each contribution is
attributable rather than bundled.

## Error budget being spent

Measured on COSMOS with the real operators, no learned weights except where
stated. Each row idealizes one more stage of the pipeline.

| Scenario | χ NRMSE |
|---|---:|
| Unmasked full-grid ground-truth field + oracle null correction | 0.00003% |
| **Masked** ground-truth field + oracle null correction | 15.31% |
| Masked ground-truth field + the trained null network | 22.39% |
| Trained filter end to end (local NRMSE 46.74%) | 55.34% |

Two measurements set the strategy. The filter's failure is a function of
background amplitude and not of background character: at COSMOS's own
background/local ratio of 22, a synthetic background is as hard (43.71–49.60%
local NRMSE across three held-out subjects) as the real one (46.74%). And the
exterior field only has to be approximately right — at τ = 0.2 an 80% relative
error in the exterior still yields a 0.90% oracle floor, against 7.51% when the
exterior is zeroed. Unmasking buys most of its headroom cheaply.

## Stage A — background regime

Config-only. Both arms start from the same stopped checkpoint
(`dnqsm_depth6_…_tkd005`, step 10440), run 400 filtering-only updates with fresh
AdamW state, and are scored on the **same** validation suite at the deployment
regime. Only the training mixture differs. The primary readout is
`bucket_strong_local_nrmse`; aggregate χ is not informative here because the
masking floor still caps it.

| Arm | clean | weak | strong | strong_min_ratio |
|---|---:|---:|---:|---:|
| baseline | 0.50 | 0.10 | 0.05 | 0.001 |
| regime | 0.05 | 0.05 | 0.40 | 0.005 |

### Result: coverage fixed, external effect not demonstrated

| Readout | baseline | regime |
|---|---:|---:|
| validation strong-bucket local NRMSE, epoch 3 | 49.37% | **45.51%** |
| validation overall local NRMSE, epoch 3 | 37.64% | **35.36%** |
| COSMOS local NRMSE | 47.43% | 47.55% |
| COSMOS χ NRMSE | 56.26% | 55.65% |

The synthetic validation favours the new mixture by 3.86 points on the strong
bucket, and the baseline arm *degrades* between epochs 2 and 3 (47.36 → 49.37)
while the regime arm holds. On COSMOS the two arms are indistinguishable.

Both arms also sit about five points below the checkpoint they started from
(COSMOS local 42.39%), because a fresh AdamW state and a restarted warmup
displace the weights faster than 400 updates can recover. That budget is
therefore dominated by re-equilibration rather than by adaptation, and the
trial is **inconclusive on external performance** in both directions.

What it does settle: the previous training distribution did not contain the
benchmark's regime in the metric the filter consumes, and the new one does.
What it does not settle: whether training on that regime improves external
reconstruction. The earlier estimate that this was worth roughly 25 points was
an inference from the difficulty curve — local NRMSE as a function of
background amplitude — not a measurement of the intervention, and this trial
does not support it. A from-scratch comparison is the fair test.

Note that the unmasking change in Stage B does not depend on this question: its
justification is an operator measurement taken with the exact ground-truth
field, independent of any training distribution.

The realized mixture is audited before any GPU time with
`scripts/audit_background_regime.py`. The amplitude fractions are not the
quantity that matters: the filter's only input is a Laplacian, and the
synthesized exterior sources are much smoother than real ones, so the audit
reports the ratio of the background's interior Laplacian RMS to the clean
field's. Under the previous profile its median was 0.015 and its p90 was 0.126,
while COSMOS sits at 0.357 — outside the training distribution entirely. Under
the regime profile the median is 0.086 and p90 is 0.576, which covers it.

## Stage B — unmasking, and an objective that matches the inversion

Three changes, each independently switchable:

1. **Exterior supervision actually reaches the filter.** `_predict_training_batch`
   accepted `include_extension` but its filtering-only branch returned only the
   interior terms, and χ branches zero the extension weight. Under the
   alternating schedule that meant `full_grid_field` with `lambda_extension > 0`
   supervised the exterior on no batch at all. `local_field_loss_terms` now owns
   the term and both paths share it.
2. **`full_grid_field` with `filter_input_version=1`**, which also gives the
   filter the normalized field as a second input channel. The existing
   output-skip path does not substitute for this: its raw-field coefficient
   measured 2.8e-5 after 75 epochs, having been initialized at zero and never
   moved, so the filter has been reconstructing an absolute field by
   integrating a Laplacian alone.
3. **`--lambda-local-tkd`**, penalizing TKD of the *predicted* field against
   ground-truth susceptibility. The filter's error is 18× over-represented,
   relative to signal, in the |D| < 0.05 band that the inversion multiplies by
   up to 1/τ, while its objective weights every frequency equally.

Objective version 6. Every default is unchanged, so existing configs and
checkpoints behave exactly as before.

### Result: unmasking works; the input change it was coupled to does not

Four fixed-COSMOS fits, τ = 0.05, 300 steps, identical learning rate and width,
each arm changing one thing from the one above it.

| arm | field to TKD | filter input | `local_tkd` | local | TKD | χ |
|---|---|---|---:|---:|---:|---:|
| control | masked | v2 Laplacian-only, cond 32 | 0 | 26.68% | 41.87% | 33.23% |
| A | masked | v1 two-channel, cond 0 | 0 | 33.45% | 47.86% | 41.65% |
| B | full grid | v1 two-channel, cond 0 | 0 | 28.46% | 44.12% | 37.12% |
| C | full grid | v1 two-channel, cond 0 | 1 | 28.36% | 42.27% | 36.19% |

| attributed change | local | TKD | χ |
|---|---:|---:|---:|
| input v2+conditional → v1 two-channel | +6.77 | +5.99 | **+8.41** |
| mask removed before TKD | −4.99 | −3.75 | **−4.53** |
| TKD supervision added | −0.10 | −1.85 | −0.93 |

The deltas sum to +2.95 against a measured C − control of +2.96, so the
decomposition is self-consistent. Unmasking is worth 4.5 points of
susceptibility NRMSE and, unexpectedly, 5.0 points of *local-field* NRMSE —
exterior supervision appears to regularize the interior fit rather than only
extending it. TKD supervision is worth about a point, concentrated where it
aims: 1.85 points on the TKD output itself.

The version-1 two-channel input costs nearly twice what unmasking gains. This
refutes the diagnosis's reading of the dead raw-field skip coefficient
(2.8e-5 after 75 epochs) as evidence that the filter needed the field as an
input. The optimizer had driven that coefficient to zero because on a strong
background the raw field is mostly background; a deliberate optimum was
misread as a pathology.

### Result: the interventions overlap, and tau = 0.05 wins

With the coupling relaxed, arm E takes the unmasked output while keeping the
Laplacian-only input and conditional scaling. Arm F is arm E at tau = 0.2.

| arm | local | TKD | chi | null network closes |
|---|---:|---:|---:|---|
| control | 26.68% | 41.87% | 33.23% | 8.64 pts (21% of TKD) |
| **E** (tau 0.05) | **23.95%** | **37.94%** | **31.35%** | 6.59 pts (17%) |
| F (tau 0.20) | 23.80% | 57.92% | 34.78% | 23.14 pts (40%) |

Arm E is the best configuration measured: 1.89 points of susceptibility and
2.73 points of local-field NRMSE below the control, with the best field
estimate of any arm.

**The deltas do not compose.** Unmasking plus TKD supervision measured 5.46
points on top of the version-1 input and delivered 1.89 on top of version 2
with conditional scaling. Both address the same underlying problem — the filter
not knowing the field's absolute scale — so they overlap rather than add.
Expect the same discount on anything further stacked here.

**tau = 0.05 is the answer at this field accuracy.** This is the fallback
declared before the arms ran, and the evidence meets its condition. tau = 0.2
does what the architecture's premise predicts, recovering 23.14 points through
the null network against 6.59 at tau = 0.05, but it starts from a TKD 20 points
worse and does not catch up. Local NRMSE is indifferent to tau (23.80 against
23.95), as expected. Record that the (I - B_tau A)N premise is measurably
active and not yet strong enough to pay for the larger null band.

Nothing physical required the two to be bundled. `ModelConfig` rejected
`full_grid_field` under `filter_input_version=2`, so unmasking could only be
reached by also adopting the worse input. Input preprocessing and output
masking are separable — the filter receives the Laplacian of the masked field
either way, and `full_grid_field` only decides whether its output is truncated
before TKD — so that rule is now relaxed and the combination is tested.

## Stage C — gate, ablate, commit

1. `dnqsm-overfit-cosmos` on the new architecture at τ = 0.2. It trains the
   complete model from the first update, so the new terms are exercised. No
   long run until its gate passes.
2. Four matched short arms, one seed each, fresh weights (the input-channel
   change forbids initialization from any existing checkpoint):

   | Arm | mixture | full grid | `lambda_local_tkd` | τ |
   |---|---|---|---:|---:|
   | A | regime | no | 0 | 0.05 |
   | B | regime | yes | 0 | 0.20 |
   | C | regime | yes | 1 | 0.20 |
   | D | regime | yes | 1 | 0.05 |

   B − A isolates unmasking, C − B the TKD objective, C − D the threshold.
3. Re-profile VRAM: two input channels and an unmasked field change the
   activation footprint.
4. Long run on the winning arm, 10 pretraining plus 100 alternating epochs.

**Declared fallback.** If arm C's null network still closes under 40% of the
oracle gap, τ = 0.2's advantage does not materialize — TKD alone at τ = 0.2 is
53.62% even with a perfect field — and arm D is the correct choice. Record in
that case that the (I − B_τA)N premise is not being realized by this null
network, rather than continuing to raise τ.

## Limitations

One volume per external benchmark, so nothing here establishes a
population-wide effect; the held-out controls used three subjects. The oracle
floors bound what the current operator composition can reach, not what
alternative near-null coefficients could achieve inside the mask. Dropping
conditional Laplacian scaling is a real trade: it improved held-out χ from
64.75% to 58.08% in the October adaptation trials, and arm B against the Stage A
baseline is what measures whether the second input channel replaces it.
sim2 is reported with its 51% forward-consistency caveat and does not select
checkpoints.
