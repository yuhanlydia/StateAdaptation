# Task subspace and actuation-site diagnostics

These experiments use query labels to analyze mechanism.  They are explicitly
**oracle diagnostics**, not valid test-time adaptation estimates and not part
of the clean main comparison.  All runs use InternVL3-8B, rank 16 Full
controllers, layers 14 and 27, and the fixed seed-0 support/query manifests.

The diagnostics test three hypotheses:

1. whether a basis estimated from support captures query correctness-gradient
   energy;
2. whether replacing the support basis with a query-label oracle basis closes
   the LoRA gap; and
3. whether residuals on Q or O, alone or together with K/V, reveal an
   actuation-site limitation.

For support basis `B_s` and query gradient second moment `C_q`, the reported
energy overlap is

```
rho = sum_m tr(B_s,m^T C_q,m B_s,m) / sum_m tr(C_q,m),
```

where the sum is over the selected layers and projection modules.

## Cross-task seed-0 results

| Dataset / site | support-to-query rho | Macro-F1 | NLL |
|---|---:|---:|---:|
| Camelyon17 / KV | **0.9362** | 0.5994 | 0.6629 |
| Camelyon17 / Q | 0.5942 | 0.5125 | 0.7057 |
| Camelyon17 / O | 0.4884 | 0.4172 | 0.7590 |
| Camelyon17 / QKVO | 0.7786 | **0.6664** | **0.6333** |
| RoboFail / KV | **0.7441** | 0.4389 | **0.7219** |
| RoboFail / Q | 0.5039 | **0.4811** | 0.8908 |
| RoboFail / O | 0.2876 | 0.4506 | 0.9488 |
| RoboFail / QKVO | 0.6077 | 0.4538 | 0.7446 |
| ManipBench droid-pick-place / KV | **0.7242** | 0.2193 | 1.4898 |
| ManipBench droid-pick-place / Q | 0.4873 | 0.1548 | 1.8415 |
| ManipBench droid-pick-place / O | 0.3306 | 0.1631 | 1.8497 |
| ManipBench droid-pick-place / QKVO | 0.6122 | **0.2625** | **1.4610** |

The Camelyon run uses the formal four-step learning-rate-0.05 setting.  The
RoboFail run uses the eight-step learning-rate-0.01 follow-up.  ManipBench uses
its formal eight-step learning-rate-0.05 follow-up.  Values should therefore be
interpreted within dataset; the projection-site comparison is controlled
within each row block.

## Query-basis oracle

The query-label oracle changes only the basis: controller coefficients are
still fitted on the original support set and have the same rank and scalar
count.

| Dataset | support-basis KV F1 | query-basis KV F1 | Change |
|---|---:|---:|---:|
| Camelyon17 | 0.5994 | **0.6632** | +0.0638 |
| RoboFail | **0.4389** | 0.4119 | -0.0270 |
| ManipBench droid-pick-place | 0.2193 | **0.2482** | +0.0289 |

This falsifies the strongest version of the subspace-estimation-only
explanation.  Better query geometry helps Camelyon and slightly helps
ManipBench, but the ManipBench oracle remains far below the formal LoRA mean
of 0.4304.  RoboFail becomes worse with the query basis.  Therefore the
remaining gaps cannot be attributed only to `B_s != B_q`.

## Guardian overlap boundary check

The same full-query KV overlap diagnostic on the three Guardian seed-0 folds
gives:

| Split | KV rho | KV Macro-F1 |
|---|---:|---:|
| UR5-Fail | 0.6862 | 0.4601 |
| RoboFail | **0.7441** | 0.4389 |
| RoboVQA execution | 0.6868 | **0.5128** |

Aggregate `rho` does not rank performance within Guardian: RoboFail has the
highest overlap but the lowest F1 of these three diagnostic runs.  The proposed
claim that overlap alone predicts adaptation gain is therefore not supported.
Per-module results point to a more specific issue: on RoboFail, layer-14 K
overlap is 0.8564 while layer-14 V overlap is only 0.3940.  Future analysis
should retain module-wise geometry and class-conditional gradients instead of
compressing them to one scalar.

## Interpretation

The results support a narrower mechanism story:

- Camelyon, where the target capability is already present and the shift is
  largely visual evidence interpretation, has very high KV transfer.  A query
  oracle improves it, and QKVO gives a further bounded diagnostic gain.
- On ManipBench, query-basis KV remains far below LoRA.  QKVO improves over KV
  by 0.0432, showing some actuation-site mismatch, but it does not close the
  much larger gap.  This is consistent with a task/decision-function and
  capacity mismatch rather than an optimization-only failure.
- RoboFail is not explained by low aggregate overlap.  Q-only is strongest on
  seed 0, whereas Q-only is harmful on ManipBench.  A universal switch from KV
  to Q is therefore not justified.

These are seed-0 mechanism diagnostics selected with query information.  They
must be replicated on held-out support seeds before supporting a statistical
claim.  In particular, the QKVO rows are diagnostic upper-bound candidates,
not a replacement for the preregistered KV method.

## Reproduction

The reusable runner is `scripts/diagnose_task_subspace.py`.  Example:

```bash
PYTHONPATH=src python3 scripts/diagnose_task_subspace.py \
  --family internvl3 \
  --model-id artifacts/models/InternVL3-8B-Instruct \
  --support data/prepared/guardian_execution/robofail/seed_0/support.jsonl \
  --query data/prepared/guardian_execution/robofail/seed_0/query.jsonl \
  --output runs/diagnostics/robofail_seed0_full.json \
  --sites KV Q O QKVO --steps 8 --learning-rate 0.01
```

Large covariances and prediction artifacts remain under ignored `runs/`.

## Module-wise signed and class-conditional geometry

The energy overlap above is unsigned.  We therefore also compute the cosine
agreement between support and query mean correctness gradients after projecting
both into the support-derived basis:

```
kappa = cos(P_Bs mean(G_support), P_Bs mean(G_query)).
```

Positive `kappa` means that support and query request the same correction in
the transferred subspace; negative `kappa` means that the subspace contains
query energy but the mean correction is reversed.  The same statistics are
computed separately per projection, layer, and class.  This remains an oracle
analysis because query labels define `G_query`.

### Aggregate projection geometry

| Dataset | Quantity | Q | K | V | O | All |
|---|---|---:|---:|---:|---:|---:|
| Camelyon17 | rho | 0.600 | **0.964** | 0.751 | 0.497 | 0.784 |
| Camelyon17 | kappa | 0.982 | 0.960 | 0.991 | **0.994** | **0.974** |
| RoboFail | rho | 0.507 | **0.863** | 0.394 | 0.290 | 0.610 |
| RoboFail | kappa | **+0.034** | -0.101 | -0.298 | -0.361 | **-0.207** |
| ManipBench droid-pick-place | rho | 0.493 | **0.820** | 0.480 | 0.338 | 0.617 |
| ManipBench droid-pick-place | kappa | 0.731 | 0.441 | 0.756 | **0.815** | **0.609** |

This signed statistic resolves the apparent RoboFail contradiction.  K has the
highest energy overlap, but its mean support/query correction is opposed.
Q is the only projection without a clearly negative aggregate direction, which
matches the separate actuation result where Q-only has the best seed-0 F1.
Camelyon is qualitatively different: every projection has strongly positive
directional agreement.  ManipBench is not an aggregate sign-reversal case;
its residual-versus-LoRA gap therefore still points to insufficient local
functional authority or task-mapping capacity.

### Layer boundary

| Dataset | Layer 14 rho / kappa | Layer 27 rho / kappa |
|---|---:|---:|
| Camelyon17 | 0.614 / +0.989 | 0.989 / +0.956 |
| RoboFail | 0.599 / -0.186 | 0.954 / **-0.892** |
| ManipBench droid-pick-place | 0.617 / +0.609 | 0.913 / +0.175 |

High late-layer overlap is not sufficient: layer 27 has high `rho` on all
three tasks, but its direction is strongly reversed on RoboFail and only weakly
aligned on ManipBench.  This is direct evidence for separating transferable
geometry from actuator suitability.

### Class-conditional geometry

| Dataset / class | rho_c | kappa_c |
|---|---:|---:|
| Camelyon17 / normal | 0.780 | +0.984 |
| Camelyon17 / tumor | 0.743 | +0.980 |
| RoboFail / success | 0.586 | +0.957 |
| RoboFail / failure | 0.647 | +0.974 |
| ManipBench / A | 0.588 | +0.910 |
| ManipBench / B | 0.545 | **+0.225** |
| ManipBench / C | 0.540 | +0.832 |
| ManipBench / D | 0.514 | +0.746 |

RoboFail's two classes each transfer cleanly when analyzed separately, yet the
aggregate direction is negative.  The support is class-balanced (8/8) while
the query is strongly imbalanced (116 success / 21 failure for seed 0), so the
relative class-gradient mixture changes between adaptation and evaluation.
This supports a class-mixture cancellation/reweighting explanation rather than
failure of both class-specific geometries.  ManipBench instead exposes one
weakly aligned action class (B), consistent with heterogeneous action-semantic
mapping.

The corresponding runner is `scripts/analyze_directional_geometry.py`.  These
seed-0 findings are strong mechanism evidence but require replication across
support seeds before a paper-level predictive correlation claim.

## Three-seed directional replication

We replicated the signed analysis over the three fixed support seeds.  Values
below are mean +/- sample standard deviation across seeds.

| Dataset | rho | aggregate kappa | Q kappa | K kappa | V kappa | O kappa |
|---|---:|---:|---:|---:|---:|---:|
| RoboFail | 0.620+/-0.008 | 0.209+/-0.361 | 0.425 | 0.438 | 0.034 | -0.120 |
| ManipBench droid-pick-place | 0.620+/-0.003 | 0.741+/-0.142 | 0.863 | 0.626 | 0.838 | 0.881 |

RoboFail's aggregate sign reversal is not stable across support seeds: its
aggregate kappas are -0.207, +0.387, and +0.448.  The stable result is instead
class conditional.  Success kappa is 0.957/0.958/0.972 (mean 0.962), and
failure kappa is 0.974/0.958/0.978 (mean 0.970).  Thus the data support robust
within-class geometry together with seed-sensitive aggregate mixing, not a
universal negative RoboFail direction.

### Query-prior mixture oracle

To test the mixing explanation directly, we keep the support covariance and
support-derived basis fixed, but reweight the support class-mean gradients by
the oracle query class prior (116 success / 21 failure).  Query labels affect
only these two mixture weights; this is not a deployable adaptation rule.

| Seed | Original aggregate kappa | Query-prior-reweighted kappa |
|---:|---:|---:|
| 0 | -0.2065 | **0.9070** |
| 1 | 0.3866 | **0.9375** |
| 2 | 0.4475 | **0.9574** |
| Mean +/- SD | 0.2092+/-0.3613 | **0.9340+/-0.0254** |

After reweighting, mean module-wise kappas are Q=0.9521, K=0.9433,
V=0.9143, and O=0.9267.  The recovery is therefore not isolated to one
projection.  This provides direct evidence that support/query class-mixture
shift causes much of the unstable aggregate direction on RoboFail, while the
strong within-class correction geometry remains transferable.  It does not
show that the query prior is available at test time, nor that prior
reweighting alone closes the F1 gap.

ManipBench has positive aggregate alignment on every seed
(0.609/0.722/0.892), while the same local activation actuator remains well
below the formal LoRA result.  This strengthens the actuator-sufficiency/task-
mapping interpretation: correct directional transfer alone is not sufficient.

## Class-conditional controller probe

We tested one deliberately small optional extension on RoboFail seed 0.  It
constructs a separate KV basis and controller for each support class, then
scores each candidate label with its corresponding controller.  This is a
diagnostic extension; the frozen backbone and core KV residual are unchanged.

The probe failed: Macro-F1 fell to **0.1625** (balanced accuracy 0.4306), with
success recall 0.0517 and failure recall 0.8095.  Candidate-specific
controllers create score scales that are not directly comparable and strongly
bias the argmax.  Moreover, the existing RoboFail support is already balanced
8/8, so simple class-normalized coefficient fitting is algebraically identical
to the current aggregate objective.  We therefore reject this extension and
do not promote it to the main method or spend additional seeds on it.

The probe is reproducible with `scripts/run_class_conditional_kv.py`; its
prediction artifacts remain under ignored `runs/diagnostics/`.

## Camelyon three-seed replication and BRIGHT transfer

Camelyon remains stable over all three fixed support seeds: aggregate rho is
**0.7846+/-0.0049** and aggregate kappa is **0.9773+/-0.0079**.  Mean
module-wise kappas are Q=0.9801, K=0.9676, V=0.9905, and O=0.9904.  The
class-conditional means are 0.9876 for normal and 0.9802 for tumor.  This is
the cleanest replicated positive regime: both geometry transfer and signed
directional agreement are high and insensitive to support selection.

We also ran the identical paired-image diagnostic on the archived Hugging Face
BRIGHT splits (`humanlong/EventTune-BRIGHT`).  These archived 4-shot folds use
12 support examples (4 per class); for the oracle diagnostic, the locked query
is deterministically truncated to its first 100 examples per class.  Gradients
are restricted to post-event visual tokens.  This is a mechanism analysis of
the archived split, not a replacement for the formal support-24 benchmark.

| BRIGHT fold | rho | kappa | Q kappa | K kappa | V kappa | O kappa |
|---|---:|---:|---:|---:|---:|---:|
| Hawaii wildfire | 0.4328 | **0.9754** | 0.9667 | 0.8241 | 0.9864 | 0.9835 |
| Libya flood | 0.3689 | **0.6769** | 0.6522 | 0.4691 | 0.7940 | 0.6842 |

Hawaii's class kappas are 0.9794/0.9201/0.9861 for intact/damaged/destroyed;
Libya's are 0.9648/0.9827/0.9423.  Thus class-specific directions transfer
strongly in both folds even though Libya's aggregate direction is weakened by
class mixing.  As in RoboFail, aggregate kappa can conceal transferable
within-class geometry.  Unlike ManipBench, the BRIGHT task keeps fixed damage
semantics and requires evidence interpretation rather than action remapping,
which is consistent with the strong signed class geometry and the existing
BRIGHT advantage of Gradient-Covariance KV over LoRA.

The paired-image runner is `scripts/analyze_bright_directional_geometry.py`.
It uses query labels only to compute oracle diagnostic statistics and never to
select or fit a deployable adapter.

## Cross-regime Geometry x Actuator synthesis

The combined evidence does not support using either rho or kappa as a scalar
leaderboard predictor.  The useful diagnostic is a decision sequence: first
ask whether the signed correction transfers; then ask whether a query-derived
basis with the same local actuator can approach the weight-adaptation result.

| Regime | Geometry evidence | Same-actuator oracle evidence | Formal Ours - LoRA F1 | Diagnosis |
|---|---|---|---:|---|
| Camelyon17 | rho 0.7846+/-0.0049; kappa 0.9773+/-0.0079 | query-basis KV 0.6632 vs LoRA 0.6325 | -0.1218 | geometry and actuator are sufficient; support estimation/coefficient margin remains |
| RoboFail | rho 0.6196; aggregate kappa seed-sensitive, class kappa 0.962/0.970 | query-basis KV 0.4119; Q-only seed-0 0.4811 vs LoRA 0.5010 | -0.1058 | class-mixture and actuation-site mismatch |
| ManipBench droid-pick | rho 0.6199; kappa 0.7412 | query-basis KV 0.2482 vs LoRA 0.4304 | -0.1465 | local activation actuator lacks task-remapping authority |
| BRIGHT Hawaii | rho 0.4328; kappa 0.9754; all class kappas >=0.920 | not run (formal KV already wins) | +0.1714 | fixed-semantics evidence shift |
| BRIGHT Libya | rho 0.3689; kappa 0.6769; all class kappas >=0.942 | not run (formal KV already wins) | +0.0578 | class directions transfer despite aggregate mixing |

The F1 differences use the corrected InternVL3 formal results.  BRIGHT's formal
rows use support 24, whereas its geometry columns use the archived Hugging Face
support-12 diagnostic folds, so BRIGHT is a qualitative cross-protocol check
rather than a matched correlation point.  The other oracle rows use query
labels and are never valid deployable scores.

This table sharpens the paper claim.  High signed transfer is not sufficient:
ManipBench has positive kappa but its query-basis oracle remains far below
LoRA.  Nor does high kappa guarantee that finite-support coefficient fitting
will realize the available correction, as Camelyon shows.  The defensible
mechanism statement is therefore:

> Internal-state TTA can substitute for weight adaptation when the correction
> direction transfers and the chosen activation site has enough functional
> authority; finite-support estimation and margin realization remain separate
> sources of error.
