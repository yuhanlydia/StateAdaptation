# Aperture — final validation protocol (v1)

**Purpose.** Strengthen the existing visual-residual-control paper without changing
its main method, introducing another backbone, or searching new datasets for wins.
The evidence questions below are fixed before this round. A negative or tied result
changes the corresponding claim, not the evaluation population.

**Source base:** `4465c460350ddf2ce3c8e65f5b2cdb247ac7497d`,
`yuhanlydia/StateAdaptation`. Existing reports and the nine-page manuscript are not
rewritten by these scripts. This additive code does not implement a new Aperture
architecture. It reuses the repository's model loader, visual masks, two-stage
controller, support loss, candidate scorer, LoRA trainer, and real-model audit.

## Why these experiments, rather than another large sweep?

The official ICLR 2027 reviewer guide emphasizes the problem, motivation, validity
of supporting evidence, significance, and reproducibility; it does not require a
leaderboard win for every task. It recommends bounded requests that validate the
existing contribution. The uploaded formatting template is a different document:
its reproducibility guidance does not constitute a numerical review-score rubric.

Official source (checked 2026-09-21):
https://iclr.cc/Conferences/2027/ReviewerGuidelines

Calibration reference: Guo et al., *On Calibration of Modern Neural Networks*,
ICML 2017, https://proceedings.mlr.press/v70/guo17a.html . Temperature scaling is a
one-parameter post-processing baseline. Our small-support grid and shrinkage term
below are fixed experimental choices, not a claim to reproduce that paper's entire
protocol.

| Evidence question | Added comparison | Where it belongs in the manuscript |
|---|---|---|
| Is low NLL only an advantage over uncalibrated, overconfident LoRA? | Raw and positive-temperature-calibrated Frozen, LoRA-1, LoRA-4 and Aperture; calibration-selected LoRA schedules | Predictive-quality subsection and its supplementary table |
| Does a fixed visual controller remain useful when image evidence is perturbed? | Same fitted states, clean/JPEG/Gaussian inputs; paired POST-only changes | Visual-dependence subsection; a robustness probe, not proof of semantic denoising |
| Does residual amplitude affect the learned correction? | Multipliers 0, 0.5, 1 on the same frozen controller and same query probe | Intervention analysis; no query-based selection of a replacement method |
| What is the actual adaptation footprint? | Optimized and stored scalars, fit/extraction time, state bytes, repeat latency and device | Efficiency discussion / appendix |
| Can another agent rerun and regenerate the tables? | Locked split/plan/state files, complete-only exports, editable CSV/TeX and measurement-based plots | Reproducibility statement / appendix |

## 1. Fixed population and total supervision

One local **Qwen2.5-VL-7B-Instruct**, BF16, original model settings and visual-token
budget. Existing support seeds **0, 1, 2**: these are not new blind seeds. Query
sets have been inspected in earlier work and must not be called an untouched test
set. The within-support fitting/calibration split is new.

| Existing dataset block | Domains | Total labeled support / domain | Fit / calibration | Locked queries |
|---|---:|---:|---:|---:|
| BRIGHT | Hawaii, Libya, Noto, Turkey | 24 (8/class) | 18 / 6 | 300 / domain |
| Camelyon17 | hospital 2 | 16 (8/class) | 12 / 4 | 300 |
| ManipBench Q1 | Bridge, DROID-place, DROID-arti | 32 (8/class) | 24 / 8 | 400 / domain |

All four arms use the same fit, calibration and query IDs within a comparison.
No additional labels are requested. Calibration examples are held out from basis
extraction and LoRA/controller fitting. **Calibration is sample/crop-held-out, not
necessarily patient/tile-group-held-out.** Fit/calibration shared groups are
recorded. Query groups are disjoint from the entire original support. Missing
group identity or a failed separation check stops preparation; it does not trigger
a silent fallback to sample IDs.

Splitting is class-stratified and deterministic: SHA-256 of split seed and sample
ID, with two calibration examples/class. Neither query scores nor query outcomes
choose the partition. Different BRIGHT building crops on one source tile are
recognized by their bounding boxes. Duplicate identical image/crop inputs are
rejected. Original source manifests and query ordering are never changed.

These new numbers must be labeled **within-support calibration protocol**. Do not
replace historical results that fitted on all 24/16/32 examples with these scores
without changing the table's stated protocol.

## 2. Fixed fitting recipes

Aperture is unchanged: K/V at decoder layers 14 and 27, rank 16 Full controller,
alpha 3, four full-support optimizer updates, learning rate 0.05, raw-matrix L2
0.001, zero initialization, fixed support-derived directions. The configured
full-answer token span is preserved. BRIGHT uses mean support reduction; the
single-image recipe uses sum. No query labels enter basis estimation or fitting.

LoRA keeps rank 16 / alpha 32 and the existing projection targets. There are two
predefined schedules (one and four support passes), not a growing learning-rate
search. BRIGHT LR is 1e-4 with accumulation 3; single-image LR is 2e-4 with
accumulation 1. The smaller fitting partition makes the step counts:

| Task | LoRA 1 pass | LoRA 4 passes | Aperture fit updates / visits (+basis visits) |
|---|---:|---:|---|
| BRIGHT fit18 | 6 updates / 18 visits | 24 / 72 | 4 / 72 (+18) |
| Camelyon fit12 | 12 / 12 | 48 / 48 | 4 / 48 (+12) |
| ManipBench fit24 | 24 / 24 | 96 / 96 | 4 / 96 (+24) |

Same unique labels is not the same number of gradient updates or FLOPs. Exact
counts are exported. The nonparametric eigenbasis extraction also consumes
backward passes and storage.

## 3. Calibration and model selection BEFORE query scoring

For each arm, save its held-out calibration candidate log scores, then fit a
strictly positive temperature on a fixed grid:

`T in unique({1} union geomspace(0.2, 20.0, 81))`

The objective is `calibration_NLL(T) + 0.01 * log(T)^2`. The shrinkage penalty is
fixed, not query-tuned. There are no clinical calibration guarantees from 4–8
calibration examples. Three support seeds reveal some of this variability.

Positive temperature preserves candidate argmax; thus it must not change F1 or
accuracy for a fixed arm. The implementation tests this invariant. If the optimal
grid value reaches an endpoint, record it; do not extend the grid after seeing
query results in this run.

Two LoRA choices are sealed per domain/seed: the raw schedule with lower held-out
calibration NLL, and the temperature-adjusted schedule with lower held-out
calibration NLL. Ties prefer one pass. Both schedules are still reported. The
selected LoRA rows are not the better query result.

A simple output-bias comparator learns K-1 zero-sum offsets using **all original
support labels** and the frozen predictor. No representation was fitted in this
arm, so using all labels is legitimate and avoids artificially weakening it. Its
label count is the same total budget as every fit+calibrate method. Frozen
positive-temperature calibration, by contrast, uses the common held-out partition.

Saving every fit and calibration record is a barrier: the evaluator refuses to
start before the comparison's calibration decisions have been sealed. No query
forward is executed by `execute_fit`.

## 4. Metrics and reporting

Primary probability metric: candidate NLL computed stably from the **summed
answer-token log scores**, using logsumexp, without clipping away confident errors.
Also store the legacy 1e-12-clipped NLL for inspection. The legacy field name
`mean_log_scores` contains sums in the reused scorer. Historical model-specific
scoring contracts are not silently mixed into this experiment.

Report Macro-F1, balanced accuracy, NLL, Brier and ten-bin ECE for all arms, raw and
calibrated. Export mean and sample SD across three support seeds on the same query
IDs. Do not pool the repeated query population into three times as many independent
examples. No missing result becomes zero, no single seed receives an invented SD,
and no result-dependent dataset removal is built into the exporter.

The output includes all LoRA schedules, calibration-selected schedules, Frozen,
Frozen+bias, Aperture and temperature-adjusted rows. A manuscript may use compact
subtables, but it must describe consequential F1/NLL trade-offs honestly.

## 5. Fixed image and controller probes

Only **seed 0**, four BRIGHT events plus Camelyon, receive the image-degradation
probe. Select **60 query IDs** by a fixed hash of ID only, without truth labels or
model scores. Apply Gaussian pixel noise (sigma 4, 8 on the 0–255 scale) and JPEG
compression (qualities 70, 40). Keep image shape, candidate text and token contract
fixed. Paired BRIGHT changes POST only; PRE remains unchanged.

All arms see exactly the same deterministic degraded inputs. Fit on clean support,
fit T on clean calibration, then reuse unchanged. Do not fit on perturbed query
images. These transformations may affect clinically/semantically relevant details;
there is no blanket label-preservation claim. They are diagnostic stress tests,
not an official corruption benchmark or direct evidence of filtering particular
nuisance pixels. Export clean probe scores alongside perturbed scores, rather than
comparing full-query clean metrics with a smaller perturbed subset.

For Aperture seed 0 on all eight domains, reuse the same fitted matrices and run
residual multipliers `{0, 0.5, 1}` on the same 60-ID probe. Zero removes the local
residual, while the other matrices stay fixed. Raw-dose metrics are the principal
interpretation: the full-dose temperature is not refitted for each multiplier.
These curves do not select alpha or a new reported model.

## 6. Resource and execution scope

Default: **24 units × 4 arms = 96 states**, of which 72 fit a non-frozen adapter or
controller. Each state has fitting/calibration and evaluation phases; two real-model
support-only audit jobs precede them. At full completion: **194 subprocess phases**
(2 audits + 96 fitting/calibration + 96 evaluation), not 194 separate model trainings.
Temperature/bias optimization is CPU work on saved scores. Clean query scoring
covers 32,400 state-query pairs, before candidate enumeration. Fixed corruption
probes add 4,800 pairs; residual-dose probes add 960 pairs. Timings add repeated
scoring and are separately recorded. No runtime in GPU-hours is promised without
an actual pilot measurement on the user's machine.

One process per visible GPU; 24GB-or-larger CUDA GPU required for the fixed BF16
protocol. No automatic quantization, smaller checkpoint, resolution reduction,
CPU offload, expanded memory approximation or hard-coded 16GB fallback. A small
control matrix does not remove backbone activation memory. On 16GB only, use a
separate validated protocol rather than silently calling it this one.

Timing: seed 0, two warm-up queries then 12 complete clean candidate-scoring queries
repeated three times. Report device, software, preprocessing/score-inclusive
latency, peak memory, basis scalars, optimized coefficients, state bytes and fitting
phase time. Latency repeat SD is not support-seed uncertainty. Historical rank-5
latency must not be presented as this rank-16 runtime.

## 7. Decision rules and ending the round

If Aperture remains better in NLL after both arms receive held-out calibration,
that strengthens the claim beyond uncalibrated confidence. If the difference
closes, state that output calibration explains that part of the old contrast and
keep supported compactness or classification claims. If corruption sensitivity is
not better, report it as a boundary; do not tune a new corruption-friendly variant.
A failed technical audit stops execution; a bad finite metric does not stop or
expand the experiment matrix.

The round ends when all planned states are valid and exported. No automatic
p-value target, Bayesian winner search, added seed until significant, or expansion
to new domains is part of this protocol. Finishing this experiment is not a
guarantee of acceptance or of a favorable result on every row.
