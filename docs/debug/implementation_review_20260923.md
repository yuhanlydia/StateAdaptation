# Implementation and support-only optimization review

Date: 2026-09-23 (Asia/Shanghai). Reviewer: tuning_audit agent.

Scope: read-only review of `/home/root123/.local/share/stateadaptation/aperture_tuning_20260922/StateAdaptation` and `/data/cwj/Aperture_Tuning_20260922/aperture_tuning_v1`. No query predictions, query metrics, or query-based rankings were read for this review. No source files, locked plans, GPU jobs, or old results were changed. All performance numbers below are **held-out support calibration**, not query/test performance. Recommendations are hypotheses for a separately versioned protocol and cannot guarantee beating LoRA.

## Main conclusion

No confirmed incorrect forward, image mask, saved-state restore, or label mapping bug was found. There are genuine optimization problems (especially high-LR LoRA collapse), material objective/selection limitations, and instrumentation gaps. Continuing an unrestricted search against H2's four calibration images is not an adequate fix: each seed's four images all come from **the same patient_046**, and all twelve q34 H2 Aperture candidate/seed evaluations already have calibration F1=1.0. The two seeds use different image IDs (zero overlap), but not independent validation patients.

## Concrete implementation findings

1. `src/aperture_table/backend.py:125-127` optimizes **negative mean answer-token log probability**, while `:129-140` scores **sum answer-token log probability**, then softmaxes only the candidate set. Local tokenizer inspection established Qwen normal=1 token and tumor=2; Gemma normal/tumor and Path letters are single-token. Consequently Qwen H2's training example weighting differs from its sequence-score objective. Both Aperture and LoRA share this contract. It is a documented inherited objective choice, not evidence of corrupted results. Changing it requires a named new objective ablation and equal treatment of LoRA; it must not silently replace v1.

2. A separate, larger distinction exists between vocabulary generative loss and candidate-conditional classification loss. Gemma H2 seed0 a4 has calibration conditional NLL **0.017 (rounded)** but correct-answer absolute sequence NLL **5.403**, with mean log total candidate probability **-5.386**. Thus high vocabulary loss can coexist with perfect binary calibration decisions. Do not infer 'classification undertrained' from Gemma's support loss alone. A candidate-normalized cross-entropy ablation would directly target classification, but is a method/objective change, not merely another LR, and costs up to nine candidate forwards on Path unless a verified equivalent implementation is used.

3. `src/vigor_handoff/core.py:244-264` uses Adam, full-support SUM gradients, L2 on raw controller matrices, and global norm clipping at 1. All reported Path Aperture updates inspected are clipped. Gemma Path a5 peaks at norms **1743.2 / 1765.2** (seeds 0/1), while loss remains finite and subsequently decreases. This is evidence of strong transient gradients, not numerical divergence. Switching sum to mean would also change relative regularization/clipping behavior; it must be a separate controlled candidate, not a supposed no-op fix. Adam does not turn a 1000x preclip norm into a 1000x parameter update.

4. The same function records each step's loss **before** `optimizer.step()` but saves the final controller **after** the last step. Consequently the final history entry is not the serialized state's final training loss. LoRA history in `src/vigor_handoff/worker.py:168-187` averages online losses along many successive parameter states and also includes training-mode adapter dropout. Direct cross-method comparison of those history numbers is invalid. Add final frozen-state support scoring and per-class support metrics in a new version, without changing v1's source fingerprint.

5. `src/aperture_table/backend.py:172` hardcodes LoRA r=16, alpha=32, dropout=.05. Current tuning correctly only changes LR and passes, but future changes to the corresponding job fields would be ignored unless loader code is updated and audited. `score()` names the sum score field `mean_log_scores`; downstream code consistently uses sums, so this is a misleading field name rather than a calculation bug.

6. The initial fixed layers were not depth-aligned. Current candidates correct this in an explicit search: q25 depth28, q34/q38 depth36, Gemma depth34. Local Transformers forward inspection shows Qwen3 and Gemma normalize K per head after k_proj; projection-output alpha therefore has architecture-dependent effects. Gemma original layers14/27 are sliding-attention layers; relative layer17 is full attention and layer33 is sliding. Depth and attention type are intertwined. Comparisons cannot attribute all improvements solely to depth.

7. `src/aperture_tuning/protocol.py:73-87` correctly selects calibration Macro-F1 first, then raw NLL, then candidate ID. On q34 H2 all candidates tie at F1=1.0 in both seeds. Therefore nominal F1-first selection reduces entirely to choosing the lowest NLL on one patient. Gemma Path seed1 chooses a0 despite NLL4.019 versus a2 NLL2.912 because a0 has F1 .201 versus .093. This is the declared rule, not a selection bug. A new composite/stability criterion must be fixed before new query access.

## Task-specific support evidence and targeted next steps

### Qwen2.5 Path

- Best v1 Aperture calibration F1 is .404 (s0 a2) and .274 (s1 a1); best LoRA is .633/.567 (both l3).
- a5 reduces the last logged training loss to 1.321/1.299, yet calibration F1 is .385/.163. In seed1 this is worse than a0's .274 despite a lower training loss. Simply training longer or fitting more strongly is unsupported.
- a3/a4 (alpha1, LR.01) show slower fitting and weaker validation. The old layers with alpha3/low LR also do not give consistent F1 gains. a2 [7,21] helps s0 but not s1. No single location is validated by both seeds.
- **Next limited search:** keep alpha3 and original generative loss; compare the already motivated single [14] and [7,21] at LR.02, with predeclared checkpoints 2/4/8. Refit each candidate under a small stratified support-only inner split rather than repeatedly using the same 18 calibration images. Count every checkpoint as a candidate. Keep rank16 until these position/stop comparisons are resolved. A subsequent V-only versus K/V mechanism ablation is more interpretable than immediately increasing rank, but requires a new implementation/audit and is not asserted to improve accuracy.

### Qwen2.5 H2

- a5 wins s0 (F1=1, NLL.102); a4 wins s1 (F1=1, NLL.237). Five candidates reach F1=1 in s0; only a4 does so in s1. Lower-strength and longer updates have support evidence here, but four images are insufficient to decide their generalization.
- LoRA LR5e-5 beats higher LRs on these support splits; l1 wins both seeds. At 2e-4, s0 l2/l3 collapse to F1 .333; at 8e-4 both schedules collapse in both seeds. This supports eliminating the high-LR region in a new **prespecified** search rather than adjusting a baseline using query outcomes.
- **Next priority:** patient-grouped support validation first. Within the existing 16 labels, leave each support patient out in turn (subject to both classes remaining available), aggregate out-of-fold predictions, and compare alpha1/3, LR.01, checkpoints4/8/12 at [14,27]. This changes the training/selection protocol; use the identical folds and label budget for LoRA. Do not advertise it as the same fixed-fit/calibration experiment. Separately test sequence-sum versus token-mean objective for both methods if resources permit.

### Qwen3-4B H2

- All six candidates have calibration F1=1 in both seeds. Conservative a3/a4 achieve lower NLL than more aggressively fitted a5, while a5 training losses are much smaller. a3 NLL .022/.004; a4 .019/.045; a5 .168/.117. Longer training is not consistently better even on existing calibration NLL.
- **Next priority:** do not expand a hyperparameter grid on patient_046. Grouped support validation is needed. Thereafter retain [18,35], alpha .5/1, LR .005/.01, checkpoints2/4/8 as a small sequential search (not a full product); compare conservative endpoints using held-out patients. Rank increase or LR increase has no support-only justification here.

### Gemma H2

- a0 gives F1 1/.333 across seeds, while a2 [17,33] gives 1/1 and NLL .024/.012. a3/a4 also reach 1/1; a1 is inconsistent (.333/1). Therefore relative dual layers have support evidence, but there is no proof that more training is the solution.
- a4 seed0's training loss remains7.376 even though calibration F1=1 and NLL .017; vocabulary probability mass explains part of the mismatch (above). a5 fits vocabulary loss much further without consistently improving conditional validation NLL.
- **Next limited search:** first compare a2 and conservative a3/a4 under patient-grouped support validation. If a mechanism probe is affordable, hold layer17 fixed and compare late global29 versus late sliding33 with identical alpha/LR/steps, and separately inspect absolute candidate mass versus conditional margins. This tests attention type/depth with an explicit confound, not a claimed architecture bug. Candidate-normalized loss should be its own matched-method experiment, not mixed into the hyperparameter winner table.

### Gemma Path

- a2 improves training much faster than a0; last logged training loss3.091/2.855 versus6.160/4.918. Validation remains low and seed-dependent: a2 F1 .206/.093, a0 .178/.201. a5 at12 steps has loss3.043/2.825 and F1 .177/.119. a3/a4 are severely underfit by both absolute and conditional held-out loss (NLL roughly6.34–10.91).
- Predicted-class histograms on calibration reveal narrow coverage: s0 a0 predicts only 3 of9 labels; s0 a2 only4 of9. This is a real support calibration failure, not evidence of mislabeled query data. Correct outputs must report per-class recall and count, not only mean F1.
- **Next limited search:** keep [17,33], alpha3; compare LR.02 and .05 at checkpoints4/8, recording final-state support classification and class coverage. Then, if both remain narrow, test one predeclared layer pair [17,29] at the selected support-only LR. Rank changes should follow, not precede, an objective/coverage diagnostic. A classification-objective ablation may be useful, but nine-way candidate CE is more expensive and must include equally trained LoRA. Existing support evidence does not establish that simply increasing steps or alpha will close the gap.

## Fairness and execution constraints

- Preserve v1 and its locked source/hash. Any changes to loss, fitting splits, layer-site kind, reduction, or stopping/selection rule belong to a new run root and named protocol.
- Choose candidates and stopping rule once before new query evaluation. Count checkpoints, folds, retries due to intentional parameter changes, and method variants in the search budget. Report all candidates, including collapsed LoRA candidates.
- Equal candidate counts are not equal compute. Report model loads, processed examples, optimizer updates, basis extraction, support forwards, and actual GPU seconds.
- H2 patient-group validation can use the same original label budget, but tiny per-patient class counts still limit reliable selection. Path only has image-content isolation in this data release, not verified patient isolation; do not call its stratified folds patient-separated.
- For a claim of independent improvement, evaluate once on a newly locked query partition not used in previous analysis. Reuse of the existing query remains exploratory, even if this review itself never reads those scores.

## Evidence locations

- Plan/candidate definitions: `aperture_tuning_v1/plan.json` and repository `src/aperture_tuning/protocol.py`.
- Per-unit held-out metrics and winners: `aperture_tuning_v1/selections/{model}--{domain}--s{seed}.json`.
- Optimization history: `aperture_tuning_v1/states/{unit}--{method}--{candidate}/fit/fitting.json`.
- Absolute scores, normalized probabilities and class coverage: the same fit directory's `calibration_predictions.jsonl`.
- Split roles and patient IDs: immutable support/calibration manifest paths embedded in plan jobs.

The compact evidence table below is generated only from selection records and support-fitting histories; its last loss is pre-final-update for Aperture and an online epoch mean for LoRA.

| Unit | Candidate | Calibration F1 | Calibration NLL | First/last logged fit loss | Maximum preclip gradient norm |
|---|---|---:|---:|---:|---:|
| q25--path--s0 | a0 | 0.366667 | 1.644650 | 2.433 / 1.580 | 356.050 |
| q25--path--s0 | a1 | 0.366667 | 1.652943 | 2.433 / 1.594 | 331.654 |
| q25--path--s0 | a2 | 0.403704 | 1.532574 | 2.433 / 1.553 | 87.135 |
| q25--path--s0 | a3 | 0.317460 | 2.081882 | 2.433 / 2.128 | 118.683 |
| q25--path--s0 | a4 | 0.388889 | 1.863650 | 2.433 / 1.813 | 118.683 |
| q25--path--s0 | a5 | 0.385185 | 1.516537 | 2.433 / 1.321 | 356.050 |
| q25--path--s0 | l0 | 0.407407 | 1.501963 | 2.049 / 2.049 | not logged |
| q25--path--s0 | l1 | 0.540741 | 2.009279 | 2.049 / 1.137 | not logged |
| q25--path--s0 | l2 | 0.563492 | 1.334325 | 2.145 / 2.145 | not logged |
| q25--path--s0 | l3 | 0.633333 | 2.132338 | 2.145 / 1.091 | not logged |
| q25--path--s0 | l4 | 0.022222 | 2.594906 | 2.860 / 2.860 | not logged |
| q25--path--s0 | l5 | 0.022222 | 2.714367 | 2.860 / 3.142 | not logged |
| q25--path--s1 | a0 | 0.274074 | 1.611308 | 2.303 / 1.435 | 272.018 |
| q25--path--s1 | a1 | 0.274074 | 1.596120 | 2.303 / 1.446 | 239.288 |
| q25--path--s1 | a2 | 0.262963 | 1.743784 | 2.303 / 1.524 | 64.902 |
| q25--path--s1 | a3 | 0.164815 | 2.195942 | 2.303 / 2.036 | 90.673 |
| q25--path--s1 | a4 | 0.225926 | 1.960594 | 2.303 / 1.762 | 90.673 |
| q25--path--s1 | a5 | 0.162963 | 1.575932 | 2.303 / 1.299 | 272.018 |
| q25--path--s1 | l0 | 0.285714 | 1.619741 | 2.052 / 2.052 | not logged |
| q25--path--s1 | l1 | 0.537037 | 1.260226 | 2.052 / 1.333 | not logged |
| q25--path--s1 | l2 | 0.382011 | 1.327359 | 2.337 / 2.337 | not logged |
| q25--path--s1 | l3 | 0.567196 | 1.620489 | 2.337 / 1.164 | not logged |
| q25--path--s1 | l4 | 0.022222 | 2.283193 | 3.541 / 3.541 | not logged |
| q25--path--s1 | l5 | 0.022222 | 2.502881 | 3.541 / 3.784 | not logged |
| q25--h2--s0 | a0 | 0.733333 | 0.238915 | 1.745 / 0.532 | 219.544 |
| q25--h2--s0 | a1 | 1.000000 | 0.202081 | 1.745 / 0.568 | 52.377 |
| q25--h2--s0 | a2 | 1.000000 | 0.186504 | 1.745 / 0.500 | 33.285 |
| q25--h2--s0 | a3 | 1.000000 | 0.149605 | 1.745 / 0.998 | 73.181 |
| q25--h2--s0 | a4 | 1.000000 | 0.122947 | 1.745 / 0.715 | 73.181 |
| q25--h2--s0 | a5 | 1.000000 | 0.101647 | 1.745 / 0.288 | 219.544 |
| q25--h2--s0 | l0 | 1.000000 | 0.227986 | 0.949 / 0.949 | not logged |
| q25--h2--s0 | l1 | 1.000000 | 0.063320 | 0.949 / 0.821 | not logged |
| q25--h2--s0 | l2 | 0.333333 | 0.563358 | 0.947 / 0.947 | not logged |
| q25--h2--s0 | l3 | 0.333333 | 3.254043 | 0.947 / 0.346 | not logged |
| q25--h2--s0 | l4 | 0.333333 | 0.745187 | 2.108 / 2.108 | not logged |
| q25--h2--s0 | l5 | 0.333333 | 0.537928 | 2.108 / 1.605 | not logged |
| q25--h2--s1 | a0 | 0.733333 | 0.260712 | 1.947 / 0.550 | 254.155 |
| q25--h2--s1 | a1 | 0.733333 | 0.363975 | 1.947 / 0.667 | 65.626 |
| q25--h2--s1 | a2 | 0.733333 | 0.280429 | 1.947 / 0.518 | 37.248 |
| q25--h2--s1 | a3 | 0.733333 | 0.265717 | 1.947 / 1.130 | 117.569 |
| q25--h2--s1 | a4 | 1.000000 | 0.237376 | 1.947 / 0.893 | 117.569 |
| q25--h2--s1 | a5 | 0.733333 | 0.225112 | 1.947 / 0.416 | 254.155 |
| q25--h2--s1 | l0 | 1.000000 | 0.268718 | 1.086 / 1.086 | not logged |
| q25--h2--s1 | l1 | 1.000000 | 0.191759 | 1.086 / 0.512 | not logged |
| q25--h2--s1 | l2 | 0.333333 | 0.758706 | 0.904 / 0.904 | not logged |
| q25--h2--s1 | l3 | 0.733333 | 0.732966 | 0.904 / 0.473 | not logged |
| q25--h2--s1 | l4 | 0.333333 | 1.235730 | 2.734 / 2.734 | not logged |
| q25--h2--s1 | l5 | 0.333333 | 4.050112 | 2.734 / 1.947 | not logged |
| q34--h2--s0 | a0 | 1.000000 | 0.072889 | 0.935 / 0.098 | 23.357 |
| q34--h2--s0 | a1 | 1.000000 | 0.113823 | 0.935 / 0.132 | 39.979 |
| q34--h2--s0 | a2 | 1.000000 | 0.174361 | 0.935 / 0.192 | 40.012 |
| q34--h2--s0 | a3 | 1.000000 | 0.021544 | 0.935 / 0.665 | 13.337 |
| q34--h2--s0 | a4 | 1.000000 | 0.019066 | 0.935 / 0.427 | 13.337 |
| q34--h2--s0 | a5 | 1.000000 | 0.168137 | 0.935 / 0.087 | 40.012 |
| q34--h2--s0 | l0 | 0.733333 | 0.325010 | 1.771 / 1.771 | not logged |
| q34--h2--s0 | l1 | 1.000000 | 0.098813 | 1.771 / 0.504 | not logged |
| q34--h2--s0 | l2 | 1.000000 | 0.099675 | 1.818 / 1.818 | not logged |
| q34--h2--s0 | l3 | 0.733333 | 0.235669 | 1.818 / 0.715 | not logged |
| q34--h2--s0 | l4 | 0.333333 | 5.998579 | 3.328 / 3.328 | not logged |
| q34--h2--s0 | l5 | 0.333333 | 1.197306 | 3.328 / 3.930 | not logged |
| q34--h2--s1 | a0 | 1.000000 | 0.098070 | 2.038 / 0.179 | 51.963 |
| q34--h2--s1 | a1 | 1.000000 | 0.104364 | 2.038 / 0.220 | 89.988 |
| q34--h2--s1 | a2 | 1.000000 | 0.095865 | 2.038 / 0.270 | 90.101 |
| q34--h2--s1 | a3 | 1.000000 | 0.004160 | 2.038 / 1.454 | 30.034 |
| q34--h2--s1 | a4 | 1.000000 | 0.044594 | 2.038 / 0.877 | 30.034 |
| q34--h2--s1 | a5 | 1.000000 | 0.116535 | 2.038 / 0.138 | 90.101 |
| q34--h2--s1 | l0 | 1.000000 | 0.006250 | 1.792 / 1.792 | not logged |
| q34--h2--s1 | l1 | 1.000000 | 0.013079 | 1.792 / 0.329 | not logged |
| q34--h2--s1 | l2 | 0.333333 | 1.348744 | 2.975 / 2.975 | not logged |
| q34--h2--s1 | l3 | 1.000000 | 0.000000 | 2.975 / 0.002 | not logged |
| q34--h2--s1 | l4 | 0.333333 | 0.678005 | 3.934 / 3.934 | not logged |
| q34--h2--s1 | l5 | 0.333333 | 0.711383 | 3.934 / 2.220 | not logged |
| g34--h2--s0 | a0 | 1.000000 | 0.035397 | 15.292 / 3.279 | 280.285 |
| g34--h2--s0 | a1 | 0.333333 | 0.490708 | 15.292 / 4.186 | 254.457 |
| g34--h2--s0 | a2 | 1.000000 | 0.024122 | 15.292 / 1.588 | 303.377 |
| g34--h2--s0 | a3 | 1.000000 | 0.042267 | 15.292 / 12.559 | 103.948 |
| g34--h2--s0 | a4 | 1.000000 | 0.017339 | 15.292 / 7.376 | 119.613 |
| g34--h2--s0 | a5 | 1.000000 | 0.039807 | 15.292 / 1.221 | 386.136 |
| g34--h2--s0 | l0 | 1.000000 | 0.022152 | 6.093 / 6.093 | not logged |
| g34--h2--s0 | l1 | 1.000000 | 0.000819 | 6.093 / 0.776 | not logged |
| g34--h2--s0 | l2 | 0.333333 | 1.680496 | 4.356 / 4.356 | not logged |
| g34--h2--s0 | l3 | 1.000000 | 0.001248 | 4.356 / 0.709 | not logged |
| g34--h2--s0 | l4 | 0.333333 | 11.468750 | 5.472 / 5.472 | not logged |
| g34--h2--s0 | l5 | 0.333333 | 0.533611 | 5.472 / 3.283 | not logged |
| g34--h2--s1 | a0 | 0.333333 | 1.175165 | 15.424 / 4.517 | 236.714 |
| g34--h2--s1 | a1 | 1.000000 | 0.034567 | 15.424 / 5.416 | 201.435 |
| g34--h2--s1 | a2 | 1.000000 | 0.011572 | 15.424 / 2.295 | 245.714 |
| g34--h2--s1 | a3 | 1.000000 | 0.025977 | 15.424 / 12.988 | 83.636 |
| g34--h2--s1 | a4 | 1.000000 | 0.021256 | 15.424 / 8.241 | 102.949 |
| g34--h2--s1 | a5 | 1.000000 | 0.060970 | 15.424 / 1.766 | 356.347 |
| g34--h2--s1 | l0 | 1.000000 | 0.070736 | 7.109 / 7.109 | not logged |
| g34--h2--s1 | l1 | 0.733333 | 0.864931 | 7.109 / 0.057 | not logged |
| g34--h2--s1 | l2 | 0.333333 | 1.522058 | 4.964 / 4.964 | not logged |
| g34--h2--s1 | l3 | 0.333333 | 4.531401 | 4.964 / 0.564 | not logged |
| g34--h2--s1 | l4 | 0.333333 | 2.583886 | 6.180 / 6.180 | not logged |
| g34--h2--s1 | l5 | 0.333333 | 1.355663 | 6.180 / 1.593 | not logged |
| g34--path--s0 | a0 | 0.177778 | 4.676140 | 14.178 / 6.160 | 397.551 |
| g34--path--s0 | a1 | 0.141270 | 3.868726 | 14.178 / 5.445 | 794.642 |
| g34--path--s0 | a2 | 0.206349 | 2.233776 | 14.178 / 3.091 | 893.458 |
| g34--path--s0 | a3 | 0.136142 | 9.770497 | 14.178 / 12.264 | 371.858 |
| g34--path--s0 | a4 | 0.136142 | 6.340162 | 14.178 / 7.200 | 429.133 |
| g34--path--s0 | a5 | 0.176720 | 2.363359 | 14.178 / 3.043 | 1743.160 |
| g34--path--s0 | l0 | 0.136040 | 2.142030 | 3.711 / 3.711 | not logged |
| g34--path--s0 | l1 | 0.452381 | 1.768234 | 3.711 / 1.569 | not logged |
| g34--path--s0 | l2 | 0.218695 | 1.927454 | 3.232 / 3.232 | not logged |
| g34--path--s0 | l3 | 0.385714 | 2.025463 | 3.232 / 2.225 | not logged |
| g34--path--s0 | l4 | 0.022222 | 2.566808 | 3.494 / 3.494 | not logged |
| g34--path--s0 | l5 | 0.022222 | 2.389128 | 3.494 / 2.866 | not logged |
| g34--path--s1 | a0 | 0.201058 | 4.019353 | 14.462 / 4.918 | 466.636 |
| g34--path--s1 | a1 | 0.179012 | 4.754158 | 14.462 / 5.013 | 672.508 |
| g34--path--s1 | a2 | 0.092593 | 2.911607 | 14.462 / 2.855 | 838.867 |
| g34--path--s1 | a3 | 0.129630 | 10.909862 | 14.462 / 12.642 | 356.537 |
| g34--path--s1 | a4 | 0.081481 | 6.999005 | 14.462 / 7.850 | 404.225 |
| g34--path--s1 | a5 | 0.119048 | 2.944014 | 14.462 / 2.825 | 1765.218 |
| g34--path--s1 | l0 | 0.182540 | 2.127634 | 3.896 / 3.896 | not logged |
| g34--path--s1 | l1 | 0.319048 | 3.106701 | 3.896 / 1.154 | not logged |
| g34--path--s1 | l2 | 0.333333 | 1.852854 | 3.005 / 3.005 | not logged |
| g34--path--s1 | l3 | 0.272222 | 2.994847 | 3.005 / 1.572 | not logged |
| g34--path--s1 | l4 | 0.022222 | 2.497089 | 3.297 / 3.297 | not logged |
| g34--path--s1 | l5 | 0.022222 | 2.709667 | 3.297 / 2.620 | not logged |
