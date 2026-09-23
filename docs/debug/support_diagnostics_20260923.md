# Support-only tuning diagnostics

No evaluation/query files or query metrics were read. This report inspects the 192 candidate fit histories and saved calibration predictions; selection perturbations below are diagnostics only, not replacement selections.

## Per model/domain/seed

F1 is on calibration only. LOO agreement = fraction of single-calibration-image deletions retaining the original selected candidate (same F1, then raw NLL, then ID rule; all classes retained in macro averaging). It measures fragility, not performance on independent data. Loss is first → last recorded pre-update fitting loss; LoRA records pass averages, so it is not directly comparable to Aperture step snapshots.

| Unit | Method / selected | Cal F1 | NLL | F1-tied configs | LOO agreement | Fit loss first→last |
|---|---|---:|---:|---:|---:|---:|
| g34--h2--s0 | aperture a4 | 1.0000 | 0.0173 | 5/6 | 2/4 | 15.292→7.376 |
| g34--h2--s0 | lora l1 | 1.0000 | 0.0008 | 3/6 | 4/4 | 6.093→0.776 |
| g34--h2--s1 | aperture a2 | 1.0000 | 0.0116 | 5/6 | 3/4 | 15.424→2.295 |
| g34--h2--s1 | lora l0 | 1.0000 | 0.0707 | 1/6 | 3/4 | 7.109→7.109 |
| g34--path--s0 | aperture a2 | 0.2063 | 2.2338 | 1/6 | 17/18 | 14.178→3.091 |
| g34--path--s0 | lora l1 | 0.4524 | 1.7682 | 1/6 | 17/18 | 3.711→1.569 |
| g34--path--s1 | aperture a0 | 0.2011 | 4.0194 | 1/6 | 17/18 | 14.462→4.918 |
| g34--path--s1 | lora l2 | 0.3333 | 1.8529 | 1/6 | 12/18 | 3.005→3.005 |
| q25--h2--s0 | aperture a5 | 1.0000 | 0.1016 | 5/6 | 3/4 | 1.745→0.288 |
| q25--h2--s0 | lora l1 | 1.0000 | 0.0633 | 2/6 | 4/4 | 0.949→0.821 |
| q25--h2--s1 | aperture a4 | 1.0000 | 0.2374 | 1/6 | 3/4 | 1.947→0.893 |
| q25--h2--s1 | lora l1 | 1.0000 | 0.1918 | 2/6 | 2/4 | 1.086→0.512 |
| q25--path--s0 | aperture a2 | 0.4037 | 1.5326 | 1/6 | 15/18 | 2.433→1.553 |
| q25--path--s0 | lora l3 | 0.6333 | 2.1323 | 1/6 | 18/18 | 2.145→1.091 |
| q25--path--s1 | aperture a1 | 0.2741 | 1.5961 | 2/6 | 15/18 | 2.303→1.446 |
| q25--path--s1 | lora l3 | 0.5672 | 1.6205 | 1/6 | 16/18 | 2.337→1.164 |
| q34--h2--s0 | aperture a4 | 1.0000 | 0.0191 | 6/6 | 3/4 | 0.935→0.427 |
| q34--h2--s0 | lora l1 | 1.0000 | 0.0988 | 2/6 | 1/4 | 1.771→0.504 |
| q34--h2--s1 | aperture a3 | 1.0000 | 0.0042 | 6/6 | 4/4 | 2.038→1.454 |
| q34--h2--s1 | lora l3 | 1.0000 | 0.0000 | 3/6 | 4/4 | 2.975→0.002 |
| q34--path--s0 | aperture a5 | 0.6370 | 0.7134 | 1/6 | 16/18 | 6.554→1.284 |
| q34--path--s0 | lora l1 | 0.6444 | 0.8778 | 1/6 | 17/18 | 2.326→0.821 |
| q34--path--s1 | aperture a0 | 0.5852 | 2.2479 | 1/6 | 14/18 | 5.711→2.181 |
| q34--path--s1 | lora l2 | 0.6487 | 1.2840 | 1/6 | 16/18 | 2.393→2.393 |
| q38--h2--s0 | aperture a4 | 1.0000 | 0.0428 | 5/6 | 3/4 | 1.170→0.455 |
| q38--h2--s0 | lora l1 | 1.0000 | 0.0540 | 2/6 | 4/4 | 1.463→0.508 |
| q38--h2--s1 | aperture a3 | 1.0000 | 0.0452 | 3/6 | 4/4 | 1.476→1.001 |
| q38--h2--s1 | lora l0 | 1.0000 | 0.0378 | 2/6 | 2/4 | 1.123→1.123 |
| q38--path--s0 | aperture a2 | 0.7333 | 0.9355 | 2/6 | 16/18 | 4.533→1.289 |
| q38--path--s0 | lora l3 | 0.9407 | 0.1287 | 1/6 | 18/18 | 2.120→0.582 |
| q38--path--s1 | aperture a5 | 0.5741 | 1.5846 | 1/6 | 13/18 | 4.899→1.247 |
| q38--path--s1 | lora l1 | 0.7259 | 1.5604 | 1/6 | 16/18 | 2.636→0.606 |

## Training numerical audit

192/192 fitting files read; 0 histories contain nonfinite loss/recorded gradient. LoRA histories do not save gradient norms: absence of recorded anomalies is not a gradient audit for LoRA.

| Model / domain | Aperture trajectories with any loss increase /12 | Preclip grad min/max | LoRA trajectories with loss increase /12 |
|---|---:|---:|---:|
| q25/h2 | 0/12 | 3.08/254 | 5/12 |
| q25/path | 0/12 | 15.9/356 | 5/12 |
| q34/h2 | 1/12 | 0.811/90.1 | 2/12 |
| q34/path | 0/12 | 38.6/603 | 4/12 |
| q38/h2 | 4/12 | 2.55/144 | 3/12 |
| q38/path | 4/12 | 30/2.01e+03 | 2/12 |
| g34/h2 | 0/12 | 20.2/386 | 3/12 |
| g34/path | 0/12 | 41.5/1.77e+03 | 3/12 |

## Evidence and limitations

- H2 has 12 fit + 4 calibration images per seed. All four calibration images come from patient_046, for both seeds. Thus two seeds do not provide two independent calibration patients. Every selected method/model/seed obtains calibration F1=1; Aperture has 1–6 equally high-F1 candidates. Qwen3-4B ties all six in both seeds. NLL usually selects among indistinguishable F1 candidates, and deleting one image frequently changes selection. This supports fixing validation diversity before enlarging the hyperparameter grid.
- Path has 54 fit + 18 calibration images, only 2 calibration examples per class; patient/slide IDs are unavailable. It cannot establish patient-disjoint generalization. Aperture's best calibration F1 is below LoRA's in all eight model/seed units, so the deficit exists before query and is not merely an unlucky query comparison. Neither method's best-of-six calibration score is an unbiased generalization estimate.
- No nonfinite recorded training loss/gradient was found. Aperture clipping is 1.0, while almost all recorded preclip norms exceed this (max approximately 2010). Clipping is expected under summed support losses; it is evidence of clipping, not proof of numerical collapse. Sum-reduction also changes regularization strength relative to the data term when sample count changes. Log both data loss and penalty and hold the convention fixed in CV.
- No rank ablation was run. The evidence does not justify increasing rank as the first fix. Model differences, loss decreases and calibration differences do not by themselves establish insufficient controller capacity.
- Gemma Path a3→a4 (same alpha=1 and lr=.01, 4→12 updates) reduces fit loss and NLL but fails to improve calibration F1 reliably: s0 F1 .136→.136; s1 .130→.081. At alpha=3, relative layers and 12 steps, a5 fit loss reaches 3.04/2.83, yet F1 .177/.119. Merely extending updates is not proven beneficial for classification. Comparing a0→a2 (fixed alpha/lr/steps, changed sites) gives s0 F1 .178→.206 but s1 .201→.093: position sensitivity is real, and no universally superior position is identified.

## Label-token objective check

Source `aperture_table/backend.py` fits `-lp.mean()` of the correct answer across the full language vocabulary; scoring stores `lp.sum()` per candidate, then normalizes across candidates. The field name `mean_log_scores` is misleading: the actual values are sums. This is not, by itself, a model execution fault.

CPU-only local tokenizer checks (no model/GPU load) found Qwen2.5/Qwen3 `normal`=[8252], `tumor`=[83,68261], whereas Gemma has one token for both. All four models encode each Path label A–I as one token. These are standalone-token checks; actual native full-prompt answer spans must also be checked before declaring a production mismatch. For Qwen H2, a two-token tumor answer receives mean token loss during fitting but sum sequence scoring at inference. This can change relative class emphasis. Test sum-answer fitting as a separately declared objective, applied to both Aperture and LoRA, rather than silently changing historical results or changing scoring after seeing query.

For Path, mean vs sum cannot explain the problem if full-prompt label spans remain one token. However, full-vocabulary answer NLL and candidate-normalized classification NLL differ: loss can fall by moving probability mass into all allowed letters without improving discrimination among them. High Gemma initial answer losses (about 14 nats) and much smaller calibration class-normalized NLL demonstrate that the quantities are not interchangeable, but do not prove this mechanism caused the F1 deficit. A candidate-normalized CE diagnostic with fixed logits for A–I can isolate this, and must be a new objective row for both methods. Raw-vocabulary and restricted-candidate support losses should both be logged.

## Feasible H2 patient-grouped three-fold design

Use only each seed's original 16 labeled support images (fit + calibration), not query. Every patient has balanced normal/tumor labels; deterministic greedy bin packing is sufficient: sort groups by decreasing image count, tie-break patient ID, place the next group in the fold with fewest images, tie-break fold index.

| Seed | Fold | Validation patients | Validation normal/tumor | Training normal/tumor |
|---|---|---|---|---|
| 0 | 0 | 046, 044 | 3/3 | 5/5 |
| 0 | 1 | 048, 052 | 3/3 | 5/5 |
| 0 | 2 | 040, 041 | 2/2 | 6/6 |
| 1 | 0 | 041, 044 | 3/3 | 5/5 |
| 1 | 1 | 046, 052 | 3/3 | 5/5 |
| 1 | 2 | 040, 048 | 2/2 | 6/6 |

All folds retain both classes in train and validation, and patient sets are disjoint within a fold. This is a changed support-selection protocol and needs a new run root. Fit preprocessing/bases/controller only on each fold's training subset. Aggregate held-out predictions according to a prespecified rule (e.g. pooled 16-example macro-F1, then NLL, then candidate ID); additionally report all three fold scores to reveal patient dependence. If selecting on patient-balanced scores instead, specify this before running. Final refit on all 16 uses the same total label budget as before but a different fitted subset, so identify it explicitly; fit any temperature from out-of-fold predictions, never in-sample final fit. New query evaluation remains exploratory because prior query results have already been viewed.

## Bounded next experiment: hypotheses per combination

| Combination | Support-based hypothesis | Small next test |
|---|---|---|
| q25/H2 | Single-patient validation and sequence-length objective mismatch can dominate selected settings. Current a4/a5 attain perfect calibration F1 but selection changes after deleting one image. | Patient CV first. Four Aperture configs: historical a0; a4; a5; single-mid alpha=1, lr=.01, 12 steps. Four LoRA configs: lr 5e-5/2e-4 × 1/4 passes. Token objective is a separate paired diagnostic, not another hidden selection dimension. |
| q34/H2 | Six-way perfect-F1 ties mean the current held-out patient provides essentially no F1 selection signal. | Same patient CV and compact four-config set (a0, a3, a4, single-mid alpha=1 lr=.01 steps12). No evidence favors rank increase. |
| q38/H2 | Perfect-F1 ties and some nonmonotone support trajectories suggest conservative strength and patient CV matter more than capacity. | Defer in the five-combination priority run; apply the same four-way patient-CV design later, without selecting settings from query. |
| g34/H2 | High vocabulary loss coexists with perfect candidate F1; model selection is patient-limited, not evidence that more training necessarily helps. | Four configs a0,a2,a4,a5 under patient CV. Same four LoRA configs. Do not stop based on vocabulary loss alone. |
| q25/Path | Both seeds favor changed/single-layer positions; selected F1 .404/.274 is below LoRA .633/.567, while fit loss falls. This supports location/strength testing more than rank expansion. | Six Aperture configs: retain a0,a1,a2; add single-mid alpha=1 lr=.01 steps12; quarter/three-quarter alpha=1 lr=.01 steps12; single-mid alpha=3 lr=.01 steps12. Keep six LoRA settings from v1 as equal nominal search budget. Original 18-image calibration only is weak validation; require future independent evaluation before confirmatory claims. |
| q34/Path | s0 nearly ties LoRA (.637 vs .644), s1 has a larger gap; best position/budget differs by seed. | Defer; replicate compact position/strength comparisons on stronger validation rather than a broad rank grid. |
| q38/Path | Candidate choice differs across seeds and some loss paths rise; no rank-capacity evidence despite F1 deficit. | Defer; retain a2/a5, add a conservative single-mid setting in future support-only comparison. |
| g34/Path | Training improves but F1 remains low; low-strength settings underfit answer loss, and position/longer training have seed-dependent classification effects. Objective alignment deserves an isolated test. | Six Aperture configs: a0,a2,a5; single-mid alpha=3 lr=.01 steps12; relative alpha=3 lr=.01 steps24; relative alpha=3 lr=.01 steps12 with candidate-normalized CE. For LoRA: four standard lr/pass settings plus two candidate-normalized CE settings at lr5e-5 with 1/4 passes. Objective change must be explicitly separated in reporting; this is exploratory model development, not a clean rank-only hyperparameter ablation. |

The strict hyperparameter-only alternative for Gemma replaces the candidate-CE entries with a lower-lr/longer-budget setting and runs CE separately afterward. Prefer this if the current paper must retain a single objective definition. In either design, record the six proposals before executing; do not add configurations until one beats LoRA on the old query. The immediate five-combination priority is q25/q34/g34 H2 and q25/g34 Path. There is no support-only evidence guaranteeing an Aperture win.
