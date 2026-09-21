# Prespecified Aperture medical main study

## Motivation and current evidence

The last uploaded calibration-controlled H2 run (results branch `355c0b1`) reports Aperture NLL 0.6076 +/- 0.0198 and F1 0.6166 +/- 0.0929. LoRA-4 has F1 0.6429 +/- 0.0737 and raw NLL 1.3259 +/- 0.5812. This motivates testing reusable probability-quality benefits beyond one center. It does not establish universal classification superiority. New medical cohorts are not included because their results are favorable: all populations below are locked before execution.

The primary question is whether a fixed visual-state correction recipe improves prediction at new hospitals under small, matched label budgets, beyond output bias and calibrated weight adaptation. The second question is whether the recipe transfers to a different tissue-label vocabulary. This is a new prospective expansion, separate from historical full-support and final-round calibration tables.

## Population and isolation

### CAMELYON17-WILDS

Use the official `metadata.csv` and original 96x96 PNGs. Labels refer to tumor in the CENTRAL 32x32 region, not arbitrary tumor anywhere. Use a fixed normal/tumor prompt that explicitly refers to the central region; this is recorded as a new prompt contract. Target centers are all of 0/1/2/3/4. No source-hospital training is introduced; each center receives its own labeled adaptation episode. This is NOT the official WILDS source-training/domain-generalization estimator.

Within each center, hash patient IDs with split seed 310927 into fixed pools: approximately 40% fitting, 20% calibration, 40% query. A patient and all their nodes/slides belong to only one pool. This is stronger isolation than splitting by patient-slide pair. All seeds and label budgets reuse the same query pool. Query selection is balanced at 250 per class and spreads sampling over patients. Insufficient eligible patients/classes stop preparation; no fallback to patch-level mixing.

H2 has been examined previously and is reported as an anchor. Centers 0/1/3/4 form the prespecified expansion summary, alongside the all-five-center summary. Neither grouping depends on measured outcomes. The study samples balanced patches; it does not estimate clinical disease prevalence or patient-level diagnosis.

### PathMNIST-224 external tissue study

Use the official 224px NPZ, pinned URL and MD5 in `data.py`. Read only the 7,180-image CRC-VAL-HE-7K test cohort; no model is trained on its upstream 100K source cohort here. Use all nine tissue classes, with a fixed A-I mapping written in every question. Candidate assignments do not change between questions. Native 224px images remain 224px before the existing processor; no 28px replacement.

The NPZ does not expose patient or slide identifiers. Identical RGB images are grouped by content hash and deduplicated; conflicting labels fail. Content groups are partitioned into 40/20/40 fit/calibration/query pools. Query selection takes 100 per class. These are TARGET-COHORT support/query episodes, not the official MedMNIST train/test score and not a patient-disjoint clinical validation. This limitation is visible in every split record and must appear in the paper.

## Label budget and optimization

Use support seeds 0/1/2, nested 8 or 16 labels per class. Exactly two labels per class are held out for temperature calibration. Thus CAMELYON totals 16 -> (12 fit,4 cal) and 32 -> (28 fit,4 cal); PathMNIST totals 72 -> (54 fit,18 cal) and 144 -> (126 fit,18 cal). Query images do not change across these budgets. All arms share identical support partitions.

Aperture is unchanged: Qwen2.5-VL-7B, BF16, decoder layers 14/27 K/V, rank16, Full matrix, alpha3, four full-support gradient accumulations, Adam lr0.05, l2=0.001, clip1, support SUM loss consistent with the recorded single-image recipe. Optimizes 1,024 coefficients; fixed bases are additional stored state. Random visual uses the identical controller/fit recipe with a seeded random basis and no gradient-based extraction.

LoRA is rank16, alpha32, Q/K/V/O, dropout0.05, AdamW lr2e-4, accumulation1, one or four complete fit-support passes. Both schedules are retained. Selection uses held-out calibration NLL, independently for raw and temperature-scaled scores. Positive temperature grid 0.2..20 with fixed log-T penalty0.01; query labels never fit or choose T. Frozen output bias uses ALL the original support labels, matching the total label budget without discarding usable supervision. No query-selected lens amplitude, layer, rank or early stopping.

## Reproducibility gate

Use one process per GPU, Python hash seed0, CPU BLAS threads1, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch algorithms, TF32 off, cuDNN benchmarking off and math SDPA when applicable. These are explicit NEW execution settings, not a claim that historical reruns were bitwise reproducible.

Two actual-model support-only gradient audit gates cover the binary and nine-class prompt interfaces. Before the full fit matrix, repeat the representative H0 n16 seed0 and PathMNIST n72 seed0 Aperture and LoRA-4 fits. Run these representative fits sequentially on the first declared GPU. Compare held-out support score vectors and class decisions, tolerance1e-3 and zero flips. Failure stops new launches, retains both fits and requires numerical investigation. The later saved-state reload also must reproduce calibration scores BEFORE any query forward. This is not a query-based success filter. If strict deterministic kernels are unsupported or OOM, stop; do not change precision/resolution silently.

## Outcomes and planned tables

Primary: candidate NLL, unscaled Aperture versus fixed LoRA schedules, calibration-selected LoRA, output bias and Random visual; separate raw and +T results. Report all five hospitals and both budgets, plus all-center and expansion-center means and worst-center endpoints. The baseline portfolio is not cherry-picked from query performance.

Complementary: Macro-F1, BA, binary AUROC/AUPRC; nine-class macro one-vs-rest AUROC and macro AUPRC; Brier and ECE. AUC/PR values are missing rather than fabricated when a subgroup has only one class. Report per-patient query NLL for CAMELYON, and support-seed sample SD; patches are not independent hospitals/patients. Temperature may alter multiclass AUC even while preserving argmax. No significance threshold determines inclusion.

Main table: five hospitals (columns), methods (rows), NLL and F1 panels at n16. A second n32 panel or figure shows nested label-budget sensitivity. Include AUROC as a discrimination check against pure calibration explanations. The external table reports all nine classes jointly at n72/n144. A compact resources table includes both basis and learned coefficients. Expected improvements are NOT filled in ahead of measurement.

The executable plan has 6 domains x 2 budgets x 3 seeds x 5 arms = 180 states, of which 144 are trainable, plus four repeat-fit diagnostics and two GPU audit gates. Postprocessing does not train another VLM. No new backbone or architectural change is introduced. Stop after the fixed matrix; no outcome-driven hospital/dataset expansion.

## Data sources and reuse

- WILDS: https://wilds.stanford.edu/datasets/ and https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py (CC0; full download is roughly 10.7GB compressed).
- MedMNIST: https://medmnist.com/ and https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py (PathMNIST CC BY 4.0).
- ICLR reviewing guidance: https://iclr.cc/Conferences/2027/ReviewerGuidelines . This study focuses on motivation, reliable comparisons and reproducibility rather than adding unrelated datasets.

Only code/configuration/documentation are published here. Store downloaded images and prepared manifests locally. No clinical deployment claim follows from these few-shot patch experiments.
