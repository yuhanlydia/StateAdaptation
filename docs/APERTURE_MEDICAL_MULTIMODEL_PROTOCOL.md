# Aperture: prospective multicenter pathology main experiment

## Why this extension
The completed H2 calibration-controlled experiment gives Aperture NLL
0.6076 +/- 0.0198, compared with Frozen 0.6669, all-support output bias 0.6526,
LoRA-1pass 0.7371 and LoRA-1pass+temperature 0.7199. Aperture F1 is 0.6166,
compared with LoRA-4passes 0.6429. These support predictive-quality research,
not a claim that every accuracy measure exceeds LoRA or that the system is
clinically ready. The next main experiment tests whether the useful behavior
extends beyond the previously inspected hospital. No favorable result is promised.

## Hypothesis and primary endpoint
A fixed, reusable visual-state controller can improve target-center candidate
probability prediction without a broad model-weight update. Primary endpoint:
NLL averaged equally over the five centers and three support seeds, separately
for each backbone. Report every center. Primary method is RAW Aperture (no
query-selected temperature/strength). Report raw AND temperature-calibrated
LoRA-1/LoRA-4, calibration-selected LoRA, Frozen, Frozen+T, and all-support bias.
Random visual residuals match rank, sites, bound, steps and support in the primary
Qwen matrix. Report F1, balanced accuracy, AUROC, average precision, Brier, ECE
and patient-macro NLL as complementary results, with support-seed mean/sample SD.
No query-tuned threshold, LR, layer, alpha, support seed or dataset dropping.

## Dataset and scope
One dataset: CAMELYON17-WILDS, all five hospital indices 0/1/2/3/4 from the
OFFICIAL `metadata.csv`. Do not call them five datasets. WILDS code identifies
TEST_CENTER=2 and VAL_CENTER=1; never reinterpret indices from a paper figure.
The label is tumor presence in the central 32x32 area of the original 96x96
patch. Retain the existing question and candidate strings normal/tumor.
This is supervised target-center few-shot adaptation on new patient partitions,
not the official WILDS domain-generalization setting or whole-slide diagnosis.
H2 has been used for exploratory design. Other centers have not been examined
in the supplied experiment records; this is not a claim about VLM pretraining.

## Fixed patient partitions
Load complete official metadata, NOT the prior 3,395-file release subset or an
early-stopped Hugging Face stream. Hash-order patients. Reserve approximately
40% of patients for query, 20% for calibration and the rest for fitting; require
at least two patients per role and at least six available patients per center.
A deterministic metadata-only feasibility search ensures both support classes
have enough samples. It never reads model outputs. Failing feasibility stops;
do not relax to patch-level splits or choose a successful hospital instead.

All fit/calibration/query roles are pairwise PATIENT- and SLIDE-disjoint.
Different slides from the same patient cannot cross roles. All sample IDs,
image paths and label/index contracts are checked. Keep one query list per
center across every seed, budget and backbone. Query sampling cycles over
reserved patients without forcing class balance. Thus the evaluation is
patient-balanced patch sampling, not clinical prevalence or pooled patient
classification. `patient_macro_nll` is the mean of per-patient patch losses.

## Main and follow-up blocks
- Primary: Qwen2.5-VL-7B; 5 hospitals x 3 support seeds (10,11,12) x 5 arms =
  75 states; 60 trainable fits, 15 Frozen states.
- Backbone replication: InternVL3-8B; same 5 hospitals x 3 seeds x 4 arms =
  60 states; 45 trainable fits. Random is not repeated in this block.
- Main total: 135 states, 105 trainable fits; 32 labels per center/seed,
  24 fitting + 8 calibration (four examples/class), 500 query patches/center.
- Optional label curve: Qwen only, all 5 hospitals x 3 seeds x budgets16/64 x
  4 arms =120 states. Budget16 gives12+4; budget64 gives48+16. Samples are nested
  within their fixed patient roles; query stays fixed. This is a whole-recipe
  learning curve, not a compute-matched experiment.
- Two additional repeated-fit controls (one per backbone) use SUPPORT only and
  are excluded from the scientific performance table.

The default run includes primary and replication. The curve is optional and
explicitly requested by `--phases curve`; it is not triggered by weak results.
A 500-query center evaluated in three support seeds is NOT 1,500 independent
patients or queries. Do not create a significance claim by treating patches
from the same patient as independent clinical replicates.

## Fixed method and baseline settings
Keep two decoder K/V layers14+27, rank16, Full bounded residual, alpha3,
four complete-support coefficient updates, LR.05, L2.001, full answer span and
legacy single-image support-sum loss. No backbone update in Aperture/Random.
LoRA rank16, alpha32, Q/K/V/O, dropout.05, LR2e-4; one/four passes, accumulation1.
At32labels this means24/96 optimizer steps, respectively, on the SAME24fit rows;
Aperture has4updates and96fitting visits plus24basis-extraction visits. These are
same-label-budget comparisons, not same-compute or equal optimizer-step claims.
Frozen+output-bias gets all32labels; no labels are discarded to weaken this control.
Positive temperature, fixed grid and shrinkage, is fitted on the8calibration
labels for each already fitted method. LoRA schedule selection uses calibration
NLL only and is sealed before any query forward. Publish all alternatives.

## Numerical integrity before launching the large matrix
One GPU process per device. Set PYTHONHASHSEED=0, CUBLAS_WORKSPACE_CONFIG=:4096:8,
OMP/MKL/OPENBLAS thread counts1, deterministic PyTorch algorithms, no TF32 or
cuDNN autotuning. Unsupported deterministic kernels raise, not silently fallback.
Reuse the support-only real-VLM gradient/checkpoint/mask/save-load audit.
Then fit the H0/seed10/support32 Aperture reference and an independent fresh
process repeat ON THE SAME GPU. Compare held-out SUPPORT candidate scores, not
query accuracy. Max absolute score difference must be <=.005; retain both fits
and fail before the large matrix on disagreement. This tests one support unit
per backbone, not a universal bitwise reproducibility guarantee. Do not pick a
better repeat or increase the tolerance to continue. Descriptive small-matrix
operator norms are computed on CPU to avoid the previously observed cuSOLVER
workspace failure. Old source and result files are not changed.

## Hardware and input contracts
Qwen2.5-VL-7B uses the existing448pixel single-image interface. InternVL3 uses
its existing224pixel/64visual-token processor contract. BF16,24GB-or-larger GPU;
no silent quantization, model replacement, resolution reduction, or CPU offload.
Keep the already validated model environment. A CPU unit-test pass is NOT a
real-model forward/backward or an end-to-end benchmark pass. Report hardware,
model-content fingerprints, source signature, temperature, optimizer updates,
fit/extraction time and serialized adaptation size. Existing result roots remain
untouched. Prospective root: `runs/aperture_medical_multimodel_v1`.

## Paper use
Main table: per-center NLL and F1, mean +/- SD across supports; all methods;
separate panels for Qwen and InternVL. Figure: per-center Aperture vs calibrated
LoRA and output bias; retain all centers, not only wins. Appendix: full metrics,
patient/slide counts, class counts, support budgets, settings and repeatability
checks. Optional figure:16/32/64label curves using the full predeclared curve.
A broad medical-performance claim needs a genuinely independent medical dataset;
this experiment supports multicenter pathology generalization only. Keep BRIGHT
as the already measured nonmedical evaluation instead of relabeling centers as
datasets. No new clinical safety or acceptance assertion follows from completion.

## Sources checked on 2026-09-21
- Official WILDS dataset implementation:
  https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py
- Official CAMELYON17 data description:
  https://camelyon17.grand-challenge.org/Data/
- Existing results (fixed snapshot):
  https://github.com/yuhanlydia/StateAdaptation/tree/355c0b103ecc5130a31aaa93a8705828fe7027cd/reports/aperture_final_2026-09-21/results/paper_exports
