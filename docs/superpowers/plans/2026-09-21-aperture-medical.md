# Aperture medical main-study implementation plan

Goal: extend the existing frozen-VLM medical result into a prespecified multicenter study, retaining all outcomes and strong calibration controls. No change to the Aperture operator or historical outputs.

Design: five CAMELYON17 centers; patient-disjoint fitting/calibration/query pools; fixed class-balanced query of 500 per center; nested 16/32-label budgets; three support seeds. H2 is a previously examined anchor, not a new blind discovery. Independent PathMNIST-224 CRC target-test cohort uses fixed-label nine-class tissue classification, 72/144 labels, 900 queries, image-content-disjoint pools; no patient-disjointness claim without metadata. The pathology extension is not the official WILDS/MedMNIST estimator.

Models: existing Qwen2.5-VL-7B loader and candidate scoring. BF16 24GB+; five arms Frozen, LoRA-1, LoRA-4, Random visual, Aperture; output bias and positive temperature postprocessing. No query-tuned schedules, images or budgets. All methods share exact fit/calibration/query at each comparison unit.

Implementation tasks:
1. Test deterministic group pools, nested budgets, missing groups, content overlap and source tampering; implement preparation.
2. Test protocol counts and strong controls; reuse audited loader/controller/trainer in an additive medical runner.
3. Test AUC/AUPRC, replay checks, support-only selection and incomplete-export rejection; implement metrics/reporting.
4. Add full-matrix CLI, per-GPU serial scheduling, download instructions and locked provenance. Run CPU tests and syntax checks before publishing.

Review focus: central 32x32 CAMELYON label; patient/node must not become slide-only leakage protection; PathMNIST patient metadata absent; source hash includes inherited code; temperature argmax invariance; missing assets cannot cause silent hospital removal; historical H2 results not overwritten; fitting nondeterminism not declared solved without GPU replay.
