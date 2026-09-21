# Aperture medical main experiment implementation plan

## Goal and fixed scope
Expand the existing CAMELYON17 binary visual task to all five official hospitals.
Preserve Aperture's operator and archived results. Primary cohort: total support32,
24 fit + 8 calibration, 500 query patches per hospital, support seeds 10/11/12.
Primary backbone Qwen2.5-VL-7B (five fitted arms), independent InternVL3-8B replication
(four arms). Optional Qwen support16/support64 curve; no hospital or seed selection
based on query performance. This is one dataset, five centers, not five datasets.

## Architecture
New additive aperture_medical_multimodel package. Reuse validated eventttt loaders,
vigor_handoff controller/fitting and visual_lens support-only gradient audit.
Do not change any old protocol or result. Build strict three-way patient-disjoint
manifests from complete official metadata, excluding all query patients from
ALL fit/calibration budgets and seeds. Fresh outputs are versioned and sealed.

## Tasks
- [x] Write failing tests for official metadata, disjointness, label budget,
  deterministic nested supports and result aggregation.
- [x] Implement patient partition, deterministic sample selection, content locks.
- [x] Implement model/budget plan and preflight, without a GPU import in planning.
- [x] Implement fitting, held-out calibration, random-basis arm, query scoring,
  AUROC/AP/patient-macro metrics and support-only repeatability gate.
- [x] Implement multi-GPU subprocess scheduling and complete-only table export.
- [x] Run all legacy and new CPU tests and synthetic end-to-end export test.
- [ ] Publish additively on a branch; preserve remote advances and all old files.

## Review focus
Patient IDs are not slide IDs; official hospital IDs are 0..4 (official test=2).
Fit/calibration are patient-disjoint as well as disjoint from query. Missing
images or labels stop rather than changing inclusion. Strong LoRA budgets and
calibration use the same fixed label budget. Patient averaging is not clinical
patient-level diagnosis. No score threshold is selected on query. H2 is already
exploratory; cross-hospital results are prospective, not a pretraining OOD claim.
