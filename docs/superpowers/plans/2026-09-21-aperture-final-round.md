# Aperture final experimental round — implementation plan

Goal: add a finite, support-budget-matched last-round evaluation to the existing
Aperture implementation; never change the published historical result roots.

Architecture: additive `aperture_final` package using the existing VLM loaders,
Backend, residual controller, LoRA training and support-only audit. Each domain/
support seed uses a fixed within-support fit/calibration split; query labels never
select temperature, checkpoint or an arm. All query metrics are exported, not
only wins. GPU integration is run by the user's agent after CPU verification.

User brief: strengthen the existing paper, not invent a new model or expand to
new datasets. Finished outputs must become paper tables/plots without manual
transcription. No promise that further evaluations will all outperform LoRA.

Review basis: ICLR 2027 ReviewerGuidelines (read 2026-09-21): claims supported by
correct, rigorous, reproducible evidence; novelty/significance need not be SOTA.
The supplied template is a formatting/reproducibility source, not a scoring rubric.

## Fixed scope
- Qwen2.5-VL-7B, existing eight domains: four BRIGHT events, Camelyon hospital 2,
  and the three existing ManipBench Q1 domains. Original support seeds 0/1/2.
- All arms retain the same total labels. Reserve 2 examples/class within each
  original support for calibration: BRIGHT 18+6, Camelyon 12+4, ManipBench 24+8.
  Calibration is sample-held-out, not guaranteed tile/patient-held-out; query
  group separation is enforced. This is a new protocol, not replacement P0 data.
- Frozen, LoRA-1pass, LoRA-4passes, Aperture-r16: 24 units, 96 states, 72 fits.
- Fit positive temperature for EACH arm on held-out support; fit an output-bias
  comparator on all original support scores (the predictor was never fitted); select LoRA schedule with calibration
  NLL only. Seal all choices BEFORE any query scoring.
- Four fixed image degradations on a hash-chosen 60-query probe, BRIGHT and
  Camelyon seed0 only; same clean-trained state and calibrator for all conditions.
- Aperture residual-dose probe {0, .5, 1} on the same 60-query subset (seed0).
- Record fitting wall time, optimized/storage scalars and repeated clean-query
  latency; no promised hardware hours or implicit lower-resolution fallback.

## Tasks
1. Write failing tests for support/calibration separation, finite temperature,
   invariant argmax, deterministic corruption and query-independent selection.
2. Implement pure calibration, split preparation, fixed plan and provenance.
3. Implement fit/calibrate and sealed read-only query evaluation via current APIs.
4. Implement barriers, one process/GPU, partial-run support and safe resume.
5. Implement complete-only paper CSV/LaTeX exports and optional plots.
6. Test numeric paths and orchestration on CPU/fake backend, full old CPU suites,
   syntax, dry plan and package apply. No synthetic metric becomes a paper result.
7. Publish through available GitHub write interface, or explicitly deliver tested
   patch if the connection is read-only. Verify remote commit before claiming push.

## Review focus
Missing group IDs, overlapping fit/cal images, different query order across seeds,
resume after changing sources/calibrators, unhandled GPU OOM or partial results.
