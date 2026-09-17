# Visual Lens rebuttal evidence export — 2026-09-17

This directory is the compact, public export of the completed local GPU campaign. Raw predictions, fitted controllers, datasets, and base-model weights remain in ignored local storage.

## Coverage

- Canonical P0: 83/83 sealed jobs (3 audit, 63 matched, 8 objective, 9 evidence).
- Follow-up debugging: 29/29 sealed jobs, plus fresh current-source Qwen2.5-VL and InternVL3 audits.
- Final local verification: 57/57 CPU tests passed; no failed artifacts or stale run locks.

## Files

- `p0_results.md`: canonical matched and objective tables.
- `p0_visual_dependence.md`: fixed-state visual intervention contrasts.
- `p0_summary.json`: exact paired bootstrap results and coverage metadata.
- `p0_final_findings.md`: interpretation, fixes, and reproduction limits.
- `debug_findings.md`: Hawaii, Turkey, RoboFail, and fixed layer-14 analysis.
- `debug_summary.json`: exact metrics for all 29 exploratory jobs.
- `historical_hawaii_protocols.md`: retained Hawaii Lens-over-LoRA results and their query-oracle boundary.
- `SHA256SUMS`: hashes of this export.

## Interpretation boundary

The canonical 83-job P0 matrix remains the primary reproduction. Follow-up jobs are explicitly exploratory. In particular, layer 14-only was selected after diagnosing Turkey, then fixed and evaluated across all four BRIGHT events. It improves the 12-state aggregate while showing substantial event interaction; it is not presented as a preregistered universal replacement.

BRIGHT uses official checksum-verified Zenodo assets. Historical private support manifests were unavailable, so the fixed folds were deterministically regenerated from the original repository procedure with `PYTHONHASHSEED=0`. These are controlled reproduction results, not a byte-for-byte restoration of the private manifests.
