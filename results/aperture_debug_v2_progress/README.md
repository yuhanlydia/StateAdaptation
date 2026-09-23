# Targeted debug v2 — interim snapshot

Snapshot UTC: 2026-09-23T09:00:00.154446+00:00. Completed query states: 16/20. This is an interim export; missing states are pending, not estimated.

Previously inspected query; targeted adaptive exploratory debugging, not independent test evidence.

All 192 search fits and 20 selected-fit repeat gates completed. Completed evaluation metrics were recomputed from sealed predictions before export. The active run continues independently; the final result will be published on `aperture-targeted-debug-20260923` under `results/aperture_debug_v2/`.

| Model | Domain | Completed / 4 | Aperture F1 % | v2 LoRA F1 % | v1 LoRA F1 % |
|---|---|---:|---:|---:|---:|
| q25 | h2 | 4 / 4 | 52.96 | 64.35 | 64.35 |
| q34 | h2 | 4 / 4 | 64.71 | 61.96 | 75.61 |
| g34 | h2 | 4 / 4 | 69.17 | 67.56 | 67.70 |
| q25 | path | 4 / 4 | 40.18 | 64.35 | 64.35 |
| g34 | path | 0 / 4 | pending | pending | 34.12 |

Two-seed means are shown only when both states completed. Gemma H2 exceeds both tuned LoRA references in this snapshot; Qwen3-4B H2 exceeds v2 LoRA but remains below v1 LoRA. Do not interpret a weaker selected baseline as resolving the gap. Tiny support selection and repeated query inspection prevent independent-test claims.

Protocol and diagnosis: ../../docs/APERTURE_DEBUG_V2.md and ../../docs/debug/. All attempted support candidates, including unfavorable ones, are retained.
