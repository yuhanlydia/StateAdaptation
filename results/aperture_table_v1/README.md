# Aperture table v1: measured results

Completed on 2026-09-22. All 64 prescribed states and 112 numeric table cells passed the official `summarize` validation, including sealed artifact hashes, eight GPU audits, four support-only repeat gates, shared query identities, and recomputation from saved predictions. The code source fingerprint is recorded in `paper_exports/coverage.json`.

This is the final ZIP protocol: four BF16 backbones, Camelyon17-WILDS hospital 2 (200 query images), PathMNIST-224 (180 query images), support seeds 0 and 1, and Frozen / LoRA one pass / LoRA four passes / Aperture. Output bias is computed from the frozen state. These are not the historical 300/500/900-query experiments.

## Main finding

Aperture has lower mean Macro-F1 than four-pass LoRA in all eight backbone/dataset combinations. It improves over Frozen in five combinations and declines in three. These measurements do not establish an overall advantage for Aperture. No query-based hyperparameter selection or replacement of unfavorable results was performed.

Macro-F1 below is a percentage, averaged over the two fixed support seeds. Full sample standard deviations, NLL, temperature-scaled NLL, and per-seed values are in [the CSV](paper_exports/clean_mean_sd.csv) and [the complete table](paper_exports/main_table_cells.json). Two-seed SD describes support-seed variability, not a patient-level confidence interval.

| Backbone | Dataset | Frozen | LoRA 1 pass | LoRA 4 passes | Aperture |
|---|---|---:|---:|---:|---:|
| q25 | h2 | 51.75 | 33.33 | 61.32 | 55.34 |
| q25 | path | 35.07 | 39.51 | 64.35 | 39.87 |
| q34 | h2 | 66.52 | 52.93 | 78.73 | 57.13 |
| q34 | path | 44.62 | 64.63 | 70.56 | 59.07 |
| q38 | h2 | 59.19 | 58.12 | 75.50 | 57.61 |
| q38 | path | 38.46 | 50.09 | 81.40 | 61.01 |
| g34 | h2 | 55.97 | 33.33 | 56.05 | 44.16 |
| g34 | path | 17.09 | 24.32 | 34.43 | 23.19 |

Backbones: q25 = Qwen2.5-VL-7B-Instruct; q34 = Qwen3-VL-4B-Instruct; q38 = Qwen3-VL-8B-Instruct; g34 = Gemma-3-4B-it.

## Diagnostics and limitations

- Qwen2.5 H2 LoRA1 predicts one class for both support seeds. Gemma H2 seed 0 LoRA1 also collapses to one class. Independent fixed-configuration replay reports are retained under `diagnostics/`.
- Qwen3-VL-4B H2 Aperture decreases Macro-F1 relative to Frozen in both seeds. Both independent replays reproduced calibration scores and query metrics exactly.
- Gemma Path predictions are strongly class-biased, including the frozen baseline. Aperture and LoRA1 seed-0 independent replays reproduced the original results exactly.
- Support-selected temperatures sometimes worsen query NLL without changing F1. The calibration sets contain only 4 H2 or 18 Path images per seed. Distribution mismatch is a possible explanation, not an established causal finding.
- Deterministic reproduction and state/audit checks did not identify an execution failure explaining these declines; they do not prove the method implementation is free of every possible issue.
- A server reboot interrupted earlier fits; only incomplete states were resumed. One later diagnostic fit encountered OOM during concurrent external GPU use and was retried unchanged. These events did not replace main-table results.

## Files and data

- `paper_exports/`: verified table, CSV, 64 full metric records, resource/optimizer counts, and coverage hashes.
- `states/`: per-example query predictions, calibration predictions, calibration settings, and actual fitting records; adapter/base-model weights are excluded.
- `audits/`, `repeat/`, `diagnostics/`: audit and repeat evidence.
- [Fixed data bundle](../../data/aperture_table_v1_release/): all 555 distinct selected images, exact manifests, source provenance, image hashes, and a portable restoration script.
- `SHA256SUMS.json`: checksums for the published result files.

The generated LaTeX table is provided directly. The checkout and supplied ZIP do not contain a manuscript with the required 112 `\pending{key}` markers, so no manuscript source was overwritten or represented as automatically filled.
