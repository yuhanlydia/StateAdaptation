# Support-selected Aperture tuning v1

Exploratory follow-up to the fixed table at 41738d9. Six predetermined candidates per method/unit; calibration Macro-F1 selects, raw NLL breaks ties. All 192 candidate fits and 32 selected-fit repeats completed before the 32 selected query evaluations. The same previously inspected query sets are reused, so these results are not new blind-test evidence.

| Model | Dataset | Tuned Aperture F1 (%) | Tuned LoRA F1 (%) | Difference (pp) |
|---|---|---:|---:|---:|
| q25 | h2 | 54.76 | 64.35 | -9.59 |
| q25 | path | 37.48 | 64.35 | -26.87 |
| q34 | h2 | 61.19 | 75.61 | -14.42 |
| q34 | path | 63.02 | 67.61 | -4.60 |
| q38 | h2 | 59.33 | 68.17 | -8.84 |
| q38 | path | 65.46 | 83.50 | -18.04 |
| g34 | h2 | 68.72 | 67.70 | +1.02 |
| g34 | path | 29.97 | 34.12 | -4.15 |

Aperture exceeds tuned LoRA in 1/8 combinations by mean Macro-F1. Full per-seed outcomes, SD, candidate IDs and NLL are in summary.json. All candidate validation scores and training records are included, including unfavorable candidates.

H2 uses only four calibration examples per seed (Path uses 18). Selection uncertainty is substantial; candidate count is matched but compute and parameter count are not. No guarantee of reaching an optimal configuration or of beating LoRA is made.

Protocol and audit findings: ../../docs/APERTURE_TUNING_V1.md. Fixed images/manifests: ../../data/aperture_table_v1_release/. No model weights or credentials are included.
