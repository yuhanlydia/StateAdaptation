# Targeted debug v2

Previously inspected query; targeted adaptive exploratory debugging, not independent test evidence.

H2 uses three patient-separated support folds; Path uses the original 18-image calibration. Final training and temperature calibration reuse the original roles. Loss/rank unchanged. Every attempted candidate is reported. Details: ../../docs/APERTURE_DEBUG_V2.md.

| Model | Domain | Aperture F1 % | v2 LoRA F1 % | v1 LoRA F1 % |
|---|---|---:|---:|---:|
| q25 | h2 | 52.96 | 64.35 | 64.35 |
| q34 | h2 | 64.71 | 61.96 | 75.61 |
| g34 | h2 | 69.17 | 67.56 | 67.70 |
| q25 | path | 40.18 | 64.35 | 64.35 |
| g34 | path | 30.24 | 36.13 | 34.12 |

A higher v2 score is exploratory evidence only. Report both LoRA references; do not claim success solely by replacing a stronger old baseline with a weaker selected baseline. Expanded cumulative search and tiny support sets limit conclusions.
