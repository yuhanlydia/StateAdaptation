# Five unit reruns

All 20 arm comparisons use exactly the same query IDs and labels as the
canonical run. Frozen scores and predictions match byte for byte in all five
units. The four `recheck_v1` units have the canonical source signature. The
`droid_arti` seed-0 `recheck_v3_cpu_norm` unit is a **diagnostic source change**:
only descriptive operator norms were calculated on CPU after fitting, to avoid
an otherwise fatal cuSOLVER allocation under concurrent GPU load. It cannot
serve as an exact-source reproducibility claim.

| Unit | Canonical source? | Aperture Δ raw macro-F1 | Aperture Δ raw NLL | Changed Aperture predictions |
|---|---|---:|---:|---:|
| Bridge pick/place, seed 1 | yes | +0.0019 | -0.0037 | 19/400 |
| DROID articulation, seed 0 | diagnostic | -0.0129 | +0.0011 | 35/400 |
| DROID pick/place, seed 2 | yes | +0.0117 | -0.0036 | 21/400 |
| Hawaii wildfire, seed 1 | yes | +0.0207 | -0.2953 | 187/300 |
| Noto earthquake, seed 2 | yes | +0.0260 | -0.0015 | 116/300 |

The trainable arms are not bitwise deterministic. Hawaii Aperture and Bridge
LoRA4 show notably larger score changes than most reruns; these are recorded in
`summary.csv` and the complete comparison JSONs, rather than replacing the
sealed primary scores. DROID articulation Aperture macro-F1 remains low in the
diagnostic rerun (0.2772 canonical, 0.2642 rerun); NLL is nearly unchanged
(1.4717 to 1.4728), with no failed or incomplete evaluation. No query-based
hyperparameter changes were made.

The archived `diagnosis.json` files explain the failed v1 and stopped v2
attempts. The v3 run passed the audit and sealed all four fit and evaluation
states. The 39 Aperture CPU tests passed after the descriptive-statistic fix.
