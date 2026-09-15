# Multiseed statistical significance

Three balanced support24 seeds; paired comparisons use identical query IDs.

| Comparison | Mean Δ Macro-F1 | Stratified bootstrap 95% CI | Paired permutation p |
|---|---:|---:|---:|
| full_vs_baseline | +0.0457 | [+0.0046, +0.0847] | 0.0167 |
| diagonal_vs_baseline | +0.0399 | [+0.0063, +0.0734] | 0.006099 |
| full_vs_diagonal | +0.0058 | [-0.0433, +0.0506] | 0.7347 |

## Per-event, per-seed results

| Event | Method | Seed | Macro-F1 | ΔF1 | ΔBalAcc | NLL reduction |
|---|---|---:|---:|---:|---:|---:|
| hawaii-wildfire | full | 0 | 0.2847 | +0.0855 | +0.0200 | +0.0604 |
| hawaii-wildfire | full | 1 | 0.2455 | +0.0462 | -0.0100 | +0.0867 |
| hawaii-wildfire | full | 2 | 0.3082 | +0.1090 | +0.0433 | +0.1147 |
| hawaii-wildfire | diagonal | 0 | 0.2790 | +0.0798 | +0.0267 | +0.0954 |
| hawaii-wildfire | diagonal | 1 | 0.3263 | +0.1271 | +0.0133 | +0.1165 |
| hawaii-wildfire | diagonal | 2 | 0.2594 | +0.0602 | +0.0133 | +0.0723 |
| libya-flood | full | 0 | 0.3181 | +0.0596 | +0.0000 | +0.1988 |
| libya-flood | full | 1 | 0.2787 | +0.0201 | +0.0100 | +0.1380 |
| libya-flood | full | 2 | 0.3165 | +0.0579 | +0.0100 | +0.2006 |
| libya-flood | diagonal | 0 | 0.3295 | +0.0710 | +0.0300 | +0.1284 |
| libya-flood | diagonal | 1 | 0.2513 | -0.0072 | -0.0033 | +0.0635 |
| libya-flood | diagonal | 2 | 0.3525 | +0.0939 | +0.0600 | +0.1542 |
| noto-earthquake | full | 0 | 0.3029 | +0.0943 | +0.0367 | +0.1326 |
| noto-earthquake | full | 1 | 0.2835 | +0.0749 | +0.0300 | +0.1185 |
| noto-earthquake | full | 2 | 0.1967 | -0.0120 | +0.0067 | +0.0733 |
| noto-earthquake | diagonal | 0 | 0.2116 | +0.0030 | +0.0100 | +0.0565 |
| noto-earthquake | diagonal | 1 | 0.1912 | -0.0175 | +0.0033 | +0.0467 |
| noto-earthquake | diagonal | 2 | 0.1917 | -0.0170 | +0.0033 | +0.0455 |
| turkey-earthquake | full | 0 | 0.2799 | +0.0526 | -0.0600 | +0.1822 |
| turkey-earthquake | full | 1 | 0.1805 | -0.0468 | -0.0200 | +0.0691 |
| turkey-earthquake | full | 2 | 0.2345 | +0.0072 | -0.0567 | +0.1762 |
| turkey-earthquake | diagonal | 0 | 0.2797 | +0.0524 | -0.0133 | +0.1388 |
| turkey-earthquake | diagonal | 1 | 0.2643 | +0.0370 | -0.0167 | +0.1208 |
| turkey-earthquake | diagonal | 2 | 0.2231 | -0.0042 | +0.0000 | +0.0679 |
