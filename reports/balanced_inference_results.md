# Balanced same-event inference adaptation results

> **Legacy comparison protocol.** The LoRA adapters in this table use
> per-event support-CV-selected training steps (8 or 32), not the later clean
> fixed-8 protocol. For the current paper table, see
> `fixed8_and_scaling_diagnostics.md` and
> `rank16_full_diagonal_alpha_sweep.md`.

Primary baseline: raw VLM `original_eval`; all adaptations use the same
event's 24 support examples and evaluate the identical balanced query IDs.

| Event | Arm | N | Original F1 | Adapted F1 | ΔF1 | ΔBalAcc | NLL reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| hawaii-wildfire | support24_lora | 300 | 0.1992 | 0.3375 | +0.1383 | +0.0200 | +0.1540 |
| hawaii-wildfire | kv_full_a3 | 300 | 0.1992 | 0.2847 | +0.0855 | +0.0200 | +0.0604 |
| hawaii-wildfire | kv_diagonal_a3 | 300 | 0.1992 | 0.2790 | +0.0798 | +0.0267 | +0.0954 |
| hawaii-wildfire | kv_unsupervised_a3 | 300 | 0.1992 | 0.1805 | -0.0188 | -0.0067 | -0.0079 |
| hawaii-wildfire | kv_full_a0p5_ablation | 300 | 0.1992 | 0.2163 | +0.0171 | +0.0033 | +0.0374 |
| hawaii-wildfire | kv_diagonal_a0p5_ablation | 300 | 0.1992 | 0.2014 | +0.0021 | +0.0033 | +0.0137 |
| hawaii-wildfire | kv_unsupervised_a0p5_ablation | 300 | 0.1992 | 0.1819 | -0.0173 | -0.0067 | +0.0034 |
| hawaii-wildfire | kv_unsupervised_full_a3_ablation | 300 | 0.1992 | 0.1781 | -0.0211 | -0.0100 | -0.1523 |
| hawaii-wildfire | kv_unsupervised_full_a0p5_legacy | 300 | 0.1992 | 0.1808 | -0.0185 | -0.0067 | -0.0083 |
| libya-flood | support24_lora | 300 | 0.2585 | 0.2861 | +0.0275 | -0.0200 | +0.2187 |
| libya-flood | kv_full_a3 | 300 | 0.2585 | 0.3181 | +0.0596 | +0.0000 | +0.1988 |
| libya-flood | kv_diagonal_a3 | 300 | 0.2585 | 0.3295 | +0.0710 | +0.0300 | +0.1284 |
| libya-flood | kv_unsupervised_a3 | 300 | 0.2585 | 0.2321 | -0.0264 | -0.0133 | +0.0138 |
| libya-flood | kv_full_a0p5_ablation | 300 | 0.2585 | 0.2675 | +0.0090 | +0.0033 | +0.0602 |
| libya-flood | kv_diagonal_a0p5_ablation | 300 | 0.2585 | 0.2531 | -0.0054 | +0.0000 | +0.0140 |
| libya-flood | kv_unsupervised_a0p5_ablation | 300 | 0.2585 | 0.2478 | -0.0107 | -0.0033 | +0.0062 |
| libya-flood | kv_unsupervised_full_a3_ablation | 300 | 0.2585 | 0.1724 | -0.0862 | -0.0300 | -0.9111 |
| libya-flood | kv_unsupervised_full_a0p5_legacy | 300 | 0.2585 | 0.2245 | -0.0341 | -0.0133 | -0.0360 |
| noto-earthquake | support24_lora | 300 | 0.2087 | 0.2667 | +0.0580 | +0.0000 | -0.1116 |
| noto-earthquake | kv_full_a3 | 300 | 0.2087 | 0.3029 | +0.0943 | +0.0367 | +0.1326 |
| noto-earthquake | kv_diagonal_a3 | 300 | 0.2087 | 0.2116 | +0.0030 | +0.0100 | +0.0565 |
| noto-earthquake | kv_unsupervised_a3 | 300 | 0.2087 | 0.1727 | -0.0359 | +0.0000 | -0.0614 |
| noto-earthquake | kv_full_a0p5_ablation | 300 | 0.2087 | 0.2390 | +0.0303 | +0.0033 | +0.0693 |
| noto-earthquake | kv_diagonal_a0p5_ablation | 300 | 0.2087 | 0.2019 | -0.0068 | +0.0033 | +0.0167 |
| noto-earthquake | kv_unsupervised_a0p5_ablation | 300 | 0.2087 | 0.1891 | -0.0196 | -0.0033 | +0.0048 |
| noto-earthquake | kv_unsupervised_full_a3_ablation | 300 | 0.2087 | 0.1667 | -0.0420 | +0.0000 | -0.7800 |
| noto-earthquake | kv_unsupervised_full_a0p5_legacy | 300 | 0.2087 | 0.1927 | -0.0160 | +0.0067 | +0.0020 |
| turkey-earthquake | support24_lora | 300 | 0.2273 | 0.4301 | +0.2028 | +0.0900 | +0.1537 |
| turkey-earthquake | kv_full_a3 | 300 | 0.2273 | 0.2799 | +0.0526 | -0.0600 | +0.1822 |
| turkey-earthquake | kv_diagonal_a3 | 300 | 0.2273 | 0.2797 | +0.0524 | -0.0133 | +0.1388 |
| turkey-earthquake | kv_unsupervised_a3 | 300 | 0.2273 | 0.1654 | -0.0619 | -0.0100 | -0.7455 |
| turkey-earthquake | kv_full_a0p5_ablation | 300 | 0.2273 | 0.2306 | +0.0033 | +0.0000 | +0.0286 |
| turkey-earthquake | kv_diagonal_a0p5_ablation | 300 | 0.2273 | 0.2397 | +0.0124 | +0.0067 | +0.0227 |
| turkey-earthquake | kv_unsupervised_a0p5_ablation | 300 | 0.2273 | 0.2111 | -0.0163 | -0.0067 | -0.0247 |
| turkey-earthquake | kv_unsupervised_full_a3_ablation | 300 | 0.2273 | 0.1724 | -0.0550 | -0.0067 | -0.8255 |
| turkey-earthquake | kv_unsupervised_full_a0p5_legacy | 300 | 0.2273 | 0.1852 | -0.0421 | -0.0033 | -0.1153 |

## Macro means

| Arm | Events | Mean F1 | Mean ΔF1 | Mean ΔBalAcc | Mean NLL reduction |
|---|---:|---:|---:|---:|---:|
| support24_lora | 4 | 0.3301 | +0.1066 | +0.0225 | +0.1037 |
| kv_full_a3 | 4 | 0.2964 | +0.0730 | -0.0008 | +0.1435 |
| kv_diagonal_a3 | 4 | 0.2750 | +0.0515 | +0.0133 | +0.1048 |
| kv_unsupervised_a3 | 4 | 0.1877 | -0.0358 | -0.0075 | -0.2002 |
| kv_full_a0p5_ablation | 4 | 0.2384 | +0.0149 | +0.0025 | +0.0489 |
| kv_diagonal_a0p5_ablation | 4 | 0.2240 | +0.0006 | +0.0033 | +0.0168 |
| kv_unsupervised_a0p5_ablation | 4 | 0.2075 | -0.0160 | -0.0050 | -0.0026 |
| kv_unsupervised_full_a3_ablation | 4 | 0.1724 | -0.0511 | -0.0117 | -0.6672 |
| kv_unsupervised_full_a0p5_legacy | 4 | 0.1958 | -0.0277 | -0.0042 | -0.0394 |
