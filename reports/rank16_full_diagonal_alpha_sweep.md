# Rank-16 Full versus Diagonal alpha sweep

All 48 runs completed without failure or OOM. Every arm uses the original
balanced same-event support24 and fixed query300, covariance basis, decoder
layers 14+27, rank 16, four coefficient updates, and coefficient learning rate
0.05. Full has 1,024 trainable scalars and Diagonal has 64. The comparison
baseline is clean fixed-8 LoRA trained without support CV.

## Full KV-TTT

| Event | LoRA fixed8 | alpha 0.5 | alpha 1 | alpha 2 | alpha 3 | alpha 5 | alpha 10 | Oracle best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hawaii wildfire | 0.3526 | 0.2622 | 0.3338 | **0.3869** | 0.2957 | 0.3239 | 0.3476 | 0.3869 (alpha 2) |
| Libya flood | 0.2776 | 0.3249 | 0.2760 | **0.3852** | 0.3739 | 0.3394 | 0.2943 | 0.3852 (alpha 2) |
| Noto earthquake | 0.3495 | 0.2354 | **0.3270** | 0.3095 | 0.3112 | 0.3021 | 0.2770 | 0.3270 (alpha 1) |
| Turkey earthquake | 0.2147 | 0.2627 | 0.1876 | 0.2527 | **0.3025** | 0.2472 | 0.2578 | 0.3025 (alpha 3) |

The mean of per-event oracle best Full values is 0.3504, versus 0.2986 for
fixed-8 LoRA. Full exceeds LoRA on Hawaii, Libya, and Turkey; it trails on Noto.

## Diagonal KV-TTT

| Event | LoRA fixed8 | alpha 0.5 | alpha 1 | alpha 2 | alpha 3 | alpha 5 | alpha 10 | Oracle best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hawaii wildfire | 0.3526 | 0.2880 | 0.2972 | 0.3021 | 0.2918 | 0.3211 | **0.3341** | 0.3341 (alpha 10) |
| Libya flood | 0.2776 | 0.3110 | 0.3033 | 0.3041 | 0.2560 | **0.3390** | 0.3278 | 0.3390 (alpha 5) |
| Noto earthquake | 0.3495 | 0.2629 | 0.2677 | 0.2278 | 0.2820 | **0.3404** | 0.3310 | 0.3404 (alpha 5) |
| Turkey earthquake | 0.2147 | 0.2637 | 0.2698 | 0.2604 | **0.2879** | 0.2843 | 0.2601 | 0.2879 (alpha 3) |

The mean of per-event oracle best Diagonal values is 0.3254. Diagonal exceeds
fixed-8 LoRA on Libya and Turkey, and trails by 0.0091 on Noto.

## Interpretation and validity

The per-event maxima above are query-oracle ablation results, not a valid
test-independent primary estimator. A formal primary table must select alpha
using support-only evidence and then evaluate query300 once. The sweep shows
that one global alpha is not uniformly optimal and identifies where the
controller has capacity to outperform fixed-8 LoRA.

There is also a reproducibility warning: an earlier nominally identical Hawaii
rank-16 Full alpha-3 run scored 0.3295, while the new run scored 0.2957. Model
fingerprint, support, and recorded hyperparameters match. The alpha-2 Hawaii
peak and Noto alpha-5 Diagonal near-tie therefore require independent repeats
before being promoted beyond oracle diagnostics.
