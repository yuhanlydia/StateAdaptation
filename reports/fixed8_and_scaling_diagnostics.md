# Fixed-8 LoRA and KV scaling diagnostics

All fixed-8 LoRA arms train directly on the balanced same-event support24
without support CV. Batch size 1 and gradient accumulation 3 give 8 optimizer
updates over exactly 24 support examples, followed by evaluation on the fixed
balanced query300.

## Clean fixed-8 LoRA

| Event | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|
| Hawaii wildfire | 0.3526 | 0.3767 | 1.0974 |
| Libya flood | 0.2776 | 0.3467 | 1.1175 |
| Noto earthquake | 0.3495 | 0.3800 | 1.0977 |
| Turkey earthquake | 0.2147 | 0.3233 | 1.1980 |

These values supersede CV-selected LoRA only for the explicitly fixed-8
protocol. The CV-selected adapters remain valid results under a different
selection protocol.

## Turkey alpha by steps interaction

All arms use Full covariance KV-TTT, rank 5, layers 14+27, learning rate 0.05,
and eight full-support coefficient updates.

| Alpha | Macro-F1 |
|---:|---:|
| 0.5 | 0.2560 |
| 1 | 0.2336 |
| 2 | 0.2793 |
| 3 | **0.2977** |
| 5 | 0.2401 |
| 10 | 0.2602 |

## Naive all-layer scaling

Expanding Full KV-TTT to all 28 decoder layers at rank 16 creates 14,336
trainable scalars. Directly retaining alpha 3 and learning rate 0.05 is
unstable and performs poorly.

| Event | Macro-F1 |
|---|---:|
| Hawaii wildfire | 0.2400 |
| Libya flood | 0.1670 |
| Noto earthquake | 0.1704 |
| Turkey earthquake | 0.1935 |

On Hawaii, reducing alpha to 0.3 and the learning rate to 0.005 makes support
loss decrease monotonically (0.6629 to 0.5496), but query Macro-F1 is only
0.2923 and damaged recall is zero. Stable support optimization therefore does
not make dense all-layer adaptation generalize. Sparse middle+last layer
selection remains the supported design.

## Hawaii support-selection diagnostic

The original support is balanced but concentrated in three tiles. A first
farthest-point diversity heuristic selected bbox outliers and produced KV
Macro-F1 0.1909; it is invalid as a representative support intervention. A
corrected representative-medoid selection filters scale outliers, treats tile
coverage as a soft constraint, remains 8/8/8 and query-tile-disjoint, and gives
KV Macro-F1 0.3084, balanced accuracy 0.3800, and NLL 1.1206. It does not exceed
the original-support rank-5 alpha-2 result (0.3278), so it is reported only as
a support-sensitivity diagnostic, not as a replacement main split.
