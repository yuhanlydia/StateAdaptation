# InternVL3 all-layer KV pilot

One-seed (seed 0) pilot with the corrected `Answer: <label>` format and the formal support/query split. All hyperparameters match the formal run; only the KV layer set changes. This run injects full-rank-16 KV residuals into every InternVL3 decoder layer (0--27).

| Domain | Middle-to-last Ours (14--27) | All-layer Ours (0--27) | Same-seed LoRA | All-layer Δ vs middle-to-last |
|---|---:|---:|---:|---:|
| bridge_pick_place | 0.3582 | 0.1967 | 0.5370 | -0.1615 |
| droid_arti | 0.4590 | 0.1906 | 0.6803 | -0.2684 |
| droid_pick_place | 0.2629 | 0.1956 | 0.1815 | -0.0673 |

All-layer injection causes a strong collapse toward a narrow class distribution and is substantially worse than middle-to-last injection in all three domains. The result argues against using every layer; the early decoder layers should remain frozen for this KV residual method.
