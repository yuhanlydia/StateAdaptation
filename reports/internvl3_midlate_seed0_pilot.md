# InternVL3 middle-to-last-layer KV pilot

This is a one-seed pilot, not a replacement for the three-seed formal result. It uses the corrected `Answer: <label>` format and exactly the formal support/query split for seed 0. The only changed hyperparameter is the KV layer set: all InternVL3 decoder layers 14--27 (14 middle-to-last layers), rank 16, full coefficient mode, alpha 3.0, 4 updates, learning rate 0.05.

| Domain | Previous Ours (layers 14,27) | Middle-to-last Ours | Same-seed LoRA | Middle-to-last Δ vs previous | Middle-to-last Δ vs LoRA |
|---|---:|---:|---:|---:|---:|
| bridge_pick_place | 0.3301 | 0.3582 | 0.5370 | +0.0281 | -0.1788 |
| droid_arti | 0.3776 | 0.4590 | 0.6803 | +0.0814 | -0.2213 |
| droid_pick_place | 0.2397 | 0.2629 | 0.1815 | +0.0232 | **+0.0814** |

The result supports the layer-coverage hypothesis: middle-to-last injection improves all three domains, and beats LoRA on droid_pick_place for this seed. The LoRA baseline is seed-sensitive there, so a three-seed rerun is required before claiming a formal win.
