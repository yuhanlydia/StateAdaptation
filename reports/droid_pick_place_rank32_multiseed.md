# ManipBench droid_pick_place rank32 middle-to-last result

Three-seed result for the selected middle-to-last configuration: InternVL3 layers 14--27, full KV coefficient mode, rank 32, alpha 1, four support updates, learning rate 0.05. The same corrected answer-format protocol and support/query folds are used as the formal baseline.

| Method | Macro-F1 mean±sd | BA mean±sd | NLL mean±sd |
|---|---:|---:|---:|
| LoRA rank16 | 0.4304±0.2223 | 0.4625±0.1786 | 2.6912±1.4393 |
| Ours rank32/alpha1 | 0.3279±0.0533 | 0.3342±0.0506 | 1.3784±0.0423 |

Ours wins seed 0 (0.2700 vs 0.1815) but not the three-seed mean. The single-seed win was therefore a high-variance baseline effect rather than a formal domain-level win.
