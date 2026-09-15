# InternVL3 middle-to-last hyperparameter pilot

All runs use the corrected `Answer: <label>` format, the same seed-0 support/query split, all KV layers 14--27, and four adaptation steps at learning rate 0.05. LoRA is the unchanged same-seed baseline (`rank=16`, `alpha=32`, Q/K/V/O targets).

| Configuration | bridge F1 | droid_arti F1 | droid_pick_place F1 |
|---|---:|---:|---:|
| LoRA baseline | 0.5370 | 0.6803 | 0.1815 |
| Ours rank16 / alpha3 | 0.3582 | 0.4590 | 0.2629 |
| Ours rank16 / alpha1 | 0.3640 | 0.3870 | 0.2216 |
| Ours rank32 / alpha1 | **0.3915** | **0.4496** | **0.2700** |
| Ours rank32 / alpha1 / l2=1e-2 | 0.3901 | 0.4346 | 0.2663 |
| Ours rank32 / alpha0.5 | 0.3456 | 0.3200 | 0.2064 |

The best tested configuration is rank32/alpha1. It improves on the original middle-to-last setting in bridge and droid_pick_place, and beats LoRA on droid_pick_place for this seed. It does not beat LoRA on bridge or droid_arti. The alpha0.5 run underfits, while stronger l2 is slightly harmful. These are seed-0 screening results; a three-seed formal claim requires selecting the configuration without query labels and then rerunning all seeds.
