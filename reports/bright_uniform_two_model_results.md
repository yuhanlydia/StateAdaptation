# BRIGHT uniform two-model comparison

Protocol: four target events (`hawaii-wildfire`, `libya-flood`,
`noto-earthquake`, `turkey-earthquake`), 24 support examples and 300 query
examples per event (100 per class), 448px paired crops, and the same four
methods: Frozen, support-only LoRA, Random-KV, and Gradient-Cov KV (Ours).

The table reports the unweighted mean across the four events.

| Model | Method | Macro-F1 | Balanced accuracy | NLL |
|---|---|---:|---:|---:|
| Qwen3-VL-8B | Frozen | 0.166667 | 0.333333 | 9.424526 |
| Qwen3-VL-8B | LoRA | 0.203375 | 0.331667 | 1.389132 |
| Qwen3-VL-8B | Random-KV | 0.168736 | 0.334167 | 8.191588 |
| Qwen3-VL-8B | Gradient-Cov KV (Ours) | **0.349152** | **0.355833** | 2.151542 |
| InternVL3-8B | Frozen | 0.184816 | 0.325833 | 1.476283 |
| InternVL3-8B | LoRA | 0.166667 | 0.333333 | 1.692657 |
| InternVL3-8B | Random-KV | 0.221434 | 0.336667 | **1.308482** |
| InternVL3-8B | Gradient-Cov KV (Ours) | **0.285996** | **0.358333** | 1.368377 |

Qwen3-VL's best Macro-F1 and balanced accuracy are obtained by Gradient-Cov
KV. InternVL3's Gradient-Cov KV has the best Macro-F1 and balanced accuracy,
while Random-KV has the lowest NLL. These are descriptive four-event means;
no claim of significance is made here.

InternVL3 was run in bf16 on the 24GB GPU with gradient checkpointing enabled
for LoRA. All 16 InternVL3 metric files completed without OOM or NaN failures.
The complete local predictions, adapters, and KV states remain under
`runs/bright_uniform/`; this tracked report keeps the repository lightweight.
