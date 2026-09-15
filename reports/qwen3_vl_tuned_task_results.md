# Qwen3-VL support-tuned task KV results

Hyperparameters were selected using only a stratified 25% holdout from each support set. Query labels were not read during selection. The formal frozen/LoRA/random-KV rows are unchanged baselines.

| Dataset/domain | tuned KV config (alpha/steps/lr) | Frozen F1 | LoRA F1 | Original Ours F1 | Tuned Ours F1 | Δ tuned-original |
|---|---:|---:|---:|---:|---:|---:|
| camelyon17/hospital_2 | 6.0/8/0.05 | 0.3407 | 0.6857 | 0.5857 | 0.8106±0.0531 | +0.2249 |
| manipbench_q1/bridge_pick_place | 3.0/8/0.05 | 0.7006 | 0.8567 | 0.7388 | 0.7662±0.0015 | +0.0274 |
| manipbench_q1/droid_arti | 6.0/4/0.05 | 0.5072 | 0.7860 | 0.5959 | 0.6303±0.0107 | +0.0343 |
| manipbench_q1/droid_pick_place | 3.0/8/0.05 | 0.5202 | 0.7624 | 0.5760 | 0.5717±0.0037 | -0.0044 |

Tuned Ours per-domain BA and NLL are included in the tracked JSON. The method remains below LoRA on ManipBench Macro-F1, but tuning improves Ours over the original setting on bridge_pick_place and droid_arti.
