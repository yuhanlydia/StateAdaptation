# Formal Camelyon17 / ManipBench task-VLM results

**Important correction:** the historical InternVL3 ManipBench rows in this file used a mismatched raw-label answer format and are superseded by [`internvl3_corrected_task_results.md`](internvl3_corrected_task_results.md). The Qwen3-VL rows remain valid.

Protocol: identical support/query splits, seeds, and adaptation methods across model families. Qwen3-VL uses a deterministic 448px image budget (256 visual tokens); InternVL3 uses 224px (64 visual tokens) to fit the 24GB GPU. Camelyon17 hospital_2 has 3 support seeds and 300 queries; each ManipBench Q1 domain has 3 support seeds and 400 queries. Each fold evaluates Frozen, four-pass LoRA, Random-KV, and Gradient-Cov KV (Ours).

Completed metric files: 96/96.

| Family | Dataset/domain | Method | folds | Macro-F1 mean±sd | BA mean±sd | NLL mean±sd |
|---|---|---:|---:|---:|---:|---:|
| internvl3 | camelyon17/hospital_2 | frozen | 3 | 0.3832±0.0000 | 0.5233±0.0000 | 1.5977±0.0000 |
| internvl3 | camelyon17/hospital_2 | lora | 3 | 0.6382±0.2492 | 0.6967±0.1767 | 3.5048±2.0994 |
| internvl3 | camelyon17/hospital_2 | ours | 3 | 0.7829±0.1646 | 0.7989±0.1377 | 0.4130±0.1121 |
| internvl3 | camelyon17/hospital_2 | random_kv | 3 | 0.3692±0.0184 | 0.5167±0.0088 | 1.6120±0.0711 |
| internvl3 | manipbench_q1/bridge_pick_place | frozen | 3 | 0.1047±0.0004 | 0.2508±0.0014 | 1.7946±0.0859 |
| internvl3 | manipbench_q1/bridge_pick_place | lora | 3 | 0.5176±0.0365 | 0.5233±0.0364 | 5.7100±3.1827 |
| internvl3 | manipbench_q1/bridge_pick_place | ours | 3 | 0.1016±0.0031 | 0.2500±0.0025 | 1.8347±0.0824 |
| internvl3 | manipbench_q1/bridge_pick_place | random_kv | 3 | 0.1557±0.0268 | 0.2475±0.0115 | 1.6733±0.0747 |
| internvl3 | manipbench_q1/droid_arti | frozen | 3 | 0.0992±0.0000 | 0.2475±0.0000 | 1.9205±0.0000 |
| internvl3 | manipbench_q1/droid_arti | lora | 3 | 0.6664±0.0323 | 0.6700±0.0303 | 2.4280±0.5835 |
| internvl3 | manipbench_q1/droid_arti | ours | 3 | 0.1086±0.0107 | 0.2542±0.0052 | 1.9012±0.0412 |
| internvl3 | manipbench_q1/droid_arti | random_kv | 3 | 0.1999±0.0272 | 0.2642±0.0126 | 1.6773±0.0182 |
| internvl3 | manipbench_q1/droid_pick_place | frozen | 3 | 0.0994±0.0000 | 0.2475±0.0000 | 1.9686±0.0000 |
| internvl3 | manipbench_q1/droid_pick_place | lora | 3 | 0.6155±0.0913 | 0.6208±0.0833 | 2.5074±0.2815 |
| internvl3 | manipbench_q1/droid_pick_place | ours | 3 | 0.1015±0.0031 | 0.2492±0.0014 | 1.8871±0.4672 |
| internvl3 | manipbench_q1/droid_pick_place | random_kv | 3 | 0.1250±0.0120 | 0.2217±0.0255 | 1.8137±0.1006 |
| qwen3_vl | camelyon17/hospital_2 | frozen | 3 | 0.3407±0.0000 | 0.5033±0.0000 | 3.2979±0.0000 |
| qwen3_vl | camelyon17/hospital_2 | lora | 3 | 0.6857±0.2026 | 0.7244±0.1458 | 2.9835±2.1273 |
| qwen3_vl | camelyon17/hospital_2 | ours | 3 | 0.5857±0.0902 | 0.6333±0.0590 | 0.8225±0.1847 |
| qwen3_vl | camelyon17/hospital_2 | random_kv | 3 | 0.3922±0.0039 | 0.5278±0.0019 | 2.8555±0.0154 |
| qwen3_vl | manipbench_q1/bridge_pick_place | frozen | 3 | 0.7006±0.0000 | 0.7050±0.0000 | 1.7561±0.0000 |
| qwen3_vl | manipbench_q1/bridge_pick_place | lora | 3 | 0.8567±0.0304 | 0.8567±0.0302 | 1.7662±0.4357 |
| qwen3_vl | manipbench_q1/bridge_pick_place | ours | 3 | 0.7388±0.0146 | 0.7383±0.0142 | 1.3819±0.0952 |
| qwen3_vl | manipbench_q1/bridge_pick_place | random_kv | 3 | 0.7004±0.0012 | 0.7042±0.0014 | 1.7492±0.0104 |
| qwen3_vl | manipbench_q1/droid_arti | frozen | 3 | 0.5072±0.0000 | 0.5325±0.0000 | 2.8505±0.0000 |
| qwen3_vl | manipbench_q1/droid_arti | lora | 3 | 0.7860±0.0409 | 0.7867±0.0401 | 3.0010±1.2395 |
| qwen3_vl | manipbench_q1/droid_arti | ours | 3 | 0.5959±0.0139 | 0.5992±0.0128 | 2.4057±0.0708 |
| qwen3_vl | manipbench_q1/droid_arti | random_kv | 3 | 0.5187±0.0024 | 0.5433±0.0014 | 2.8541±0.0100 |
| qwen3_vl | manipbench_q1/droid_pick_place | frozen | 3 | 0.5202±0.0000 | 0.5300±0.0000 | 3.0225±0.0000 |
| qwen3_vl | manipbench_q1/droid_pick_place | lora | 3 | 0.7624±0.0145 | 0.7625±0.0152 | 1.6482±0.5276 |
| qwen3_vl | manipbench_q1/droid_pick_place | ours | 3 | 0.5760±0.0045 | 0.5758±0.0038 | 2.8409±0.1083 |
| qwen3_vl | manipbench_q1/droid_pick_place | random_kv | 3 | 0.5279±0.0017 | 0.5358±0.0014 | 3.0185±0.0068 |

Raw per-fold JSON is tracked in `reports/task_vlm_formal_metrics.json`; large predictions and adapters remain under ignored `runs/`.
