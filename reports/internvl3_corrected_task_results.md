# Corrected InternVL3 task-VLM results

The previous InternVL3 ManipBench rows were invalidated: the processor asked for a raw label, while the benchmark prompt requires `Answer: <label>`. This rerun uses the matching answer format for both support adaptation and candidate likelihood scoring.

Protocol: same support/query splits, seeds, and four methods as the Qwen3-VL run. InternVL3 uses the validated 224px/64-visual-token budget on the 24GB GPU; no query labels or query-time tuning were used. The rerun contains 3 seeds per fold.

Completed metric files: 48/48.

| Dataset/domain | Method | folds | Macro-F1 mean±sd | BA mean±sd | NLL mean±sd |
|---|---:|---:|---:|---:|---:|
| camelyon17/hospital_2 | frozen | 3 | 0.3333±0.0000 | 0.5000±0.0000 | 0.9693±0.0000 |
| camelyon17/hospital_2 | lora | 3 | 0.6325±0.2725 | 0.6956±0.1846 | 1.9900±1.0781 |
| camelyon17/hospital_2 | ours | 3 | 0.5107±0.0837 | 0.5689±0.0267 | 0.7841±0.1762 |
| camelyon17/hospital_2 | random_kv | 3 | 0.4935±0.0071 | 0.5578±0.0069 | 0.7276±0.0260 |
| manipbench_q1/bridge_pick_place | frozen | 3 | 0.2502±0.0000 | 0.2500±0.0000 | 1.6049±0.0000 |
| manipbench_q1/bridge_pick_place | lora | 3 | 0.5551±0.0159 | 0.5575±0.0152 | 3.4351±0.6396 |
| manipbench_q1/bridge_pick_place | ours | 3 | 0.2993±0.0366 | 0.3142±0.0138 | 1.4500±0.0161 |
| manipbench_q1/bridge_pick_place | random_kv | 3 | 0.2416±0.0117 | 0.2483±0.0080 | 1.5296±0.0050 |
| manipbench_q1/droid_arti | frozen | 3 | 0.1504±0.0000 | 0.1675±0.0000 | 1.7993±0.0000 |
| manipbench_q1/droid_arti | lora | 3 | 0.6759±0.0150 | 0.6775±0.0139 | 2.2237±0.3739 |
| manipbench_q1/droid_arti | ours | 3 | 0.3307±0.0691 | 0.3367±0.0648 | 1.4513±0.0634 |
| manipbench_q1/droid_arti | random_kv | 3 | 0.1485±0.0167 | 0.1775±0.0087 | 1.7158±0.0121 |
| manipbench_q1/droid_pick_place | frozen | 3 | 0.1541±0.0000 | 0.1650±0.0000 | 1.8426±0.0000 |
| manipbench_q1/droid_pick_place | lora | 3 | 0.4304±0.2223 | 0.4625±0.1786 | 2.6912±1.4393 |
| manipbench_q1/droid_pick_place | ours | 3 | 0.2839±0.0729 | 0.2958±0.0686 | 1.4663±0.0517 |
| manipbench_q1/droid_pick_place | random_kv | 3 | 0.1556±0.0047 | 0.1958±0.0123 | 1.6881±0.0118 |

The old InternVL3 rows remain only in the historical formal report; use this report for corrected comparisons.
