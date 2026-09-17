# Visual Lens P0 results

Completed 83/83 planned jobs. Missing results are not zero.

Primary reproduction and objective controls are separated. Mean ± sample SD is support-seed variation, not a confidence interval.

| Stage | Backbone | Domain | Method | Seeds | Macro-F1 | NLL |
|---|---|---|---|---|---:|---:|
| matched | internvl3 | hospital_2 | frozen | 0 | 0.3333 | 0.9662 |
| matched | internvl3 | hospital_2 | lora | 0 | 0.5454 | 3.0991 |
| matched | internvl3 | hospital_2 | ours | 0 | 0.5457 | 0.7252 |
| matched | qwen2 | hawaii-wildfire | frozen | 0,1,2 | 0.1944 ± 0.0000 | 1.2446 ± 0.0000 |
| matched | qwen2 | hawaii-wildfire | lora | 0,1,2 | 0.3504 ± 0.0382 | 1.1059 ± 0.0049 |
| matched | qwen2 | hawaii-wildfire | lora_passes4 | 0,1,2 | 0.2845 ± 0.0384 | 1.1548 ± 0.0247 |
| matched | qwen2 | hawaii-wildfire | ours | 0,1,2 | 0.3329 ± 0.0443 | 1.1124 ± 0.0142 |
| matched | qwen2 | hawaii-wildfire | random_kv | 0,1,2 | 0.3667 ± 0.0280 | 1.1052 ± 0.0017 |
| matched | qwen2 | libya-flood | frozen | 0,1,2 | 0.2728 ± 0.0000 | 1.3231 ± 0.0000 |
| matched | qwen2 | libya-flood | lora | 0,1,2 | 0.3592 ± 0.0287 | 1.1059 ± 0.0121 |
| matched | qwen2 | libya-flood | lora_passes4 | 0,1,2 | 0.2439 ± 0.0376 | 1.2067 ± 0.0675 |
| matched | qwen2 | libya-flood | ours | 0,1,2 | 0.3404 ± 0.0316 | 1.1235 ± 0.0146 |
| matched | qwen2 | libya-flood | random_kv | 0,1,2 | 0.3052 ± 0.0440 | 1.1384 ± 0.0144 |
| matched | qwen2 | noto-earthquake | frozen | 0,1,2 | 0.2060 ± 0.0000 | 1.2739 ± 0.0000 |
| matched | qwen2 | noto-earthquake | lora | 0,1,2 | 0.3126 ± 0.0291 | 1.1203 ± 0.0188 |
| matched | qwen2 | noto-earthquake | lora_passes4 | 0,1,2 | 0.2679 ± 0.0392 | 1.1465 ± 0.0200 |
| matched | qwen2 | noto-earthquake | ours | 0,1,2 | 0.3242 ± 0.0272 | 1.1350 ± 0.0058 |
| matched | qwen2 | noto-earthquake | random_kv | 0,1,2 | 0.2803 ± 0.0776 | 1.1620 ± 0.0752 |
| matched | qwen2 | turkey-earthquake | frozen | 0,1,2 | 0.2288 ± 0.0000 | 1.4337 ± 0.0000 |
| matched | qwen2 | turkey-earthquake | lora | 0,1,2 | 0.2363 ± 0.0085 | 1.1946 ± 0.0112 |
| matched | qwen2 | turkey-earthquake | lora_passes4 | 0,1,2 | 0.2992 ± 0.0299 | 1.1840 ± 0.0211 |
| matched | qwen2 | turkey-earthquake | ours | 0,1,2 | 0.2557 ± 0.0351 | 1.2010 ± 0.0283 |
| matched | qwen2 | turkey-earthquake | random_kv | 0,1,2 | 0.2550 ± 0.0204 | 1.2427 ± 0.0036 |
| objective | internvl3 | hospital_2 | lens_mean_full | 0 | 0.5458 | 0.6861 |
| objective | internvl3 | hospital_2 | lens_mean_label | 0 | 0.3333 | 1.0251 |
| objective | internvl3 | hospital_2 | lens_sum_full | 0 | 0.5457 | 0.7252 |
| objective | internvl3 | hospital_2 | lens_sum_label | 0 | 0.3333 | 1.1276 |
| objective | internvl3 | robofail | lens_mean_full | 0 | 0.3005 | 1.7445 |
| objective | internvl3 | robofail | lens_mean_label | 0 | 0.3080 | 1.8279 |
| objective | internvl3 | robofail | lens_sum_full | 0 | 0.3080 | 1.7745 |
| objective | internvl3 | robofail | lens_sum_label | 0 | 0.3080 | 1.8507 |
