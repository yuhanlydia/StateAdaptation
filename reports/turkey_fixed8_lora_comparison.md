# Turkey fixed-8 LoRA comparison

This diagnostic removes per-event support CV from the Turkey LoRA baseline.
LoRA is pre-registered at 8 optimizer updates: batch size 1 with gradient
accumulation 3, so all 24 balanced support examples are processed exactly once.
The balanced 300-example query is used only after training.

| Method | Support epochs | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|---:|
| Pure VLM | 0 | 0.2273 | 0.3400 | 1.4333 |
| LoRA fixed8, no support CV | 1 | 0.2147 | 0.3233 | **1.1980** |
| Full KV-TTT, steps1 | 1 | 0.2352 | 0.3400 | 1.3662 |
| Full KV-TTT, alpha3 steps4 | 4 | 0.2799 | 0.2800 | 1.2512 |
| Full KV-TTT, alpha3 steps8 | 8 | **0.2977** | 0.3000 | 1.2443 |
| LoRA support-CV selected32 (reference) | 4 | 0.4301 | **0.4300** | 1.2796 |

Under the requested fixed8/no-CV LoRA protocol, Full KV-TTT is better in
Macro-F1: +0.0205 at the matched one-support-epoch budget, +0.0652 for the
primary four-step KV configuration, and +0.0830 for the eight-step KV arm.
The old 0.4301 LoRA result remains a valid support-CV-tuned reference, but it
is a different protocol and must not be labeled fixed8.
