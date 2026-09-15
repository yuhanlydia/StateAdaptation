# Failure and efficiency analysis

The LoRA row below is the original fixed-support, per-event-CV-selected LoRA,
not the later clean fixed-8 baseline. KV rows average three support seeds of the
compact rank-5 alpha-3 controller.

## Mean per-class F1

| Method | Intact | Damaged | Destroyed | Macro-F1 |
|---|---:|---:|---:|---:|
| baseline | 0.0196 | 0.4983 | 0.1524 | 0.2234 |
| lora | 0.3032 | 0.3009 | 0.3862 | 0.3301 |
| full | 0.2696 | 0.2178 | 0.3201 | 0.2692 |
| diagonal | 0.0999 | 0.3529 | 0.3371 | 0.2633 |

Raw VLM and LoRA use the original fixed support24 manifest.

## Efficiency

| Method | Trainable scalars | Mean artifact size | Extraction | Coefficient fit |
|---|---:|---:|---:|---:|
| lora | 10,092,544 | 39453.3 KiB | n/a | not recorded |
| full | 100 | 44.0 KiB | 17.2s | 53.6s |
| diagonal | 20 | 43.8 KiB | 17.1s | 53.3s |

Full uses 100 trainable controller scalars and Diagonal uses 20, versus 10,092,544 LoRA parameters. KV extraction and coefficient-fit times exclude model loading and 300-query evaluation.
