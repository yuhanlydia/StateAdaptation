# BRIGHT uniform-event cross-model comparison

This is the apples-to-apples follow-up to the earlier Qwen BRIGHT suite.
It uses the same four uniform events: Hawaii wildfire, Libya flood, Noto
earthquake, and Turkey earthquake.

## Locked protocol

- `target_support.jsonl`: 24 labelled examples, 8 per class.
- `target_query.jsonl`: 300 examples, 100 per class; query labels are never
  used during fitting.
- Paired pre/post building crop, 448px, candidate likelihood over
  `intact`/`damaged`/`destroyed`.
- LoRA: rank 16, alpha 32, dropout 0.05, learning rate 2e-4, one pass over
  support24 with gradient accumulation 3 (exactly 8 optimizer updates).
- Random-KV and Gradient-Cov KV/Ours: layers 14 and 27, rank 16, full
  coefficient matrix, alpha maximum 3, four support updates, learning rate
  0.05, L2 1e-3. Random-KV changes only the basis from covariance to random.

## Four-event mean metrics

Values are mean event Macro-F1 / balanced accuracy / NLL over the four events;
each cell is based on 1,200 query examples in total.

| backbone | Frozen | LoRA | Random-KV | Gradient-Cov KV/Ours |
|---|---:|---:|---:|---:|
| Phi | 0.1665 / 0.3325 / 1.2575 | 0.2364 / 0.3450 / 1.3757 | 0.1665 / 0.3325 / 1.2453 | 0.2075 / 0.3475 / 1.1456 |
| Gemma 3 4B | 0.1737 / 0.3333 / 9.9892 | 0.1970 / 0.3442 / 2.2780 | 0.1820 / 0.3367 / 9.0449 | **0.2607 / 0.3517 / 2.3852** |
| Llama-backed LLaVA | **0.2134** / 0.3300 / 1.2574 | 0.1667 / 0.3333 / 1.8285 | 0.1872 / 0.3342 / 1.2425 | 0.2034 / **0.3450** / **1.1691** |

The Qwen reference from the registered suite is mean Macro-F1 0.3208 for
Gradient-Cov KV/Ours, versus 0.2986 for LoRA and 0.2234 for Frozen. The new
backbone results therefore show that the method transfers best to Gemma in
this controlled comparison, while LoRA remains stronger on Phi Macro-F1 and
Frozen remains strongest on Llama-backed Macro-F1. NLL and Macro-F1 do not
always rank methods identically.

Artifacts are under `runs/bright_uniform/<family>/<event>/`; each method has
`metrics.json`, `predictions.jsonl`, and an adaptation configuration. The
earlier support12/diagonal/alpha0.5 runs remain separate diagnostic artifacts
and are not included in this table.
