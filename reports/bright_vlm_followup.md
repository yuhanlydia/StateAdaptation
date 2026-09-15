# BRIGHT VLM follow-up

This is a follow-up BRIGHT-only evaluation requested after the earlier
cross-task gate study. It does not modify the completed Qwen/Phi cross-task
tables.

## Fixed protocol

- Dataset: the official BRIGHT release, with the existing NeurIPS 11-event
  splits in `data/prepared/neurips/<event>/`.
- Input: the same building crop from pre-event and post-event imagery,
  448x448, with the existing `intact`/`damaged`/`destroyed` candidate-label
  likelihood scorer.
- Frozen evaluation: every target query is scored once; no target query is
  used for fitting or model selection.
- Adaptation: support-only, with `support_12.jsonl`/`support_48.jsonl` fixed
  before query evaluation. LoRA uses rank 16, alpha 32, dropout 0.05, four
  passes and learning rate 2e-4. KV controls use the same selected-layer and
  coefficient budget; all failures and OOMs are recorded rather than hidden.
- Runtime: each batch is wrapped by the six-hour timeout and is resumable at
  event-fold boundaries.

## Backbones

- Phi: `artifacts/models/Phi-3.5-vision-instruct`.
- Gemma: `artifacts/models/gemma-3-4b-it`.
- Llama arm: `artifacts/models/llava-llama-3-8b-v1_1-transformers`, an open
  LLaVA checkpoint backed by Llama 3. This is **not** the gated official
  `meta-llama/Llama-3.2-11B-Vision-Instruct`; the official download returned
  HTTP 403 for the authenticated account and was not bypassed.

Transformers was upgraded to 4.57.x to support Gemma 3. The Phi loader
disables the legacy cache path required by the remote Phi implementation.

## Reproducibility

Frozen runs write `predictions.jsonl`, `metrics.json`, and `eval_config.json`
under `runs/bright_vlm/<family>/<event>/`. The support-only LoRA pilot uses
`scripts/adapt_bright_lora.py` and writes its adapter, losses, config and
query metrics in the output directory. The exact model identifiers and split
paths are retained in every config.

## Status

All three frozen 11-fold batches are complete. Aggregate target-query results
(2,752 examples per backbone; macro-F1 / balanced accuracy / NLL) are:

| backbone | mean event macro-F1 | mean event BA | mean event NLL |
|---|---:|---:|---:|
| Phi | 0.127808 | 0.302727 | 1.331756 |
| Gemma 3 4B | 0.158282 | 0.348763 | 11.339470 |
| Llama-backed LLaVA | 0.213170 | 0.347639 | 1.178409 |

The pooled (all events) macro-F1 values are 0.149172, 0.163429 and 0.227931
for Phi, Gemma and Llama-backed LLaVA respectively. These Frozen baselines
are weak and should not be treated as evidence that adaptation helps.

Hawaii full-query support-only adaptation was then run after the smoke/debug
gate. Macro-F1 / balanced accuracy / NLL:

| backbone | Frozen | LoRA | Random-KV | Gradient-Cov KV/Ours |
|---|---:|---:|---:|---:|
| Phi | 0.1671 / 0.3333 / 1.2389 | 0.2848 / 0.3200 / 1.1128 | 0.1667 / 0.3333 / 1.2412 | 0.1671 / 0.3333 / 1.2332 |
| Gemma 3 4B | 0.1667 / 0.3333 / 10.0532 | 0.3081 / 0.3700 / 2.6264 | 0.1667 / 0.3333 / 10.0216 | 0.1667 / 0.3333 / 9.9343 |
| Llama-backed LLaVA | 0.2252 / 0.3267 / 1.2679 | 0.1667 / 0.3333 / 1.1141 | 0.2252 / 0.3267 / 1.2678 | 0.2098 / 0.3267 / 1.2691 |

These Hawaii numbers are one-event diagnostics, not a claim of generalization
or a direct comparison to the registered Qwen BRIGHT suite. In particular,
this follow-up used `support_12`, a diagonal rank-8 controller with
`alpha_max=0.5`, and architecture-relative middle/final layers. The registered
Qwen balanced suite uses the uniform-event subset, `target_support` (24
examples), and its locked full rank-16/alpha-3 configuration. Therefore the
adapter table above must not be used to claim that Gradient-Cov KV loses to a
baseline; an apples-to-apples Phi/Gemma/Llama rerun is still required.
