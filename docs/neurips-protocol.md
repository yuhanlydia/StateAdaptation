# NeurIPS Protocol: baselines, main experiment, ablations

Reproducibility plan for the EventTune + Post-Visual Residual KV-TTT
evaluation on BRIGHT, one latest experiment arm per section. Everything below
runs from a clean checkout with a 24 GB single GPU once `data/bright.jsonl`
exists (see `README.md` -> "Reproduce from scratch" / `scripts/download_bright.sh`
+ `scripts/prepare_dataset.py`).

## Environment

- Python 3.10, packages from `requirements.txt` (`transformers==4.49.0`,
  `peft`, `torch`, `qwen-vl-utils`).
- **`PYTHON_BIN` defaults to `python3`** in `scripts/run_event_kv.sh` and
  `scripts/run_neurips_batch.sh` (no `.venv` assumption).
- GPU: bf16, `device_map=auto`, LoRA batch size 1, gradient-accumulation 6.

## Data cuts (regenerate with `scripts/prepare_neurips_splits.py all`)

Leave-one-event-out over the _11_ BRUTE events; per-event outputs under
`data/prepared/neurips/<event>/`:

| Manifest | Content | Size rule |
|---|---|---|
| `source_train.jsonl` | other-event clips | balanced 1350 / label -> 4050 |
| `target_support.jsonl` | target tile-disjoint support | 8 / label -> **24** (main) |
| `support_12.jsonl`, `support_48.jsonl` | same protocol | 4 / 16 shots per label |
| `target_query.jsonl` | held-out queries | balanced, 100 / label -> <=300 |
| `target_natural.jsonl` | held-out, raw distribution | full, capped at 60k/event |

448 px crops are taken at runtime from the raw tiles; nothing is materialised.

## Baselines

1. **Pre-trained zero-shot** (raw Qwen2.5-VL-7B, no LoRA) -> `original_eval/`.
   Enabled with `RUN_ORIGINAL_EVAL=1` in `scripts/run_event_kv.sh`.
2. **Source-only LoRA** (`source_adapter/` -> `source_eval/`): state-of-the-art
   VLM fine-tuned on the other-event source clips, evaluated on the held-out
   target query set. This is the KV-TTT baseline with the KV controller
   switched off.

## Main experiment (EventTune + Residual KV-TTT)

`scripts/run_event_kv.sh <split_dir> <run_dir> 1000` runs, per event:

1. `source_adapter/` - LoRA SFT (r16/alpha32/dropout05, steps 1000, lr 2e-4,
   class-circle sampling, grad-accum 6, crop 448).
2. `source_eval/` - baseline **source-only** evaluation of `target_query.jsonl`
   (D4 views=1, product-of-experts label log-likelihood).
3. `event_kv/` - Post-Visual Residual KV-TTT: correctness-gradient KV subspace
   `C = sum G^T G` on the target support24, top-`rank` eigenbasis, 32 scalar
   coefficients fit by Adam (lr .05, 4 full-support updates, L2 1e-3, alpha_max .5),
   intervention layers `14 27`, post-image tokens only.
4. `kv_eval/` - the same `target_query.jsonl` scored with the KV controller;
   repeated for natural ablations below.
5. `kv_gain.json` - prediction-level + metrics deltas (KV vs source baseline).

n_links parameters: `KV_RANK=5`, `KV_STEPS=4`, `KV_ALPHA_MAX=0.5`,
`SOURCE_STEPS=1000`, `EVAL_D4_VIEWS=1`.

## Ablations

| Arm | Manifest(s) | Definition |
|---|---|---|
| Support budget 24 => 12 / 48 | `support_12/48.jsonl` | same protocol but 4 or 16 shots / label; `adapt_event_kv` only |
| Natural (unbalanced) distribution | `target_natural.jsonl` | source baseline + KV controller evaluated on the raw class mix |
| Balanced (main) | `target_query.jsonl` | 100/label balanced set |
| Model backbone | n/a | Ablation of VLM family is tracked as a future arm, not part of the current pull |

Default support budget pivot events: `hawaii-wildfire`, `la_palma-volcano`
(`PIVOT_EVENTS` in the runner).

## Orchestration

`scripts/run_neurips_batch.sh` runs all folds above sequentially with resume
safety: each stage skips when its output already exists (`source_adapter/
train_summary.json`, `kv_gain.json`, `*/metrics.json`), the runner waits for a
free GPU, and logs to `logs/neurips_batch.log` + per-fold `runs/neurips/<event>/
pipeline.log`.

```bash
# fresh clone, after data download + prepare_dataset + prepare_neurips_splits
bash scripts/run_neurips_batch.sh
```

To run a single event:

```bash
bash scripts/run_event_kv.sh data/prepared/neurips/<event> runs/neurips/<event> 1000
```