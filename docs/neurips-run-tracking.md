# NeurIPS Protocol Run Tracking

Live status log for the long-running NeurIPS batch (main pipeline + ablations;
natural eval skipped). Updated periodically during the run.

- **Started**: 2026-08-08 17:17 UTC (batch log: `logs/neurips_batch.log`)
- **Command**: `bash scripts/run_neurips_batch.sh` (setsid, detached)
- **Runner log**: `/tmp/opencode/neurips_batch.out`
- **GPU**: RTX A5000 24GB, bf16, `device_map=auto`
- **Scope**: 11-fold leave-one-event-out, main pipeline + support-budget ablations
  (pivot: hawaii-wildfire, la_palma-volcano). `target_natural` eval disabled via
  `RUN_NATURAL=0`.

## Per-event pipeline (per fold)

1. `source_adapter/` — LoRA SFT 1000 steps (~85 min)
2. `original_eval/` — pretrained zero-shot baseline
3. `source_eval/` — source-only LoRA baseline on `target_query`
4. `event_kv/` — KV-TTT subspace + coefficients on 24-shot support
5. `kv_eval/` — source + KV controller on `target_query`
6. `kv_gain.json` — KV vs source delta
7. Pivot only: support-budget ablation 12/48 shots

## Fold status

| Event | Main | Ablation 12 | Ablation 48 | kv_gain | Notes |
|---|---|---|---|---|---|
| bata-explosion | in_progress | - | - | - | SFT running |
| beirut-explosion | pending | - | - | - | |
| congo-volcano | pending | - | - | - | |
| haiti-earthquake | pending | - | - | - | |
| hawaii-wildfire | pending | pending | pending | - | pivot |
| la_palma-volcano | pending | pending | pending | - | pivot |
| libya-flood | pending | - | - | - | |
| marshall-wildfire | pending | - | - | - | |
| morocco-earthquake | pending | - | - | - | |
| noto-earthquake | pending | - | - | - | |
| turkey-earthquake | pending | - | - | - | |

## Batches / logs

- `logs/neurips_batch.log` — orchestration trace
- `runs/neurips/<event>/pipeline.log` — per-fold progress
- `runs/neurips/<event>/kv_gain.json` — results when done
