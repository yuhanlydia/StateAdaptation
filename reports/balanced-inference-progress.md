# Balanced same-event inference progress

Last updated: 2026-08-11 UTC.

Status: **complete**. All four uniform events, supervised ablations, multiseed
robustness, clean fixed-8 LoRA, and rank-16 Full/Diagonal alpha sweeps have
finished. The current publication-facing outputs are
`fixed8_and_scaling_diagnostics.md`, `rank16_full_diagonal_alpha_sweep.md`, and
`paper/main.tex`. `balanced_inference_results.md/json` preserve the earlier
CV-selected-LoRA protocol and are not the clean fixed-8 primary table.

The extended mechanism and robustness suite is also complete: mean-gradient,
centered-covariance, three-seed random-subspace controls, alpha
0.5/1/2/3/5/10, and coefficient learning rates 0.01/0.05/0.1/0.2 were all run
on all four events. See `extended_neurips_results.md`. Exact 300-query latency
and memory measurements for pure VLM, LoRA, Full KV-TTT, and Diagonal KV-TTT
are in `inference_efficiency_benchmark.md`.

A controlled rank-16 Full/Diagonal alpha sweep is complete (48/48): six alpha
values on both controller types across all four events. See
`rank16_full_diagonal_alpha_sweep.md`. Per-event query-oracle means are 0.3504
for Full and 0.3254 for Diagonal, versus 0.2986 for clean fixed-8 LoRA; these
maxima are explicitly not treated as support-selected primary estimates.

The NeurIPS-level supervised ablation expansion is 40/40 complete. It covers support budget 12/24/48,
layers 14/27/14+27, ranks 5/8/16/32, and update steps 1/2/4/8 using raw VLM,
same-event labeled support, full KV-TTT, and alpha 3. No run has failed or hit
CUDA OOM.

A multiseed robustness suite is also complete (16/16). It adds two independently sampled, balanced,
query-tile-disjoint support24 seeds for Full and Diagonal alpha-3 KV-TTT on all
four events (16 new runs). The final three-seed paired analysis is in
`multiseed_significance.md` and `multiseed_significance.json`: Full versus raw
VLM has mean delta Macro-F1 +0.0457 (95% CI +0.0046 to +0.0847, p=0.0167),
and Diagonal versus raw VLM has +0.0399 (95% CI +0.0063 to +0.0734,
p=0.0061). Full versus Diagonal is not significant (p=0.7347).

The experiment
uses only the four strictly uniform BRIGHT events (support 8/8/8 and query
100/100/100). Every adaptation starts independently from raw
`Qwen/Qwen2.5-VL-7B-Instruct` and uses the same event's support24. All deltas
below are against raw-VLM `original_eval` on the identical 300 query IDs.

## Completed raw-VLM baselines

| Event | Query N | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|---:|
| Hawaii wildfire | 300 | 0.1992 | 0.3433 | 1.2491 |
| Libya flood | 300 | 0.2585 | 0.3633 | 1.3331 |
| Noto earthquake | 300 | 0.2087 | 0.3333 | 1.2779 |
| Turkey earthquake | 300 | 0.2273 | 0.3400 | 1.4333 |

## Current fixed-configuration main results

| Event | Raw VLM | LoRA fixed8 | Full rank16 alpha3 | Diagonal rank16 alpha3 |
|---|---:|---:|---:|---:|
| Hawaii wildfire | 0.1992 | **0.3526** | 0.2957 | 0.2918 |
| Libya flood | 0.2585 | 0.2776 | **0.3739** | 0.2560 |
| Noto earthquake | 0.2087 | **0.3495** | 0.3112 | 0.2820 |
| Turkey earthquake | 0.2273 | 0.2147 | **0.3025** | 0.2879 |
| Mean | 0.2234 | 0.2986 | **0.3208** | 0.2794 |

## Primary queue status

- Completed: raw-VLM originals and support24 LoRA for all four events.
- Completed primary KV-TTT: supervised full alpha 3, supervised diagonal
  alpha 3, and unsupervised diagonal alpha 3 for all four events.
- Completed ablations: alpha-0.5 amplitude variants and unsupervised
  full-controller variants for all four events.
- No suite process remains active and all result comparisons contain 300
  query examples per event and arm.

## Resume and failure behavior

The suite is artifact-resumable: completed adapters, KV states, evaluations,
and comparisons are skipped only when their completion artifacts exist, while
partial evaluations resume through the configuration-aware evaluation path.
Restarting

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_balanced_inference_suite.sh
```

therefore continues the queue without discarding completed results. A network
failure does not affect local training or evaluation because the model and
BRIGHT data are cached locally. It can delay only GitHub/Hugging Face
publication; the local Git commit and run artifacts remain available for a
later retry.

The legacy suite summary can be regenerated with
`scripts/summarize_balanced_inference.py`; the clean fixed-8 and rank-16 sweep
reports are maintained separately to prevent protocol mixing.
