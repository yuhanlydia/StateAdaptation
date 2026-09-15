# EventTune

Few-shot internal-state adaptation of frozen vision--language models, with
cross-domain evidence on remote sensing, medical imaging, and robotics.

## ICLR generalization expansion

The current expansion tests a broader principle: few-shot test-time adaptation
can modify task-relevant internal visual evidence states instead of model
weights. It keeps one gradient-second-moment KV-TTT operator across remote
sensing, medical imaging, and robot manipulation, changing only the visual-token
mask. Qwen2.5-VL-7B is the primary BRIGHT backbone; Qwen3-VL-8B and
InternVL3-8B provide the main architecture transfer, with additional
Phi/Gemma/LLaVA BRIGHT controls.

**Current evidence freeze (2026-08-26).** The GPU experiment phase is closed.
The ICLR 2027 draft asks when internal-state adaptation can replace weight
adaptation, with BRIGHT as the positive evidence-shift regime, Camelyon as a
geometry/realization diagnostic, RoboFail as a class-mixture/site mismatch,
and ManipBench as an actuator/capability boundary. See
[`reports/iclr2027_evidence_audit.md`](reports/iclr2027_evidence_audit.md) for
the protocol/source-of-truth matrix and [`paper/main.pdf`](paper/main.pdf) for
the current compiled draft. Query-label results are mechanism oracles only.

To reproduce the pinned model and dataset preparation:

```bash
bash scripts/download_oral_assets.sh
```

The script downloads both backbones, the official ManipBench code and simplified
dataset, and materializes three leakage-safe support seeds for Camelyon17-WILDS
and ManipBench Q1. Large assets stay outside Git. See
[`docs/ORAL_ASSET_STATUS.md`](docs/ORAL_ASSET_STATUS.md) for exact revisions,
checksums, counts, and local paths.

The preregistered configuration, experiment matrix, admission gates, leakage
rules, statistical tests, artifact contract, and exact GPT-Luna handoff order
are in [`configs/oral_generalization.yaml`](configs/oral_generalization.yaml)
and [`docs/LUNA_ORAL_EXECUTION.md`](docs/LUNA_ORAL_EXECUTION.md).

## Project status and history

The dated snapshot below is retained as historical provenance and is
superseded by the current evidence freeze above.

**Snapshot: 2026-08-08 UTC.** The end-to-end system is implemented and has
completed one full 7B diagnostic, but the research hypothesis is **not yet
validated**. The first result was negative and is preserved rather than hidden.
The new Post-Visual Residual KV-TTT path is implemented, all eight pre-run
sanity gates pass, and a smoke GPU run is complete; a Hawaii query run has not
started. No GPU training job is currently running.

| Area | Current state | Evidence |
|---|---|---|
| Repository | Private, flattened, and recoverable on a fresh GPU node | [`cf34db1`](https://github.com/Yunbo-max/EventTune/commit/cf34db1291619f7cc86cbfcad258260f13f35714) |
| Model path | Qwen2.5-VL-7B, frozen BF16 base, rank-16 LoRA; no 4-bit loading or QLoRA | [`701cf7b`](https://github.com/Yunbo-max/EventTune/commit/701cf7b) |
| Data | BRIGHT instance manifest plus event-held-out, tile-disjoint splits are stored in the private dataset Hub | [EventTune-BRIGHT](https://huggingface.co/datasets/humanlong/EventTune-BRIGHT) |
| Completed run | Hawaii wildfire, seed 0, 100 source updates, 12 support and 3,443 query examples | [diagnostic report](reports/20260804-hawaii-100step-diagnostic.md) |
| Durable artifacts | Source/event LoRA adapters, complete predictions, metrics, and hashes are stored privately | [EventTune adapters](https://huggingface.co/humanlong/EventTune-Qwen2.5-VL-7B) |
| Corrective controls | Per-update class cycling and an independent 150-example source gate | [`63a36c5`](https://github.com/Yunbo-max/EventTune/commit/63a36c5), [`2a66683`](https://github.com/Yunbo-max/EventTune/commit/2a66683) |
| Verification | 32 unit tests pass (incl. 9 KV-TTT); shell and Python syntax checks pass | [`tests/`](tests) |
| KV-TTT path | Post-Visual Residual KV-TTT implemented, all sanity gates pass, smoke GPU run complete | [`src/eventttt/kv_ttt.py`](src/eventttt/kv_ttt.py) |
| Next experiment | 11-fold NeurIPS main + baselines + ablations (EVAL_D4_VIEWS=1, KV rank 5, support 24/12/48) | [docs/neurips-protocol.md](docs/neurips-protocol.md) + `scripts/run_neurips_batch.sh` |

### Result so far

The first 100-update systems run proved that the data, training, adaptation,
evaluation, export, and Hub recovery paths work end to end. It did **not** show
a model improvement:

| Hawaii query result | Source LoRA | Event-adapted LoRA |
|---|---:|---:|
| Macro-F1 | 0.145947 | 0.057340 |
| Balanced accuracy | 0.333333 | 0.333333 |
| Prediction collapse | all `intact` | all `damaged` |

The failure is consistent with under-training and update-level class imbalance:
only 200 examples were sampled from a 241,442-example source manifest. The
adapted result is worse, so it is recorded as a negative diagnostic and not as
evidence for EventTune. The exact weights and results remain available for
reproduction, with the model card explicitly warning against production use.

### Post-Visual Residual KV-TTT (experimental path)

An independent, experimental adaptation path implemented alongside the LoRA
EventTune baseline. The original LoRA pipeline is untouched and remains the
reference baseline. KV-TTT freezes the entire VLM **and** the source LoRA, and
learns only a tiny event-specific coefficient vector operating in a
correctness-gradient-derived K/V subspace:

| Component | Setting |
|---|---|
| Intervention site | Qwen2.5-VL language-decoder `k_proj` / `v_proj` outputs (post-image tokens only) |
| Prompt layout | first image = pre-event optical; second image = post-event optical/SAR; then text; label-only loss on the assistant class-label span |
| Layers | `num_layers // 2` and `num_layers - 1` (middle + last; 14 and 27 for the 7B) |
| Subspace | correctness-gradient covariance `C = Σ G^T G` over post-image K/V rows, top-`rank` eigenvectors of `C` (= SVD right singular directions) |
| Adaptation | residual `Z' = Z + M ⊙ (Z B) diag(α·tanh(a)) B^T`, `α_max = 0.5`, only post-image token rows `M` are touched |
| Trainable | 2 layers × 2 (K/V) × rank 8 = **32 scalar coefficients**; base VLM = 0, source LoRA = 0 |
| Coefficient fit | Adam, lr 0.05, 4 full-support updates (no sampler), L2 1e-3, grad clip 1.0 |
| Saved artifact | `event_kv/kv_state.pt` + `extraction.json` + `adaptation.json` (no new LoRA weights) |

New files:

```text
src/eventttt/kv_ttt.py        mask builder, K/V discovery, gradient collector,
                              subspace extraction, ResidualKVController,
                              coefficient fitting, serialization
scripts/adapt_event_kv.py     extract subspace + fit coefficients from support
scripts/sanity_check_kv.py    pre-run gate checks (CPU/GPU)
scripts/evaluate.py           gained --kv-state PATH, same D4 scorer
tests/test_kv_ttt.py          9 unit tests (mask grouping, C↔SVD, identity, isolation)
```

**Sanity gates — all pass (RTX A5000, BF16).** These gate a full Hawaii query run:

| Gate | Result |
|---|---:|
| `a = 0` reproduces source logits | max diff = 0.00 |
| Post-image K/V gradients present | `14:K` 1.7e-2, `14:V` 1.5e-2, `27:K` 1.1e-3, `27:V` 2.3e-4 |
| Basis orthogonality `B^T B ≈ I` | max dev 8.3e-7 |
| Parameter isolation | base 0 / LoRA 0 / KV scalars 32 |
| Mask isolation | pre-image tokens strictly unchanged; post-image tokens changed; write is masked to post rows only |
| Reset restores source logits | max diff = 0.00 |
| Save/load round trip | max diff = 0.00 |
| Support loss after fitting | 0.120 → 0.094 |

Note: text tokens that follow the post-image group legitimately change through
causal attention (their queries attend to the edited post K/V) — that is the
adaptation mechanism, not a mask violation.

**Smoke GPU run (12 support / 6 query, synthetic smoke data, seed 0).** The
pipeline ran end to end: 7B source LoRA → KV subspace extraction (10.2 s) → 32
coefficient fit (31.2 s) → query evaluation with and without the KV state.

| Query result | Source LoRA | Source + KV-TTT |
|---|---:|---:|
| Macro-F1 | 1.000 | 1.000 |
| Balanced accuracy | 1.000 | 1.000 |
| NLL | 0.3004 | 0.2737 |

The toy smoke set is saturated (both reach F1 1.0), so the observable signal so
far is the NLL improvement (0.300 → 0.274) plus the support-loss reduction;
macro-F1 does not discriminate on this data. Eigenvalue spectra show a clean
rank-8 drop-off, strongest at layer 14 (K > V). Reproduce with:

```bash
python scripts/make_smoke_data.py --output-dir data/smoke
# source LoRA (tiny, 4 updates)
python scripts/adapt_event.py --support-manifest data/smoke/source.jsonl \
  --output-dir reports/source_smoke --fixed-steps 4
# KV-TTT adaptation
python scripts/adapt_event_kv.py --support-manifest data/smoke/support.jsonl \
  --source-adapter reports/source_smoke --output-dir reports/kv_tune_smoke
# evaluation with / without the KV state
python scripts/evaluate.py --manifest data/smoke/query.jsonl \
  --adapter reports/source_smoke --output-dir reports/eval_source
python scripts/evaluate.py --manifest data/smoke/query.jsonl \
  --adapter reports/source_smoke --kv-state reports/kv_tune_smoke/kv_state.pt \
  --output-dir reports/eval_kv
```

Support-CV rule: the KV basis itself uses support labels, so a naive
"full-support basis → support CV" selection is invalid. Hyperparameters
(rank/layers/lr/steps/α_max) are pre-registered; any future CV must re-extract
the basis per fold from that fold's training subset only.

### Development timeline

| Date | Milestone | Outcome |
|---|---|---|
| 2026-08-04 | Initial remote-sensing adaptation prototype imported ([`c4eaf3e`](https://github.com/Yunbo-max/EventTune/commit/c4eaf3e)) | Established the original paired-image EventTTT experiment. |
| 2026-08-04 | Reproducible compute and private Hub workflow added ([`6116055`](https://github.com/Yunbo-max/EventTune/commit/6116055)) | A disposable GPU can recover code, manifests, splits, and adapters from GitHub and Hugging Face. |
| 2026-08-04 | Portable manifests and EventTune naming completed ([`e40d1e1`](https://github.com/Yunbo-max/EventTune/commit/e40d1e1), [`c52ac48`](https://github.com/Yunbo-max/EventTune/commit/c52ac48), [`cf34db1`](https://github.com/Yunbo-max/EventTune/commit/cf34db1)) | Removed machine-specific paths and the old outer project folder. |
| 2026-08-04 | Switched to Qwen2.5-VL-7B BF16 LoRA ([`701cf7b`](https://github.com/Yunbo-max/EventTune/commit/701cf7b), [`8e04f03`](https://github.com/Yunbo-max/EventTune/commit/8e04f03)) | Confirmed the unquantized 7B path fits an RTX A5000 at 448 px. |
| 2026-08-04 | Long evaluation resume and portable export added ([`ae1dc0a`](https://github.com/Yunbo-max/EventTune/commit/ae1dc0a), [`71786c9`](https://github.com/Yunbo-max/EventTune/commit/71786c9)) | Long evaluations can resume; completed runs can be hashed and moved to private Hub storage. |
| 2026-08-04 | First Hawaii diagnostic completed | Found source and event-adapter single-class collapse; published the full negative result. |
| 2026-08-04 | Class-cycle sampler added ([`63a36c5`](https://github.com/Yunbo-max/EventTune/commit/63a36c5)) | Every six-microbatch update now contains two examples from each damage class. |
| 2026-08-04 | ARC Prize 2025 review and fail-closed source gate added ([`2a66683`](https://github.com/Yunbo-max/EventTune/commit/2a66683)) | Gate samples are excluded from training; a failed source gate stops before target-query evaluation. |

The [ARC Prize 2025 / ARC-AGI-2 review](reports/20260804-arc-prize-2025-review.md)
records which refinement, validation, and reliability ideas transfer to
EventTune and which reported ARC results contain incompatible assumptions or
query leakage.

### Next run

The next run will use a new directory and will not reuse the failed diagnostic
adapter. It starts with the 7B BF16 LoRA, 1,000 source updates, gradient
accumulation six, and the held-out source gate. The pipeline requires gate
macro-F1 of at least `0.2` and at least two predicted classes before evaluating
the target query. The 3B model is only a CUDA/memory fallback; a learning-gate
failure triggers diagnosis rather than an unrecorded model change.

Project history is append-only at the experiment level: positive and negative
runs keep immutable configurations, complete denominators, artifact hashes, and
links to their Git and Hub revisions.

## Quick start on a fresh GPU node

GitHub is the source of truth for code and reproducibility instructions. The
private Hugging Face collections hold durable dataset derivatives and trained
adapters:

- [EventTune-BRIGHT dataset](https://huggingface.co/datasets/humanlong/EventTune-BRIGHT)
- [EventTune Qwen2.5-VL-7B adapters](https://huggingface.co/humanlong/EventTune-Qwen2.5-VL-7B)
- [Qwen2.5-VL-7B base model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [BRIGHT official release](https://zenodo.org/records/20072020)

Authenticate once, clone the code, choose a PyTorch wheel compatible with the
node, and restore the persistent artifacts:

```bash
git clone https://github.com/Yunbo-max/EventTune.git
cd EventTune

python3 -m venv .venv
# Example for a CUDA 12.1-compatible node; select another official PyTorch
# wheel index when the host driver requires it.
.venv/bin/python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -e '.[train,test]'

.venv/bin/hf auth login
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh pull
.venv/bin/python -m pytest
.venv/bin/python scripts/preflight.py
```

`hub_sync.sh pull` restores `data/manifests/`, `data/splits/`, checksums, and
available model adapters. Transformers obtains the immutable base model directly from
`Qwen/Qwen2.5-VL-7B-Instruct` on the first model run. If the dataset Hub does
not yet contain a required raw asset, follow [Data preparation](#data-preparation)
to acquire it from the official source and rebuild the derivative.

Run a prepared event split with:

```bash
bash scripts/run_event.sh \
  data/splits/bright_4shot/TARGET_EVENT/seed_0 \
  runs/TARGET_EVENT/seed_0 \
  1000
```

Before releasing an ephemeral node, publish approved outputs and push code:

```bash
.venv/bin/python scripts/export_run.py \
  --run-dir runs/TARGET_EVENT/seed_0 \
  --destination runs/TARGET_EVENT/seed_0/run-v1
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh push-data
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh push-model
git add -A && git commit -m "Describe the experiment update" && git push
```

## What this project studies

Disaster-damage models often face a large domain shift at deployment. A new event can have a different hazard, country, sensor, season, image resolution, viewing geometry, or post-event modality. A model trained on earlier disasters may therefore perform poorly on the new event.

This project tests one focused hypothesis:

> Can a source-trained Qwen2.5-VL-7B model improve on an unseen disaster by fitting a small temporary adapter to only a few labeled buildings from that event?

Each example contains:

- a pre-event optical crop;
- a post-event optical or SAR crop of the same building;
- the building bounding box and event/tile identifiers;
- one normalized label: `intact`, `damaged`, or `destroyed`.

The primary dataset is BRIGHT. Dataset adapters also exist for xBD and DisasterM3 so the same protocol can later be evaluated across datasets.

## What method is implemented

The method is **support-selected event-time LoRA**, named EventTune. It has seven stages. The Python package retains the historical `eventttt` namespace for compatibility.

### 1. Leakage-safe event split

One disaster is held out as the target event. All other events form `source_train`. A small class-balanced target support set is sampled, and the remaining target buildings form `target_query`.

The split is tile-disjoint: every tile used to construct the support pool is removed from the query set. Consequently, nearby buildings from the same image tile cannot appear on both sides of evaluation.

### 2. Source supervised fine-tuning

`Qwen/Qwen2.5-VL-7B-Instruct` is trained on the source events with label-only supervised loss. The prompt supplies the paired pre/post images and asks for exactly one of the three severity labels.

Training uses standard BF16 LoRA:

- LoRA rank 16, alpha 32, dropout 0.05;
- adapters on `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
- the 7B base VLM remains frozen in bfloat16;
- only the approximately 10.1 million LoRA parameters are optimized.
- source examples use a randomized class-cycle sampler. With the default
  gradient accumulation of six, every optimizer update contains exactly two
  examples from each class instead of relying on long-run balance.

No 4-bit quantization or QLoRA path is used. On the reference RTX A5000 24 GB,
one 448-pixel paired-image training step peaks at approximately 17.0 GiB
allocated and 18.5 GiB reserved CUDA memory.

### 3. Source-only target baseline

Before seeing target support examples, the source adapter is evaluated on the held-out target query set. This produces the baseline against which event adaptation is compared.

Classification uses candidate-label likelihood instead of unconstrained text generation. The model scores the full assistant answer for each possible label and normalizes the three scores into probabilities.

### 4. Support-only adaptation selection

The target query set is never used to choose hyperparameters. Candidate update counts `[0, 4, 8, 16, 32]` are evaluated with stratified cross-validation entirely inside the labeled support set.

Selection maximizes mean support-CV macro-F1, then breaks ties using lower NLL, lower ECE, and fewer update steps. Including zero steps lets the selector reject harmful adaptation.

### 5. Final temporary event adapter

After selection, the source adapter is reset to its original state and fitted on all target support examples for the selected number of updates. This produces a separate, temporary adapter for that event.

### 6. Paired D4 inference

At evaluation time, the same rotation/reflection is applied to both pre- and post-event images. Per-view label evidence is combined with a product of experts: log-scores are averaged across views and then normalized.

### 7. Adaptation-gain evaluation

The source and adapted adapters are evaluated on exactly the same unseen query buildings. `adaptation_gain.json` reports:

- gains where higher is better: macro-F1, balanced accuracy, quadratic weighted kappa;
- error reductions where positive is better: ordinal MAE, NLL, Brier score, and ECE.

This design evaluates **supervised few-shot event-time adaptation**. The current implementation is not unlabeled test-time training, reinforcement learning, or self-distillation.

## Experimental protocol

The recommended study uses:

- leave-one-disaster-event-out evaluation;
- 1, 2, 4, and 8 labeled examples per class;
- seeds 0–4;
- operational support budgets of 12, 24, and 48 examples;
- target events with different hazards and sensors;
- a source-only LoRA baseline and an adapted LoRA model under identical inference settings.

The first one-GPU Hawaii wildfire diagnostic used four support examples per
class, 100 source updates, and one inference view. It completed the full 3,443
query denominator but exposed single-class collapse in both the source and
adapted adapters. See
[`reports/20260804-hawaii-100step-diagnostic.md`](reports/20260804-hawaii-100step-diagnostic.md).
The run is a failed systems diagnostic, not positive evidence for the method.
The corrected protocol uses exact per-update class cycling and must pass a
balanced source-domain gate before another full target evaluation.

The design review that motivated the gate and stricter experiment protocol is
documented in
[`reports/20260804-arc-prize-2025-review.md`](reports/20260804-arc-prize-2025-review.md).
It focuses on ARC Prize 2025 / ARC-AGI-2; ARC Prize 2024 is used only as a
methodological predecessor.

## Repository layout

```text
configs/                 Experiment configuration
data/README.md           Dataset locations and checksums
scripts/download_bright.sh
                         BRIGHT downloader
scripts/prepare_dataset.py
                         BRIGHT, xBD, and DisasterM3 normalization
scripts/make_splits.py   Event-held-out, tile-disjoint splits
scripts/train_source.py  Source-event LoRA SFT
scripts/adapt_event.py   Support-only CV and final event adapter
scripts/evaluate.py      Candidate scoring and D4 evaluation
scripts/compare_predictions.py
                         Baseline/adaptation comparison
scripts/run_event.sh     Complete single-event pipeline
scripts/launch_long_job.py
                         Detached, logged one-GPU launcher
src/eventttt/            Reusable Python implementation
tests/                   Unit tests
```

## Detailed setup

Python 3.10+ and an NVIDIA GPU are expected.

Use a PyTorch CUDA build compatible with the host NVIDIA driver. Do not assume
every compute node has the same CUDA stack. For example, a CUDA 12.1-compatible
node can use:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -e '.[train,test]'
.venv/bin/python scripts/preflight.py
```

The preflight command stops early if CUDA, bfloat16, Qwen-VL, or PEFT is unavailable.

## Ephemeral compute and persistent assets

GitHub is the source of truth for code, configuration, and reproduction
instructions. Hugging Face stores durable dataset derivatives and model
artifacts. A GPU node is disposable and should be recoverable from those two
services.

The private Hub repositories are:

- dataset derivatives: `humanlong/EventTune-BRIGHT`;
- model adapters and run summaries: `humanlong/EventTune-Qwen2.5-VL-7B`.

On a fresh node, clone/pull GitHub, install a PyTorch build compatible with the
node, and then restore persistent assets:

```bash
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh pull
```

Before a node is released, keep approved manifests/splits under `data/`, and
export each completed run into `artifacts/model/`, then run:

```bash
.venv/bin/python scripts/export_run.py \
  --run-dir runs/TARGET_EVENT/seed_0 \
  --destination runs/TARGET_EVENT/seed_0/run-v1
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh push-data
HF_CLI=.venv/bin/hf bash scripts/hub_sync.sh push-model
git status
```

The exporter requires both evaluations and `adaptation_gain.json`, omits raw
imagery and caches, rewrites machine-local paths in run metadata, records a
SHA-256 manifest, and refuses to overwrite a different exported run.

Override either Hub destination with `EVENTTUNE_DATASET_REPO` or
`EVENTTUNE_MODEL_REPO`. Do not upload BRIGHT, xBD, or DisasterM3 raw archives
unless their distribution terms explicitly permit it. The official acquisition
locations and checksums are documented in `data/README.md`; the generated
manifests, splits, checksums, and permitted derivatives belong in the dataset
repository.

## Data preparation

Download only the public BRIGHT instance labels:

```bash
bash scripts/download_bright.sh labels data/raw/bright
```

Download the complete official BRIGHT pre/post/target release:

```bash
bash scripts/download_bright.sh full data/raw/bright
```

Create the preferred instance-level manifest and crops:

```bash
.venv/bin/python scripts/prepare_dataset.py bright-coco \
  --root data/raw/bright \
  --annotations data/raw/bright/target_instance_level \
  --output data/manifests/bright_instances.jsonl \
  --output-crops data/crops/bright_instances
```

`prepare_dataset.py` also exposes `bright-raster`, `xbd`, and `disasterm3` subcommands. xBD minor and major damage are merged into `damaged` so all datasets share the three-class label space.

## Create splits

The following command creates five four-shot-per-class splits for all target events:

```bash
.venv/bin/python scripts/make_splits.py \
  --manifest data/manifests/bright_instances.jsonl \
  --output-dir data/splits/bright_4shot \
  --shots-per-class 4 \
  --seeds 0 1 2 3 4
```

Each split directory contains:

```text
source_train.jsonl
target_support.jsonl
target_query.jsonl
split.json
```

Create a source-only event/label-balanced gate from the source manifest. The
gate samples are automatically excluded from training when passed to the run:

```bash
.venv/bin/python scripts/make_source_gate.py \
  --manifest data/splits/bright_4shot/hawaii-wildfire/seed_0/source_train.jsonl \
  --output data/splits/diagnostics/hawaii-source-event-label-150.jsonl \
  --per-event-label 5 \
  --seed 20260804
```

## Run one complete event experiment

```bash
SOURCE_GRADIENT_ACCUMULATION=6 \
SOURCE_GATE_MANIFEST=data/splits/diagnostics/hawaii-source-event-label-150.jsonl \
CROP_SIZE=448 \
EVAL_D4_VIEWS=8 \
bash scripts/run_event.sh \
  data/splits/bright_4shot/TARGET_EVENT/seed_0 \
  runs/TARGET_EVENT/seed_0 \
  1000
```

The pipeline automatically performs source training, baseline evaluation, support-only selection, final event adaptation, adapted evaluation, and comparison.
When a source gate is configured, it first requires macro-F1 at least `0.2`
and at least two predicted classes. A failed gate stops before any target-query
evaluation, saving GPU time without inspecting target labels.

For one-shot-per-class support, cross-validation is not identifiable. Use a step count pre-registered on source/meta-events:

```bash
bash scripts/run_event.sh SPLIT_DIR RUN_DIR 1000 8
```

### NeurIPS protocol (main + baselines + ablations)

The complete experiment set -- data cuts, baselines, the main
EventTune/KV-TTT run, and all ablations -- is specified in
[docs/neurips-protocol.md](docs/neurips-protocol.md) and launched by:

```bash
PYTHON_BIN=python3 \
SOURCE_STEPS=1000 EVAL_D4_VIEWS=1 KV_RANK=5 KV_STEPS=4 KV_ALPHA_MAX=0.5 \
KV_LAYERS="14 27" RUN_ORIGINAL_EVAL=1 \
bash scripts/run_neurips_batch.sh
```

The batch is resumable per fold and per stage, and writes per-fold progress to
`runs/neurips/<event>/pipeline.log`.

## Run two events on two GPUs

Each event uses one GPU; this is not one model sharded across two GPUs.

```bash
.venv/bin/python scripts/launch_long_job.py \
  --gpu 0 \
  --split-dir data/splits/bright_4shot/hawaii-wildfire/seed_0 \
  --run-dir runs/hawaii-wildfire_seed0

.venv/bin/python scripts/launch_long_job.py \
  --gpu 1 \
  --split-dir data/splits/bright_4shot/libya-flood/seed_0 \
  --run-dir runs/libya-flood_seed0
```

## Outputs and interpretation

```text
RUN_DIR/
  source_adapter/train_summary.json
  source_eval/metrics.json
  source_eval/predictions.jsonl
  event_adapter/support_cv.json
  event_adapter/selection.json
  event_eval/metrics.json
  event_eval/predictions.jsonl
  adaptation_gain.json
```

A basic initial success signal requires positive values in:

```text
gain.macro_f1
gain.balanced_accuracy
error_reduction.nll
```

Loss alone does not validate the method. A low training loss only shows that an adapter fitted its training examples. The main claim must be based on adaptation gains on the untouched query set and should be confirmed across target events and seeds.

## CPU/unit smoke test

The model training path requires CUDA, but data/split/metric components can be tested without downloading BRIGHT:

```bash
.venv/bin/python scripts/make_smoke_data.py --output-dir data/smoke
.venv/bin/python scripts/make_splits.py \
  --manifest data/smoke/manifest.jsonl \
  --output-dir data/smoke/splits \
  --target-events flood-alpha \
  --shots-per-class 1 \
  --seeds 0
.venv/bin/python -m pytest
```

## Scope and limitations

- The code classifies/reclassifies known building instances; it does not generate instance masks.
- For the BRIGHT instance task, use official detector/segmenter proposals and `reclassify_coco.py` to replace their severity classes.
- Support examples are labeled, so results should not be described as fully unsupervised TTT.
- Support-CV is noisy with only 12 examples and must be repeated across seeds.
- D4 candidate scoring is deliberately conservative but computationally expensive.
- Cross-sensor and cross-hazard conclusions require evaluation on more than one target event.

## Source ZIP policy

The shareable source archive contains the README, configs, scripts, package source, tests, and metadata. It excludes all datasets (including generated smoke data), `.venv`, downloaded raw imagery, generated manifests/splits, model caches, adapters, and run outputs. Those artifacts are large or reproducible from the commands above and may have separate distribution terms.
