# KV-TTT oral-level generalization execution contract

This document is the handoff contract for GPT-Luna. Do not change the primary
claim, silently tune on query data, or rerun completed BRIGHT sweeps. Work in
small audited commits and preserve every negative result.

## Scientific claim

Test-time adaptation of a VLM can operate on task-relevant internal visual
evidence states instead of model weights. For support item `i`, layer `l`, and
projection `t in {K,V}`:

```text
G_i = grad_{Z_M} L_i
C   = (1/N) sum_i G_i^T G_i
B   = TopEig_r(C)
Z'  = Z + M * (Z B) A B^T
```

`C` is an uncentered correctness-gradient second moment. Do not call it a
centered covariance. The application changes only `M`:

| Family | Registered visual mask |
|---|---|
| BRIGHT | second/post-event visual group |
| Camelyon17-WILDS | all tokens of the single medical image |
| ManipBench Q1 | all tokens of the current observation image |

The fixed primary configuration is in `configs/oral_generalization.yaml`:
rank 16, Full controller, alpha 3, middle+last decoder layers, four full-support
updates, coefficient LR 0.05. Run it unchanged on new applications before any
support-only tuning.

The formal execution budget is six wall-clock hours per scheduled experiment
batch, with a separate 30-minute debug allowance. Jobs stop at the budget
boundary while preserving resumable artifacts and an explicit partial status;
they must not silently extend the run or discard completed seeds.

## Immutable experimental rules

1. Support and query IDs are written before any model evaluation.
   Within a domain, query IDs are fixed with query seed 1729; support seeds
   0/1/2 change support only. This is required for paired query-level tests.
2. Query labels may be read only by the final metric process. Adaptation,
   checkpoint choice, hyperparameter choice, and admission gates cannot access
   them.
3. Report every seed and every denominator. Never replace a failed seed.
4. Do not report per-domain query-oracle alpha maxima as primary results.
5. All arms for one row use identical support/query manifests and model hashes.
6. A nominal repeat that changes materially is investigated, not averaged away.
7. Keep BRIGHT's existing full suite. Add only the registered reviewer controls.

## Asset preparation

From the repository root:

```bash
bash scripts/download_oral_assets.sh
```

Expected local assets:

```text
artifacts/models/Phi-3.5-vision-instruct/
data/raw/manipbench-official/
data/raw/manipbench-simplified/
data/prepared/camelyon17/seed_{0,1,2}/{support,query}.jsonl
data/prepared/manipbench_q1/DOMAIN/seed_{0,1,2}/{support,query}.jsonl
```

If Google Drive refuses the ManipBench zip, use the upstream repository's
official “Simplified Dataset” browser link and place it at
`data/raw/manipbench-simplified.zip`. Do not substitute a reconstructed or
unversioned dataset. Then run:

```bash
python -m zipfile -e data/raw/manipbench-simplified.zip data/raw/manipbench-simplified
for seed in 0 1 2; do
  python scripts/prepare_manipbench_generalization.py --seed "$seed"
done
```

Camelyon source priority is: official WILDS archive, then the pinned
`wltjr1007/Camelyon17-WILDS` parquet mirror used by the preparation script.
Record dataset revision, image IDs, hospital, patient, and slide. The target is
WILDS center 2; patient/slide groups must not overlap support and query.

## Implementation sequence

### Phase 0 — generic backend

Refactor the current Qwen-specific batch construction behind a `VLMBackend`
interface with these operations:

```text
load(model_id)
build_labeled_batch(sample, candidate)
visual_token_mask(input_ids, mode)
discover_decoder_kv()
score_candidates(sample, candidates)
```

Keep the existing Qwen path bitwise-compatible. Add a Phi backend using
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and
`AutoProcessor`. Discover K/V modules by introspection; never assume Qwen's
module nesting or image token ID. Fail if each selected layer does not expose
exactly one K and one V projection.

Acceptance tests:

- zero coefficients reproduce frozen logits;
- reset and save/load reproduce logits;
- only registered visual rows change at the hooked projection;
- base VLM trainable parameters remain zero;
- candidate scoring returns probabilities summing to one;
- Qwen BRIGHT smoke output remains unchanged.

### Phase 1 — seed-0 admission gates

For each new domain, first run frozen candidate-likelihood scoring only:

```text
Camelyon: normal / tumor
ManipBench Q1: A / B / C / D
```

Stop expansion if the frozen model predicts fewer than two classes or if
Camelyon balanced accuracy is below 0.52. Record the collapse. For Camelyon,
the registered fallback is one source LoRA trained on WILDS source hospitals
0, 3, and 4, frozen before target-center KV-TTT. Do not choose this fallback
after inspecting adapted query results.

### Phase 2 — seed-0 core matrix

For each domain and Qwen-7B, run:

| Arm | What learns at test time |
|---|---|
| Frozen | nothing |
| LoRA-TTA | rank-16 q/k/v/o LoRA, one support epoch |
| Hidden residual | matched-rank residual on decoder hidden state |
| Random KV | same KV controller with seeded orthonormal random `B` |
| Gradient-Cov KV | registered method |

The comparison answers: weight space vs generic activation space vs arbitrary
KV geometry vs correctness-gradient KV geometry. Match layers, rank, alpha
bound, update count, support, and scorer wherever meaningful.

For Phi-3.5-Vision initially run only Frozen, LoRA-TTA, Random KV, and Ours.
Expand to all three seeds only after seed 0 passes the admission gate and all
artifact invariants.

### Phase 3 — BRIGHT reviewer controls

Do not rerun existing sweeps. Add exactly:

- strong support-only LoRA with an equal, preregistered search budget;
- hidden-state residual;
- activation-PCA basis;
- shuffled-label correctness-gradient basis;
- mask ablation: `second_visual`, `first_visual`, `all_visual`, `text`.

The support-only LoRA selector must use nested support folds. KV basis extraction
inside support CV must be repeated inside each training fold; a full-support
basis evaluated on held-out support is leakage.

### Phase 4 — geometry tests

Add state export for each module's `C` and `B`. Verify the proposition:

```text
TopEig_r(C) = argmax_{B^T B=I} tr(B^T C B)
captured_energy = tr(B^T C B) / tr(C)
```

Use `scripts/analyze_gradient_geometry.py` to relate captured energy to
Delta-F1 and NLL reduction across basis controls and domains. Also compute
principal angles across support seeds for correctness, random, and
shuffled-label bases.

Run the split-support control for every Qwen domain:

```text
S_B intersect S_A = empty
S_B -> extract B
S_A -> fit A
```

Maintain class balance in both halves. This is a control, not the primary
configuration.

## Required output tree

```text
runs/oral/<backbone>/<family>/<domain>/seed_<n>/<arm>/
  config.json
  environment.json
  support_manifest.sha256
  query_manifest.sha256
  model.sha256
  predictions.jsonl
  metrics.json
  extraction.json                 # KV arms
  adaptation.json                 # learned arms
  kv_state.pt                     # KV arms
```

Every command must be resumable only when configuration and content hashes
match. A stale `predictions.jsonl` from a different model, mask, prompt,
candidate order, or crop size must abort.

## Reporting

Primary metrics are macro-F1, balanced accuracy, and NLL. Add accuracy for
ManipBench because it is standard for multiple choice. Report mean and sample
standard deviation over three support seeds, paired bootstrap confidence
intervals, and paired permutation tests over identical query IDs. Stratify
bootstrap samples by domain and class.

The primary table uses the single fixed cross-application configuration. Put
support-only tuned results in a separate practical upper-bound table. Keep
query-oracle results in diagnostics only.

## Stop conditions and handoff notes

- No visible CUDA: prepare assets/tests only and leave exact resume commands.
- Dataset license or download denial: record the official URL and exact error;
  do not scrape around access controls.
- Frozen collapse: run the registered gate/fallback, not the full matrix.
- OOM: first lower image resolution or enable checkpointing consistently
  across arms; do not silently switch backbone.
- Any query-informed choice invalidates that result as primary.

At the end of every phase, update one append-only `reports/oral_progress.md`
with completed/failed counts, hashes, exact commands, and the next safe command.
