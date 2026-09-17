# Visual Lens P0: reproducible GPU execution

## What was migrated

The new repository imports the complete tracked tree and Git history of EventTune at `3150cae1eecf2ee82e508d32fe7f2e80913f8149`. Source code, scripts, configs, reports and original binary paper assets were verified byte-for-byte during the import. `legacy/eventtune-3150cae` is the unmodified source branch. The subsequent handoff adds new packages and manuscript sources; it does not modify the historical model implementation under `src/eventttt/`.

### An existing GPU checkout

First save or commit local code changes; do not delete ignored datasets or runs. Because the new main contains the old history, a clean checkout at the migrated commit can fast-forward:

```bash
git remote set-url origin https://github.com/yuhanlydia/StateAdaptation.git
git fetch origin
git switch main
git pull --ff-only origin main
```

If local commits diverged, stop and reconcile normally; never force-reset a machine that may contain unpushed work. A fresh clone is equally valid. Copy/symlink your existing `data/` and `artifacts/models/` assets without adding them to Git.

## Required local assets

Install a compatible CUDA PyTorch wheel, then `python3 -m pip install -e '.[train,test]'`. Use the existing tested environment when possible. The original pyproject defines the model integration dependencies; new P0 code adds no paid API or external training service.

Edit `configs/visual_lens_p0.json` with absolute or repository-relative model snapshot paths. Defaults expect `artifacts/models/Qwen2.5-VL-7B-Instruct`, `artifacts/models/InternVL3-8B-Instruct` and optionally Qwen3-VL-8B-Instruct. Models must be complete local snapshots on one GPU. No implicit CPU offload, 4-bit fallback or remote checkpoint substitution is accepted. An 8B BF16 model plus backward activations should be scheduled on the existing 24GB-or-larger GPU; this handoff does not promise a 16GB fit. A one-sample audit is the real hardware gate.

Data defaults:
- `data/bright.jsonl`: complete normalized BRIGHT building pool with real file paths.
- `data/prepared/neurips/<event>/target_support.jsonl` and `target_query.jsonl`: existing locked four-event cuts; do not reconstruct them with a different random split.
- `data/prepared/camelyon17/seed_0/{support,query}.jsonl`.
- `data/prepared/guardian_execution/robofail/seed_0/{support,query}.jsonl`.

The existing Hub restoration and dataset-preparation scripts remain in the repository. Hub identifiers still point at the original dataset owner; moving Git code does not migrate or publish private Hub assets.

## Step 1: produce real BRIGHT support seeds

```bash
bash scripts/run_visual_lens_all.sh prepare
```

`prepare_visual_lens_seeds.py` preserves original seed0 support24 and the locked query300. Seeds1/2 are deterministic class-balanced samples from the complete pool, excluding every locked query tile and sample ID. Image/tile separation is checked. Existing manifests with different content are not overwritten. Output: `data/prepared/visual_lens/bright/<event>/seed_<seed>/`. Use explicit CLI paths if your source pool is elsewhere. Each manifest records that the queries were previously inspected: this is an independent support replication, not a claim of a new blind test.

## Step 2: plan and fail-closed preflight

```bash
bash scripts/run_visual_lens_all.sh plan
bash scripts/run_visual_lens_all.sh check
```

Planning does not load any model. `check` verifies paths, class/support counts, group separation, local snapshots and query IDs. Missing assets are printed as BLOCKED and no GPU run begins. The canonical plan is always the complete 83-job matrix, even after a filtered invocation.

## Step 3: actual GPU gradient audits

The first inexpensive integration pass can be limited to one task:

```bash
python3 scripts/run_visual_lens.py run --stages audit \
  --families internvl3 --domains hospital_2 --gpus 0
```

The full default audit stage checks Qwen2.5/BRIGHT plus InternVL3/Camelyon and InternVL3/RoboFail. Each uses one support example per class. It checks zero/reset identity, serialization, direct selected/unselected-row intervention, finite gradients, true backward checkpoint replay, controller and activation gradient agreement, mean/sum arithmetic, contextual label suffixes, and full-answer versus label-only gradients/bases. Mask state stays alive through backward. Finite differences are reported at multiple step sizes but are not passed off as exact BF16 derivative certification. Defaults use 0.005 absolute loss/score and 5% gradient-relative tolerances; these are configurable documented tolerances, not guaranteed mathematical equivalence. A no-op checkpoint implementation fails the replay gate.

## Step 4: run the missing matrix

```bash
# One GPU
bash scripts/run_visual_lens_all.sh run 0
# Or parallel independent jobs, at most one model process per GPU
bash scripts/run_visual_lens_all.sh run 0,1,2,3
```

Stages execute with barriers: `audit -> matched -> objective -> evidence`. Any failed job stops downstream stages; logs and failure exit codes remain. Successfully sealed jobs are skipped only when fingerprints and all output hashes still match.

For a filtered run after audits:

```bash
python3 scripts/run_visual_lens.py run --stages matched \
  --families qwen2 --domains hawaii-wildfire --seeds 0 --gpus 0
```

Do not request evidence alone until its corresponding matched states have finished. Filters do not invent missing support manifests. No parameter is selected based on query scores.

### A. Strong baseline and matched replication

Four BRIGHT events × three seeds × five arms: Frozen, rank16 LoRA one-pass, rank16 LoRA four-pass, rank16 Random-KV and rank16 Full Lens. Support24, query IDs, model and token scoring are shared. One/four-pass LoRA use accumulation3, giving 8/32 updates. Lens performs four full-support updates. This is a shared-supervision comparison, not equal compute. Both budgets and scalar counts are reported. Independently development-selected LoRA is not automatically implemented: provide a separately fixed protocol in a new config if needed, and do not use these query scores to choose it.

### B. Image dependence of the same fitted state

Hawaii/Libya and Camelyon seed0: restore the exact fitted Lens/LoRA state. Score full predetermined queries with real, deterministically shuffled, and neutral128 RGB images; BRIGHT also replaces only the post image. A single label-independent cyclic permutation is shared across methods. Donor images are resized to each original input size. Prompt token IDs, candidate spans, grid layout and visual-token counts must match real versus controls exactly. LoRA or Lens is never retrained per condition.

The frozen condition also fits a K-1-dimensional zero-sum candidate-logit bias on support, with fixed L2=0.001. Compare its real/control performance with the state method. This isolates a cheap answer-prior alternative; it does not prove that either mechanism has a particular attention map.

### C. Objective and span comparison

Camelyon/RoboFail seed0 each run sum/mean × full/label-only support objectives. The prompt and candidate scoring stay unchanged. Label-only spans are found as an exact decoded suffix within the known answer span, not by blindly matching standalone tokenization. The existing full-answer behavior is preserved as a separate arm. These new corrected-backward runs cannot retrospectively replace historical numbers.

## Step 5: summary and reporting

```bash
bash scripts/run_visual_lens_all.sh summarize
```

Outputs under `<configured run_root>/summary/` (`runs/visual_lens_p0_v3/summary/` for the completed campaign):
`results.md`, `summary.json`, `coverage.md`, `visual_dependence.md`.

Paired main contrasts and visual difference-in-differences use query-cluster bootstrap within a fixed support state. SD across support seeds is reported separately, not mislabeled a CI. Evidence statistic:
`D=(F1_Lens(real)-F1_Frozen(real))-(F1_Lens(control)-F1_Frozen(control))`.
NLL differences retain their lower-is-better sign convention. No raw query is pooled as independent merely because it was rescored under multiple support seeds. Missing and failed jobs are listed, never filled with zero. The 83-job count counts top-level fitted-state/diagnostic units, not individual condition scorings or GPU hours.

## Additional existing ablations

The separately supplied `configs/vigor_iclr2027.json` and `scripts/run_vigor_all.sh` now live in GitHub. They include the larger basis/rank/layer/mask/support/hidden-state/site menu. Its legacy name is for compatibility. Run only missing, preregistered contrasts, in a distinct root. Query-label oracles remain labeled diagnostics. `docs/VIGOR_RUN_ALL.md` describes those commands; it is not a command to rerun the whole historical study.
