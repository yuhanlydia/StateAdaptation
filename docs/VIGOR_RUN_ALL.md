# VIGOR: agent execution handoff

## Status and boundaries

This is an **additive validation/reproduction package** for EventTune commit
`3150cae1eecf2ee82e508d32fe7f2e80913f8149`. It does not overwrite historical
scripts, metrics or paper/main.tex. All new outputs go to `runs/vigor_handoff_v1`.
The originating ChatGPT session could read the repository, but its GitHub write
attempt returned HTTP 403. Apply and push this patch using the coding agent's
normal authorized Git credentials; a plain `git pull` cannot retrieve a patch
that has not yet been pushed.

The new logic is CPU-tested. It has NOT been run on the user's GPUs, models or
datasets. Full-model compatibility, BF16 behavior and peak VRAM must pass the
first real job. An estimator/kernel rewrite can change floating-point results;
new outputs must not be relabeled as the previously frozen results.

## Apply, review and push

From the existing EventTune checkout, with a clean working tree:

```bash
git status --short
git pull --ff-only
git switch -c vigor/iclr-handoff-20260914
# Use the downloaded patch path; do NOT replace it with a guessed path.
git apply --check /path/to/vigor_iclr_handoff.patch
git apply /path/to/vigor_iclr_handoff.patch
python3 -m pip install -e '.[train,test]'
PYTHONPATH=src python3 -m pytest -q tests_vigor
python3 -m compileall -q src/vigor_handoff scripts/run_vigor_job.py scripts/run_vigor_suite.py
bash scripts/run_vigor_all.sh plan
git add src/vigor_handoff scripts/run_vigor_*.py scripts/run_vigor_all.sh \
  scripts/summarize_vigor_suite.py configs/vigor_iclr2027.json docs/VIGOR_*.md \
  tests_vigor paper/vigor_iclr
git commit -m "add auditable VIGOR ICLR reproduction suite and writing package"
git push -u origin vigor/iclr-handoff-20260914
```

Do not force-push or reset the collaborator's work. If any patch path already
exists, reconcile it before applying. After merging through your normal workflow,
the GPU agent can pull the shared branch/main and execute the commands below.
The overlay archive is an alternative to the patch, not an additional patch.

## Local assets

Use the already-working training environment where possible. The inspected
pyproject specifies Transformers >=4.57,<5 and PEFT >=0.14,<0.15 for the training
extra; the old document mentioning Transformers 4.49 is not adequate for Qwen3.
Record `python -m pip freeze` alongside final experimental artifacts.

Edit only the local model paths in `configs/vigor_iclr2027.json` if necessary:

- `artifacts/models/Qwen2.5-VL-7B-Instruct`
- `artifacts/models/Qwen3-VL-8B-Instruct`
- `artifacts/models/InternVL3-8B-Instruct`

The package does not include large model weights or datasets. It never tries to
bypass a gated/private dataset. The existing download/prepare scripts remain
available, but do NOT regenerate locked splits when resuming an experiment.
Use the already prepared BRIGHT `data/prepared/neurips/<event>` manifests;
Camelyon17/ManipBench/Guardian paths follow the existing formal runners.
Support12 and support48 must exist for the support-budget phase. Preflight
reports every missing image/model/manifest before GPU jobs start.

```bash
bash scripts/run_vigor_all.sh check
```

Preflight checks sample IDs, image paths, candidate orders, group separation,
class budgets and query counts. If a grouped split fails, resolve the data audit;
do not bypass it by replacing group IDs with per-image IDs. This can reveal a
previously unverified property of a locked split. Such a finding requires an
explicit new split version rather than silent regeneration.

## First real-model integration run

Run a single domain first, including the unchanged frozen baseline. The VIGOR
worker checks zero-controller identity and fails on missing projection sites,
non-finite updates, CPU/meta offload, or unknown image-token boundaries.

```bash
python3 scripts/run_vigor_suite.py --phases main --families internvl3 \
  --domains hospital_2 --arms frozen ours --seeds 0 --execute --gpus 0
```

This is a matched frozen/VIGOR seed-0 block. The first actual VIGOR job automatically checks its identity gate before fitting.
There is no fallback that converts an OOM into a different crop size or method.

## Run the full registered matrix

```bash
# Main + one-factor ablations + support-size ablations, no query-label methods:
bash scripts/run_vigor_all.sh run 0,1,2,3

# Also reproduce the explicitly separated oracle mechanism diagnostics:
bash scripts/run_vigor_all.sh run-with-oracles 0,1,2,3
```

One subprocess runs per GPU. Each job loads a fresh model; there is no controller
carryover between methods or events. Main blocks are matched within each fold.
Do not run two launchers on the same run root simultaneously.

The shipped full plan contains **357 jobs**: 180 core comparisons, 124 one-factor
BRIGHT ablations, eight support-budget comparisons, and 45 oracle diagnostics.
This is a full reproduction menu, not a recommendation to rerun all completed
historical experiments. The first invocation in a new root will not automatically
import legacy metrics whose artifacts cannot be verified. After a new job is
sealed, subsequent identical invocations skip it. Use `--families`/`--domains`
for an intentionally selected, logged subset, not outcome-dependent pruning.

## Exact experiment coverage

Core: Frozen / supervised LoRA-r16 / Random-KV / VIGOR. BRIGHT uses three
backbones and four events with the locked seed-0 support; Camelyon and ManipBench
use Qwen3/InternVL and three support seeds; Guardian uses InternVL and three
support seeds. Newly added Guardian Random-KV is a completion experiment.

One-factor BRIGHT controls, on Qwen2.5 and all four locked events:
centered gradient, mean-gradient rank1 with covariance/random rank1 controls,
activation PCA, shuffled-label basis, independent random-basis seeds,
Full/Diagonal, zero coefficients, hard projection, rank5/8/32, alpha0.5/1/2,
one/two/eight updates, K-only/V-only, middle-only/last-only, pre/all-visual/text
mask, decoder-input hidden-state residual at the same rank, no L2, loss-reduction control, and a four-pass LoRA budget control.

Support-size controls use support12/24/48 with a common query set. Query-label
probes include query-basis-only replacement and Q/O/QKVO intervention sites,
plus the existing signed, class-conditional and query-prior geometry analysis.
These are never promoted to a main result by the summarizer.

Not silently implemented or claimed: a new meta-learned/offline shared basis,
clinical deployment, closed-loop robot control, an official CLIP LoRA-TTT or
ReFT reimplementation, and a universal causal-capacity oracle. Existing negative
and architecture-control results must remain in the paper's evidence ledger.

## Resume and provenance

The job fingerprint includes the plan, model weight/config contents, relevant
source files, support/query manifests, image content and software versions.
Content hashes are cached using path/size/mtime to avoid rereading gigabytes on
every job; remove `_asset_fingerprints` to force rehash after an unusual in-place
edit that preserves timestamps. Different jobs have different output paths.
A changed identity at an existing path raises an error rather than reuses rows.

The fitted controller/LoRA and its hashes are sealed before query evaluation.
Predictions are appended one sample at a time. A truncated last JSONL record may
be repaired; corrupt interior records are rejected. Resuming partial queries
reloads the exact sealed adaptation state. A lock prevents concurrent writes;
a stale lock after a killed process must be inspected and removed manually.

## Analysis and publication safety

```bash
python3 scripts/summarize_vigor_suite.py --bootstrap 2000
```

`summary.json`, coverage.md and separate main/ablation/support/diagnostic tables
are emitted. Missing jobs remain missing; no zero imputation or best-query
selection is performed. Paired confidence intervals operate within a matched
support fold and resample the recorded clusters. They are not three-seed
population guarantees. Hardware timings include the specified protocol and
must not be converted into a generic zero-overhead claim.

For a genuinely matched three-seed BRIGHT LoRA comparison, add the already
locked seed1/2 support paths to a separate configuration and run both methods
on those same supports. The present seed-0 BRIGHT plan does not manufacture
additional folds or retroactively turn historical unmatched seeds into matched
replications.
