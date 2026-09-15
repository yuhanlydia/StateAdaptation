# StateAdaptation — A Learned Visual Lens

**A Learned Visual Lens: Low-Rank State Adaptation for Vision–Language Models**

This is the active repository migrated from `Yunbo-max/EventTune`. The complete upstream Git history through `3150cae1eecf2ee82e508d32fe7f2e80913f8149` is retained, and `legacy/eventtune-3150cae` preserves the exact original snapshot. The old repository is unchanged. Historical code, reports and results remain available; `src/eventttt` and `src/vigor_handoff` retain their import names for compatibility.

## Start here: the three missing experiments

The new P0 entry point is **`scripts/run_visual_lens_all.sh`**, not a request to rerun every historical sweep. Full instructions: [docs/VISUAL_LENS_RUN.md](docs/VISUAL_LENS_RUN.md). The original pending-experiment specification is preserved at [docs/VISUAL_LENS_EXPERIMENTS_TO_COMPLETE.md](docs/VISUAL_LENS_EXPERIMENTS_TO_COMPLETE.md).

| Stage | Implemented experiment | Default jobs |
|---|---|---:|
| `audit` | Real-model support-only identity, mask, contextual answer span, mean/sum and actual checkpoint replay/gradient checks | 3 |
| `matched` | Four BRIGHT events × three support seeds × Frozen / LoRA-r16 one-pass / LoRA-r16 four-pass / Random-KV-r16 / Lens-r16; plus Camelyon seed0 states needed for image controls | 63 |
| `objective` | Camelyon17 and RoboFail seed0: sum/mean × full-answer/label-only loss | 8 |
| `evidence` | Same fitted Frozen / LoRA / Lens state on real, shuffled, neutral and BRIGHT post-only image controls; support-fitted candidate-bias baseline | 9 |

**83 top-level jobs**, with multiple conditions per image-control job. The code is supplied for the GPU agent to execute. These are **not completed GPU results**. CPU tests cover 52 legacy tests, 31 earlier handoff tests and 9 new P0 tests. Real-model GPU gates must pass before downstream jobs run.

```bash
git clone https://github.com/yuhanlydia/StateAdaptation.git
cd StateAdaptation
# Install a PyTorch build compatible with your GPU first.
python3 -m pip install -e '.[train,test]'

# Reuse your existing local model snapshots, data and prepared task manifests.
# Correct paths in configs/visual_lens_p0.json before checking.
bash scripts/run_visual_lens_all.sh prepare
bash scripts/run_visual_lens_all.sh plan
bash scripts/run_visual_lens_all.sh check
bash scripts/run_visual_lens_all.sh run 0,1,2,3
bash scripts/run_visual_lens_all.sh summarize
```

For one GPU, use `run 0`. Existing machines can change their origin URL and pull; see the migration instructions. Raw datasets, ignored `runs/`, large model weights and private Hugging Face assets are **not stored in Git** and were not moved to a different Hub account.

## Results and reproducibility

New outputs are isolated under `runs/visual_lens_p0_v1/`. Resume checks hash the configuration, source, model, manifests, images and saved state; changed or incomplete outputs are never accepted solely because a `metrics.json` exists. Primary fitting never uses query labels. Reusing an already inspected query set is labeled reproducibility checking, not a new untouched test evaluation. Query-label diagnostics remain separate from primary performance.

`reports/iclr2027_evidence_audit.md` describes the **historical** evidence freeze. This migration adds the specifically requested P0 verification suite without altering those recorded results or implying that all earlier conclusions have been independently confirmed. See [docs/VISUAL_LENS_STATUS.md](docs/VISUAL_LENS_STATUS.md) for implemented versus still-pending work.

## Existing full ablation menu

The previously delivered code overlay is now part of this repository. It contains basis controls, rank/alpha/layer/mask/steps variants, hidden-state and K/V-site controls, support-budget runs and separately marked oracle diagnostics:

```bash
bash scripts/run_vigor_all.sh plan
```

Read [docs/VIGOR_RUN_ALL.md](docs/VIGOR_RUN_ALL.md) before invoking its larger menu. Do not run the entire menu merely to duplicate completed historical experiments. No inference that an unexecuted ablation improved performance is made.

## Manuscript

The latest single-version paper sources and figure prompts are in `paper/visual_lens/`; `main.tex` is its only manuscript entry point. The old `paper/main.tex` remains a historical source. Numerical figures are recreated from the included measured-data JSON using `plotting/make_figures.py`; no missing experiment values are filled in.

For historical setup details, dataset restoration commands and reports, see [the original EventTune README](docs/legacy/README_EventTune.md).
