# Guardian/FailCoT execution-verification results

This experiment uses the official execution-verification metadata from the Guardian/FailCoT OOD bundle. It is a fixed-semantics binary task: `success` versus `failure`. Each sample is converted to a composite image with the before observation on top and the after observation on the bottom; for UR5-Fail, the front-camera before/after pair is used. This is the single-view-pair pilot, not the full multi-view Guardian model.

Protocol: 8 support examples per class (16 total) and all remaining examples as query, with three stratified support seeds. InternVL3 uses the corrected `Answer: <label>` format. LoRA is rank 16, alpha 32, dropout 0.05, targeting Q/K/V/O. Ours is covariance KV residual, rank 16, full coefficients, alpha 3, layers 14 and 27, four support updates at learning rate 0.05. Query labels are never used for fitting or selection.

| OOD split | Samples | Frozen Macro-F1 | LoRA Macro-F1 | Ours Macro-F1 |
|---|---:|---:|---:|---:|
| UR5-Fail | 140 | 0.4513±0.0067 | 0.4432±0.0510 | 0.4375±0.0688 |
| RoboFail | 153 | 0.4476±0.0263 | **0.5010±0.0645** | 0.4231±0.0505 |
| RoboVQA execution | 357 | 0.4043±0.0068 | **0.5159±0.0087** | 0.4917±0.0275 |

Ours has substantially better NLL than LoRA on all three splits (UR5-Fail 0.7734 vs 3.0313, RoboFail 0.8340 vs 7.5011, RoboVQA 0.8158 vs 4.3961), but does not yet win Macro-F1. This is informative: fixed success/failure semantics reduce the language-selection burden, yet the current two-layer visual KV update remains weaker than LoRA on OOD classification.

## Support-only hyperparameter selection

To test whether this was only a tuning issue, each of the 9 split/seed folds was divided into 12 support-train and 4 support-validation examples (2 per class). The following four configurations were compared by validation Macro-F1: rank16/alpha3, rank16/alpha1, rank32/alpha1, and rank32/alpha1 with L2=1e-2. The global winner was the original rank16/alpha3 configuration (mean validation Macro-F1 0.4519; rank32/alpha1 scored 0.4185). Refitting that selected configuration on all 16 support examples produced the query table above; no query-informed gain was found.

The raw dataset is intentionally not tracked. Recreate it with `hf download paulpacaud/Guardian-FailCoT-OOD-datasets --repo-type dataset --local-dir data/guardian_ood/_ood_tmp` followed by the extraction layout described in `scripts/prepare_guardian_failure.py`.

## Fixed alpha=3, 8-step / low-learning-rate follow-up

At the user's request, we also ran the fixed configuration rank=16, alpha=3,
layers 14 and 27, 8 coefficient updates, learning rate 0.01, and L2=1e-3.
This configuration was evaluated on the same nine query folds; it was not
selected using query labels.

| OOD split | LoRA Macro-F1 | Ours alpha3/lr0.01/steps8 | Ours alpha3/lr0.05/steps4 |
|---|---:|---:|---:|
| UR5-Fail | 0.4432±0.0510 | **0.4905±0.0308** | 0.4375±0.0688 |
| RoboFail | **0.5010±0.0645** | 0.3952±0.0310 | 0.4231±0.0505 |
| RoboVQA execution | 0.5159±0.0087 | **0.5204±0.0108** | 0.4917±0.0275 |
| Overall nine folds | 0.4867±0.0531 | 0.4687±0.0610 | 0.4508±0.0547 |

Thus 8 steps at lr=0.01 improves Ours over the earlier four-step setting by
0.0179 overall and wins two of the three splits, but it does not exceed LoRA
on the pooled nine-fold Macro-F1 because RoboFail remains difficult. The
alpha-only support CV gave mean validation Macro-F1 0.3852 (alpha=2) and
0.4111 (alpha=4), versus 0.4519 for alpha=3.

## RoboFail-focused alpha sweep

Because RoboFail is the only split where the eight-step configuration remains
clearly below LoRA, we ran a split-specific, support-only alpha sweep while
holding rank=16, layers=(14, 27), learning rate=0.01, steps=8, and L2=1e-3
fixed. Each of the three support seeds used the same 12/4 internal
train/validation split as the earlier selection protocol.

| Alpha | Mean validation Macro-F1 | Per-seed validation Macro-F1 |
|---:|---:|---:|
| 0.5 | 0.3000 | 0.2000 / 0.5000 / 0.2000 |
| 1 | 0.3000 | 0.2000 / 0.5000 / 0.2000 |
| 2 | 0.5000 | 0.5000 / 0.5000 / 0.5000 |
| 3 | **0.6556** | 0.7333 / 0.5000 / 0.7333 |
| 4 | 0.4778 | 0.2000 / 0.5000 / 0.7333 |
| 5 | **0.6556** | 0.7333 / 0.5000 / 0.7333 |

Alpha=3 and alpha=5 tie on support validation; deterministic tie-breaking
keeps alpha=3, so changing alpha alone does not explain the RoboFail gap.
The clean query result for the selected setting remains 0.3952±0.0310, below
LoRA's 0.5010±0.0645. The next principled experiments should therefore
target optimizer/representation mismatch (for example layer gates or a
visual Q/O residual), not query-driven alpha selection.

## RoboFail learning-rate follow-up

With alpha=3, rank=16, layers=(14, 27), and steps=8 fixed, increasing the
learning rate from 0.01 to 0.05 reduced RoboFail query Macro-F1:

| Learning rate | Query Macro-F1 |
|---:|---:|
| 0.01 | 0.3952±0.0310 |
| 0.05 | 0.3793±0.0713 |

The lower learning rate is therefore retained for RoboFail; the larger rate
does not recover the LoRA gap.
