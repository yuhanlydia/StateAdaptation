# ICLR 2027 evidence freeze

This document freezes the publication-facing evidence and prevents completed
experiments from being rerun or incompatible protocols from being merged.

## Core method freeze

The main method is the correctness-gradient covariance visual KV residual:
frozen backbone, task-relevant visual tokens only, rank 16, decoder layers 14
and 27, bounded Full controller, and support-only coefficient fitting.  Q-only,
O-only, QKVO, query-basis, query-prior, and class-conditional controllers are
mechanism/oracle diagnostics and are not method variants.

## Publication-facing quantitative assets

| Evidence block | Backbone(s) | Protocol | Seed coverage | Methods | Source of truth |
|---|---|---|---|---|---|
| BRIGHT primary | Qwen2.5-VL-7B | 4 events, support24, balanced query300, tile disjoint | one locked support per event | Frozen, fixed-8 LoRA, Full KV, Diagonal KV | `fixed8_and_scaling_diagnostics.md`, `rank16_full_diagonal_alpha_sweep.md` |
| BRIGHT support robustness | Qwen2.5-VL-7B | same four fixed query sets | 3 independent support24 seeds/event | Full and Diagonal KV vs fixed Frozen | `multiseed_significance.md/json` |
| BRIGHT architecture transfer | Qwen3-VL-8B, InternVL3-8B | same 4 events, support24/query300 | one locked support per event | Frozen, LoRA, Random-KV, Ours | `bright_uniform_two_model_results.md/json` |
| BRIGHT additional backbones | Phi, Gemma 3 4B, LLaVA/Llama | same locked 4-event protocol | one support per event | Frozen, LoRA, Random-KV, Ours | `bright_uniform_cross_model.md` |
| Camelyon17 | Qwen3-VL-8B, InternVL3-8B | hospital 2, support16, query300 | 3 matched support seeds | Frozen, LoRA, Random-KV, Ours | `task_vlm_formal_results.md` for Qwen; `internvl3_corrected_task_results.md` for InternVL |
| Guardian fixed semantics | InternVL3-8B | support16, all held-out queries | 3 matched support seeds/split | Frozen, LoRA, Ours | `guardian_failure_results.md` |
| ManipBench boundary | Qwen3-VL-8B, InternVL3-8B | 3 domains, query400 | 3 matched support seeds | Frozen, LoRA, Random-KV, Ours | Qwen rows in `task_vlm_formal_results.md`; InternVL rows in `internvl3_corrected_task_results.md` |
| Geometry x Actuator | InternVL3-8B | query-label oracle diagnostics | 3 seeds for Camelyon/RoboFail/droid-pick; archived seed0 BRIGHT folds | rho, kappa, class kappa, site and query-basis oracles | `task_subspace_diagnostics.md` |

## Results that must not be merged

- Historical InternVL ManipBench rows in `task_vlm_formal_results.md` used an
  invalid raw-label answer format.  Only `internvl3_corrected_task_results.md`
  is publication facing.
- BRIGHT support-seed robustness has three KV seeds but a fixed LoRA support.
  It supports KV-vs-Frozen robustness, not a matched three-seed KV-vs-LoRA
  significance claim.
- The BRIGHT directional diagnostic uses archived support12 folds, while the
  primary table uses support24.  It is qualitative mechanism evidence, not a
  matched performance correlation point.
- Alpha maxima, query-basis results, Q/O/QKVO sites, query-prior reweighting,
  and any query-label statistic are oracle diagnostics.
- The failed RoboFail class-conditional controller (Macro-F1 0.1625) is a
  negative diagnostic and is not Ours+.
- ManipBench steps 8/16/32/64 runs are optimization diagnostics; the boundary
  conclusion uses the corrected formal table and query-basis/site oracle.

## Final experiment decision

No additional large GPU matrix is required.  In particular, do not rerun
BRIGHT, repeat completed three-seed task matrices, add a QKVO main method, or
resume ManipBench hyperparameter sweeps.  A fully matched three-seed BRIGHT
LoRA matrix would be desirable in isolation, but it is not necessary for the
paper's claim and would duplicate a mature four-event evidence block.  The
paper will state the fixed-support limitation explicitly.

The remaining work is analysis and writing: publication tables, existing
confidence intervals, the Ky Fan energy-retention proposition, the
Geometry--Actuator mechanism argument, calibration results, limitations, and
reproducibility documentation.
