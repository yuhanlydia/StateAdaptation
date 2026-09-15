# Oral cross-task experiment progress

This file is append-only. BRIGHT is frozen and is not rerun by the oral
generalization experiment.

## 2026-08-22 — protocol and backend

- Models are pinned locally: Qwen2.5-VL-7B-Instruct and Phi-3.5-Vision-Instruct.
- Camelyon17 and ManipBench manifests use query seed `1729`; support seed only
  changes the support set. Query IDs are identical across support seeds.
- CPU test suite: 52 passed.
- Qwen single-image candidate-likelihood backend passes Camelyon binary and
  ManipBench four-option GPU smoke tests on RTX 3090.
- Phi-3.5-Vision numbered-image-tag candidate scorer passes a two-example
  Camelyon GPU smoke. The initial smoke used eager attention; the completed Phi
  adaptation diagnostics use the remote model's bundled SDPA path consistently.
- Qwen seed-0 Camelyon17 Hospital 2 query300, fixed support16:

| Arm | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|
| Frozen | 0.5474 | 0.5967 | 0.6668 |
| LoRA-TTA, 4 support passes | 0.4991 | 0.5833 | 2.6356 |
| Random-KV, rank16 Full alpha3 | 0.5405 | 0.5800 | 0.6691 |
| Gradient-Cov KV, rank16 Full alpha3 | **0.6909** | **0.6967** | **0.5715** |

These are seed-0 gate results, not the final multi-seed estimate. The LoRA
negative result and Random-KV control are retained. Exact artifacts are under
`runs/oral/qwen/camelyon17/hospital_2/seed_0/`.

## 2026-08-22 — ManipBench Q1 Qwen seed-0

All three domains use the same 400-example fixed query and 32-example support
protocol.  Frozen is the resized 448px scorer; the older native-resolution
bridge run is excluded.  LoRA uses four support passes with checkpointing;
KV uses rank 16, full coefficients, alpha 3, and four coefficient steps.

| Domain / arm | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|
| bridge / Frozen | 0.2312 | 0.3200 | 1.6511 |
| bridge / LoRA-TTA | **0.7205** | **0.7225** | 4.3489 |
| bridge / Random-KV | 0.2271 | 0.3175 | 1.5663 |
| bridge / Gradient-Cov KV | 0.3561 | 0.3900 | 1.2586 |
| bridge / Hidden Residual | 0.3795 | 0.4125 | 1.2625 |
| droid-pick-place / Frozen | 0.1580 | 0.2725 | 1.8821 |
| droid-pick-place / LoRA-TTA | 0.7124 | 0.7125 | 2.6714 |
| droid-pick-place / Gradient-Cov KV | 0.2944 | 0.3275 | 1.4236 |
| droid-arti / Frozen | 0.1471 | 0.2650 | 1.9514 |
| droid-arti / LoRA-TTA | 0.7854 | 0.7875 | 1.6893 |
| droid-arti / Gradient-Cov KV | 0.2731 | 0.3075 | 1.4384 |

The droid-arti frozen gate is below the pre-registered chance+0.02 threshold
(0.27); it is retained and explicitly reported rather than silently dropped.
The bridge hidden-residual arm and all three Gradient-Cov KV arms are now
executed. Random-KV is recorded for all Qwen ManipBench domains and seeds;
the Phi matrix is gate-limited as documented above.

The fixed-seed Frozen rerun for ManipBench support seed 1 is complete:
bridge/droid-pick-place/droid-arti balanced accuracies are 0.3200/0.2725/0.2675
(macro-F1 0.2333/0.1553/0.1483). Query IDs and image paths are identical to
seed 0, and a 20-example deterministic check produced identical probabilities
across seeds. The deterministic learned-arm reruns for seed 1 are also
complete for all three domains and included in the paired summaries below.

Camelyon17 Hospital 2 is now complete for support seeds 0/1/2. The paired
summaries are in `reports/camelyon_*_multiseed.json`:

| Arm | Mean macro-F1 | Mean balanced accuracy | Mean NLL |
|---|---:|---:|---:|
| Frozen | 0.5538 | 0.6011 | 0.6658 |
| LoRA-TTA | 0.4054 | 0.5356 | 5.7935 |
| Random-KV | 0.5360 | 0.5856 | 0.6736 |
| Gradient-Cov KV | **0.6781** | **0.6822** | **0.5895** |
| Hidden Residual | 0.4696 | 0.5500 | 0.6902 |

Phi-3.5-Vision gate results (seed 0, frozen candidate scorer) are recorded
separately. Camelyon balanced accuracy is 0.5033 (below the registered 0.52
binary gate) and ManipBench bridge is 0.2850 (barely above the multiclass
chance+0.02 gate). The Phi Camelyon collapse is retained; no query-informed
fallback or tuning is applied. The initial adapter correctly failed closed on
Phi's fused `qkv_proj` layout rather than pretending it had independent K/V
modules; the debugged implementation now adapts the fused K/V slices under the
consistent SDPA path. The Qwen backbone remains the primary Gradient-Cov KV
result.

The first Phi fused-qkv backward attempt exceeded the 24-GiB RTX 3090 budget
under eager attention. Debugging switched to the remote model's bundled SDPA
attention implementation (weights and prompts unchanged); a one-sample
backward and the full 32-example support pass then completed. The original
OOM is retained in the audit trail, and no eager/SDPA runs are mixed in one
comparison.

The remaining Phi Frozen gates are also recorded: droid-pick-place balanced
accuracy 0.2550 and droid-arti 0.2375, both below the multiclass 0.27 gate.
Thus Phi expansion is stopped for all domains except the diagnostic bridge
Frozen row; Qwen is the registered primary backbone for the full matrix.

Under the consistent SDPA attention path, the Phi bridge seed-0 diagnostic
matrix is:

| Arm | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|
| Frozen | 0.1833 | 0.2850 | 1.8548 |
| LoRA-TTA | **0.5277** | **0.5275** | 4.0960 |
| Random-KV | 0.1895 | 0.2900 | 1.8287 |
| Gradient-Cov KV | 0.2593 | 0.3150 | 1.5750 |

Phi droid gates fail (0.2550 and 0.2375), and Camelyon fails the binary gate;
therefore the preregistered stop rule prevents a Phi multi-seed expansion.

ManipBench seed-2 adaptation results are now complete:

| Domain / arm | Macro-F1 | Balanced accuracy | NLL |
|---|---:|---:|---:|
| bridge / LoRA-TTA | 0.7642 | 0.7650 | 3.2599 |
| bridge / Gradient-Cov KV | 0.4551 | 0.4525 | 1.2028 |
| bridge / Random-KV | 0.2203 | 0.3125 | 1.5638 |
| bridge / Hidden Residual | 0.3058 | 0.3650 | 1.3895 |
| droid-pick-place / LoRA-TTA | 0.7340 | 0.7350 | 2.4758 |
| droid-pick-place / Gradient-Cov KV | 0.2466 | 0.3375 | 1.3718 |
| droid-pick-place / Random-KV | 0.1830 | 0.2875 | 1.7203 |
| droid-pick-place / Hidden Residual | 0.2194 | 0.3100 | 1.5973 |
| droid-arti / LoRA-TTA | 0.7139 | 0.7150 | 2.8798 |
| droid-arti / Gradient-Cov KV | 0.2637 | 0.3150 | 1.4714 |
| droid-arti / Random-KV | 0.1250 | 0.2550 | 1.9344 |
| droid-arti / Hidden Residual | 0.3078 | 0.3650 | 1.5880 |

An earlier audit checkpoint covered 51 Qwen prediction directories. The final
audit covers 67 Qwen prediction directories and all pass count, fixed-query
ID/order, candidate-label, probability-sum, manifest hash, model fingerprint,
and environment metadata checks. The six-hour batch budget and separate
30-minute debug allowance are recorded in the YAML protocol and execution
contract.

The completed ManipBench three-seed paired summaries are now in
`reports/manipbench_*_multiseed.json` (all query IDs identical):

| Domain / arm | Mean macro-F1 | Mean balanced accuracy | Mean NLL |
|---|---:|---:|---:|
| bridge / Frozen | 0.2333 | 0.3200 | 1.6544 |
| bridge / LoRA-TTA | **0.7740** | **0.7750** | 2.8392 |
| bridge / Gradient-Cov KV | 0.4260 | 0.4392 | 1.2139 |
| bridge / Random-KV | 0.2258 | 0.3167 | 1.5675 |
| bridge / Hidden Residual | 0.3449 | 0.3942 | 1.3322 |
| droid-pick-place / Frozen | 0.1553 | 0.2725 | 1.8899 |
| droid-pick-place / LoRA-TTA | **0.7137** | **0.7150** | 2.3416 |
| droid-pick-place / Gradient-Cov KV | 0.2876 | 0.3450 | 1.3944 |
| droid-pick-place / Random-KV | 0.1966 | 0.2992 | 1.7225 |
| droid-pick-place / Hidden Residual | 0.2497 | 0.3258 | 1.5725 |
| droid-arti / Frozen | 0.1483 | 0.2675 | 1.9609 |
| droid-arti / LoRA-TTA | **0.7536** | **0.7550** | 2.4298 |
| droid-arti / Gradient-Cov KV | 0.2497 | 0.2992 | 1.4584 |
| droid-arti / Random-KV | 0.1567 | 0.2683 | 1.8356 |
| droid-arti / Hidden Residual | 0.2793 | 0.3317 | 1.5757 |

The audit count is now 67 Qwen prediction directories (all passed); the
earlier count of 51 was before the deterministic seed-1 and missing seed-0
control arms were filled.
