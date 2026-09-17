# Visual Lens weak-case debugging

These are exploratory follow-up experiments. They are stored separately from the canonical 83-job P0 run and do not replace its preregistered results.

## Integrity

- 20/20 debug jobs are sealed and their full prediction sets pass coverage and probability validation.
- Fresh Qwen2.5-VL and InternVL3 gradient-path audits passed against the current source tree.
- No debug job failed, and no query result was inserted into the canonical P0 summary.

## Hawaii wildfire

Seed-0 diagnostics suggested that the original residual bound was too large for that support: alpha 1 improved Macro-F1 from 0.3630 to 0.4019, while steps 2, learning rate 0.01, and rank 8 reached 0.3481, 0.3649, and 0.3433. A visually representative support set reached only 0.3155, so simply increasing tile diversity did not fix the result.

The alpha result did not replicate consistently. Alpha 1 reached 0.2362 and 0.3352 on seeds 1 and 2, for a three-seed mean of 0.3244 versus 0.3329 for the canonical alpha-3 setting. The correct conclusion is support-dependent sensitivity to the residual bound, not a new global alpha default.

The support audit found strong scene concentration: the canonical seed-0 support has 24 samples but only three total source tiles. After excluding the fixed query, the damaged class has only one available source tile in the candidate pool. This is a real coverage limitation, though the diverse-support result shows it is not sufficient by itself to explain performance.

## Turkey earthquake

Conservative changes all hurt seed 0: alpha 1, steps 2, learning rate 0.01, and rank 8 reached 0.2697, 0.2553, 0.2517, and 0.2445. Increasing to eight updates improved Macro-F1 from 0.2845 to 0.3378.

The main issue was layer choice. Layer 14-only reached 0.3574 on seed 0, while layer 27-only reached 0.2672. Cross-seed validation confirmed the middle-layer result:

| Setting | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| Canonical Lens, layers 14+27 | 0.2845 | 0.2659 | 0.2166 | 0.2557 |
| Lens, layer 14 only | 0.3574 | 0.2867 | 0.3195 | 0.3212 |
| LoRA, four passes | 0.3272 | 0.3028 | 0.2676 | 0.2992 |

Layer 14-only improves the Lens mean by about 6.55 points and exceeds four-pass LoRA by about 2.20 points, while using 512 controller coefficients instead of 1,024. This is suitable as a targeted layer ablation. It should not be silently substituted into the canonical cross-event result without evaluating the same fixed layer rule on every event.

## RoboFail

The zero-residual baseline reaches 0.4115 Macro-F1, compared with 0.3080 for the canonical Lens objective. Alpha 1, steps 2, and layer 27-only reach 0.3553, 0.3259, and 0.3859. These changes mitigate but do not remove negative transfer.

The query contains 93 success and 20 failure examples, while support is forced to 8/8 and covers different task-variation groups. The canonical Lens predicts failure for roughly 79 of 113 queries. Alpha 1 increases failure recall from 25% for the zero-residual state to 60%, but reduces success recall from 58% to 32%. Source-field inspection confirms that `execution_reward`, before/end images, task instruction, and labels are converted consistently; this is not a label-mapping bug.

RoboFail should therefore remain a limitation: balanced few-shot adaptation shifts the decision prior and does not learn enough transferable visual evidence across task groups. Frozen-state fallback or support-only model selection is preferable to claiming a Lens improvement on this split.

Task-family analysis makes the failure mode more specific. Removing the numeric variation suffix from each `taskvar`, support covers only 5 of the 10 families present in query. On the 44 query examples from covered families, Frozen reaches 0.4653 Macro-F1 and layer-27 Lens reaches 0.4844. On the 69 examples from uncovered families, Frozen reaches 0.3611, canonical Lens 0.1711, and layer-27 Lens 0.3007. Canonical Lens reduces success recall on these unseen families to 7.0% while increasing failure recall to 66.7%.

This is cross-task-family negative transfer. A new family-stratified split or a support-only fallback gate would be a separate protocol, not a parameter repair to the current result. No further query-driven hyperparameter sweep is justified for the present paper.

## Fixed layer-14 cross-event validation

After the Turkey finding, the same layer-14-only rule was fixed and run on all remaining BRIGHT event/seed states. Across all 12 states it reaches 0.3207 mean Macro-F1, compared with 0.3133 for the canonical two-layer Lens, 0.3146 for one-pass LoRA, 0.3018 for Random-KV, and 0.2739 for four-pass LoRA. It uses 512 learned controller coefficients rather than 1,024.

| Event | Two-layer Lens | Layer 14 only | Difference |
|---|---:|---:|---:|
| Hawaii wildfire | 0.3329 | 0.2883 | -0.0446 |
| Libya flood | 0.3404 | 0.3354 | -0.0050 |
| Noto earthquake | 0.3242 | 0.3378 | +0.0137 |
| Turkey earthquake | 0.2557 | 0.3212 | +0.0655 |
| Four-event mean | 0.3133 | 0.3207 | +0.0074 |

This is a useful parameter-efficiency result, but the interaction with event is substantial. Layer 14 only should be reported as a fixed exploratory variant that improves the aggregate, especially Noto and Turkey, rather than as a universally superior layer choice. The canonical two-layer result remains the preregistered primary comparison.
