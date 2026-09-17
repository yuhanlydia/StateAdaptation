# Historical Hawaii results and protocol boundaries

The repository retains earlier Hawaii results in `reports/rank16_full_diagonal_alpha_sweep.md` and the historical manuscript. They were not overwritten by the P0 v3 campaign.

## Historical locked-support alpha sweep

The historical rank-16 sweep used the original balanced support24, fixed query300, covariance basis, layers 14+27, four Lens updates, and a fixed-eight-update LoRA baseline.

| Method/configuration | Hawaii Macro-F1 |
|---|---:|
| Fixed-8 LoRA | 0.3526 |
| Full Lens, alpha 0.5 | 0.2622 |
| Full Lens, alpha 1 | 0.3338 |
| Full Lens, alpha 2 | **0.3869** |
| Full Lens, alpha 3 | 0.2957 |
| Full Lens, alpha 5 | 0.3239 |
| Full Lens, alpha 10 | 0.3476 |

Thus the historical alpha-2 Lens result is higher than the historical fixed-8 LoRA result. Across the four events, per-event oracle-best Full Lens beats fixed-8 LoRA on Hawaii, Libya, and Turkey, with a 0.3504 versus 0.2986 mean.

## Why it is not the new primary estimator

The alpha value was selected from query performance, so 0.3869 is a query-oracle capacity diagnostic. The original report explicitly says it must not be promoted as a test-independent primary result. It also records nominally identical Hawaii alpha-3 runs at 0.3295 and 0.2957, motivating independent replication.

The P0 v3 campaign answers a different question: fixed alpha 3, regenerated deterministic supports after the unavailable private manifests, matched methods across three support seeds, and an explicitly versioned implementation. Its canonical Hawaii mean is 0.3329 for the two-layer Lens versus 0.3504 for one-pass LoRA. A seed-0 alpha-1 follow-up reaches 0.4019 but does not replicate across seeds.

Both result families are valid within their protocols:

- Use 0.3869 to demonstrate that the Lens has capacity to beat the historical fixed-8 LoRA on Hawaii under the alpha sweep.
- Use the P0 v3 three-seed table for the fixed-configuration matched replication.
- Do not merge the oracle-best value and fixed-configuration values into one primary row.
