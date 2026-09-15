# InternVL3 ManipBench steps=8 follow-up

This is the formal InternVL3 ManipBench setup with only the Ours update count
changed from 4 to 8. Rank=16, alpha=3, learning rate=0.05, L2=1e-3, layers
14 and 27, support/query IDs, and all three seeds are unchanged. LoRA is the
same formal rank-16 baseline.

| Domain | Ours steps=4 | Ours steps=8 | Change | LoRA |
|---|---:|---:|---:|---:|
| bridge_pick_place | 0.2993±0.0366 | 0.3347±0.0101 | +0.0354 | 0.5551±0.0159 |
| droid_arti | 0.3307±0.0691 | 0.4064±0.0404 | +0.0757 | 0.6759±0.0150 |
| droid_pick_place | 0.2839±0.0729 | 0.3328±0.0948 | +0.0489 | 0.4304±0.2223 |

Eight updates improve all three domains, especially droid_arti, but Ours
still remains below LoRA on every domain. The run uses the corrected
`Answer: <label>` scoring path and the same formal query sets; no query-based
selection was performed.

## Longer-optimization follow-up

We next tested whether the gains from eight coefficient updates continued with
longer optimization.  The 16-step run uses learning rate 0.01 and otherwise
keeps rank=16, alpha=3, L2=1e-3, layers 14 and 27, and the same three
support/query folds.  Its complete three-seed results are:

| Domain | Ours steps=8 | Ours steps=16, lr=0.01 | Change | LoRA |
|---|---:|---:|---:|---:|
| bridge_pick_place | 0.3347±0.0101 | 0.3333±0.0115 | -0.0014 | 0.5551±0.0159 |
| droid_arti | 0.4064±0.0404 | 0.4054±0.0481 | -0.0010 | 0.6759±0.0150 |
| droid_pick_place | 0.3328±0.0948 | 0.3347±0.1153 | +0.0019 | 0.4304±0.2223 |

Increasing from 8 to 16 updates at the lower learning rate is therefore
effectively flat.  More steps alone are not sufficient to close the LoRA gap.

## Steps=32/64 single-seed screening

To avoid spending three full folds on every long-run configuration, subsequent
settings were screened on support seed 0.  These numbers are tuning diagnostics,
not final multi-seed estimates.  Query labels were used to compare these screen
runs, so any selected configuration requires an independent or predeclared
multi-seed confirmation before it can support a primary claim.

| Configuration | bridge | droid_arti | droid_pick_place |
|---|---:|---:|---:|
| steps32, alpha3, lr0.01 | 0.3429 | 0.4135 | 0.3438 |
| steps64, alpha3, lr0.01 | -- | 0.4636 | 0.3984 |
| steps64, alpha2, lr0.01 | 0.3211 | 0.4158 | 0.3786 |
| steps64, alpha4, lr0.01 | 0.3218 | 0.4679 | 0.3907 |
| steps64, alpha3, lr0.005 | 0.3065 | 0.4288 | 0.3641 |
| **steps64, alpha3, lr0.02** | **0.3485** | **0.4782** | **0.4147** |

Among the completed seed-0 screens, steps=64, alpha=3, learning rate 0.02 is
the best common configuration across all three domains.  It is especially close
to the three-seed LoRA mean on droid_pick_place (0.4147 versus 0.4304), but this
is not a like-for-like statistical comparison and it remains below LoRA on the
screened seed.  The attempted seed-1/2 confirmation is not reported here because
no completed metrics were available in the retained run record.  The next valid
step is to finish that confirmation rather than further select on query results.
