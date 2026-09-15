# Hawaii wildfire 100-step diagnostic

Status: completed failure analysis. This report preserves a negative result;
it is not a paper-level claim.

## Frozen setup

- Experiment code: `cf34db1291619f7cc86cbfcad258260f13f35714`.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`, frozen BF16 base with rank-16 LoRA;
  no 4-bit weights and no QLoRA.
- Target: `hawaii-wildfire`, seed 0, tile-disjoint BRIGHT instance split.
- Data: 241,442 source examples, 12 balanced support examples, 3,443 query
  examples.
- Source training: 100 optimizer updates, learning rate `2e-4`, batch size 1,
  gradient accumulation 2, inverse-frequency class sampling, 448-pixel crops.
- Adaptation selection: support-only 3-fold CV over 0/4/8/16/32 updates;
  16 updates selected with mean support-CV macro-F1 `0.3783068783`.
- Evaluation: candidate-label likelihood, one D4 view, identical complete query
  denominator before and after adaptation.

## Full query result

| Metric | Source-only | Adapted | Adapted - source |
|---|---:|---:|---:|
| Macro-F1 | 0.145947 | 0.057340 | -0.088607 |
| Balanced accuracy | 0.333333 | 0.333333 | 0.000000 |
| Quadratic weighted kappa | 0.000000 | 0.000000 | 0.000000 |
| Ordinal MAE | 1.345338 | 0.905896 | -0.439442 |
| NLL | 1.107205 | 1.447254 | +0.340049 |
| Brier | 0.708569 | 0.934736 | +0.226167 |
| ECE | 0.272988 | 0.502099 | +0.229111 |

The source adapter predicted all 3,443 queries as `intact`. The adapted adapter
predicted all 3,443 as `damaged`. The lower adapted ordinal MAE is a mechanical
effect of always choosing the middle class; macro-F1, NLL, Brier, and ECE show
that adaptation was harmful.

## Source-domain diagnostic

A separately selected post-hoc diagnostic set contains 150 examples: 10 source
events times 3 labels times 5 examples. It was not excluded from this first
run's training pool and is used only to localize the failure, not for model
selection. The source adapter again predicted every item
as `intact` (macro-F1 `0.166667`, balanced accuracy `0.333333`, NLL
`1.214619`). This rules out an explanation based only on Hawaii domain shift.

The 100-update run consumed only 200 sampled examples from a 241,442-example
source manifest. Long-run inverse-frequency sampling did not guarantee balanced
labels within an individual optimizer update. The evidence supports an
under-training/update-instability diagnosis.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| source adapter | `9de0d486b4fad06450fccc467cc6885b4d4cf94ee741d0d5f974a124fb25b67d` |
| source query metrics | `15038a529fea883c3f4d00426207258ea6bed18d97b84ec80d4b548398aec059` |
| source query predictions | `7575d3d6c5374809db6758b81b9b802e3a261fb789eba03494f2261063b786f5` |
| event adapter | `54be2ff986f3a5140ef40ad1f45ddc4780808bcf122dd545f28e759317375c17` |
| event selection | `c9df1090bc12b628bd14323466e37f901dd580ce8404cf9601df706cc910ad2c` |
| adapted query metrics | `220cf30b4467ef99b506000fdb29eacef189c0b89ba2af137e204778c2eeb1c5` |
| adapted query predictions | `78c74770839e60f8b337762bb3fa29c5ff1ae014a3c57b6b3bb93f1a5cdc0a89` |
| adaptation gain | `29ee735c58ba08c01cb2a3867e138a08a482fa4ec58f96862c043c78f3ec6a3b` |
| balanced source diagnostic metrics | `ee7b7d482e72f22c21167eae1b5e30b8003c1791eca7d952f3d81f85c58a5474` |
| balanced source diagnostic predictions | `237037432fb6c3a445709887dde002ee235217d339c597d04de7103fc6992c2e` |

## Corrective gate

The next run uses a randomized class-cycle sampler, six microbatches per source
update (two per class), and substantially more source updates. It must predict
more than one class and improve over macro-F1 `1/6` on the frozen balanced
source diagnostic before a new full Hawaii query evaluation is allowed.
