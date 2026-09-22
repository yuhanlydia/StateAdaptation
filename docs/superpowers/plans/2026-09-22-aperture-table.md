# Four-backbone main-table implementation plan

**Goal:** Implement the already approved 64-state main-table experiment, not another medical campaign.
**Architecture:** Add isolated `aperture_table` namespace. Reuse the executed bounded controller, stable scoring metrics, output-bias fitter and medical_v1 patient/content partitions. Implement explicit native Qwen2.5/Qwen3/Gemma3 single-image adapters, a sealed fit/calibration/evaluation lifecycle and a 112-cell manuscript exporter.
**Specification:** Four models q25/q34/q38/g34, H2/path, two support seeds (0/1), four fitted arms (frozen/lora1/lora4/aperture). H2 12+4 labels/200 query, Path 54+18/180 query. Old protocols and result files remain unchanged.

- [ ] RED: test model registry, 64 jobs, split identity, query subsets, decoder site filtering, answer boundaries, attention masks, controller backward, artifact integrity, table cell mapping and incomplete exports.
- [ ] GREEN: implement deterministic protocol and prepared-manifest adapters.
- [ ] GREEN: implement explicit native processors/loaders, decoder-only LoRA and bounded controller reuse. All four backbones must have real-model support-only audits on the execution host.
- [ ] GREEN: implement fit + calibration sealing before query access; repeat fit and saved-state score checks; stable candidate scoring.
- [ ] GREEN: implement full-coverage exports and non-destructive main.tex filling, with exact unique pending keys.
- [ ] Verify CPU tests, synthetic end-to-end lifecycle, command entry points, source/asset locks, no non-additive diff; document absence of real GPU results.
- [ ] Publish via available GitHub write action or git; verify actual remote commit before saying pushed. If no write action/network is available, state precisely that publication remains pending.

No query-based changes to models, population, loss, schedule, metric, tolerance or inclusion. No private data, images, model weights, tokens, run artifacts or patient identifiers are added to Git.
