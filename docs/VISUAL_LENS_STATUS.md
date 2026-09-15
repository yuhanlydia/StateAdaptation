# Implementation and evidence status

## Completed repository work

- Full tracked EventTune code/report/paper tree and upstream history imported into StateAdaptation; old repository untouched.
- Original snapshot pinned at 3150cae1eecf2ee82e508d32fe7f2e80913f8149 on legacy/eventtune-3150cae.
- Previously local-only vigor_handoff code overlay integrated; obsolete multiple-draft paper variants omitted in favor of the latest single manuscript.
- P0-A deterministic matched BRIGHT seeds and 1-pass/4-pass LoRA matrix implemented.
- P0-B fixed-state image interventions and support-only candidate-bias alternative implemented.
- P0-C actual checkpoint-replay, mask, objective and answer-span audit implemented; downstream jobs require a passed sealed gate.
- Aggregation keeps protocols and diagnostic stages separate and reports missing/failed units.

## Validation versus results

The development checkout passed 92 CPU tests (52 original,31 earlier overlay,9 new) and syntax/planning checks. The default plan has 83 top-level jobs: 3 audit,63 matched,8 objective,9 evidence. No model training, full-model backward test or new benchmark evaluation was run in this environment. CPU tests do not certify model-family GPU integration or hardware fit. The first real-GPU audit must decide that.

## Still requires the GPU agent

All new P0 numerical outcomes, including formal matched rank16 means and image-control differences. Existing published-facing historical results are preserved separately and should not be silently replaced. The archival original task fitting code and new explicit objective code differ; record this when comparing.

## Not silently claimed complete

- Independent development-domain hyperparameter selection: the fixed four-pass LoRA stronger baseline is coded; selecting an additional learned schedule requires an agreed development set/config.
- Full three-repeat rank16 end-to-end efficiency study: per-job timing/memory is logged, but controlled warm-up/repeated timing and phase-specific engineering measurements remain a separately designed P1 study.
- Full ReFT reproduction: the existing hidden-state residual is a custom control, not the author's full algorithm.
- General controller-gradient transfer across domains (P2): support-only actual controller gradients and a finite-difference curve are audited, but no predictive theorem or across-domain empirical law is claimed.
- Git migration does not move private Hub datasets, untracked runs or multi-gigabyte model files.

The historical evidence-freeze decision and the current request are different scopes: preserve the old evidence; run only the newly requested verification questions and required controls.
