# Implementation and evidence status

## Completed repository work

- Full tracked EventTune code/report/paper tree and upstream history imported into StateAdaptation; old repository untouched.
- Original snapshot pinned at 3150cae1eecf2ee82e508d32fe7f2e80913f8149 on legacy/eventtune-3150cae.
- Previously local-only vigor_handoff code overlay integrated; obsolete multiple-draft paper variants omitted in favor of the latest single manuscript.
- P0-A deterministic matched BRIGHT seeds and 1-pass/4-pass LoRA matrix implemented.
- P0-B fixed-state image interventions and support-only candidate-bias alternative implemented.
- P0-C actual checkpoint-replay, mask, objective and answer-span audit implemented; downstream jobs require a passed sealed gate.
- Aggregation keeps protocols and diagnostic stages separate and reports missing/failed units.

## Completed GPU validation and results

The final checkout passes 57 collected CPU tests. The canonical GPU plan completed all 83 top-level jobs: 3 audit, 63 matched, 8 objective, and 9 evidence. All sealed artifact hashes, 110 prediction sets, query coverage, labels, finite normalized probabilities, and stage return codes were verified. The follow-up weak-case campaign completed 29/29 exploratory jobs and fresh current-source Qwen2.5-VL and InternVL3 audits. Compact public results are exported under `reports/rebuttal_2026-09-17/`.

## Main empirical outcome

Across the 12 BRIGHT event/support-seed states, canonical Visual Lens reaches 0.3133 mean Macro-F1 versus Frozen 0.2255, one-pass LoRA 0.3146, four-pass LoRA 0.2739, and Random-KV 0.3018. The exploratory fixed layer-14-only variant reaches 0.3207 with 512 learned coefficients, but its effect is event-dependent: it improves Noto and Turkey, is close on Libya, and hurts Hawaii. Canonical results remain primary; exploratory results are labeled separately.

## Not silently claimed complete

- Independent development-domain hyperparameter selection: the fixed four-pass LoRA stronger baseline is coded; selecting an additional learned schedule requires an agreed development set/config.
- Full three-repeat rank16 end-to-end efficiency study: per-job timing/memory is logged, but controlled warm-up/repeated timing and phase-specific engineering measurements remain a separately designed P1 study.
- Full ReFT reproduction: the existing hidden-state residual is a custom control, not the author's full algorithm.
- General controller-gradient transfer across domains (P2): support-only actual controller gradients and a finite-difference curve are audited, but no predictive theorem or across-domain empirical law is claimed.
- Git migration does not move private Hub datasets, untracked runs or multi-gigabyte model files.

## Recorded limitations

- Historical private BRIGHT support manifests were unavailable. Official Zenodo assets were checksum-verified, and folds were deterministically regenerated from the original procedure with `PYTHONHASHSEED=0`.
- Hawaii is sensitive to support state and residual strength; no alternative alpha consistently improves all three seeds.
- RoboFail shows cross-task-family negative transfer. Lens helps slightly on covered task families but degrades unseen families, so Frozen remains stronger on the full split.
- Layer choice has a substantial event interaction; the layer-14 aggregate improvement is not a universal per-event guarantee.

The historical evidence-freeze decision and the current request are different scopes: preserve the old evidence; run only the newly requested verification questions and required controls.
