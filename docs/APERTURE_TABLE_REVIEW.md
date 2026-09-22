# Implementation review

Reviewed locally; no independent reviewer subagent or real GPU was available.

- The existing main branch does not implement the proposed Gemma/Qwen3 medical
  table. Added an isolated new namespace instead of renaming either old study.
- Native Gemma image token types preserve its multimodal attention behavior;
  no Qwen loader/mask is substituted. Qwen grids are checked against expansion.
- LoRA collection excludes vision/projection impostors and uses actual decoder
  modules. Selective vocabulary logits preserve complete candidate likelihoods.
- LoRA and Aperture share fitting IDs, held-out labels and queries, but not the
  same optimizer-step count. Both LoRA budgets and the all-support bias remain.
- Reused existing controller/fitter math. Actual base core.py blob SHA1 matches
  inspected main: b83089a783b02ca84448b14d8c965dc453226bea.
- Stage seals protect all four methods, not just the method currently scoring.
  Saved-state calibration replay occurs before queries. Fit code uses query
  manifests for integrity only, never model query predictions.
- Exact112-key exporter is covered by a synthetic64-state test using the real
  DONE/identity/selection/prediction file layout. Tampering revokes table output.
- Added tests package marker after joint test discovery exposed name collisions
  with existing test_audit.py/test_protocol.py. The whole selected CPU suite
  then passed; no old test modules were edited.
- Running matrices from CPU tests does not establish native model integration,
  Gemma download permissions, patient-pool feasibility or CUDA determinism.
  Those are mandatory host-side gates, not checked boxes in this report.
