# Agent execution contract

Active project: A Learned Visual Lens; repository yuhanlydia/StateAdaptation.

1. Read README.md, docs/VISUAL_LENS_RUN.md and docs/VISUAL_LENS_STATUS.md before GPU execution.
2. Preserve existing reports and runs. New P0 outputs belong only to runs/visual_lens_p0_v1 (or a new explicitly versioned root after changing the protocol).
3. Do not change dataset inclusion, support seed, candidate order, learning rate, span, image resolution or stopping budget based on query scores. Missing assets or a failed GPU audit stop execution; do not silently substitute a smaller model or quantization.
4. Execute prepare, plan, check, then audit gates. A checkpoint flag without observed backward recomputation does not count as passing.
5. Fixed-state image controls must restore the exact sealed model/controller. Do not refit on shuffled or neutral query images; fit candidate bias on support only. Keep text and visual-token contracts equal.
6. Any oracle using query labels is diagnostic and cannot enter primary tables. Changed loss definitions go in separate rows, not replacements for historical measurements.
7. Report actual optimizer steps, processed support examples, per-class metrics, failures, source fingerprints and completed job count. Never fill missing results with expected values.
8. The old package names eventttt and vigor_handoff are intentional compatibility names. Do not mass-rename imports.
9. Do not push raw medical/robot data, private model files, tokens or ignored runs into this public repository. Export only appropriate compact metrics and provenance.
10. Stopping a failed stage is required. Diagnose in a new branch/run root and record the reason before resuming. Do not relax audit tolerances solely to make a test pass.
