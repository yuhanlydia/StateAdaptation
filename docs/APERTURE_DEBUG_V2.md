# Targeted debugging after the first support search

This is a new exploratory experiment, motivated by the two support-only reviews in `docs/debug/`. It does not replace the fixed table or v1. Five combinations are prioritized: Qwen2.5 H2/Path, Qwen3-4B H2, and Gemma H2/Path. No forward/mask/restore bug was established. The concrete issues are unrepresentative H2 validation, optimization/conditional-classification mismatch, and missing final-state fit classification diagnostics.

## Locked selection design

The candidate registry is `configs/debug/aperture_debug_v2.json`. Rank16, original generative loss, BF16, image preprocessing, original labels, query IDs, and two support seeds are preserved. H2 has four candidates per method and seed; Path has six. Aperture candidates are task-specific rather than another shared cross-model recipe. LoRA uses the same candidate count and validation folds; removed high-LR regions had collapsed on support validation, not query. The reports document the rationale.

For H2 the original 12 fit + 4 calibration images are partitioned into three patient-separated folds. Each of the 16 labels is held out once, with both classes in every train/validation partition. Folds are chosen deterministically from group IDs and label balance only; query is read solely for data-integrity checks. Candidate selection maximizes pooled out-of-fold Macro-F1, then minimizes pooled NLL, then uses candidate ID. This corrects the original dependence on four calibration images from a single patient. It is a changed selection protocol, not the same fixed-fit/calibration study. Pooling scores from separately fitted folds is explicitly reported; fold-specific metrics remain available.

Path retains its original 54 fit / 18 calibration images and F1/NLL selection. Its tiny, already used validation set can still overfit; image isolation is not patient isolation. No independent generalization claim is made.

After selection, both methods are refitted on the original fit split (12 H2 / 54 Path), preserving the final training budget, and independently repeated on the same GPU. All 20 selected states must pass the unchanged 0.005 replay/repeat tolerance before any v2 query scoring. Final-state support classification, class coverage and confusion matrices are saved: the old last Aperture history loss was pre-update, and LoRA history was an online average, so neither was the saved state's final fit classification.

## Budget and comparison

The plan comprises 192 search fits (144 H2 fold fits + 48 Path fits), 20 final fits, 20 selected-fit repeats, architecture-specific support audits, and 20 selected evaluations. Each checkpoint/step count is counted as a separate candidate. No hidden query-guided retries or selective deletion of failed candidates. All compact candidate histories and validation predictions will be exported, including unfavorable outcomes.

The published report includes both v2 LoRA and the previously tuned v1 LoRA reference. Do not claim success by weakening the baseline. Candidate count is matched within this round, but FLOPs, parameter count and cumulative historical search are not identical. The test/query set has already been inspected; the entire follow-up is adaptive exploratory debugging. A future independent locked test is needed for an independent generalization claim. A loss change (candidate-normalized CE or token-sum loss) would require a separate named experiment with the same treatment of LoRA and is not silently included here.

## Execution

Use the persistent tested venv and set `PYTHONPATH=src`. Prepare with the original runtime base config and existing asset manifests, then run `python -m aperture_debug.cli --root NEW_ROOT check` before queueing. The source fingerprint locks the debug package and original model implementation. Modifications require a new versioned root. The current queue waits for v1 to finish and upload before acquiring free GPUs; it uses the same per-GPU cooperative locks and does not stop unrelated processes. GPU availability requires at least 23000 MiB free and no compute processes. A stage has a 24-hour deadline and any failure stops dispatch without changing parameters.

`scripts/run_aperture_debug_queue.py --root NEW_ROOT --previous V1_ROOT` runs the locked experiment and publishes to `aperture-targeted-debug-20260923` on success. Check `status.json`, `assignments.jsonl`, `failure.json`, and `UPLOAD_STATUS.json`. A publish failure after export/commit requires inspection and a direct `git push` retry; the exporter deliberately refuses to overwrite existing results. Do not remove a live RUNNING lock or reinterpret a failed gate as a success.
