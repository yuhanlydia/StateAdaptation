# Bounded support-selected tuning (exploratory v1)

This is a new experiment after the fixed table at commit 41738d9. Its purpose is to test whether a small, specified hyperparameter search improves Aperture relative to an equally explicit LoRA search. Improvement or superiority is not assumed. Historical states and table results remain unchanged.

## Evidence and scope

The read-only review found no clear answer-span, visual-mask or restoration failure explaining the old scores. It identified architecture and optimization choices worth testing:

- Decoder depths are 28 (q25), 36 (q34 and q38), and 34 (Gemma). Identical layer IDs do not align relative depth. Gemma's attention types also differ by layer.
- Qwen3/Gemma apply head normalization downstream of the K projection. Identical projection-space alpha need not induce the same change as in Qwen2.5.
- Many Aperture updates reach the existing gradient clipping threshold, and four steps need not be appropriate for every backbone/task.
- Qwen H2 `normal` and `tumor` have different token lengths. The inherited mean-token training loss and summed-token candidate score are intentionally preserved for BOTH methods in this hyperparameter experiment. Changing this objective requires a separately labeled comparison.
- LoRA rank/alpha/dropout are hard-coded by the current backend to 16/32/0.05. This search varies only parameters that the backend actually uses: learning rate and passes.

The only modification to the old audit is generalizing its exact expected projection-key set from four sites to the sites explicitly requested. A single K/V layer has two sites. All mask, identity and gradient tolerances are unchanged; new CPU tests reject missing, wrong, extra, leaking, and nonfinite sites.

## Locked candidates

Each model/dataset/support-seed unit receives six Aperture candidates and six LoRA candidates. Labels, image processing, backbone BF16 weights, rank, optimization seed, candidate answer order, and train/calibration/query membership are identical to the old fixed table. This matches candidate count and labels, **not** GPU time or trainable parameter count.

For depth L, `mid = floor(L/2)` and `last = L-1` (zero-based):

| ID | Aperture layers | Alpha | Learning rate | Full-support updates |
|---|---|---:|---:|---:|
| a0 | 14,27 (legacy) | 3 | .05 | 4 |
| a1 | mid only | 3 | .05 | 4 |
| a2 | mid,last | 3 | .05 | 4 |
| a3 | mid,last | 1 | .01 | 4 |
| a4 | mid,last | 1 | .01 | 12 |
| a5 | mid,last | 3 | .01 | 12 |

For q25 only, a2 would duplicate a0. The preregistered deterministic substitute is `[floor(L/4), floor(3L/4)] = [7,21]`, retaining six distinct configurations. These are configuration bundles, not an exhaustive grid or a causal one-factor ablation. Rank remains 16 and L2 .001.

LoRA candidates l0–l5 enumerate learning rates `[5e-5, 2e-4, 8e-4]`, with one and four passes at each rate, in that order. Rank/alpha/dropout remain 16/32/.05. No optimizer budget is extended after looking at query results.

## Selection and query barrier

Selection maximizes **held-out support calibration Macro-F1**, then minimizes raw calibration NLL, then uses the candidate ID as a deterministic tie break. Both methods use this rule. All six validation records and fit seals are saved. The calibration sets are small: four H2 or 18 Path images per seed. Six-way selection can overfit these sets; this is a bounded exploratory search, not proof of an optimal configuration.

All 16 units must finish candidate fits and select winners before the global query barrier can be sealed. Each of the 32 winning fits receives a fresh same-GPU support-only repeat, with the inherited .005 score tolerance. Only those 32 winners can subsequently read query through the evaluation entry point. Saved-state calibration replay is checked again before query scoring.

The old query sets have already been inspected. Re-evaluating them is explicitly **exploratory replication**, not independent blind validation. No result may silently replace the old table or establish cross-model superiority merely by choosing a favorable test cell. Independent generalization claims need a separately locked, previously unused evaluation set.

## Execution

Use the validated environment with `PYTHONPATH=src` from this checkout:

```bash
python -m aperture_tuning.cli --root /new/root/aperture_tuning_v1 prepare \
  --base-config /path/to/original/runtime_config.json \
  --asset-manifests /path/to/original/run/assets
python -m aperture_tuning.cli --root /new/root/aperture_tuning_v1 check
python -m aperture_tuning.cli --root /new/root/aperture_tuning_v1 run --gpus 0,1,2,3 --wait-hours 24
```

The plan contains 192 candidate fits, 32 selected-fit repeats, and 32 selected query evaluations, plus architecture/strength-specific support-only audits. Model byte checks, manifest/image checks, all fresh GPU audits, selection/repeat barriers, and query metric recomputation must pass. Candidate and selected-state identities include the tuning source fingerprint.

The dispatcher waits until a GPU has at least 23,000 MiB free and no compute process, then acquires the cooperative per-GPU lock and rechecks availability. It does not stop another task or reserve VRAM while waiting. External jobs can still arrive after the check; any failure is recorded and stops new dispatch rather than changing parameters. Each stage has a 24-hour deadline by default. Resumed incomplete units keep their original GPU for same-device repeat checks; completed units do not require that GPU again. A leftover `RUNNING.lock` is fail-closed: inspect its host/PID and preserve the failure record before manual recovery. Do not remove a live lock.

Outputs include `plan.json`, `CHECKED.json`, `assignments.jsonl`, `status.json`, `selections/`, `repeat_gates/`, `SELECTION_COMPLETE.json`, all candidate fitting records, selected predictions, and `summary.json`. `status.json` reports waiting versus active work; no GPU result is inferred from CPU test success.

For the prepared server checkout, `scripts/run_aperture_tuning_queue.py --root ...` runs the GPU queue and, only after successful verification, exports all candidate calibration records, selected predictions, repeat evidence, and the actual comparison to `results/aperture_tuning_v1/`. It then commits and pushes **aperture-support-tuning-20260922**, without rewriting the historical main table. Failures stop the pipeline and are recorded in `pipeline_status.json`; success records the uploaded commit in `UPLOAD_STATUS.json`.

If export/commit succeeds but the network push fails, the completed local result commit is retained. Inspect `pipeline_status.json` and the local commit, then retry only `git push origin HEAD:refs/heads/aperture-support-tuning-20260922`; do not regenerate or delete the nonempty result export. GPU device affinity uses local numeric IDs; after a reboot or device remapping, verify that the physical GPU is unchanged before resuming a partially completed unit.
