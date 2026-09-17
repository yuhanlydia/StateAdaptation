# Visual Lens P0 Experiment Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and verify the canonical 83-job Visual Lens P0 experiment matrix on the local RTX 5090.

**Architecture:** Preserve the fail-closed stage barriers `audit -> matched -> objective -> evidence`, while allowing already-ready InternVL objective jobs to run before BRIGHT acquisition finishes. Store the final results produced after the Guardian group-disjoint and audit-tolerance fixes under `runs/visual_lens_p0_v3` and never mix them with earlier fingerprints.

**Tech Stack:** Python 3.10, PyTorch 2.11.0+cu128, Transformers 4.57.6, PEFT 0.14.0, RTX 5090, Hugging Face Hub, official BRIGHT Zenodo release.

**Spec:** `docs/VISUAL_LENS_RUN.md`

## Global Constraints

- Preserve fixed support seeds, candidate order, objectives, image resolution, optimizer budgets, and query sets.
- Do not tune protocol choices from query scores.
- Do not substitute quantization, CPU offload, smaller checkpoints, or alternate datasets.
- Stop downstream stages on a failed audit or job and diagnose the recorded failure.
- Keep raw data, model weights, tokens, and ignored run outputs out of Git.
- Record negative outcomes as limitations after implementation and data errors are excluded.

---

### Task 1: Environment and fixed assets

**Files:**
- Verify: `configs/visual_lens_p0.json`
- Verify: `artifacts/models/Qwen2.5-VL-7B-Instruct`
- Verify: `artifacts/models/InternVL3-8B-Instruct`

- [x] Install the project training and test dependencies without replacing CUDA PyTorch.
- [x] Download Qwen2.5-VL-7B revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- [x] Download InternVL3-8B revision `ddb3a169d5582e5c76e0809a128e55ab63686ada`.
- [x] Prepare and validate Camelyon17 seed 0.
- [x] Prepare a group-disjoint RoboFail seed 0 and pass its regression test.
- [x] Download, checksum, normalize, and prepare the official BRIGHT four-event assets.

### Task 2: GPU audit gates

**Files:**
- Output: `runs/visual_lens_p0_v3/audit/`

- [x] Pass InternVL3 Camelyon gradient-path audit.
- [x] Pass InternVL3 RoboFail gradient-path audit.
- [x] Pass Qwen2.5-VL BRIGHT gradient-path audit after BRIGHT preflight succeeds.

### Task 3: Objective and answer-span matrix

**Files:**
- Output: `runs/visual_lens_p0_v3/objective/`

- [x] Complete all 8 Camelyon/RoboFail sum/mean by full-answer/label-only jobs.
- [x] Verify all outputs are sealed with matching fingerprints and hashes.

### Task 4: Matched BRIGHT and Camelyon states

**Files:**
- Output: `runs/visual_lens_p0_v3/matched/`

- [x] Complete all 63 Frozen, LoRA, Random-KV, and Lens jobs on their fixed supports.
- [x] Record optimizer steps, support examples, timing, memory, and per-class metrics.

### Task 5: Fixed-state image evidence

**Files:**
- Output: `runs/visual_lens_p0_v3/evidence/`

- [x] Complete all 9 real, shuffled, neutral, post-only, and support-only bias jobs.
- [x] Verify each control restores the exact sealed fitted state without refitting.

### Task 6: Aggregate and verify coverage

**Files:**
- Output: `runs/visual_lens_p0_v3/summary/`

- [x] Run `bash scripts/run_visual_lens_all.sh summarize` with the v3 configuration.
- [x] Inspect `coverage.md`, `results.md`, `visual_dependence.md`, and `summary.json`.
- [x] Confirm the canonical matrix has 83 completed top-level jobs and no unreported failures.
- [x] Run the full CPU test suite and final asset/result fingerprint checks.
- [x] Report implementation failures separately from empirical limitations.
