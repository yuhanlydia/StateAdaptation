# ARC Prize 2025 review and implications for EventTune

Status: evidence review and code audit completed on 2026-08-04. This is a
design note, not a new EventTune experiment. ARC Prize 2025 / ARC-AGI-2 is the
primary subject; 2024 is considered only where it explains the evolution of a
method.

## Executive conclusion

ARC Prize 2025 was not won by one universally superior architecture. The
strongest offline systems combined task-time adaptation, invertible
augmentation, multiple candidate paths, support-example feedback, and careful
ranking. The transferable idea for EventTune is therefore a **verified
refinement loop** around a frozen target query, not an ARC-specific solver or a
larger unvalidated training run.

The current EventTune direction remains appropriate: Qwen2.5-VL-7B in BF16,
standard LoRA, no 4-bit loading, and no QLoRA. The immediate correction is to
prove that source learning has not collapsed before spending compute on a full
target evaluation. This repository now implements that source-only gate.

## What the 2025 competition actually measured

The final competition used 240 hidden ARC-AGI-2 tasks: 120 semi-private tasks
for the live leaderboard and a distinct 120-task private final set. Submissions
ran offline for at most 12 hours on four NVIDIA L4 GPUs. Each test pair allowed
two output attempts.

The official score should not be confused with a global cell or example
micro-average. For each task, a test pair is correct when either attempt exactly
matches its target; pair accuracy is averaged inside the task and then tasks
are averaged equally. A stricter “task solved” count requires every test pair
of that task to be correct. EventTune should likewise report both macro-event
and micro-sample metrics and never substitute one for the other.

The final private leaderboard was substantially lower than many public or
ARC-AGI-1 claims:

| Team | Private score | Main mechanism |
|---|---:|---|
| NVARC | 24.03% | synthetic program corpus, Qwen3-4B BF16 LoRA per task, multi-view ranking |
| ARChitects | 16.53% | LLaDA-8B diffusion model, 2-D position encoding, task-time LoRA and recursive sampling |
| MindsAI | 12.64% | CodeT5-Large task-time training, leave-one-out validation, augmented inference |
| Lonnie | 6.67% | Tiny Recursive Model task-time adaptation |
| Barbadillo | 6.53% | 2024-style test-time QLoRA pipeline |

These are competition-private scores. ARC-AGI-1 evaluation scores, ARC-AGI-2
public evaluation scores, semi-private API results, and private final results
are not interchangeable.

## What changed from 2024 to 2025

ARC Prize 2024 established a useful template: adapt on the demonstrations,
generate several transformed views, invert predictions back to the original
frame, and rank candidates by demonstration consistency. ARC-AGI-2 then made
single-rule pattern matching much less effective by emphasizing multi-rule
composition, sequential dependencies, contextual control, and symbols whose
meaning is defined inside the task.

More updates are not a substitute for evidence that the update rule is
correct. The 2025 systems that held up best created several hypotheses and used
verifiable support feedback to decide which ones survived. The public-to-private
drops also show why a clean gate and frozen query are more valuable than
repeatedly tuning against a public score.

## Code-level findings from representative 2025 systems

### NVARC

NVARC generated roughly 103,000 description/program-derived puzzles and used a
Qwen3-4B model with per-task BF16 LoRA, rank 256, 16 inference views, and a
second-stage likelihood/frequency ranker. It is the strongest direct evidence
that a 4B-class model plus adaptation can outperform larger generic inference
under a fixed offline budget. It does not imply that rank 256 is appropriate
for EventTune: ARC outputs are discrete programs/grids, while EventTune has
only twelve labeled support examples and a three-label classifier.

The released NVARC material also contains manual descriptions for part of the
public evaluation set. Those descriptions may explain public performance, but
they cannot be used as evidence of clean unseen-task generalization. The
private score is the meaningful competition result.

### ARChitects

ARChitects adapted an 8B masked-diffusion model with explicit two-dimensional
position information and recursively resampled uncertain outputs. The paper
reports rank 32 LoRA, while the released competition notebook uses rank 64 in
the exact path audited here. This is a reminder to treat executed configuration
as the record of an experiment rather than copying a paper table. Its expected
public score, live score, and private score also declined materially, so public
selection pressure must be recorded as a source of overfit.

### MindsAI

MindsAI used full task-time fine-tuning of CodeT5-Large rather than LoRA. The
most useful part for EventTune is its orchestration: leave-one-out pseudo-test
selection, deterministic inverse augmentations, a valid fallback submission,
per-run counters, error and OOM handling, and incremental result writes. The
architecture is not the transferable contribution; the failure-safe evaluation
loop is.

### Tiny Recursive Models and related entries

TRM showed that a roughly 7M-parameter recursive network can be competitive
when trained for the ARC representation. Its final path updates the full model,
and its task embedding is large. A later TRM task-time study found LoRA weaker
than full-trunk adaptation in that narrow regime. This does not establish that
LoRA is weak for a pretrained 7B vision-language model; the parameterization,
pretraining, and data regime are different.

Barbadillo's final notebook uses an `unsloth_4bit` base, rank-4 QLoRA, and an
8-bit optimizer. It is outside EventTune's required BF16/no-quantization path
and is not a candidate implementation.

## Reproducibility and leakage audit

Several attractive 2025-era claims do not survive a hidden-query audit:

- Some papers described as 2025 submissions evaluate the 400 ARC-AGI-1 tasks,
  not the ARC-AGI-2 private set.
- The visual-diffusion path in *Rethinking Visual Intelligence* uses the test
  target grid when choosing render scale and later decodes with exact target
  render metadata. Its reported ARC-AGI-1 result is therefore not a deployable
  hidden-target measurement. Its Qwen3-4B BF16 LoRA configuration is still a
  useful engineering reference, separate from that result.
- The public CoreThink code includes public evaluation outputs; one jigsaw path
  reads the test output directly, its available result file covers only a
  subset of tasks, and the implementation does not match several “deterministic
  perception” claims. Its self-reported public score is not a private
  competition result.
- VARC's released inference augmentation accesses the query output and uses its
  dimensions. This can change the valid transformation set, especially when
  input and output shapes differ.
- ArcMemo's dynamic-memory experiment accepts memories only after checking the
  query answer, which is an oracle analysis rather than a deployable update
  rule.
- External Grok-4 systems such as ARC Lang and Ed Pang report strong
  semi-private/API results, but they do not satisfy the offline four-L4 final
  environment and cannot be compared directly with the private leaderboard.

For EventTune, a valid adaptation path must satisfy all of the following:

1. Target-query labels, label-derived metadata, and query metrics are
   inaccessible until the adapter and all hyperparameters are frozen.
2. Every support-CV fold refits without the held-out support example; merely
   hiding it from a final scorer is not leave-one-out validation.
3. Failed examples and tasks remain in the denominator with zero credit.
4. Any memory, synthetic library, or cached candidate records its provenance
   and is checked against evaluation events.
5. Resume logic restores the same configuration and state; the presence of a
   partial output file is not evidence that a stage completed.

## EventTune protocol derived from the review

The next run should execute this sequence:

1. Reserve an event/label-balanced source gate and remove its sample IDs from
   source training.
2. Train the 7B BF16 rank-16 source LoRA with six microbatches per update so
   each update sees two examples from each class.
3. Evaluate the independent source gate first. Require macro-F1 at least 0.2
   and at least two predicted classes. On failure, stop before any target-query
   evaluation.
4. If the gate passes, produce the frozen source-only target baseline.
5. Select event update count only inside target support, including zero updates
   as the rollback candidate; reset to the source adapter before the final fit.
6. Apply only paired, domain-valid geometric views to pre/post imagery and
   aggregate label evidence. ARC color permutations have no remote-sensing
   meaning and must not transfer.
7. Evaluate the untouched query exactly once with the frozen source and event
   adapters, require complete sample coverage, and publish negative as well as
   positive results.

The 3B model is a resource fallback only if the 7B BF16 path fails CUDA or
memory preflight. A failed learning gate should be diagnosed rather than hidden
by silently changing the model.

## Tomorrow's prepared 7B launch

The gate is deterministic and can be restored from the private dataset Hub.
Its 150-line manifest has SHA-256
`331c5d37b731932322accd78d17e0d72a5fa30bcc29fbe3add427d41812f33de`.
Use a new run directory so no checkpoint from the failed 100-step diagnostic is
silently reused:

```bash
.venv/bin/python scripts/launch_long_job.py \
  --gpu 0 \
  --split-dir data/splits/bright_4shot/hawaii-wildfire/seed_0 \
  --run-dir runs/hawaii-wildfire_seed0_7b_lora_gate_v1 \
  --source-steps 1000 \
  --source-gradient-accumulation 6 \
  --crop-size 448 \
  --d4-views 8 \
  --source-gate-manifest data/splits/diagnostics/hawaii-source-event-label-150.jsonl
```

No training was started while preparing this review.

## Primary sources

- [ARC Prize 2025 competition](https://arcprize.org/competitions/2025)
- [ARC Prize 2025 results and analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [ARC-AGI-2 dataset and task format](https://github.com/arcprize/ARC-AGI-2)
- [ARC Prize 2025 technical report](https://arxiv.org/abs/2601.10904)
- [ARC-AGI-2 technical report](https://arxiv.org/abs/2505.11831)
- [Official model baseline and scorer](https://github.com/arcprize/model_baseline)
- [NVARC implementation](https://github.com/1ytic/NVARC)
- [ARChitects solution](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)
- [Tiny Recursive Models](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [Rethinking Visual Intelligence](https://arxiv.org/abs/2510.24448)
- [VARC](https://github.com/lillian039/VARC)
- [ARC Lang](https://github.com/jerber/arc-lang-public)
- [Ed Pang ARC-AGI system](https://github.com/epang080516/arc_agi)
