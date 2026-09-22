# Aperture four-backbone main-table protocol v1

This implements the previously designed empty main table, not the 135-state
Qwen/InternVL five-hospital study or the 180-state single-backbone study.
Source base inspected: `92a6b54ec4b393276c5df97f729e06decfe578dc`.
No historical method, result, data split or manuscript is overwritten.

## Registered matrix

| Key | Exact checkpoint | Native model class |
|---|---|---|
| q25 | Qwen/Qwen2.5-VL-7B-Instruct | Qwen2_5_VLForConditionalGeneration |
| q34 | Qwen/Qwen3-VL-4B-Instruct | Qwen3VLForConditionalGeneration |
| q38 | Qwen/Qwen3-VL-8B-Instruct | Qwen3VLForConditionalGeneration |
| g34 | google/gemma-3-4b-it | Gemma3ForConditionalGeneration |

Four checkpoints, two families, two tasks, support seeds 0 and 1, four base
arms (Frozen, LoRA-1pass, LoRA-4passes, Aperture): **64 states / 48 fitted
states**. Four extra support-only repeat fits and eight model/task audits do
not enter the scientific result table. No new GPU results are supplied here.

| Task key | Dataset | Fit | Calibration | Query | Isolation |
|---|---|---:|---:|---:|---|
| h2 | CAMELYON17-WILDS Hospital 2 | 12 | 4 | 200 | Patient |
| path | PathMNIST-224 CRC cohort | 54 | 18 | 180 | Image content only |

Support is six fitting and two calibration examples per class. The source
roles and support IDs come from the existing `aperture_medical_v1` protocol,
split seed 310927, budget eight labels/class. The new query lists are fixed
class-balanced hash subsamples of that protocol's locked query lists (100 H2
or 20 Path queries/class). All models/arms/seeds share each task's query IDs.
Original files are immutable. H2 is a previously examined anchor, not a fresh
hospital blind test. Path's supplied NPZ has no patient IDs; this is not a
patient-isolated Path experiment. Each backbone-task lens is fitted separately.

Preparation can call the existing data-only medical preparation on these TWO
domains; it does not launch either old medical experiment matrix. Full official
metadata/images and the prescribed 224px Path NPZ are required; no 28px fallback.

## Fixed fitting

Aperture reuses `vigor_handoff.core`: K/V projection outputs in decoder layers
14 and 27 (zero based), rank16 Full residual, alpha3, four full-support updates,
Adam LR0.05, raw-controller L2=0.001, clipping1, support SUM of the per-answer
mean token losses. The backbone is frozen. Norm-bounded residual and basis
estimation are unchanged. Bases are support-derived and detached.

LoRA: decoder-only Q/K/V/O, rank16, alpha32, dropout0.05, LR2e-4,
accumulation1, one or four complete fitting-support passes. The inherited
baseline fitter uses AdamW. All actual parameter counts, optimizer updates,
example visits, basis size and saved adaptation bytes are recorded, rather
than copying Qwen2's 10.1M count to other models. Parameter counts are not speed.

Frozen output bias uses ALL original support labels (16 or 72); this comparator
is not weakened by discarding the fitting partition. Temperature uses only
held-out support for each fixed base arm: positive fixed grid .2--20 with
shrinkage0.01 toward1. It cannot change argmax/F1. Raw and temperature-scaled
LoRA schedules are selected separately on calibration NLL; ties prefer1pass.
Both schedules remain visible. Query labels are never used for these choices.

## Actual native multimodal integration

The new backend supports all three native Transformers classes explicitly.
It does not route Gemma through an InternVL/Qwen loader. Qwen uses 448px input
resizing followed by its native processor/grid. Gemma uses its native 896px
processor with pan-and-scan disabled. This is within-model preprocessing
control, not equal visual-token counts across architectures.

Gemma `token_type_ids` are retained for its visual attention mask. Image rows
must exactly match the native image token ID and the expected expansion count.
Qwen masks must match `image_grid_thw` and spatial merge size. Decoder K/V sites
are discovered by actual module objects; vision/projector modules are excluded.
The LoRA target collection must contain all decoder Q/K/V/O modules, not vision
projections that happen to share suffixes. Missing/ambiguous modules fail.

Scoring uses the exact same prompt prefix and candidate answer tokens for
fitting/evaluation. Candidates must not re-tokenize the prompt boundary. Native
image tensors and token contracts are checked across candidates. Selective
`logits_to_keep=answer_tokens+1` avoids constructing vocabulary logits for all
visual/prompt positions. Scores are SUM answer-token log probabilities, and
NLL is computed by float64 logsumexp without legacy clipping. Keep-cache is off.

API sources reviewed: upstream Transformers v4.57.1
- `src/transformers/models/gemma3/processing_gemma3.py`
- `src/transformers/models/gemma3/modeling_gemma3.py`
- `src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Qwen3-VL official 4B/8B model cards and native model class.

Recommended isolated environment: Transformers4.57.1 and PEFT0.17.1 with a
compatible CUDA PyTorch build. Loader rejects Transformers outside [4.57.1,5).
Weights must be BF16 unquantized safetensors on a24GB+ CUDA GPU, locally complete.
Gemma access/license must be obtained before the execution window. No secret
or credential is stored in this repository.

## Execution, integrity and reproducibility

One model process per assigned GPU. Parent orchestrator creates new child
processes with CUBLAS_WORKSPACE_CONFIG=:4096:8, PYTHONHASHSEED=0 and one BLAS
thread. Model workers request deterministic PyTorch algorithms, disable TF32
and autotuning, and use the SDPA math backend. Unsupported kernels raise.

Before fitting the full matrix, every model/task passes an actual support-only
audit: native image masks, zero-residual identity, nonzero visual/controller
gradients, direct mask isolation, save/restore, and observed non-reentrant
checkpoint recomputation with bounded gradient differences. There are eight
audits, not a claim of prior GPU validation. A configuration flag alone is not
evidence that checkpoint replay occurred.

For each model, its H2 seed0 Aperture is fitted twice in independent processes
on the SAME GPU; compare held-out-support scores at max-absolute tolerance0.005.
Both states are retained; no better-repeat selection. One reference pair does
not prove global determinism. Failed audits/repeats stop; do not widen tolerances.

All fitting, temperatures and selection records are sealed before evaluation.
Restoring each state must reproduce its saved calibration scores before its
first query forward. Model files are content-hashed at preflight and checked
for changes before each job. Source, library versions, split and image content,
recorded scores, probabilities and completion artifacts are checked. Per-query
resume accepts only an exact saved prefix under the same fingerprints.

The source-data roles are read for integrity during fitting, but no query is
forwarded through a model or used by an optimizer/calibrator at that stage.

## Export and manuscript integration

Run root: `runs/aperture_table_v1/`. Complete exports contain all64 states,
40 model--task--method records (20 displayed table rows),112 numeric cells, raw prediction-derived statistics and the
measured per-state resource records. Bias+T is n/a, not zero. Two-seed sample SD
is descriptive repeatability, not an independent patient-level confidence bound.

`summarize` validates all eight audits, four repeat pairs, all state hashes,
selections, shared query IDs and recomputed metrics before marking `paper_ready`.
On failure it revokes previous table exports rather than leaving stale valid
looking tables. No partial-completion means or best-seed replacement.

`fill-main` requires the112 unique `\\pending{key}` markers in the latest
manuscript. It creates a NEW file and a provenance record; never overwrites
source. It replaces only measured numeric cells. Pending prose still needs
editing, and the manuscript needs recompilation. The original 300/500/900-query
historical means cannot be pasted into these200/180-query cells.

## Runtime scope

Day-scale execution assumes assets/environment are prepared and2--4 suitable
GPUs. The64-state matrix makes12,160 complete query evaluations (64,640
candidate forwards), plus fitting/calibration/audits. Support-only timings are
exported before query evaluation to forecast query cost; they are estimates,
not a guarantee. This code does not silently reduce the model list, query count,
image resolution or seeds to fit a clock budget. No135/180-state campaign is
launched. Scientific performance remains to be measured on the execution host.
