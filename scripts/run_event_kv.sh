#!/usr/bin/env bash
set -euo pipefail

# EventTune experiment pipeline for the Post-Visual Residual KV-TTT path.
#
# The KV-TTT baseline is the source adapter itself: the KV subspace B is
# extracted from the target support with the source weights frozen and the 32
# coefficients are fitted at "test time", so everything downstream of
# source_eval compares Residual KV-TTT against the exact same source model.
#
# pipeline: source -> [source_gate] -> source_eval -> event_kv(kv_state.pt)
#           -> kv_eval -> kv_gain
#
# Iteration controls the LoRA ground-truth on source and the KV adaptation on
# the event_kv stage, matching the original sequence in run_event.sh.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SPLIT_DIR RUN_DIR [SOURCE_STEPS]" >&2
  exit 2
fi

SPLIT_DIR="$1"
RUN_DIR="$2"
SOURCE_STEPS="${3:-1000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_GRADIENT_ACCUMULATION="${SOURCE_GRADIENT_ACCUMULATION:-6}"
CROP_SIZE="${CROP_SIZE:-448}"
EVAL_D4_VIEWS="${EVAL_D4_VIEWS:-8}"
SOURCE_GATE_MANIFEST="${SOURCE_GATE_MANIFEST:-}"
SOURCE_GATE_MIN_MACRO_F1="${SOURCE_GATE_MIN_MACRO_F1:-0.2}"
SOURCE_GATE_MIN_CLASSES="${SOURCE_GATE_MIN_CLASSES:-2}"
RUN_ORIGINAL_EVAL="${RUN_ORIGINAL_EVAL:-0}"

# KV-TTT knobs (defaults mirror the smoke run / README).
KV_RANK="${KV_RANK:-8}"
KV_ALPHA_MAX="${KV_ALPHA_MAX:-0.5}"
KV_STEPS="${KV_STEPS:-4}"
KV_LR="${KV_LR:-0.05}"
KV_L2="${KV_L2:-1e-3}"
KV_LAYERS="${KV_LAYERS:-}"

mkdir -p "${RUN_DIR}"

# ---- 1. source SFT ---------------------------------------------------------
if [[ ! -f "${RUN_DIR}/source_adapter/train_summary.json" ]]; then
  SOURCE_ARGUMENTS=(
    --train-manifest "${SPLIT_DIR}/source_train.jsonl"
    --steps "${SOURCE_STEPS}"
    --gradient-accumulation "${SOURCE_GRADIENT_ACCUMULATION}"
    --crop-size "${CROP_SIZE}"
    --output-dir "${RUN_DIR}/source_adapter"
  )
  if [[ -n "${SOURCE_GATE_MANIFEST}" ]]; then
    SOURCE_ARGUMENTS+=(--exclude-manifest "${SOURCE_GATE_MANIFEST}")
  fi
  "${PYTHON_BIN}" scripts/train_source.py "${SOURCE_ARGUMENTS[@]}"
else
  echo "resume: source training already complete"
fi

if [[ -n "${SOURCE_GATE_MANIFEST}" ]]; then
  if [[ ! -f "${RUN_DIR}/source_gate/metrics.json" ]]; then
    "${PYTHON_BIN}" scripts/evaluate.py \
      --manifest "${SOURCE_GATE_MANIFEST}" \
      --adapter "${RUN_DIR}/source_adapter" \
      --d4-views 1 \
      --crop-size "${CROP_SIZE}" \
      --output-dir "${RUN_DIR}/source_gate"
  fi
  "${PYTHON_BIN}" scripts/check_source_gate.py \
    --metrics "${RUN_DIR}/source_gate/metrics.json" \
    --predictions "${RUN_DIR}/source_gate/predictions.jsonl" \
    --output "${RUN_DIR}/source_gate/gate.json" \
    --minimum-macro-f1 "${SOURCE_GATE_MIN_MACRO_F1}" \
    --minimum-predicted-classes "${SOURCE_GATE_MIN_CLASSES}"
fi

if [[ "${RUN_ORIGINAL_EVAL}" == "1" ]]; then
  if [[ ! -f "${RUN_DIR}/original_eval/metrics.json" ]]; then
    "${PYTHON_BIN}" scripts/evaluate.py \
      --manifest "${SPLIT_DIR}/target_query.jsonl" \
      --no-lora \
      --d4-views "${EVAL_D4_VIEWS}" \
      --crop-size "${CROP_SIZE}" \
      --output-dir "${RUN_DIR}/original_eval"
  fi
fi

# ---- 2. source_eval (the KV-TTT baseline) ---------------------------------
if [[ ! -f "${RUN_DIR}/source_eval/metrics.json" ]]; then
  "${PYTHON_BIN}" scripts/evaluate.py \
    --manifest "${SPLIT_DIR}/target_query.jsonl" \
    --adapter "${RUN_DIR}/source_adapter" \
    --d4-views "${EVAL_D4_VIEWS}" \
    --crop-size "${CROP_SIZE}" \
    --output-dir "${RUN_DIR}/source_eval"
else
  echo "resume: source evaluation already complete"
fi

# ---- 3. event_kv: extract B on source + fit a on the 12 support -----------
KV_ADAPT_ARGUMENTS=(
  --support-manifest "${SPLIT_DIR}/target_support.jsonl"
  --source-adapter "${RUN_DIR}/source_adapter"
  --output-dir "${RUN_DIR}/event_kv"
  --crop-size "${CROP_SIZE}"
  --rank "${KV_RANK}"
  --alpha-max "${KV_ALPHA_MAX}"
  --coefficient-mode "${KV_COEFFICIENT_MODE:-diagonal}"
  --steps "${KV_STEPS}"
  --learning-rate "${KV_LR}"
  --l2 "${KV_L2}"
)
if [[ -n "${KV_LAYERS}" ]]; then
  # shellcheck disable=SC2086
  KV_ADAPT_ARGUMENTS+=(--layers ${KV_LAYERS})
fi
if [[ ! -f "${RUN_DIR}/event_kv/kv_state.pt" ]]; then
  "${PYTHON_BIN}" scripts/adapt_event_kv.py "${KV_ADAPT_ARGUMENTS[@]}"
else
  echo "resume: KV extraction already complete"
fi

# ---- 4. kv_eval: source + KV residual controller ---------------------------
if [[ ! -f "${RUN_DIR}/kv_eval/metrics.json" ]]; then
  "${PYTHON_BIN}" scripts/evaluate.py \
    --manifest "${SPLIT_DIR}/target_query.jsonl" \
    --adapter "${RUN_DIR}/source_adapter" \
    --kv-state "${RUN_DIR}/event_kv/kv_state.pt" \
    --d4-views "${EVAL_D4_VIEWS}" \
    --crop-size "${CROP_SIZE}" \
    --output-dir "${RUN_DIR}/kv_eval"
else
  echo "resume: KV evaluation already complete"
fi

# ---- 5. kv_gain: KV-TTT vs source (and original if available) --------------
if [[ ! -f "${RUN_DIR}/kv_gain.json" ]]; then
  "${PYTHON_BIN}" scripts/compare_predictions.py \
    --baseline "${RUN_DIR}/source_eval/predictions.jsonl" \
    --adapted "${RUN_DIR}/kv_eval/predictions.jsonl" \
    --output "${RUN_DIR}/kv_gain.json"
fi

if [[ -f "${RUN_DIR}/original_eval/metrics.json" ]] && [[ ! -f "${RUN_DIR}/original_vs_kv.json" ]]; then
  "${PYTHON_BIN}" scripts/compare_predictions.py \
    --baseline "${RUN_DIR}/original_eval/predictions.jsonl" \
    --adapted "${RUN_DIR}/kv_eval/predictions.jsonl" \
    --output "${RUN_DIR}/original_vs_kv.json"
fi
