#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 SPLIT_DIR RUN_DIR [SOURCE_STEPS] [FIXED_EVENT_STEPS]" >&2
  exit 2
fi

SPLIT_DIR="$1"
RUN_DIR="$2"
SOURCE_STEPS="${3:-1000}"
FIXED_EVENT_STEPS="${4:-}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SOURCE_GRADIENT_ACCUMULATION="${SOURCE_GRADIENT_ACCUMULATION:-6}"
CROP_SIZE="${CROP_SIZE:-448}"
EVAL_D4_VIEWS="${EVAL_D4_VIEWS:-8}"
SOURCE_GATE_MANIFEST="${SOURCE_GATE_MANIFEST:-}"
SOURCE_GATE_MIN_MACRO_F1="${SOURCE_GATE_MIN_MACRO_F1:-0.2}"
SOURCE_GATE_MIN_CLASSES="${SOURCE_GATE_MIN_CLASSES:-2}"
RUN_ORIGINAL_EVAL="${RUN_ORIGINAL_EVAL:-0}"

mkdir -p "${RUN_DIR}"

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

ADAPT_ARGUMENTS=()
if [[ -n "${FIXED_EVENT_STEPS}" ]]; then
  ADAPT_ARGUMENTS+=(--fixed-steps "${FIXED_EVENT_STEPS}")
fi
if [[ ! -f "${RUN_DIR}/event_adapter/selection.json" ]]; then
  "${PYTHON_BIN}" scripts/adapt_event.py \
    --support-manifest "${SPLIT_DIR}/target_support.jsonl" \
    --source-adapter "${RUN_DIR}/source_adapter" \
    --output-dir "${RUN_DIR}/event_adapter" \
    --crop-size "${CROP_SIZE}" \
    "${ADAPT_ARGUMENTS[@]}"
else
  echo "resume: event adaptation already complete"
fi

if [[ ! -f "${RUN_DIR}/event_eval/metrics.json" ]]; then
  "${PYTHON_BIN}" scripts/evaluate.py \
    --manifest "${SPLIT_DIR}/target_query.jsonl" \
    --adapter "${RUN_DIR}/event_adapter" \
    --d4-views "${EVAL_D4_VIEWS}" \
    --crop-size "${CROP_SIZE}" \
    --output-dir "${RUN_DIR}/event_eval"
else
  echo "resume: event evaluation already complete"
fi

if [[ ! -f "${RUN_DIR}/adaptation_gain.json" ]]; then
  "${PYTHON_BIN}" scripts/compare_predictions.py \
    --baseline "${RUN_DIR}/source_eval/predictions.jsonl" \
    --adapted "${RUN_DIR}/event_eval/predictions.jsonl" \
    --output "${RUN_DIR}/adaptation_gain.json"
fi

if [[ -f "${RUN_DIR}/original_eval/metrics.json" ]]; then
  if [[ ! -f "${RUN_DIR}/original_vs_source.json" ]]; then
    "${PYTHON_BIN}" scripts/compare_predictions.py \
      --baseline "${RUN_DIR}/original_eval/predictions.jsonl" \
      --adapted "${RUN_DIR}/source_eval/predictions.jsonl" \
      --output "${RUN_DIR}/original_vs_source.json"
  fi

  if [[ -f "${RUN_DIR}/event_eval/metrics.json" && ! -f "${RUN_DIR}/original_vs_event.json" ]]; then
    "${PYTHON_BIN}" scripts/compare_predictions.py \
      --baseline "${RUN_DIR}/original_eval/predictions.jsonl" \
      --adapted "${RUN_DIR}/event_eval/predictions.jsonl" \
      --output "${RUN_DIR}/original_vs_event.json"
  fi
fi
