#!/usr/bin/env bash
set -euo pipefail

# Complete label-free KV-TTT experiment for one prepared fold. The manifest is
# used only for image paths/sample metadata; ground-truth labels are never read
# by the adaptation objective. Evaluation remains labeled and held out.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SPLIT_DIR FOLD_RUN_DIR [OUTPUT_NAME]" >&2
  exit 2
fi

SPLIT_DIR="$1"
FOLD_RUN_DIR="$2"
OUTPUT_NAME="${3:-event_kv_unsupervised}"
OUTPUT_DIR="${FOLD_RUN_DIR}/${OUTPUT_NAME}"
EVAL_DIR="${OUTPUT_DIR}/eval"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UNLABELED_MANIFEST="${UNLABELED_MANIFEST:-${SPLIT_DIR}/target_support.jsonl}"
QUERY_MANIFEST="${QUERY_MANIFEST:-${SPLIT_DIR}/target_query.jsonl}"
CROP_SIZE="${CROP_SIZE:-448}"
EVAL_D4_VIEWS="${EVAL_D4_VIEWS:-1}"
KV_D4_VIEWS="${KV_D4_VIEWS:-2}"
KV_RANK="${KV_RANK:-5}"
KV_ALPHA_MAX="${KV_ALPHA_MAX:-0.5}"
KV_COEFFICIENT_MODE="${KV_COEFFICIENT_MODE:-diagonal}"
KV_STEPS="${KV_STEPS:-4}"
KV_LR="${KV_LR:-0.05}"
KV_L2="${KV_L2:-1e-3}"
KV_LAYERS="${KV_LAYERS:-}"

for required in \
  "${UNLABELED_MANIFEST}" \
  "${QUERY_MANIFEST}" \
  "${FOLD_RUN_DIR}/source_adapter/train_summary.json" \
  "${FOLD_RUN_DIR}/source_eval/predictions.jsonl"; do
  [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
adapt_args=(
  --unlabeled-manifest "${UNLABELED_MANIFEST}"
  --source-adapter "${FOLD_RUN_DIR}/source_adapter"
  --output-dir "${OUTPUT_DIR}"
  --rank "${KV_RANK}"
  --alpha-max "${KV_ALPHA_MAX}"
  --coefficient-mode "${KV_COEFFICIENT_MODE}"
  --steps "${KV_STEPS}"
  --learning-rate "${KV_LR}"
  --l2 "${KV_L2}"
  --d4-views "${KV_D4_VIEWS}"
  --crop-size "${CROP_SIZE}"
)
if [[ -n "${KV_LAYERS}" ]]; then
  # shellcheck disable=SC2206
  layer_args=(${KV_LAYERS})
  adapt_args+=(--layers "${layer_args[@]}")
fi

if [[ ! -f "${OUTPUT_DIR}/kv_state.pt" ]]; then
  "${PYTHON_BIN}" scripts/adapt_event_kv_unsupervised.py "${adapt_args[@]}"
fi

if [[ ! -f "${EVAL_DIR}/metrics.json" ]]; then
  "${PYTHON_BIN}" scripts/evaluate.py \
    --manifest "${QUERY_MANIFEST}" \
    --adapter "${FOLD_RUN_DIR}/source_adapter" \
    --kv-state "${OUTPUT_DIR}/kv_state.pt" \
    --d4-views "${EVAL_D4_VIEWS}" \
    --crop-size "${CROP_SIZE}" \
    --output-dir "${EVAL_DIR}"
fi

if [[ ! -f "${OUTPUT_DIR}/gain_vs_source.json" ]]; then
  "${PYTHON_BIN}" scripts/compare_predictions.py \
    --baseline "${FOLD_RUN_DIR}/source_eval/predictions.jsonl" \
    --adapted "${EVAL_DIR}/predictions.jsonl" \
    --output "${OUTPUT_DIR}/gain_vs_source.json"
fi
