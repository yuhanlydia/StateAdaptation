#!/usr/bin/env bash
set -euo pipefail

# Re-test bounded full-subspace mixing on the three events where diagonal
# supervised KV-TTT previously did not improve. Reuses each fold's frozen
# source adapter and source_eval so this job does not retrain or change the
# evaluation denominator.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"
EVENTS="${EVENTS:-beirut-explosion congo-volcano haiti-earthquake}"
EVAL_D4_VIEWS="${EVAL_D4_VIEWS:-1}"
CROP_SIZE="${CROP_SIZE:-448}"
KV_RANK="${KV_RANK:-5}"
KV_ALPHA_MAX="${KV_ALPHA_MAX:-0.5}"
KV_STEPS="${KV_STEPS:-4}"
KV_LR="${KV_LR:-0.05}"
KV_L2="${KV_L2:-1e-3}"
KV_LAYERS="${KV_LAYERS:-14 27}"

for event in ${EVENTS}; do
  split_dir="${PREP_ROOT}/${event}"
  fold_dir="${RUNS_ROOT}/${event}"
  output_dir="${fold_dir}/kv_fullmix/locked_r${KV_RANK}_a${KV_ALPHA_MAX}_lr${KV_LR}"
  eval_dir="${output_dir}/eval"

  for required in \
    "${split_dir}/target_support.jsonl" \
    "${split_dir}/target_query.jsonl" \
    "${fold_dir}/source_adapter/train_summary.json" \
    "${fold_dir}/source_eval/predictions.jsonl"; do
    [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 1; }
  done

  mkdir -p "${output_dir}"
  if [[ ! -f "${output_dir}/kv_state.pt" ]]; then
    "${PYTHON_BIN}" scripts/adapt_event_kv.py \
      --support-manifest "${split_dir}/target_support.jsonl" \
      --source-adapter "${fold_dir}/source_adapter" \
      --output-dir "${output_dir}" \
      --crop-size "${CROP_SIZE}" \
      --rank "${KV_RANK}" \
      --alpha-max "${KV_ALPHA_MAX}" \
      --coefficient-mode full \
      --steps "${KV_STEPS}" \
      --learning-rate "${KV_LR}" \
      --l2 "${KV_L2}" \
      --layers ${KV_LAYERS}
  fi

  if [[ ! -f "${eval_dir}/metrics.json" ]]; then
    "${PYTHON_BIN}" scripts/evaluate.py \
      --manifest "${split_dir}/target_query.jsonl" \
      --adapter "${fold_dir}/source_adapter" \
      --kv-state "${output_dir}/kv_state.pt" \
      --d4-views "${EVAL_D4_VIEWS}" \
      --crop-size "${CROP_SIZE}" \
      --output-dir "${eval_dir}"
  fi

  if [[ ! -f "${output_dir}/gain_vs_source.json" ]]; then
    "${PYTHON_BIN}" scripts/compare_predictions.py \
      --baseline "${fold_dir}/source_eval/predictions.jsonl" \
      --adapted "${eval_dir}/predictions.jsonl" \
      --output "${output_dir}/gain_vs_source.json"
  fi
done
