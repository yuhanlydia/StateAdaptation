#!/usr/bin/env bash
set -euo pipefail

# Raw-VLM baselines for events whose support is exactly 8/class and whose query
# is exactly 100/class. These predictions define the primary denominator for
# the same-event support24 inference suite.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"
EVENTS="${EVENTS:-hawaii-wildfire libya-flood noto-earthquake turkey-earthquake}"
EVAL_D4_VIEWS="${EVAL_D4_VIEWS:-1}"
CROP_SIZE="${CROP_SIZE:-448}"

for event in ${EVENTS}; do
  manifest="${PREP_ROOT}/${event}/target_query.jsonl"
  output="${RUNS_ROOT}/${event}/original_eval"
  log="${RUNS_ROOT}/${event}/original_eval.log"
  [[ -f "${manifest}" ]] || { echo "missing ${manifest}" >&2; exit 1; }
  mkdir -p "${RUNS_ROOT}/${event}"
  if [[ ! -f "${output}/metrics.json" ]]; then
    "${PYTHON_BIN}" scripts/evaluate.py \
      --manifest "${manifest}" --no-lora --d4-views "${EVAL_D4_VIEWS}" \
      --crop-size "${CROP_SIZE}" --output-dir "${output}" >>"${log}" 2>&1
  fi
done
