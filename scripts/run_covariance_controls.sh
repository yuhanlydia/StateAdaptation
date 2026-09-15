#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EVENTS="${EVENTS:-hawaii-wildfire libya-flood noto-earthquake turkey-earthquake}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"

run_arm() {
  local event="$1" mode="$2" rank="$3" name="$4"
  local split fold arm eval
  split="${PREP_ROOT}/${event}"
  fold="${RUNS_ROOT}/${event}"
  arm="${fold}/${name}"
  eval="${arm}/eval"
  mkdir -p "${arm}"
  [[ -f "${arm}/kv_state.pt" ]] || "${PYTHON_BIN}" scripts/adapt_event_kv.py \
    --support-manifest "${split}/target_support.jsonl" --output-dir "${arm}" \
    --basis-mode "${mode}" --rank "${rank}" --alpha-max 3.0 \
    --coefficient-mode full --steps 4 --learning-rate 0.05 --l2 1e-3 --layers 14 27
  [[ -f "${eval}/metrics.json" ]] || "${PYTHON_BIN}" scripts/evaluate.py \
    --manifest "${split}/target_query.jsonl" --no-lora --kv-state "${arm}/kv_state.pt" \
    --d4-views 1 --crop-size 448 --output-dir "${eval}"
  [[ -f "${arm}/gain_vs_original.json" ]] || "${PYTHON_BIN}" scripts/compare_predictions.py \
    --baseline "${fold}/original_eval/predictions.jsonl" \
    --adapted "${eval}/predictions.jsonl" --output "${arm}/gain_vs_original.json"
}

for event in ${EVENTS}; do
  run_arm "${event}" random 5 control_random_rank5_full_a3
  run_arm "${event}" mean_gradient 1 control_mean_gradient_rank1_full_a3
done
