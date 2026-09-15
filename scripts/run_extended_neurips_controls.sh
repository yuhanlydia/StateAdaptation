#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EVENTS="${EVENTS:-hawaii-wildfire libya-flood noto-earthquake turkey-earthquake}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"

run_arm() {
  local event="$1" basis="$2" seed="$3" alpha="$4" name="$5"
  local lr="${6:-0.05}"
  local split fold arm eval
  split="${PREP_ROOT}/${event}"; fold="${RUNS_ROOT}/${event}"
  arm="${fold}/${name}"; eval="${arm}/eval"; mkdir -p "${arm}"
  [[ -f "${arm}/kv_state.pt" ]] || "${PYTHON_BIN}" scripts/adapt_event_kv.py \
    --support-manifest "${split}/target_support.jsonl" --output-dir "${arm}" \
    --basis-mode "${basis}" --seed "${seed}" --rank 5 --alpha-max "${alpha}" \
    --coefficient-mode full --steps 4 --learning-rate "${lr}" --l2 1e-3 --layers 14 27
  [[ -f "${eval}/metrics.json" ]] || "${PYTHON_BIN}" scripts/evaluate.py \
    --manifest "${split}/target_query.jsonl" --no-lora --kv-state "${arm}/kv_state.pt" \
    --d4-views 1 --crop-size 448 --output-dir "${eval}"
  [[ -f "${arm}/gain_vs_original.json" ]] || "${PYTHON_BIN}" scripts/compare_predictions.py \
    --baseline "${fold}/original_eval/predictions.jsonl" \
    --adapted "${eval}/predictions.jsonl" --output "${arm}/gain_vs_original.json"
}

for event in ${EVENTS}; do
  run_arm "${event}" centered_covariance 0 3.0 control_centered_covariance_rank5_full_a3
  run_arm "${event}" random 1 3.0 control_random_seed1_rank5_full_a3
  run_arm "${event}" random 2 3.0 control_random_seed2_rank5_full_a3
  for alpha in 1 2 5 10; do
    run_arm "${event}" covariance 0 "${alpha}" "ablation_alpha${alpha}_full"
  done
  run_arm "${event}" covariance 0 3.0 ablation_lr0p01_full_a3 0.01
  run_arm "${event}" covariance 0 3.0 ablation_lr0p1_full_a3 0.1
  run_arm "${event}" covariance 0 3.0 ablation_lr0p2_full_a3 0.2
done
