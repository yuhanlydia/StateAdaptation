#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"
EVENTS="${EVENTS:-hawaii-wildfire libya-flood noto-earthquake turkey-earthquake}"
CROP_SIZE="${CROP_SIZE:-448}"
ALPHA="${ALPHA:-3.0}"
LR="${LR:-0.05}"
L2="${L2:-1e-3}"

run_arm() {
  local event="$1" manifest="$2" name="$3" rank="$4" steps="$5" layers="$6"
  local split_dir fold_dir arm_dir eval_dir
  split_dir="${PREP_ROOT}/${event}"
  fold_dir="${RUNS_ROOT}/${event}"
  arm_dir="${fold_dir}/${name}"
  eval_dir="${arm_dir}/eval"
  mkdir -p "${arm_dir}"
  if [[ ! -f "${arm_dir}/kv_state.pt" ]]; then
    "${PYTHON_BIN}" scripts/adapt_event_kv.py \
      --support-manifest "${split_dir}/${manifest}" --output-dir "${arm_dir}" \
      --crop-size "${CROP_SIZE}" --rank "${rank}" --alpha-max "${ALPHA}" \
      --coefficient-mode full --steps "${steps}" --learning-rate "${LR}" \
      --l2 "${L2}" --layers ${layers}
  fi
  if [[ ! -f "${eval_dir}/metrics.json" ]]; then
    "${PYTHON_BIN}" scripts/evaluate.py \
      --manifest "${split_dir}/target_query.jsonl" --no-lora \
      --kv-state "${arm_dir}/kv_state.pt" --d4-views 1 \
      --crop-size "${CROP_SIZE}" --output-dir "${eval_dir}"
  fi
  if [[ ! -f "${arm_dir}/gain_vs_original.json" ]]; then
    "${PYTHON_BIN}" scripts/compare_predictions.py \
      --baseline "${fold_dir}/original_eval/predictions.jsonl" \
      --adapted "${eval_dir}/predictions.jsonl" \
      --output "${arm_dir}/gain_vs_original.json"
  fi
}

"${PYTHON_BIN}" scripts/prepare_balanced_support_ablations.py \
  --prep-root "${PREP_ROOT}" --events ${EVENTS}

for event in ${EVENTS}; do
  log="${RUNS_ROOT}/${event}/supervised_ablations.log"
  {
    echo "[$(date -u +%FT%TZ)] ${event}: supervised ablations start"
    run_arm "${event}" support_12_strict.jsonl ablation_support12_full_a3 5 4 "14 27"
    run_arm "${event}" support_48_strict.jsonl ablation_support48_full_a3 5 4 "14 27"
    run_arm "${event}" target_support.jsonl ablation_layer14_full_a3 5 4 "14"
    run_arm "${event}" target_support.jsonl ablation_layer27_full_a3 5 4 "27"
    for rank in 8 16 32; do
      run_arm "${event}" target_support.jsonl "ablation_rank${rank}_full_a3" "${rank}" 4 "14 27"
    done
    for steps in 1 2 8; do
      run_arm "${event}" target_support.jsonl "ablation_steps${steps}_full_a3" 5 "${steps}" "14 27"
    done
    echo "[$(date -u +%FT%TZ)] ${event}: supervised ablations done"
  } >>"${log}" 2>&1
done
