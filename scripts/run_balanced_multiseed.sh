#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"
EVENTS="${EVENTS:-hawaii-wildfire libya-flood noto-earthquake turkey-earthquake}"

run_arm() {
  local event="$1" seed="$2" mode="$3"
  local split_dir fold_dir arm_dir eval_dir
  split_dir="${PREP_ROOT}/${event}"; fold_dir="${RUNS_ROOT}/${event}"
  arm_dir="${fold_dir}/support24_seed${seed}_kv_${mode}_a3"; eval_dir="${arm_dir}/eval"
  mkdir -p "${arm_dir}"
  [[ -f "${arm_dir}/kv_state.pt" ]] || "${PYTHON_BIN}" scripts/adapt_event_kv.py \
    --support-manifest "${split_dir}/target_support_seed${seed}.jsonl" \
    --output-dir "${arm_dir}" --crop-size 448 --rank 5 --alpha-max 3.0 \
    --coefficient-mode "${mode}" --steps 4 --learning-rate 0.05 --l2 1e-3 --layers 14 27
  [[ -f "${eval_dir}/metrics.json" ]] || "${PYTHON_BIN}" scripts/evaluate.py \
    --manifest "${split_dir}/target_query.jsonl" --no-lora --kv-state "${arm_dir}/kv_state.pt" \
    --d4-views 1 --crop-size 448 --output-dir "${eval_dir}"
  [[ -f "${arm_dir}/gain_vs_original.json" ]] || "${PYTHON_BIN}" scripts/compare_predictions.py \
    --baseline "${fold_dir}/original_eval/predictions.jsonl" \
    --adapted "${eval_dir}/predictions.jsonl" --output "${arm_dir}/gain_vs_original.json"
}

"${PYTHON_BIN}" scripts/prepare_balanced_support_seeds.py --prep-root "${PREP_ROOT}" \
  --events ${EVENTS} --seeds 1 2
for event in ${EVENTS}; do
  for seed in 1 2; do
    for mode in full diagonal; do
      run_arm "${event}" "${seed}" "${mode}"
    done
  done
done
