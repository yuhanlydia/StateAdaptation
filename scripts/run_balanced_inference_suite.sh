#!/usr/bin/env bash
set -euo pipefail

# Same-event, support-24 inference adaptation suite. Every arm starts from the
# raw base VLM; no cross-event source LoRA is loaded. Evaluation is restricted
# to the balanced target_query manifest and compared with an existing raw-VLM
# original_eval on the identical sample IDs.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"
EVENTS="${EVENTS:-hawaii-wildfire libya-flood noto-earthquake turkey-earthquake}"
CROP_SIZE="${CROP_SIZE:-448}"
EVAL_D4_VIEWS="${EVAL_D4_VIEWS:-1}"
KV_RANK="${KV_RANK:-5}"
KV_STEPS="${KV_STEPS:-4}"
KV_LR="${KV_LR:-0.05}"
KV_L2="${KV_L2:-1e-3}"
KV_LAYERS="${KV_LAYERS:-14 27}"
KV_ALPHA_MAX="${KV_ALPHA_MAX:-3.0}"
UNSUPERVISED_D4_VIEWS="${UNSUPERVISED_D4_VIEWS:-2}"
RUN_DIAGONAL_ABLATION="${RUN_DIAGONAL_ABLATION:-1}"
RUN_DIAGONAL_ALPHA3_ABLATION="${RUN_DIAGONAL_ALPHA3_ABLATION:-1}"
RUN_FULL_ALPHA3_ABLATION="${RUN_FULL_ALPHA3_ABLATION:-1}"
RUN_UNSUPERVISED_FULL_ABLATION="${RUN_UNSUPERVISED_FULL_ABLATION:-1}"

mkdir -p "${RUNS_ROOT}" logs
[[ -n "${EVENTS// }" ]] || { echo "no prepared events under ${PREP_ROOT}" >&2; exit 1; }

"${PYTHON_BIN}" scripts/audit_balanced_inference_assets.py \
  --prep-root "${PREP_ROOT}" --runs-root "${RUNS_ROOT}" --events ${EVENTS}

run_kv_arm() {
  local event="$1" split_dir="$2" fold_dir="$3" name="$4" mode="$5" alpha="$6" objective="$7"
  local arm_dir eval_dir
  arm_dir="${fold_dir}/${name}"
  eval_dir="${arm_dir}/eval"
  if [[ ! -f "${arm_dir}/kv_state.pt" ]]; then
    if [[ "${objective}" == "supervised" ]]; then
      "${PYTHON_BIN}" scripts/adapt_event_kv.py \
        --support-manifest "${split_dir}/target_support.jsonl" \
        --output-dir "${arm_dir}" --crop-size "${CROP_SIZE}" \
        --rank "${KV_RANK}" --alpha-max "${alpha}" \
        --coefficient-mode "${mode}" --steps "${KV_STEPS}" \
        --learning-rate "${KV_LR}" --l2 "${KV_L2}" \
        --layers ${KV_LAYERS}
    else
      "${PYTHON_BIN}" scripts/adapt_event_kv_unsupervised.py \
        --unlabeled-manifest "${split_dir}/target_support.jsonl" \
        --output-dir "${arm_dir}" --crop-size "${CROP_SIZE}" \
        --rank "${KV_RANK}" --alpha-max "${alpha}" \
        --coefficient-mode "${mode}" --steps "${KV_STEPS}" \
        --learning-rate "${KV_LR}" --l2 "${KV_L2}" \
        --d4-views "${UNSUPERVISED_D4_VIEWS}" --layers ${KV_LAYERS}
    fi
  fi
  if [[ ! -f "${eval_dir}/metrics.json" ]]; then
    "${PYTHON_BIN}" scripts/evaluate.py \
      --manifest "${split_dir}/target_query.jsonl" --no-lora \
      --kv-state "${arm_dir}/kv_state.pt" --d4-views "${EVAL_D4_VIEWS}" \
      --crop-size "${CROP_SIZE}" --output-dir "${eval_dir}"
  fi
  if [[ ! -f "${arm_dir}/gain_vs_original.json" ]]; then
    "${PYTHON_BIN}" scripts/compare_predictions.py \
      --baseline "${fold_dir}/original_eval/predictions.jsonl" \
      --adapted "${eval_dir}/predictions.jsonl" \
      --output "${arm_dir}/gain_vs_original.json"
  fi
}

# Two passes guarantee full event coverage for the primary arms before GPU time
# is spent on ablations.
for phase in main ablations; do
  for event in ${EVENTS}; do
    split_dir="${PREP_ROOT}/${event}"
    fold_dir="${RUNS_ROOT}/${event}"
    log="${fold_dir}/balanced_inference_suite.log"
    mkdir -p "${fold_dir}"
    for required in target_support.jsonl target_query.jsonl; do
      [[ -f "${split_dir}/${required}" ]] || { echo "missing ${split_dir}/${required}" >&2; exit 1; }
    done
    [[ -f "${fold_dir}/original_eval/predictions.jsonl" ]] || {
      echo "missing existing original baseline: ${fold_dir}/original_eval/predictions.jsonl" >&2
      exit 1
    }

    {
      echo "[$(date -u +%FT%TZ)] ${event}: ${phase} start"
      if [[ "${phase}" == "main" ]]; then
        # Same-event supervised LoRA trained only on this event's labeled support24.
        if [[ ! -f "${fold_dir}/support24_lora/selection.json" ]]; then
          "${PYTHON_BIN}" scripts/adapt_event.py \
            --support-manifest "${split_dir}/target_support.jsonl" \
            --output-dir "${fold_dir}/support24_lora" --crop-size "${CROP_SIZE}"
        fi
        if [[ ! -f "${fold_dir}/support24_lora_eval/metrics.json" ]]; then
          "${PYTHON_BIN}" scripts/evaluate.py \
            --manifest "${split_dir}/target_query.jsonl" \
            --adapter "${fold_dir}/support24_lora" --d4-views "${EVAL_D4_VIEWS}" \
            --crop-size "${CROP_SIZE}" --output-dir "${fold_dir}/support24_lora_eval"
        fi
        if [[ ! -f "${fold_dir}/support24_lora/gain_vs_original.json" ]]; then
          "${PYTHON_BIN}" scripts/compare_predictions.py \
            --baseline "${fold_dir}/original_eval/predictions.jsonl" \
            --adapted "${fold_dir}/support24_lora_eval/predictions.jsonl" \
            --output "${fold_dir}/support24_lora/gain_vs_original.json"
        fi
        run_kv_arm "${event}" "${split_dir}" "${fold_dir}" \
          support24_kv_full_a3 full "${KV_ALPHA_MAX}" supervised
        if [[ "${RUN_DIAGONAL_ABLATION}" == "1" ]]; then
          run_kv_arm "${event}" "${split_dir}" "${fold_dir}" \
            support24_kv_diagonal_a3 diagonal "${KV_ALPHA_MAX}" supervised
        fi
        run_kv_arm "${event}" "${split_dir}" "${fold_dir}" \
          support24_kv_unsupervised_a3 diagonal "${KV_ALPHA_MAX}" unsupervised
      else
        if [[ "${RUN_FULL_ALPHA3_ABLATION}" == "1" ]]; then
          run_kv_arm "${event}" "${split_dir}" "${fold_dir}" \
            support24_kv_full_a0p5 full 0.5 supervised
        fi
        if [[ "${RUN_DIAGONAL_ALPHA3_ABLATION}" == "1" ]]; then
          run_kv_arm "${event}" "${split_dir}" "${fold_dir}" \
            support24_kv_diagonal_a0p5 diagonal 0.5 supervised
        fi
        if [[ "${RUN_UNSUPERVISED_FULL_ABLATION}" == "1" ]]; then
          run_kv_arm "${event}" "${split_dir}" "${fold_dir}" \
            support24_kv_unsupervised_full_a3 full "${KV_ALPHA_MAX}" unsupervised
        fi
      fi
      echo "[$(date -u +%FT%TZ)] ${event}: ${phase} done"
    } >>"${log}" 2>&1
  done
done

"${PYTHON_BIN}" scripts/summarize_balanced_inference.py \
  --runs-root "${RUNS_ROOT}" --events ${EVENTS} \
  --output-json reports/balanced_inference_results.json \
  --output-markdown reports/balanced_inference_results.md
