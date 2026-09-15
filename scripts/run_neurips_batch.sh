#!/usr/bin/env bash
set -euo pipefail

# Long-running NeurIPS protocol batch for the ER- & KV-TTT path.
# For every prepared fold in data/prepared/neurips/<event>:
#   1) run_event_kv.sh (source SFT -> source_eval -> event_kv -> kv_eval -> gain)
#   2) unbalanced "natural" eval with the SAME source adapter and SAME KV state
#   3) support-budget ablations (12/48 shots) on the pivot events
#
# Waits for each fold's manifest set to be prepared before starting that fold.
# Skips any fold whose kv_gain.json is already present (resume).

PYTHON_BIN="${PYTHON_BIN:-python3}"
PREP_DIR="data/prepared/neurips"
RUNS_ROOT="runs/neurips"
BATCH_LOG="logs/neurips_batch.log"
SOURCE_STEPS="${SOURCE_STEPS:-1000}"
EVAL_D4_VIEWS="${EVAL_D4_VIEWS:-1}"
KV_RANK="${KV_RANK:-5}"
KV_STEPS="${KV_STEPS:-4}"
KV_ALPHA_MAX="${KV_ALPHA_MAX:-0.5}"
KV_COEFFICIENT_MODE="${KV_COEFFICIENT_MODE:-diagonal}"
KV_LAYERS="${KV_LAYERS:-14 27}"
PIVOT_EVENTS="${PIVOT_EVENTS:-hawaii-wildfire la_palma-volcano}"

# Guard: do not start heavy GPU work while the pre-existing libya kv_eval is
# still occupying the GPU. Wait until it writes its metrics file (max ~2h).
LIBYA_GUARD="runs/all_datasets/libya-flood/kv_eval/metrics.json"
echo "[$(date +%T)] batch start (guard libya: ${LIBYA_GUARD})" >>"${BATCH_LOG}"
if [[ -f "${LIBYA_GUARD}" ]]; then
  echo "[$(date +%T)] libya guard already complete, proceeding" >>"${BATCH_LOG}"
else
  for i in $(seq 1 360); do
    if [[ -f "${LIBYA_GUARD}" ]]; then
      echo "[$(date +%T)] libya guard satisfied after ${i} min" >>"${BATCH_LOG}"
      break
    fi
    sleep 60
  done
fi

mkdir -p "${RUNS_ROOT}"

# Generic GPU-idle gate: do not start source SFT while the previous all_datasets
# continuation job is still training/evaluating (would OOM on a 24GB GPU).
GPU_JOB_RE="python3 scripts/(train_source|evaluate|adapt_event)"
for i in $(seq 1 360); do
  if ! pgrep -f "${GPU_JOB_RE}" >/dev/null; then
    echo "[$(date +%T)] gpu idle after ${i} min" >>"${BATCH_LOG}"
    break
  fi
  sleep 60
done

for SPLIT_DIR in "${PREP_DIR}"/*/; do
  event="$(basename "${SPLIT_DIR}")"
  RUN_DIR="${RUNS_ROOT}/${event}"
  mkdir -p "${RUN_DIR}"

  # wait until the fold's manifests are ready
  for f in source_train.jsonl target_support.jsonl target_query.jsonl target_natural.jsonl; do
    for i in $(seq 1 120); do
      [[ -f "${SPLIT_DIR}/${f}" ]] && break
      sleep 30
    done
  done
  if [[ ! -f "${SPLIT_DIR}/target_natural.jsonl" ]]; then
    echo "[$(date +%T)] SKIP ${event}: prep never produced manifests" >>"${BATCH_LOG}"
    continue
  fi

  echo "[$(date +%T)] ==== FOLD ${event} start" >>"${BATCH_LOG}"

  # ---- 1. main pipeline -----------------------------------------------------
  if [[ ! -f "${RUN_DIR}/kv_gain.json" ]]; then
    (
      PYTHON_BIN="${PYTHON_BIN}" \
      SOURCE_STEPS="${SOURCE_STEPS}" \
      EVAL_D4_VIEWS="${EVAL_D4_VIEWS}" \
      SOURCE_GATE_MIN_MACRO_F1="${SOURCE_GATE_MIN_MACRO_F1:-0.2}" \
      SOURCE_GATE_MIN_CLASSES="${SOURCE_GATE_MIN_CLASSES:-2}" \
      KV_RANK="${KV_RANK}" \
      KV_STEPS="${KV_STEPS}" \
      KV_ALPHA_MAX="${KV_ALPHA_MAX}" \
      KV_COEFFICIENT_MODE="${KV_COEFFICIENT_MODE}" \
      RUN_ORIGINAL_EVAL="${RUN_ORIGINAL_EVAL:-1}" \
      KV_LAYERS="${KV_LAYERS}" \
      bash scripts/run_event_kv.sh "${SPLIT_DIR}" "${RUN_DIR}" "${SOURCE_STEPS}"
    ) >>"${RUN_DIR}/pipeline.log" 2>&1
  else
    echo "[$(date +%T)] ${event}: kv_gain exists, skipping main pipeline" >>"${BATCH_LOG}"
  fi
  [[ -f "${RUN_DIR}/source_adapter/train_summary.json" ]] || { echo "FAIL ${event} source" >>"${BATCH_LOG}"; continue; }

  # ---- 2. natural (unbalanced) evaluation, source vs source+KV -------------
  # Skipped by default (~137h across all folds). Enable with RUN_NATURAL=1.
  if [[ "${RUN_NATURAL:-0}" == "1" ]]; then
    if [[ ! -f "${RUN_DIR}/source_natural/metrics.json" ]]; then
      "${PYTHON_BIN}" scripts/evaluate.py \
        --manifest "${SPLIT_DIR}/target_natural.jsonl" \
        --adapter "${RUN_DIR}/source_adapter" \
        --d4-views "${EVAL_D4_VIEWS}" \
        --output-dir "${RUN_DIR}/source_natural" >>"${RUN_DIR}/pipeline.log" 2>&1
    fi
    if [[ ! -f "${RUN_DIR}/kv_natural/metrics.json" ]]; then
      "${PYTHON_BIN}" scripts/evaluate.py \
        --manifest "${SPLIT_DIR}/target_natural.jsonl" \
        --adapter "${RUN_DIR}/source_adapter" \
        --kv-state "${RUN_DIR}/event_kv/kv_state.pt" \
        --d4-views "${EVAL_D4_VIEWS}" \
        --output-dir "${RUN_DIR}/kv_natural" >>"${RUN_DIR}/pipeline.log" 2>&1
    fi
  fi

  # ---- 3. support-budget ablation on pivot events ---------------------------
  if [[ " ${PIVOT_EVENTS} " == *" ${event} "* ]]; then
    for shots in 12 48; do
      AD="event_kv_s${shots}"
      EV="kv_eval_s${shots}"
      if [[ ! -f "${RUN_DIR}/${AD}/kv_state.pt" ]]; then
        "${PYTHON_BIN}" scripts/adapt_event_kv.py \
          --support-manifest "${SPLIT_DIR}/support_${shots}.jsonl" \
          --source-adapter "${RUN_DIR}/source_adapter" \
          --output-dir "${RUN_DIR}/${AD}" \
          --rank "${KV_RANK}" --alpha-max "${KV_ALPHA_MAX}" --steps "${KV_STEPS}" \
          --coefficient-mode "${KV_COEFFICIENT_MODE}" \
          --layers ${KV_LAYERS} >>"${RUN_DIR}/pipeline.log" 2>&1
      fi
      if [[ ! -f "${RUN_DIR}/${EV}/metrics.json" ]]; then
        "${PYTHON_BIN}" scripts/evaluate.py \
          --manifest "${SPLIT_DIR}/target_query.jsonl" \
          --adapter "${RUN_DIR}/source_adapter" \
          --kv-state "${RUN_DIR}/${AD}/kv_state.pt" \
          --d4-views "${EVAL_D4_VIEWS}" \
          --output-dir "${RUN_DIR}/${EV}" >>"${RUN_DIR}/pipeline.log" 2>&1
      fi
    done
  fi

  echo "[$(date +%T)] ==== FOLD ${event} done" >>"${BATCH_LOG}"
done

echo "[$(date +%T)] neurips batch ALL DONE" >>"${BATCH_LOG}"
