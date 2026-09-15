#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EVENT="${EVENT:-hawaii-wildfire}"
PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"
BENCH_ROOT="${BENCH_ROOT:-runs/efficiency_benchmark/${EVENT}}"
manifest="${PREP_ROOT}/${EVENT}/target_query.jsonl"
mkdir -p "${BENCH_ROOT}"
raw="${BENCH_ROOT}/timings.tsv"
printf 'method\twall_seconds\tseconds_per_sample\tpeak_gpu_mib\n' >"${raw}"

run_one() {
  local method="$1"; shift
  local output="${BENCH_ROOT}/${method}"
  local start_ns end_ns wall peak current child
  start_ns=$(date +%s%N)
  "${PYTHON_BIN}" scripts/evaluate.py --manifest "${manifest}" --crop-size 448 \
    --d4-views 1 --output-dir "${output}" "$@" >"${BENCH_ROOT}/${method}.log" 2>&1 &
  child=$!
  peak=0
  while kill -0 "${child}" 2>/dev/null; do
    current=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F, -v pid="${child}" '$1+0 == pid {gsub(/ /,"",$2); print $2}' | head -1)
    if [[ -n "${current}" && "${current}" -gt "${peak}" ]]; then peak="${current}"; fi
    sleep 0.2
  done
  wait "${child}"
  end_ns=$(date +%s%N)
  wall=$(awk -v start="${start_ns}" -v end="${end_ns}" 'BEGIN {printf "%.6f", (end-start)/1000000000}')
  per_sample=$(awk -v wall="${wall}" 'BEGIN {printf "%.6f", wall/300}')
  printf '%s\t%s\t%s\t%s\n' "${method}" "${wall}" "${per_sample}" "${peak}" >>"${raw}"
}

run_one pure_vlm --no-lora
run_one support24_lora --adapter "${RUNS_ROOT}/${EVENT}/support24_lora"
run_one kv_full_a3 --no-lora --kv-state "${RUNS_ROOT}/${EVENT}/support24_kv_full_a3/kv_state.pt"
run_one kv_diagonal_a3 --no-lora --kv-state "${RUNS_ROOT}/${EVENT}/support24_kv_diagonal_a3/kv_state.pt"

cat "${raw}"
