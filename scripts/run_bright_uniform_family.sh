#!/usr/bin/env bash
set -euo pipefail

# Run the locked four-event BRIGHT comparison for one multimodal family.
# The model is resolved by Hugging Face's cache; no local weight copy is made.
MODEL_ID=${1:?model id required}
FAMILY=${2:?family required}
ROOT=${3:?output root required}
PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Reduce allocator fragmentation across the repeated model subprocesses.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

events=(hawaii-wildfire libya-flood noto-earthquake turkey-earthquake)
for event in "${events[@]}"; do
  support="data/prepared/neurips/${event}/target_support.jsonl"
  query="data/prepared/neurips/${event}/target_query.jsonl"
  out="${ROOT}/${event}"
  mkdir -p "${out}"

  if [[ ! -f "${out}/frozen/metrics.json" ]]; then
    "${PYTHON_BIN}" scripts/evaluate_bright_vlm.py \
      --manifest "${query}" --model-id "${MODEL_ID}" --family "${FAMILY}" \
      --output-dir "${out}/frozen" --crop-size 448
  fi
  if [[ ! -f "${out}/lora/metrics.json" ]]; then
    "${PYTHON_BIN}" scripts/adapt_bright_lora.py \
      --support-manifest "${support}" --query-manifest "${query}" \
      --model-id "${MODEL_ID}" --family "${FAMILY}" \
      --output-dir "${out}/lora" --passes 1 --grad-accum-steps 3 \
      --learning-rate 2e-4 --crop-size 448
  fi
  for method in random_kv gradient_cov_kv; do
    if [[ ! -f "${out}/${method}/metrics.json" ]]; then
      "${PYTHON_BIN}" scripts/adapt_bright_kv.py \
        --support-manifest "${support}" --query-manifest "${query}" \
        --model-id "${MODEL_ID}" --family "${FAMILY}" --method "${method}" \
        --output-dir "${out}/${method}" --rank 16 --alpha-max 3 \
        --coefficient-mode full --layers 14 27 --steps 4 --crop-size 448
    fi
  done
done
