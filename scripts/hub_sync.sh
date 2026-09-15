#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-pull}"
DATASET_REPO="${EVENTTUNE_DATASET_REPO:-humanlong/EventTune-BRIGHT}"
MODEL_REPO="${EVENTTUNE_MODEL_REPO:-humanlong/EventTune-Qwen2.5-VL-7B}"
HF="${HF_CLI:-hf}"

case "${ACTION}" in
  pull)
    mkdir -p data artifacts/model
    "${HF}" download "${DATASET_REPO}" \
      --repo-type dataset \
      --include "manifests/**" "splits/**" "checksums/**" \
      --local-dir data
    "${HF}" download "${MODEL_REPO}" --repo-type model --local-dir artifacts/model
    ;;
  push-data)
    uploaded=false
    for folder in manifests splits checksums; do
      if [[ -d "data/${folder}" ]]; then
        "${HF}" upload "${DATASET_REPO}" "data/${folder}" "${folder}" --repo-type dataset
        uploaded=true
      fi
    done
    [[ "${uploaded}" == true ]] || { echo "No publishable data artifacts found" >&2; exit 1; }
    ;;
  push-model)
    test -d artifacts/model || { echo "artifacts/model does not exist" >&2; exit 1; }
    "${HF}" upload "${MODEL_REPO}" artifacts/model . --repo-type model
    ;;
  *)
    echo "usage: $0 {pull|push-data|push-model}" >&2
    exit 2
    ;;
esac
