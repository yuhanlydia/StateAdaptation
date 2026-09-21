#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
action="${1:?Usage: run_aperture_medical_multimodel.sh plan|prepare|check|run|summarize [GPU_IDS] [options]}"; shift
if [[ "$action" == run ]]; then
  gpus=0
  if [[ $# -gt 0 && "$1" != --* ]]; then gpus="$1"; shift; fi
  exec python3 scripts/run_aperture_medical_multimodel.py run --gpus "$gpus" "$@"
fi
exec python3 scripts/run_aperture_medical_multimodel.py "$action" "$@"
