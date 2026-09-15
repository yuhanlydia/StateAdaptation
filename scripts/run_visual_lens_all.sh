#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION="${1:-plan}"
GPUS="${2:-0}"
case "$ACTION" in
  prepare) "$PYTHON_BIN" scripts/prepare_visual_lens_seeds.py ;;
  plan|check|run) "$PYTHON_BIN" scripts/run_visual_lens.py "$ACTION" --gpus "$GPUS" ;;
  summarize) "$PYTHON_BIN" scripts/summarize_visual_lens.py ;;
  *) echo 'Usage: bash scripts/run_visual_lens_all.sh {prepare|plan|check|run|summarize} [0,1,...]' >&2; exit 2 ;;
esac
