#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${VIGOR_CONFIG:-configs/vigor_iclr2027.json}"
MODE="${1:-plan}"
GPUS="${2:-0}"
case "$MODE" in
  plan)
    "$PYTHON_BIN" scripts/run_vigor_suite.py --config "$CONFIG" --phases main ablation support diagnostic
    ;;
  check)
    "$PYTHON_BIN" -m pytest -q tests_vigor
    "$PYTHON_BIN" scripts/run_vigor_suite.py --config "$CONFIG" --phases main ablation support diagnostic --preflight
    ;;
  run)
    "$PYTHON_BIN" -m pytest -q tests_vigor
    "$PYTHON_BIN" scripts/run_vigor_suite.py --config "$CONFIG" --phases main ablation support --execute --gpus "$GPUS"
    "$PYTHON_BIN" scripts/summarize_vigor_suite.py --bootstrap 2000
    ;;
  run-with-oracles)
    "$PYTHON_BIN" -m pytest -q tests_vigor
    "$PYTHON_BIN" scripts/run_vigor_suite.py --config "$CONFIG" --phases main ablation support diagnostic --allow-oracle --execute --gpus "$GPUS"
    "$PYTHON_BIN" scripts/summarize_vigor_suite.py --bootstrap 2000
    ;;
  *) echo "Usage: $0 {plan|check|run|run-with-oracles} [GPU_IDS]" >&2;exit 2;;
esac
