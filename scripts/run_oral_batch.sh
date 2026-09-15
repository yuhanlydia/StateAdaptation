#!/usr/bin/env bash
# Run a registered experiment command with the six-hour wall-clock contract.
# Usage: ./scripts/run_oral_batch.sh <command> [args...]
set -u
budget_seconds=$((6 * 60 * 60))
debug_seconds=$((30 * 60))
mkdir -p logs
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="logs/oral_batch_${stamp}.log"
echo "budget_seconds=${budget_seconds} debug_seconds=${debug_seconds} log=${log}"
set +e
timeout --signal=INT --kill-after=120 "${budget_seconds}s" "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -eq 124 ]; then
  echo "PARTIAL: six-hour budget reached; artifacts are preserved"
  exit 124
fi
exit "$status"
