#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
action="${1:-plan}"
if (( $# > 0 )); then shift; fi
config="${APERTURE_CONFIG:-configs/aperture_final_round.json}"
if [[ "$action" == "run" ]]; then
  gpus="${1:-0}"
  if (( $# > 0 )); then shift; fi
  exec python3 scripts/run_aperture_final.py run --config "$config" --gpus "$gpus" "$@"
else
  exec python3 scripts/run_aperture_final.py "$action" --config "$config" "$@"
fi
