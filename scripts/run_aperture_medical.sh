#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
exec python3 scripts/run_aperture_medical.py "$@"
