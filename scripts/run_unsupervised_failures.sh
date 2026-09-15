#!/usr/bin/env bash
set -euo pipefail

# Label-free counterpart of the locked supervised failure-set test.

PREP_ROOT="${PREP_ROOT:-data/prepared/neurips}"
RUNS_ROOT="${RUNS_ROOT:-runs/neurips}"
EVENTS="${EVENTS:-beirut-explosion congo-volcano haiti-earthquake}"

for event in ${EVENTS}; do
  bash scripts/run_unsupervised_kv.sh \
    "${PREP_ROOT}/${event}" \
    "${RUNS_ROOT}/${event}" \
    "event_kv_unsupervised"
done
