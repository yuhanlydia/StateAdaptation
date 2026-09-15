#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np

from eventttt.io import iter_jsonl, write_json
from eventttt.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure adaptation gain against source-only predictions")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--adapted", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = {row["sample_id"]: row for row in iter_jsonl(args.baseline)}
    adapted = {row["sample_id"]: row for row in iter_jsonl(args.adapted)}
    if baseline.keys() != adapted.keys():
        raise ValueError("Baseline and adapted predictions must cover identical sample_ids")
    ordered = sorted(baseline)
    truth = [baseline[key]["label_id"] for key in ordered]
    base_metrics = classification_metrics(
        truth, np.asarray([baseline[key]["probabilities"] for key in ordered])
    )
    adapted_metrics = classification_metrics(
        truth, np.asarray([adapted[key]["probabilities"] for key in ordered])
    )
    result = {
        "count": len(ordered),
        "baseline": base_metrics,
        "adapted": adapted_metrics,
        "gain": {
            name: adapted_metrics[name] - base_metrics[name]
            for name in ("macro_f1", "balanced_accuracy", "quadratic_weighted_kappa")
        },
        "error_reduction": {
            name: base_metrics[name] - adapted_metrics[name]
            for name in ("ordinal_mae", "nll", "brier", "ece")
        },
    }
    write_json(args.output, result)
    print(json.dumps(result["gain"], indent=2))


if __name__ == "__main__":
    main()
