#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventttt.gates import assess_source_gate
from eventttt.io import iter_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stop a target evaluation when its source-domain gate fails"
    )
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-macro-f1", type=float, default=0.2)
    parser.add_argument("--minimum-predicted-classes", type=int, default=2)
    args = parser.parse_args()
    result = assess_source_gate(
        json.loads(args.metrics.read_text(encoding="utf-8")),
        iter_jsonl(args.predictions),
        args.minimum_macro_f1,
        args.minimum_predicted_classes,
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
