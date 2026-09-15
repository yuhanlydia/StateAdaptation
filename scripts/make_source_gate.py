#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json

from eventttt.gates import select_event_label_gate
from eventttt.io import read_samples, write_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic event/label-balanced source gate"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-event-label", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    selected = select_event_label_gate(
        read_samples(args.manifest), args.per_event_label, args.seed
    )
    write_samples(args.output, selected)
    print(
        json.dumps(
            {
                "samples": len(selected),
                "events": dict(sorted(Counter(row.event_id for row in selected).items())),
                "labels": dict(sorted(Counter(row.label for row in selected).items())),
                "seed": args.seed,
                "per_event_label": args.per_event_label,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
