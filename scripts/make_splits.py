#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from eventttt.io import read_samples, write_json, write_samples
from eventttt.splits import make_event_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe leave-one-event-out splits")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-events", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shots-per-class", type=int)
    group.add_argument("--support-budget", type=int)
    args = parser.parse_args()

    samples = read_samples(args.manifest)
    events = args.target_events or sorted({sample.event_id for sample in samples})
    root = Path(args.output_dir)
    for event in events:
        for seed in args.seeds:
            split = make_event_split(
                samples,
                event,
                seed,
                shots_per_class=args.shots_per_class,
                support_budget=args.support_budget,
            )
            output = root / event / f"seed_{seed}"
            write_samples(output / "source_train.jsonl", split.source)
            write_samples(output / "target_support.jsonl", split.support)
            write_samples(output / "target_query.jsonl", split.query)
            write_json(
                output / "split.json",
                {
                    "target_event": event,
                    "seed": seed,
                    "source_count": len(split.source),
                    "support_count": len(split.support),
                    "query_count": len(split.query),
                    "support_labels": dict(Counter(row.label for row in split.support)),
                    "support_tiles": sorted({row.tile_id for row in split.support}),
                    "query_tiles": sorted({row.tile_id for row in split.query}),
                },
            )
            print(output)


if __name__ == "__main__":
    main()
