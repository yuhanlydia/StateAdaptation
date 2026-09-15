#!/usr/bin/env python3
"""Build nested, query-tile-disjoint 12/48-shot support ablations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from eventttt.io import read_samples, write_samples


LABELS = ("intact", "damaged", "destroyed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/bright.jsonl")
    parser.add_argument("--prep-root", default="data/prepared/neurips")
    parser.add_argument("--events", nargs="+", required=True)
    args = parser.parse_args()

    all_samples = read_samples(args.data)
    by_event = defaultdict(list)
    for sample in all_samples:
        by_event[sample.event_id].append(sample)

    for event in args.events:
        event_dir = Path(args.prep_root) / event
        support24 = read_samples(event_dir / "target_support.jsonl")
        query = read_samples(event_dir / "target_query.jsonl")
        query_ids = {sample.sample_id for sample in query}
        query_tiles = {sample.tile_id for sample in query}
        support_ids = {sample.sample_id for sample in support24}

        by_label = defaultdict(list)
        for sample in support24:
            by_label[sample.label].append(sample)
        support12 = [sample for label in LABELS for sample in by_label[label][:4]]

        candidates = defaultdict(list)
        fallback = defaultdict(list)
        for sample in by_event[event]:
            if sample.sample_id in support_ids or sample.sample_id in query_ids:
                continue
            target = candidates if sample.tile_id not in query_tiles else fallback
            target[sample.label].append(sample)
        for label in LABELS:
            candidates[label].sort(key=lambda sample: sample.sample_id)
            fallback[label].sort(key=lambda sample: sample.sample_id)
            need = 16 - len(by_label[label])
            chosen = candidates[label][:need]
            chosen.extend(fallback[label][:need - len(chosen)])
            if len(chosen) < need:
                raise RuntimeError(f"{event}/{label}: need {need}, have {len(chosen)}")
            by_label[label].extend(chosen)
        support48 = [sample for label in LABELS for sample in by_label[label]]

        assert len(support12) == 12 and len(support48) == 48
        assert not ({sample.sample_id for sample in support48} & {sample.sample_id for sample in query})
        write_samples(event_dir / "support_12_strict.jsonl", support12)
        write_samples(event_dir / "support_48_strict.jsonl", support48)
        tile_overlap = len({sample.tile_id for sample in support48} & query_tiles)
        print(f"PREPARED {event}: support12=12 support48=48 query_tile_overlap={tile_overlap}", flush=True)


if __name__ == "__main__":
    main()
