#!/usr/bin/env python3
"""Create deterministic balanced support24 resamples disjoint from query tiles."""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from eventttt.io import read_samples, write_samples

LABELS = ("intact", "damaged", "destroyed")


def order_key(event: str, seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{event}:{seed}:{sample_id}".encode()).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/bright.jsonl")
    parser.add_argument("--prep-root", default="data/prepared/neurips")
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    samples = read_samples(args.data)
    by_event = defaultdict(list)
    for sample in samples:
        by_event[sample.event_id].append(sample)
    for event in args.events:
        event_dir = Path(args.prep_root) / event
        query = read_samples(event_dir / "target_query.jsonl")
        query_tiles = {sample.tile_id for sample in query}
        eligible = [sample for sample in by_event[event] if sample.tile_id not in query_tiles]
        for seed in args.seeds:
            by_label = defaultdict(list)
            for sample in eligible:
                by_label[sample.label].append(sample)
            chosen = []
            for label in LABELS:
                rows = sorted(by_label[label], key=lambda s: order_key(event, seed, s.sample_id))
                if len(rows) < 8:
                    raise RuntimeError(f"{event}/{label}: fewer than 8 query-tile-disjoint samples")
                chosen.extend(rows[:8])
            write_samples(event_dir / f"target_support_seed{seed}.jsonl", chosen)
            print(f"PREPARED {event} seed={seed} n={len(chosen)}", flush=True)


if __name__ == "__main__":
    main()
