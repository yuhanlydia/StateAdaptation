#!/usr/bin/env python3
"""Prepare leave-one-event-out manifest+crops for all 10 BRIGHT events.

For every target event:
  * source: balanced 450-clip (150/class) sampled from the other 9 events,
    with support+query reserved; shuffled with the event seed.
  * support: 4 shots/class (12) from target, tiles disjoint from query.
  * query: balanced 300 (100/class) from remaining target tiles.
Materialised 448px crops are written per event and the resulting manifests
point at the PNG files."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from random import Random

from eventttt.io import read_samples, write_json, write_samples
from eventttt.splits import make_event_split

ALL_EVENTS = [
    "bata-explosion",
    "beirut-explosion",
    "congo-volcano",
    "haiti-earthquake",
    "hawaii-wildfire",
    "la_palma-volcano",
    "libya-flood",
    "marshall-wildfire",
    "morocco-earthquake",
    "noto-earthquake",
    "turkey-earthquake",
]

SOURCE_PER_EVENT = 450
QUERY_PER_EVENT = 300
SHOTS_PER_CLASS = 4
CROP_SIZE = 448
CROP_MARGIN = 1.5


def balanced_cap(samples, target, per_class, rng):
    by = defaultdict(list)
    for s in samples:
        by[s.label].append(s)
    for k in by:
        rng.shuffle(by[k])
    kept = []
    for label in ("intact", "damaged", "destroyed"):
        kept.extend(by[label][:per_class])
    return kept


def main():
    root = Path("data")
    out_root = root / "prepared" / "all"
    out_root.mkdir(parents=True, exist_ok=True)
    samples = read_samples(root / "bright.jsonl")
    rng = Random(0)
    for event in sorted(set(s.event_id for s in samples)):
        if event != sys.argv[1]:
            continue
        others = [s for s in samples if s.event_id != event]
        target = [s for s in samples if s.event_id == event]
        ev_rng = Random(hash(event) & 0xFFFFFFFF)
        # source: balanced from others
        src = balanced_cap(others, event, SOURCE_PER_EVENT // 3, ev_rng)
        # support/query via leave-one-event-out tile-disjoint protocol
        tiles_by = defaultdict(list)
        for s in target:
            tiles_by[s.tile_id].append(s)
        tile_ids = list(tiles_by)
        ev_rng.shuffle(tile_ids)
        support_pool = []
        for t in tile_ids:
            support_pool.extend(tiles_by[t])
            counts = Counter(s.label for s in support_pool)
            if all(counts[l] >= SHOTS_PER_CLASS for l in ("intact", "damaged", "destroyed")):
                break
        by_label = defaultdict(list)
        for s in support_pool:
            by_label[s.label].append(s)
        support = []
        for l in ("intact", "damaged", "destroyed"):
            ev_rng.shuffle(by_label[l])
            support.extend(by_label[l][:SHOTS_PER_CLASS])
        support_tiles = {s.tile_id for s in support_pool}
        query = [s for s in target if s.tile_id not in support_tiles]
        query = balanced_cap(query, event, QUERY_PER_EVENT // 3, ev_rng)

        ev_dir = out_root / event
        ev_dir.mkdir(parents=True, exist_ok=True)
        write_samples(ev_dir / "source_train.jsonl", src)
        write_samples(ev_dir / "support.jsonl", support)
        write_samples(ev_dir / "query_300.jsonl", query)

        # copy rows to prepared crops dir, materialise images
        prepared = root / "prepared" / "all" / event
        crops = prepared / "crops"
        crops.mkdir(parents=True, exist_ok=True)
        for name in ("source_train", "support", "query_300"):
            out = prepared / f"{name}.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/materialize_pairs.py",
                    "--manifest",
                    str(ev_dir / f"{name}.jsonl"),
                    "--output-crops",
                    str(crops / name),
                    "--output",
                    str(out),
                    "--crop-size",
                    str(CROP_SIZE),
                    "--margin",
                    str(CROP_MARGIN),
                ],
                check=True,
            )
        write_json(
            prepared / "dataset_summary.json",
            {
                "event": event,
                "source": len(src),
                "support": len(support),
                "query": len(query),
                "support_labels": dict(Counter(s.label for s in support)),
                "query_labels": dict(Counter(s.label for s in query)),
            },
        )
        print("PREPARED", event, flush=True)


if __name__ == "__main__":
    main()