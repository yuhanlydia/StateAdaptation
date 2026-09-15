#!/usr/bin/env python3
"""NeurIPS-level leave-one-event-out manifests for BRIGHT.

Per target event (all 11 folds):
  * source_train:   balanced 4050 clips (1350/class) from the other events;
                    448px crops are taken at runtime from raw TIFF tiles.
  * target_support  balanced 24 (8/class) from tile-disjoint target tiles (main budget).
  * support_12/48   same protocol at 4/16 shots per class (budget ablation).
  * target_query    balanced 300 (100/class) held-out queries (MAIN balanced eval).
  * target_natural  all held-out instances in the natural (imbalanced) distribution,
                    capped at 60k per event (unbalanced evaluation set).

Randomness is seeded per event (0xFFFFFFFF AND of hash(event), matching the
existing prepare_all_events.py convention). No crops are written to disk."""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from random import Random

from eventttt.io import read_samples, write_json, write_samples

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

SOURCE_PER_CLASS = 1350               # source_train total = 4050
QUERY_PER_CLASS = 100                 # balanced main eval, up to 300 instances
SUPPORT_BUDGETS = {"12": 4, "24": 8, "48": 16}   # shots per class -> total
NATURAL_CAP = 60000                   # turkey (119k) is capped, others are full
LABEL_ORDER = ("intact", "damaged", "destroyed")


def balanced_n(samples, per_class, rng):
    by = defaultdict(list)
    for s in samples:
        by[s.label].append(s)
    for k in by:
        rng.shuffle(by[k])
    out = []
    for label in LABEL_ORDER:
        out.extend(by[label][:per_class])
    return out


def support_for(target, shots_per_class, rng):
    """Tile-disjoint support: keep adding shuffled tiles until every class has
    >= shots_per_class instances, then draw `shots_per_class` per class."""
    tiles_by = defaultdict(list)
    for s in target:
        tiles_by[s.tile_id].append(s)
    tile_ids = list(tiles_by)
    rng.shuffle(tile_ids)
    pool = []
    for t in tile_ids:
        pool.extend(tiles_by[t])
        counts = Counter(s.label for s in pool)
        if all(counts[l] >= shots_per_class for l in LABEL_ORDER):
            break
    by = defaultdict(list)
    for s in pool:
        by[s.label].append(s)
    out = []
    for label in LABEL_ORDER:
        rng.shuffle(by[label])
        out.extend(by[label][:shots_per_class])
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: prepare_neurips_splits.py <event_id> | all", file=sys.stderr)
        raise SystemExit(2)
    want = sys.argv[1]

    samples = read_samples("data/bright.jsonl")
    events = [e for e in ALL_EVENTS if want == "all" or want == e]
    if not events:
        print(f"no events selected; want={want}", file=sys.stderr)
        raise SystemExit(2)

    by_event = defaultdict(list)
    for s in samples:
        by_event[s.event_id].append(s)

    out_root = __import__("pathlib").Path("data") / "prepared" / "neurips"

    for event in events:
        if event not in by_event:
            print(f"SKIP {event}: not in BRIGHT", file=sys.stderr)
            continue
        others = [s for s in samples if s.event_id != event]
        target = by_event[event]
        ev_rng = Random(hash(event) & 0xFFFFFFFF)

        source = balanced_n(others, SOURCE_PER_CLASS, ev_rng)

        supports = {
            name: support_for(target, shots, ev_rng)
            for name, shots in SUPPORT_BUDGETS.items()
        }
        support_tiles = {s.tile_id for s in supports["24"]}
        held_out = [s for s in target if s.tile_id not in support_tiles]
        balanced = balanced_n(held_out, QUERY_PER_CLASS, ev_rng)
        natural = held_out[:NATURAL_CAP]

        ev_dir = out_root / event
        ev_dir.mkdir(parents=True, exist_ok=True)
        write_samples(ev_dir / "source_train.jsonl", source)
        write_samples(ev_dir / "target_support.jsonl", supports["24"])
        write_samples(ev_dir / "support_12.jsonl", supports["12"])
        write_samples(ev_dir / "support_48.jsonl", supports["48"])
        write_samples(ev_dir / "target_query.jsonl", balanced)
        write_samples(ev_dir / "target_natural.jsonl", natural)
        write_json(
            ev_dir / "dataset_summary.json",
            {
                "event": event,
                "source": len(source),
                "support_12": len(supports["12"]),
                "support_24": len(supports["24"]),
                "support_48": len(supports["48"]),
                "query_balanced": len(balanced),
                "query_natural": len(natural),
                "support_labels_24": dict(Counter(s.label for s in supports["24"])),
                "natural_labels": dict(Counter(s.label for s in natural)),
            },
        )
        print("PREPARED", event, flush=True)


if __name__ == "__main__":
    main()