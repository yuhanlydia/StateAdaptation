#!/usr/bin/env python3
"""Select balanced, query-tile-disjoint support with visual diversity."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from eventttt.io import read_samples, write_samples
from eventttt.schemas import DAMAGE_LABELS, Sample
from eventttt.vision import crop_pair, load_image


def descriptor(sample: Sample, size: int) -> np.ndarray:
    pre, post = crop_pair(
        load_image(sample.pre_image), load_image(sample.post_image),
        sample.bbox_xyxy, size=size,
    )
    arrays = []
    for image in (pre, post):
        x = np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        arrays.append(x)
    pre_x, post_x = arrays
    values: list[float] = []
    for x in (pre_x, post_x, np.abs(post_x - pre_x)):
        values.extend(x.mean((0, 1)).tolist())
        values.extend(x.std((0, 1)).tolist())
        gray = x.mean(2)
        values.extend(np.quantile(gray, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist())
    x1, y1, x2, y2 = sample.bbox_xyxy
    values.extend([np.log1p(max(0, x2 - x1) * max(0, y2 - y1)), (x2 - x1) / max(1, y2 - y1)])
    return np.asarray(values, dtype=np.float32)


def representative_facility_location(rows: list[Sample], count: int, size: int) -> list[Sample]:
    """Greedy medoid coverage after robust bbox-quality filtering.

    Unlike farthest-point sampling, facility location rewards candidates that
    represent many pool members and therefore does not deliberately select
    visual outliers. Tile coverage is a small bonus rather than a hard rule.
    """
    areas = np.asarray([
        max(0, row.bbox_xyxy[2] - row.bbox_xyxy[0])
        * max(0, row.bbox_xyxy[3] - row.bbox_xyxy[1]) for row in rows
    ])
    low, high = np.quantile(areas, [0.1, 0.9])
    quality = [
        i for i, row in enumerate(rows)
        if low <= areas[i] <= high
        and row.bbox_xyxy[2] - row.bbox_xyxy[0] >= 16
        and row.bbox_xyxy[3] - row.bbox_xyxy[1] >= 16
    ]
    if len(quality) < count:
        quality = list(range(len(rows)))
    rows = [rows[i] for i in quality]
    features = np.stack([descriptor(row, size) for row in rows])
    median = np.median(features, axis=0)
    scale = (np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0)).clip(1e-6)
    features = np.clip((features - median) / scale, -5, 5)
    pairwise = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    selected = [int(np.argmin(pairwise.sum(axis=1)))]
    nearest = pairwise[:, selected[0]].copy()
    selected_tiles = {rows[selected[0]].tile_id}
    tile_counts = defaultdict(int)
    tile_counts[rows[selected[0]].tile_id] = 1
    population = defaultdict(int)
    for row in rows:
        population[row.tile_id] += 1
    dominant = max(population.values())
    max_per_tile = max((count + 1) // 2, count - (len(rows) - dominant))
    while len(selected) < count:
        candidates = [
            i for i, row in enumerate(rows)
            if i not in selected and tile_counts[row.tile_id] < max_per_tile
        ]
        if not candidates:
            candidates = [i for i in range(len(rows)) if i not in selected]
        base = float(nearest.sum()) / max(1, count)
        def utility(i: int) -> tuple[float, str]:
            reduction = float((nearest - np.minimum(nearest, pairwise[:, i])).sum())
            tile_bonus = 0.05 * base if rows[i].tile_id not in selected_tiles else 0.0
            return reduction + tile_bonus, rows[i].sample_id
        choice = max(candidates, key=utility)
        selected.append(choice)
        selected_tiles.add(rows[choice].tile_id)
        tile_counts[rows[choice].tile_id] += 1
        nearest = np.minimum(nearest, pairwise[:, choice])
    return [rows[i] for i in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/bright.jsonl")
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shots-per-class", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=448)
    args = parser.parse_args()

    query_tiles = {row.tile_id for row in read_samples(args.query_manifest)}
    eligible = [
        row for row in read_samples(args.data)
        if row.event_id == args.event and row.tile_id not in query_tiles
    ]
    by_label: dict[str, list[Sample]] = defaultdict(list)
    for row in eligible:
        by_label[row.label].append(row)
    chosen = []
    summary = {}
    for label in DAMAGE_LABELS:
        rows = sorted(by_label[label], key=lambda row: row.sample_id)
        if len(rows) < args.shots_per_class:
            raise RuntimeError(f"{label}: need {args.shots_per_class}, found {len(rows)}")
        selected = representative_facility_location(rows, args.shots_per_class, args.crop_size)
        chosen.extend(selected)
        summary[label] = {
            "samples": len(selected),
            "tiles": len({row.tile_id for row in selected}),
            "tile_ids": sorted({row.tile_id for row in selected}),
        }
    write_samples(args.output, chosen)
    report = Path(args.output).with_suffix(".selection.json")
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
