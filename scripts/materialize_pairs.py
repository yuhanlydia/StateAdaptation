#!/usr/bin/env python3
"""Materialize 448px (configurable) paired crops for every sample in a manifest.

Loads each distinct (pre, post) raster pair exactly once, then writes a crop
per instance. The output manifest points at the PNG files and drops the bbox,
so later steps skip the raster/GeoTIFF decode path entirely.

Usage:
    python3 scripts/materialize_pairs.py --manifest MANIFEST \
        --output-crops DIR --output OUT.jsonl [--crop-size 448] [--margin 1.5]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from tqdm.auto import tqdm

from eventttt.io import read_samples, write_samples
from eventttt.vision import crop_pair, load_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-crops", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--margin", type=float, default=1.5)
    args = parser.parse_args()

    output = Path(args.output_crops)
    output.mkdir(parents=True, exist_ok=True)

    samples = read_samples(args.manifest)
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for sample in samples:
        groups[(sample.pre_image, sample.post_image)].append(sample)

    materialized = []
    for (pre_path, post_path), instances in tqdm(
        groups.items(), desc="materializing pairs", dynamic_ncols=True
    ):
        pre_image = load_image(pre_path)
        post_image = load_image(post_path)
        for sample in instances:
            pre, post = crop_pair(
                pre_image, post_image, sample.bbox_xyxy, margin=args.margin, size=args.crop_size
            )
            pre_out = output / f"{sample.sample_id}_pre.png"
            post_out = output / f"{sample.sample_id}_post.png"
            pre.save(pre_out)
            post.save(post_out)
            materialized.append(
                replace(
                    sample,
                    pre_image=str(pre_out.resolve()),
                    post_image=str(post_out.resolve()),
                    bbox_xyxy=None,
                )
            )
    count = write_samples(args.output, materialized)
    print(f"wrote {count} materialized samples to {args.output}")


if __name__ == "__main__":
    main()