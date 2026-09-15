#!/usr/bin/env python3
from __future__ import annotations

import argparse

from eventttt.datasets import adapt_bright_coco, adapt_bright_rasters, adapt_disasterm3, adapt_xbd
from eventttt.io import write_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize public disaster datasets to EventTTT JSONL")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    bright_raster = subparsers.add_parser("bright-raster")
    bright_raster.add_argument("--root", required=True)
    bright_raster.add_argument("--output", required=True)
    bright_raster.add_argument("--output-crops")
    bright_raster.add_argument("--crop-size", type=int, default=448)
    bright_raster.add_argument("--crop-margin", type=float, default=1.5)
    bright_raster.add_argument("--min-pixels", type=int, default=6)

    bright_coco = subparsers.add_parser("bright-coco")
    bright_coco.add_argument("--root", required=True)
    bright_coco.add_argument("--annotations", required=True)
    bright_coco.add_argument("--output", required=True)
    bright_coco.add_argument("--output-crops")
    bright_coco.add_argument("--crop-size", type=int, default=448)
    bright_coco.add_argument("--crop-margin", type=float, default=1.5)

    xbd = subparsers.add_parser("xbd")
    xbd.add_argument("--root", required=True)
    xbd.add_argument("--output", required=True)
    xbd.add_argument("--output-crops")
    xbd.add_argument("--crop-size", type=int, default=448)
    xbd.add_argument("--crop-margin", type=float, default=1.5)

    disaster = subparsers.add_parser("disasterm3")
    disaster.add_argument("--root", required=True)
    disaster.add_argument("--annotations", required=True)
    disaster.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.dataset == "bright-raster":
        samples = adapt_bright_rasters(
            args.root, args.output_crops, args.crop_size, args.crop_margin, args.min_pixels
        )
    elif args.dataset == "bright-coco":
        samples = adapt_bright_coco(
            args.root, args.annotations, args.output_crops, args.crop_size, args.crop_margin
        )
    elif args.dataset == "xbd":
        samples = adapt_xbd(
            args.root, args.output_crops, args.crop_size, args.crop_margin
        )
    else:
        samples = adapt_disasterm3(args.root, args.annotations)
    count = write_samples(args.output, samples)
    events = sorted({sample.event_id for sample in samples})
    print(f"wrote {count} samples across {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
