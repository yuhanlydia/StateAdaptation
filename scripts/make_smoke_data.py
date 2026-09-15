#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from eventttt.io import write_samples
from eventttt.schemas import DAMAGE_LABELS, Sample


def scene(label_id: int, post: bool, event_index: int) -> Image.Image:
    background = (40 + 25 * event_index, 70, 90)
    image = Image.new("RGB", (96, 96), background)
    draw = ImageDraw.Draw(image)
    if not post or label_id == 0:
        draw.rectangle((24, 24, 72, 72), fill=(190, 180, 150), outline="white", width=3)
        draw.line((24, 48, 72, 48), fill=(100, 90, 70), width=2)
    elif label_id == 1:
        draw.polygon([(24, 30), (45, 24), (72, 37), (66, 72), (29, 67)], fill=(145, 130, 110))
        draw.line((32, 31, 61, 66), fill=(25, 25, 25), width=5)
    else:
        draw.rectangle((25, 60, 70, 72), fill=(95, 85, 75))
        draw.polygon([(18, 64), (35, 38), (51, 62), (64, 30), (77, 68)], fill=(80, 75, 70))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/smoke")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    samples = []
    for event_index, event in enumerate(("flood-alpha", "quake-beta", "fire-gamma")):
        for tile_index in range(6):
            tile = f"{event}_{tile_index:08d}"
            for label_id, label in enumerate(DAMAGE_LABELS):
                sample_id = f"smoke-{tile}-{label}"
                pre = images / f"{sample_id}_pre.png"
                post = images / f"{sample_id}_post.png"
                scene(label_id, False, event_index).save(pre)
                scene(label_id, True, event_index).save(post)
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        event_id=event,
                        tile_id=tile,
                        pre_image=str(pre),
                        post_image=str(post),
                        label=label,
                        label_id=label_id,
                        dataset="smoke",
                    )
                )
    output = root / "manifest.jsonl"
    write_samples(output, samples)
    print(f"wrote {len(samples)} examples to {output}")


if __name__ == "__main__":
    main()
