#!/usr/bin/env python3
"""Restore the fixed table manifests with portable image paths."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new data/prepared/aperture_table_v1 directory")
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output must be empty: {output}")

    image_map = json.loads((bundle / "image_map.json").read_text())
    images = {item["original_image"]: item for item in image_map["images"]}
    for item in images.values():
        image = bundle / item["release_image"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != item["file_sha256"]:
            raise ValueError(f"image checksum mismatch: {image}")

    rows_by_manifest = {}
    for source in sorted((bundle / "manifests").rglob("*.jsonl")):
        relative = source.relative_to(bundle / "manifests")
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        for row in rows:
            row["image"] = str((bundle / images[row["image"]]["release_image"]).resolve())
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
        rows_by_manifest[str(relative)] = rows

    for source in sorted((bundle / "manifests").rglob("split.json")):
        relative = source.relative_to(bundle / "manifests")
        record = json.loads(source.read_text())
        domain = relative.parts[0]
        seed = relative.parts[1]
        record["content"] = {
            "support": digest(rows_by_manifest[f"{domain}/{seed}/fit.jsonl"]),
            "calibration": digest(rows_by_manifest[f"{domain}/{seed}/calibration.jsonl"]),
            "query": digest(rows_by_manifest[f"{domain}/query.jsonl"]),
        }
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    print(f"Restored {len(rows_by_manifest)} manifests and {len(images)} verified images at {output}")


if __name__ == "__main__":
    main()
