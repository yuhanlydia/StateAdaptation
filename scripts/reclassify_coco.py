#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventttt.io import iter_jsonl


def reclassify(input_path: Path, output_path: Path, predictions: dict) -> int:
    coco = json.loads(input_path.read_text(encoding="utf-8-sig"))
    categories = {str(row["name"]).lower(): int(row["id"]) for row in coco["categories"]}
    if not {"intact", "damaged", "destroyed"}.issubset(categories):
        raise ValueError("COCO categories must include intact, damaged, and destroyed")
    images = {int(row["id"]): row for row in coco["images"]}
    changed = 0
    for annotation in coco["annotations"]:
        tile = images[int(annotation["image_id"])].get("sample_id")
        key = f"bright-instance-{tile}-{annotation['id']}"
        if key in predictions:
            annotation["category_id"] = categories[predictions[key]["prediction"]]
            annotation["eventttt_probabilities"] = predictions[key]["probabilities"]
            changed += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco), encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace BRIGHT proposal categories with EventTTT building predictions"
    )
    parser.add_argument("--input-coco", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-coco", required=True)
    args = parser.parse_args()

    predictions = {row["sample_id"]: row for row in iter_jsonl(args.predictions)}
    input_path = Path(args.input_coco)
    output_path = Path(args.output_coco)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.json"))
        changed = sum(
            reclassify(path, output_path / path.name, predictions) for path in files
        )
    else:
        changed = reclassify(input_path, output_path, predictions)
    if changed == 0:
        raise ValueError("No prediction sample_id matched COCO annotation ids")
    print(f"reclassified {changed} annotations -> {output_path}")


if __name__ == "__main__":
    main()
