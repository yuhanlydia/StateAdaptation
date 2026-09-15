#!/usr/bin/env python3
"""Prepare Guardian execution verification as leakage-safe binary task folds.

The released metadata contains before/after views and an ``execution_reward``
label.  We create a single composite image (before on top, after on bottom)
so the existing frozen/LoRA/KV candidate-likelihood backend can be reused
without pretending that a single pre-action frame is sufficient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SPLITS = ("ur5fail_test", "robofail", "robovqa")
CANDIDATES = ["success", "failure"]


def _local_image(raw: str, dataset_root: Path) -> Path:
    marker = "data/failure_forge/data/"
    if marker in raw:
        relative = raw.split(marker, 1)[1]
        split_name = dataset_root.name.removesuffix("_dataset")
        expected = f"{split_name}_dataset/"
        if relative.startswith(expected):
            relative = relative[len(expected):]
        return dataset_root / relative
    path = Path(raw)
    return path if path.is_absolute() else dataset_root / path


def _make_composite(before: Path, after: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(before).convert("RGB") as first, Image.open(after).convert("RGB") as second:
        canvas = Image.new("RGB", (448, 448), (0, 0, 0))
        for image, top in ((first, 0), (second, 224)):
            fitted = ImageOps.contain(image, (448, 224), method=Image.Resampling.LANCZOS)
            left = (448 - fitted.width) // 2
            canvas.paste(fitted, (left, top + (224 - fitted.height) // 2))
        canvas.save(output, quality=95)


def _convert_row(row: dict, split: str, dataset_root: Path, composite_root: Path, index: int) -> dict:
    raw_images = list(row["images"])
    if len(raw_images) < 2:
        raise ValueError(f"{split} row {index} has fewer than two execution views")
    before_index = 0
    after_index = len(raw_images) // 2 if len(raw_images) > 2 else 1
    before = _local_image(raw_images[before_index], dataset_root)
    after = _local_image(raw_images[after_index], dataset_root)
    if not before.is_file() or not after.is_file():
        raise FileNotFoundError(f"missing views for {split} row {index}: {before}, {after}")
    sample_id = f"{split}-{row.get('episode_id', row.get('taskvar', index))}-{index}"
    composite = composite_root / f"{index:05d}_{row.get('episode_id', row.get('taskvar', index))}.jpg"
    _make_composite(before, after, composite)
    label = "success" if int(row["execution_reward"]) == 1 else "failure"
    instruction = row.get("task_instruction", "")
    subtask = row.get("detailed_subtask_name", "")
    question = (
        "The top half of the image is the observation before the manipulation and the "
        "bottom half is the observation after it. Given the task instruction and the "
        "before/after visual evidence, determine whether execution succeeded. "
        "Task instruction: " + instruction + "."
    )
    if subtask:
        question += " The verified subtask is: " + subtask + "."
    question += " Answer exactly one label: success or failure."
    return {
        "sample_id": sample_id,
        "domain_id": split,
        "group_id": str(row.get("episode_id", row.get("taskvar", index))),
        "image": str(composite.resolve()),
        "label": label,
        "label_id": CANDIDATES.index(label),
        "question": question,
        "dataset": "guardian-execution",
        "candidate_labels": CANDIDATES,
        "metadata": {
            "source_split": split,
            "taskvar": row.get("taskvar"),
            "episode_id": row.get("episode_id"),
            "view_mode": "before_after_vertical_front_view",
            "num_source_views": len(raw_images),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/guardian_ood")
    parser.add_argument("--output-root", default="data/prepared/guardian_execution")
    parser.add_argument("--support-per-class", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = parser.parse_args()
    data_root, output_root = Path(args.data_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"support_per_class": args.support_per_class, "seeds": args.seeds, "splits": {}}
    for split in args.splits:
        dataset_root = data_root / f"{split}_dataset"
        rows = [json.loads(line) for line in (dataset_root / "metadata_execution.jsonl").read_text().splitlines()]
        converted = [_convert_row(row, split, dataset_root, output_root / split / "composites", i)
                     for i, row in enumerate(rows)]
        by_label = {label: np.asarray([i for i, row in enumerate(converted) if row["label"] == label])
                    for label in CANDIDATES}
        if any(len(indices) < args.support_per_class for indices in by_label.values()):
            raise ValueError(f"{split} lacks support examples for one class: { {k: len(v) for k,v in by_label.items()} }")
        split_summary = {"total": len(converted), "labels": {k: int(len(v)) for k, v in by_label.items()}, "seeds": {}}
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            support_indices = np.concatenate([
                rng.permutation(indices)[: args.support_per_class] for indices in by_label.values()
            ])
            support_set = set(int(i) for i in support_indices)
            fold_root = output_root / split / f"seed_{seed}"
            fold_root.mkdir(parents=True, exist_ok=True)
            support = [converted[i] for i in sorted(support_set)]
            query = [row for i, row in enumerate(converted) if i not in support_set]
            (fold_root / "support.jsonl").write_text("".join(json.dumps(row) + "\n" for row in support))
            (fold_root / "query.jsonl").write_text("".join(json.dumps(row) + "\n" for row in query))
            config = {"split": split, "seed": seed, "support_per_class": args.support_per_class,
                      "support": len(support), "query": len(query), "candidate_labels": CANDIDATES,
                      "view_mode": "before_after_vertical_front_view"}
            (fold_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            split_summary["seeds"][str(seed)] = {"support": len(support), "query": len(query)}
        summary["splits"][split] = split_summary
    (output_root / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
