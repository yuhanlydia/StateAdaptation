#!/usr/bin/env python3
"""Materialize leakage-safe Camelyon17-WILDS target-center support/query sets."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


LABELS = ("normal", "tumor")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wltjr1007/Camelyon17-WILDS")
    parser.add_argument("--revision", default="d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="data/prepared/camelyon17")
    parser.add_argument("--support-per-class", type=int, default=8)
    parser.add_argument("--query-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--query-seed", type=int, default=1729,
        help="Fixed seed for the shared query set; --seed changes support only.",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    # Streaming avoids downloading 455k patches when the protocol needs 316.
    stream = load_dataset(
        args.dataset, revision=args.revision, split=args.split, streaming=True
    )
    by_label: dict[int, list[dict]] = defaultdict(list)
    required = args.support_per_class + args.query_per_class
    # Deterministic reservoir sampling limits dependence on remote row ordering.
    # Reservoir and query sampling are deliberately independent of the support
    # seed. This gives all support replicates identical query IDs.
    reservoir_rng = random.Random(args.query_seed)
    seen = defaultdict(int)
    reservoir_size = max(required * 8, 2048)
    for row in stream:
        label = int(row["label"])
        seen[label] += 1
        candidate = {
            "image": row["image"],
            "label": label,
            "patient": int(row["patient"]),
            "slide": int(row["slide"]),
            "image_id": int(row["image_id"]),
        }
        bucket = by_label[label]
        if len(bucket) < reservoir_size:
            bucket.append(candidate)
        else:
            index = reservoir_rng.randrange(seen[label])
            if index < reservoir_size:
                bucket[index] = candidate
        if all(seen[k] >= reservoir_size * 4 for k in (0, 1)):
            break

    output = Path(args.output_dir) / f"seed_{args.seed}"
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifests = {"support": [], "query": []}
    selected_by_label = {label_id: rows for label_id, rows in by_label.items()}
    grouped = defaultdict(lambda: defaultdict(list))
    for label_id, rows in selected_by_label.items():
        for row in rows:
            grouped[(row["patient"], row["slide"])][label_id].append(row)
    # Freeze a balanced query first. Any patient/slide contributing a query row
    # is excluded from every support seed.
    query_rng = random.Random(args.query_seed)
    query_group_keys = list(grouped)
    query_rng.shuffle(query_group_keys)
    query_by_label = {0: [], 1: []}
    query_groups = set()
    for group in query_group_keys:
        if all(len(query_by_label[k]) >= args.query_per_class for k in (0, 1)):
            break
        contributed = False
        for label_id in (0, 1):
            need = args.query_per_class - len(query_by_label[label_id])
            selected = grouped[group][label_id][: max(0, need)]
            query_by_label[label_id].extend(selected)
            contributed = contributed or bool(selected)
        if contributed:
            query_groups.add(group)
    if any(len(query_by_label[k]) != args.query_per_class for k in (0, 1)):
        raise RuntimeError("could not construct balanced fixed query")

    support_rng = random.Random(args.seed)
    support_group_keys = [group for group in grouped if group not in query_groups]
    support_rng.shuffle(support_group_keys)
    support_by_label = {0: [], 1: []}
    support_groups = set()
    for group in support_group_keys:
        if all(len(support_by_label[k]) >= args.support_per_class for k in (0, 1)):
            break
        contributed = False
        for label_id in (0, 1):
            need = args.support_per_class - len(support_by_label[label_id])
            selected = grouped[group][label_id][: max(0, need)]
            support_by_label[label_id].extend(selected)
            contributed = contributed or bool(selected)
        if contributed:
            support_groups.add(group)
    if any(len(support_by_label[k]) != args.support_per_class for k in (0, 1)):
        raise RuntimeError("could not construct balanced group-held-out support")
    for label_id, label_name in enumerate(LABELS):
        support_rows = support_by_label[label_id]
        query_rows = query_by_label[label_id]
        for part, selected in (("support", support_rows), ("query", query_rows)):
            for row in selected:
                sample_id = f"cam17-{row['image_id']}"
                image_path = image_dir / f"{sample_id}.png"
                row["image"].save(image_path)
                manifests[part].append({
                    "sample_id": sample_id,
                    "domain_id": "hospital_2",
                    "group_id": f"patient_{row['patient']}_slide_{row['slide']}",
                    "image": str(image_path.resolve()),
                    "label": label_name,
                    "label_id": label_id,
                    "candidate_labels": list(LABELS),
                    "question": "Does the central region contain tumor tissue? Answer exactly normal or tumor.",
                    "dataset": "camelyon17-wilds",
                    "metadata": {"patient": row["patient"], "slide": row["slide"]},
                })
    # Patient/slide overlap is reported and forbidden: it is the medical analogue
    # of BRIGHT tile separation.
    support_group_ids = {r["group_id"] for r in manifests["support"]}
    query_group_ids = {r["group_id"] for r in manifests["query"]}
    if support_group_ids & query_group_ids:
        raise RuntimeError("patient/slide overlap between support and fixed query")
    if any(sum(r["label_id"] == k for r in manifests["query"]) != args.query_per_class for k in (0, 1)):
        raise RuntimeError("group-disjoint query count differs from registered target")
    for part in manifests:
        write_jsonl(output / f"{part}.jsonl", manifests[part])
    print(json.dumps({k: len(v) for k, v in manifests.items()}, indent=2))


if __name__ == "__main__":
    main()
