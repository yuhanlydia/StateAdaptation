#!/usr/bin/env python3
"""Convert official simplified ManipBench Q1 folders to fixed domain splits."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


OPTIONS = ("A", "B", "C", "D")


def normalize_answer(text: str) -> str:
    hits = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", text.upper())
    if not hits:
        raise ValueError(f"cannot parse A/B/C/D answer from {text!r}")
    return hits[-1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/manipbench-simplified/Q1")
    parser.add_argument("--output-dir", default="data/prepared/manipbench_q1")
    parser.add_argument("--domains", nargs="+", default=["bridge_pick_place", "droid_pick_place", "droid_arti"])
    parser.add_argument("--support-per-class", type=int, default=8)
    parser.add_argument("--query-target", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--query-seed", type=int, default=1729,
        help="Fixed seed for the shared query set; --seed changes support only.",
    )
    args = parser.parse_args()

    for domain in args.domains:
        grouped = defaultdict(list)
        for folder in sorted((Path(args.root) / domain).glob("question_*")):
            required = [folder / "question.txt", folder / "answer.txt", folder / "image.png"]
            if not all(path.is_file() for path in required):
                continue
            answer = normalize_answer(required[1].read_text(encoding="utf-8", errors="replace"))
            grouped[answer].append({
                "sample_id": f"{domain}-{folder.name}", "domain_id": domain,
                "group_id": folder.name, "image": str(required[2].resolve()),
                "label": answer, "label_id": OPTIONS.index(answer),
                "candidate_labels": list(OPTIONS),
                "question": required[0].read_text(encoding="utf-8", errors="replace").strip(),
                "dataset": "manipbench-q1", "metadata": {"source_folder": str(folder.resolve())},
            })
        if set(grouped) != set(OPTIONS):
            raise RuntimeError(f"{domain}: expected A/B/C/D, found {sorted(grouped)}")
        support, query = [], []
        query_rng = random.Random(args.query_seed)
        support_rng = random.Random(args.seed)
        for option in OPTIONS:
            rows = list(grouped[option])
            query_rng.shuffle(rows)
            per_class = args.query_target // len(OPTIONS)
            if len(rows) < per_class + args.support_per_class:
                raise RuntimeError(
                    f"{domain}/{option}: need {per_class + args.support_per_class} rows, "
                    f"found {len(rows)}"
                )
            query.extend(rows[:per_class])
            support_pool = rows[per_class:]
            support_rng.shuffle(support_pool)
            support.extend(support_pool[: args.support_per_class])
        support_rng.shuffle(support)
        query_rng.shuffle(query)
        output = Path(args.output_dir) / domain / f"seed_{args.seed}"
        write_jsonl(output / "support.jsonl", support)
        write_jsonl(output / "query.jsonl", query)
        print(json.dumps({"domain": domain, "support": len(support), "query": len(query)}))


if __name__ == "__main__":
    main()
