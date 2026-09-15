#!/usr/bin/env python3
"""Aggregate the corrected InternVL3 task-VLM rerun."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("macro_f1", "balanced_accuracy", "nll", "brier", "ece")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/task_vlm_formal_v2")
    parser.add_argument("--output-json", default="reports/internvl3_corrected_task_metrics.json")
    parser.add_argument("--output-md", default="reports/internvl3_corrected_task_results.md")
    args = parser.parse_args()
    root = Path(args.run_root) / "internvl3"
    grouped = defaultdict(list)
    for path in sorted(root.glob("**/metrics.json")):
        dataset, domain, seed_dir, method, _ = path.relative_to(root).parts
        metrics = json.loads(path.read_text())
        grouped[(dataset, domain, method)].append(
            {"seed": int(seed_dir.removeprefix("seed_")), **metrics}
        )

    summaries = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: row["seed"])
        summaries.append({
            "family": "internvl3", "dataset": key[0], "domain": key[1],
            "method": key[2], "folds": len(rows),
            "seeds": [row["seed"] for row in rows],
            "mean": {m: float(np.mean([row[m] for row in rows])) for m in METRICS},
            "std": {m: float(np.std([row[m] for row in rows], ddof=1))
                    if len(rows) > 1 else 0.0 for m in METRICS},
            "fold_metrics": [{"seed": row["seed"], **{m: row[m] for m in METRICS}}
                             for row in rows],
        })
    payload = {"run_root": str(root), "fold_metric_files": sum(len(v) for v in grouped.values()),
               "expected_files": 48, "complete": sum(len(v) for v in grouped.values()) == 48,
               "format_fix": "InternVL3 scores the complete Answer: <label> span, matching the official prompt.",
               "metrics": summaries}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Corrected InternVL3 task-VLM results", "",
        "The previous InternVL3 ManipBench rows were invalidated: the processor asked for a raw label, while the benchmark prompt requires `Answer: <label>`. This rerun uses the matching answer format for both support adaptation and candidate likelihood scoring.",
        "", "Protocol: same support/query splits, seeds, and four methods as the Qwen3-VL run. InternVL3 uses the validated 224px/64-visual-token budget on the 24GB GPU; no query labels or query-time tuning were used. The rerun contains 3 seeds per fold.", "",
        f"Completed metric files: {payload['fold_metric_files']}/48.", "",
        "| Dataset/domain | Method | folds | Macro-F1 mean±sd | BA mean±sd | NLL mean±sd |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        mean, std = item["mean"], item["std"]
        lines.append(
            f"| {item['dataset']}/{item['domain']} | {item['method']} | {item['folds']} | "
            f"{mean['macro_f1']:.4f}±{std['macro_f1']:.4f} | "
            f"{mean['balanced_accuracy']:.4f}±{std['balanced_accuracy']:.4f} | "
            f"{mean['nll']:.4f}±{std['nll']:.4f} |"
        )
    lines += ["", "The old InternVL3 rows remain only in the historical formal report; use this report for corrected comparisons."]
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": payload["complete"], "fold_metric_files": payload["fold_metric_files"],
                      "output_json": args.output_json, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
