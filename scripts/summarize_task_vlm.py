#!/usr/bin/env python3
"""Aggregate formal Camelyon17/ManipBench task-VLM fold metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("macro_f1", "balanced_accuracy", "nll", "brier", "ece")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/task_vlm_formal")
    parser.add_argument("--output-json", default="reports/task_vlm_formal_metrics.json")
    parser.add_argument("--output-md", default="reports/task_vlm_formal_results.md")
    args = parser.parse_args()
    root = Path(args.run_root)
    rows = []
    for path in sorted(root.glob("*/**/metrics.json")):
        parts = path.relative_to(root).parts
        if len(parts) != 6:
            continue
        family, dataset, domain, seed_dir, method, _ = parts
        seed = int(seed_dir.removeprefix("seed_"))
        metrics = json.loads(path.read_text())
        rows.append({"family": family, "dataset": dataset, "domain": domain,
                     "seed": seed, "method": method, "metrics": metrics})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["dataset"], row["domain"], row["method"])].append(row)
    summaries = []
    for key in sorted(grouped):
        group = grouped[key]
        values = {metric: [float(item["metrics"][metric]) for item in group]
                  for metric in METRICS}
        summaries.append({"family": key[0], "dataset": key[1], "domain": key[2],
                          "method": key[3], "folds": len(group),
                          "seeds": sorted(item["seed"] for item in group),
                          "mean": {m: float(np.mean(v)) for m, v in values.items()},
                          "std": {m: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                                  for m, v in values.items()},
                          "fold_metrics": [{"seed": item["seed"], **{
                              m: item["metrics"][m] for m in METRICS}}
                              for item in sorted(group, key=lambda x: x["seed"])]})
    payload = {"run_root": str(root), "fold_metric_files": len(rows),
               "expected_files": 96, "complete": len(rows) == 96,
               "metrics": summaries}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Formal Camelyon17 / ManipBench task-VLM results", "",
             "Protocol: identical support/query splits, seeds, and adaptation "
             "methods across model families. Qwen3-VL uses a deterministic "
             "448px image budget (256 visual tokens); InternVL3 uses 224px "
             "(64 visual tokens) to fit the 24GB GPU. Camelyon17 hospital_2 "
             "has 3 support seeds and 300 queries; each ManipBench Q1 domain "
             "has 3 support seeds and 400 queries. Each fold evaluates Frozen, "
             "four-pass LoRA, Random-KV, and Gradient-Cov KV (Ours).", "",
             f"Completed metric files: {len(rows)}/96.", "",
             "| Family | Dataset/domain | Method | folds | Macro-F1 mean±sd | BA mean±sd | NLL mean±sd |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for item in summaries:
        mean, std = item["mean"], item["std"]
        lines.append(f"| {item['family']} | {item['dataset']}/{item['domain']} | "
                     f"{item['method']} | {item['folds']} | "
                     f"{mean['macro_f1']:.4f}±{std['macro_f1']:.4f} | "
                     f"{mean['balanced_accuracy']:.4f}±{std['balanced_accuracy']:.4f} | "
                     f"{mean['nll']:.4f}±{std['nll']:.4f} |")
    lines += ["", "Raw per-fold JSON is tracked in `reports/task_vlm_formal_metrics.json`; "
              "large predictions and adapters remain under ignored `runs/`."]
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": payload["complete"], "fold_metric_files": len(rows),
                      "output_json": args.output_json, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
