#!/usr/bin/env python3
"""Support-only CV for Guardian Ours, followed by clean query evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from eventttt.io import read_task_samples
from run_guardian_failure_suite import run_method


GRID = (
    {"name": "r16_a3", "rank": 16, "alpha": 3.0, "l2": 1e-3, "layers": (14, 27)},
    {"name": "r16_a2", "rank": 16, "alpha": 2.0, "l2": 1e-3, "layers": (14, 27)},
    {"name": "r16_a4", "rank": 16, "alpha": 4.0, "l2": 1e-3, "layers": (14, 27)},
    {"name": "r16_a1", "rank": 16, "alpha": 1.0, "l2": 1e-3, "layers": (14, 27)},
    {"name": "r32_a1", "rank": 32, "alpha": 1.0, "l2": 1e-3, "layers": (14, 27)},
    {"name": "r32_a1_l2e2", "rank": 32, "alpha": 1.0, "l2": 1e-2, "layers": (14, 27)},
)


def _support_holdout(rows, seed: int):
    rng = np.random.default_rng(9100 + seed)
    val = []
    train = []
    for label in ("success", "failure"):
        indices = [i for i, row in enumerate(rows) if row.label == label]
        if len(indices) < 4:
            raise ValueError(f"support class {label} has only {len(indices)} rows")
        chosen = set(int(i) for i in rng.permutation(indices)[:2])
        val.extend(rows[i] for i in sorted(chosen))
        train.extend(rows[i] for i in indices if i not in chosen)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", default="data/prepared/guardian_execution")
    parser.add_argument("--cv-root", default="runs/guardian_failure_tune_cv")
    parser.add_argument("--final-root", default="runs/guardian_failure_tuned")
    parser.add_argument("--family", default="internvl3")
    parser.add_argument("--model-id", default="OpenGVLab/InternVL3-8B-Instruct")
    parser.add_argument("--splits", nargs="+", default=["ur5fail_test", "robofail", "robovqa"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=None,
                        help="override with rank-16 alpha values for a focused sweep")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=None,
                        help="override the L2 penalty for the rank-16 focused sweep")
    args = parser.parse_args()
    if args.alpha_grid is None:
        grid = GRID
    else:
        l2 = 1e-3 if args.l2 is None else args.l2
        grid = tuple({"name": f"r16_a{str(alpha).replace('.', 'p')}",
                      "rank": 16, "alpha": alpha, "l2": l2, "layers": (14, 27)}
                     for alpha in args.alpha_grid)
    cv_root, final_root = Path(args.cv_root), Path(args.final_root)
    records = []
    for split in args.splits:
        for seed in args.seeds:
            fold = Path(args.manifest_root) / split / f"seed_{seed}"
            support = read_task_samples(fold / "support.jsonl")
            train, val = _support_holdout(support, seed)
            for cfg in grid:
                out = cv_root / cfg["name"] / split / f"seed_{seed}"
                metrics = run_method(
                    "ours", args.family, args.model_id, train, val, out,
                    rank=cfg["rank"], alpha=cfg["alpha"], steps=args.steps,
                    learning_rate=args.learning_rate, l2=cfg["l2"], layers=cfg["layers"],
                )
                records.append({"split": split, "seed": seed, **cfg, "metrics": metrics})
                print(json.dumps({"split": split, "seed": seed, "config": cfg["name"],
                                  "val_macro_f1": metrics["macro_f1"]}))

    grouped = defaultdict(list)
    for row in records:
        grouped[row["name"]].append(row)
    summary = []
    for name, rows in sorted(grouped.items()):
        summary.append({
            "name": name,
            "rank": rows[0]["rank"], "alpha": rows[0]["alpha"], "l2": rows[0]["l2"],
            "layers": rows[0]["layers"], "folds": len(rows),
            "mean_val_macro_f1": float(np.mean([r["metrics"]["macro_f1"] for r in rows])),
            "mean_val_balanced_accuracy": float(np.mean([r["metrics"]["balanced_accuracy"] for r in rows])),
            "fold_metrics": [{"split": r["split"], "seed": r["seed"],
                              "macro_f1": r["metrics"]["macro_f1"],
                              "balanced_accuracy": r["metrics"]["balanced_accuracy"]}
                             for r in rows],
        })
    selected = max(summary, key=lambda row: (row["mean_val_macro_f1"], row["mean_val_balanced_accuracy"]))
    final_records = []
    for split in args.splits:
        for seed in args.seeds:
            fold = Path(args.manifest_root) / split / f"seed_{seed}"
            support = read_task_samples(fold / "support.jsonl")
            query = read_task_samples(fold / "query.jsonl")
            out = final_root / split / f"seed_{seed}" / "ours"
            metrics = run_method(
                "ours", args.family, args.model_id, support, query, out,
                rank=selected["rank"], alpha=selected["alpha"], steps=args.steps,
                learning_rate=args.learning_rate, l2=selected["l2"], layers=tuple(selected["layers"]),
            )
            final_records.append({"split": split, "seed": seed, "metrics": metrics})
            print(json.dumps({"final_split": split, "seed": seed, "macro_f1": metrics["macro_f1"]}))
    payload = {"objective": "support-only internal validation Macro-F1",
               "support_holdout": "2 examples per class; 12 train / 4 validation",
               "grid": list(grid), "steps": args.steps, "learning_rate": args.learning_rate,
               "cv_summary": summary, "selected": selected,
               "final_records": final_records}
    cv_root.mkdir(parents=True, exist_ok=True)
    (cv_root / "tuning_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"selected": selected, "final_root": str(final_root)}, indent=2))


if __name__ == "__main__":
    main()
