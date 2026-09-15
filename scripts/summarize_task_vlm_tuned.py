#!/usr/bin/env python3
"""Summarize support-tuned Qwen3-VL Ours against the formal baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = [
    ("camelyon17", "hospital_2"),
    ("manipbench_q1", "bridge_pick_place"),
    ("manipbench_q1", "droid_arti"),
    ("manipbench_q1", "droid_pick_place"),
]


def mean_sd(values):
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned-root", default="runs/task_vlm_tuned/qwen3_vl")
    parser.add_argument("--formal-json", default="reports/task_vlm_formal_metrics.json")
    parser.add_argument("--output-json", default="reports/qwen3_vl_tuned_task_metrics.json")
    parser.add_argument("--output-md", default="reports/qwen3_vl_tuned_task_results.md")
    args = parser.parse_args()
    formal = json.loads(Path(args.formal_json).read_text())
    formal_rows = {(row["dataset"], row["domain"], row["method"]): row
                   for row in formal["metrics"] if row["family"] == "qwen3_vl"}
    rows = []
    for dataset, domain in DOMAINS:
        tuned_paths = sorted(Path(args.tuned_root, dataset, domain).glob("seed_*/ours/metrics.json"))
        if len(tuned_paths) != 3:
            raise RuntimeError(f"expected 3 tuned folds for {dataset}/{domain}, found {len(tuned_paths)}")
        tuned_metrics = [json.loads(path.read_text()) for path in tuned_paths]
        configs = [json.loads((path.parent.parent / "config.json").read_text()) for path in tuned_paths]
        config = {key: configs[0][key] for key in
                  ("kv_rank", "kv_layers", "kv_steps", "kv_alpha_max", "kv_learning_rate", "kv_l2")}
        tuned = {metric: mean_sd([item[metric] for item in tuned_metrics])
                 for metric in ("macro_f1", "balanced_accuracy", "nll", "brier", "ece")}
        rows.append({"family": "qwen3_vl", "dataset": dataset, "domain": domain,
                     "config": config, "tuned_ours": tuned,
                     "formal": {method: formal_rows[(dataset, domain, method)]
                                for method in ("frozen", "lora", "ours", "random_kv")}})
    payload = {"protocol": "support-only tuning; query never used for selection",
               "folds": 12, "rows": rows}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Qwen3-VL support-tuned task KV results", "",
        "Hyperparameters were selected using only a stratified 25% holdout "
        "from each support set. Query labels were not read during selection. "
        "The formal frozen/LoRA/random-KV rows are unchanged baselines.", "",
        "| Dataset/domain | tuned KV config (alpha/steps/lr) | Frozen F1 | LoRA F1 | Original Ours F1 | Tuned Ours F1 | Δ tuned-original |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        f = row["formal"]
        tuned = row["tuned_ours"]["macro_f1"]["mean"]
        original = f["ours"]["mean"]["macro_f1"]
        c = row["config"]
        lines.append(
            f"| {row['dataset']}/{row['domain']} | {c['kv_alpha_max']}/{c['kv_steps']}/{c['kv_learning_rate']} | "
            f"{f['frozen']['mean']['macro_f1']:.4f} | {f['lora']['mean']['macro_f1']:.4f} | "
            f"{original:.4f} | {tuned:.4f}±{row['tuned_ours']['macro_f1']['std']:.4f} | "
            f"{tuned - original:+.4f} |"
        )
    lines += ["", "Tuned Ours per-domain BA and NLL are included in the tracked JSON. "
              "The method remains below LoRA on ManipBench Macro-F1, but tuning "
              "improves Ours over the original setting on bridge_pick_place and droid_arti."]
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps({"folds": 12, "output_json": args.output_json,
                      "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
