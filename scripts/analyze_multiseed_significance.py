#!/usr/bin/env python3
"""Multiseed paired significance analysis for balanced raw-VLM comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

EVENTS = ("hawaii-wildfire", "libya-flood", "noto-earthquake", "turkey-earthquake")
METHODS = ("full", "diagonal")
LABELS = ("intact", "damaged", "destroyed")


def read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def prediction_path(root: Path, event: str, method: str, seed: int) -> Path:
    if seed == 0:
        return root / event / f"support24_kv_{method}_a3" / "eval" / "predictions.jsonl"
    return root / event / f"support24_seed{seed}_kv_{method}_a3" / "eval" / "predictions.jsonl"


def metrics(rows, indices):
    truth = [rows[i]["label"] for i in indices]
    pred = [rows[i]["prediction"] for i in indices]
    nll = -float(np.mean([np.log(max(rows[i]["probabilities"][rows[i]["label_id"]], 1e-12)) for i in indices]))
    return {
        "macro_f1": float(f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "nll": nll,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/neurips")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-json", default="reports/multiseed_significance.json")
    parser.add_argument("--output-markdown", default="reports/multiseed_significance.md")
    args = parser.parse_args()
    root = Path(args.runs_root)
    rng = np.random.default_rng(args.seed)
    baseline, adapted = {}, {}
    rows = []
    for event in EVENTS:
        base = read(root / event / "original_eval" / "predictions.jsonl")
        baseline[event] = base
        base_ids = [row["sample_id"] for row in base]
        for method in METHODS:
            for seed in range(3):
                arm = read(prediction_path(root, event, method, seed))
                assert [row["sample_id"] for row in arm] == base_ids
                adapted[event, method, seed] = arm
                idx = np.arange(len(base))
                bm, am = metrics(base, idx), metrics(arm, idx)
                rows.append({"event": event, "method": method, "support_seed": seed,
                             **{f"baseline_{k}": v for k, v in bm.items()},
                             **{k: v for k, v in am.items()},
                             **{f"delta_{k}": am[k] - bm[k] for k in bm}})

    summaries = {}
    comparisons = (("full", "baseline"), ("diagonal", "baseline"), ("full", "diagonal"))
    strata = {e: {label: np.array([i for i, row in enumerate(baseline[e]) if row["label"] == label])
                  for label in LABELS} for e in EVENTS}
    for left, right in comparisons:
        observed = []
        for event in EVENTS:
            idx = np.arange(len(baseline[event]))
            for seed in range(3):
                lrows = adapted[event, left, seed]
                rrows = baseline[event] if right == "baseline" else adapted[event, right, seed]
                observed.append(metrics(lrows, idx)["macro_f1"] - metrics(rrows, idx)["macro_f1"])
        observed_mean = float(np.mean(observed))
        boot = np.empty(args.iterations)
        null = np.empty(args.iterations)
        for iteration in range(args.iterations):
            boot_events, null_events = [], []
            for event in EVENTS:
                seed = int(rng.integers(0, 3))
                indices = np.concatenate([rng.choice(strata[event][label], len(strata[event][label]), replace=True)
                                          for label in LABELS])
                lrows = adapted[event, left, seed]
                rrows = baseline[event] if right == "baseline" else adapted[event, right, seed]
                boot_events.append(metrics(lrows, indices)["macro_f1"] - metrics(rrows, indices)["macro_f1"])
                swap = rng.random(len(indices)) < 0.5
                lp = [dict(lrows[i]) for i in indices]
                rp = [dict(rrows[i]) for i in indices]
                for j, do_swap in enumerate(swap):
                    if do_swap:
                        lp[j]["prediction"], rp[j]["prediction"] = rp[j]["prediction"], lp[j]["prediction"]
                local = np.arange(len(indices))
                null_events.append(metrics(lp, local)["macro_f1"] - metrics(rp, local)["macro_f1"])
            boot[iteration] = np.mean(boot_events)
            null[iteration] = np.mean(null_events)
        key = f"{left}_vs_{right}"
        summaries[key] = {
            "mean_delta_macro_f1": observed_mean,
            "bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "paired_permutation_p_two_sided": float((1 + np.sum(np.abs(null) >= abs(observed_mean))) / (args.iterations + 1)),
            "iterations": args.iterations,
        }

    result = {"events": list(EVENTS), "support_seeds": [0, 1, 2], "rows": rows, "comparisons": summaries,
              "estimand": "mean of event-level macro-F1 differences; bootstrap samples support seed and stratified queries"}
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Multiseed statistical significance", "",
             "Three balanced support24 seeds; paired comparisons use identical query IDs.", "",
             "| Comparison | Mean Δ Macro-F1 | Stratified bootstrap 95% CI | Paired permutation p |",
             "|---|---:|---:|---:|"]
    for name, values in summaries.items():
        lo, hi = values["bootstrap_95_ci"]
        lines.append(f"| {name} | {values['mean_delta_macro_f1']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {values['paired_permutation_p_two_sided']:.4g} |")
    lines += ["", "## Per-event, per-seed results", "",
              "| Event | Method | Seed | Macro-F1 | ΔF1 | ΔBalAcc | NLL reduction |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['event']} | {row['method']} | {row['support_seed']} | {row['macro_f1']:.4f} | {row['delta_macro_f1']:+.4f} | {row['delta_balanced_accuracy']:+.4f} | {-row['delta_nll']:+.4f} |")
    Path(args.output_markdown).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
