#!/usr/bin/env python3
"""Summarize same-event inference adaptations against raw-VLM original eval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = {
    "support24_lora": "support24_lora/gain_vs_original.json",
    "kv_full_a3": "support24_kv_full_a3/gain_vs_original.json",
    "kv_diagonal_a3": "support24_kv_diagonal_a3/gain_vs_original.json",
    "kv_unsupervised_a3": "support24_kv_unsupervised_a3/gain_vs_original.json",
    "kv_full_a0p5_ablation": "support24_kv_full_a0p5/gain_vs_original.json",
    "kv_diagonal_a0p5_ablation": "support24_kv_diagonal_a0p5/gain_vs_original.json",
    "kv_unsupervised_a0p5_ablation": "support24_kv_unsupervised/gain_vs_original.json",
    "kv_unsupervised_full_a3_ablation": "support24_kv_unsupervised_full_a3/gain_vs_original.json",
    "kv_unsupervised_full_a0p5_legacy": "support24_kv_unsupervised_full/gain_vs_original.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs/neurips")
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    root = Path(args.runs_root)
    rows = []
    for event in args.events:
        for arm, relative in ARMS.items():
            path = root / event / relative
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "event": event,
                "arm": arm,
                "count": payload["count"],
                "original_macro_f1": payload["baseline"]["macro_f1"],
                "macro_f1": payload["adapted"]["macro_f1"],
                "delta_macro_f1": payload["gain"]["macro_f1"],
                "original_balanced_accuracy": payload["baseline"]["balanced_accuracy"],
                "balanced_accuracy": payload["adapted"]["balanced_accuracy"],
                "delta_balanced_accuracy": payload["gain"]["balanced_accuracy"],
                "original_nll": payload["baseline"]["nll"],
                "nll": payload["adapted"]["nll"],
                "nll_reduction": payload["error_reduction"]["nll"],
            })

    aggregates = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        fields = ("macro_f1", "delta_macro_f1", "balanced_accuracy",
                  "delta_balanced_accuracy", "nll", "nll_reduction")
        aggregates[arm] = {
            "events": len(arm_rows),
            **{f"mean_{field}": sum(row[field] for row in arm_rows) / len(arm_rows)
               for field in fields},
        }

    result = {"baseline": "raw_vlm_original_eval", "rows": rows, "aggregates": aggregates}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Balanced same-event inference adaptation results", "",
        "Primary baseline: raw VLM `original_eval`; all adaptations use the same",
        "event's 24 support examples and evaluate the identical balanced query IDs.", "",
        "| Event | Arm | N | Original F1 | Adapted F1 | ΔF1 | ΔBalAcc | NLL reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['event']} | {row['arm']} | {row['count']} | "
            f"{row['original_macro_f1']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['delta_macro_f1']:+.4f} | {row['delta_balanced_accuracy']:+.4f} | "
            f"{row['nll_reduction']:+.4f} |"
        )
    lines.extend(["", "## Macro means", "",
                  "| Arm | Events | Mean F1 | Mean ΔF1 | Mean ΔBalAcc | Mean NLL reduction |",
                  "|---|---:|---:|---:|---:|---:|"])
    for arm, values in aggregates.items():
        lines.append(
            f"| {arm} | {values['events']} | {values['mean_macro_f1']:.4f} | "
            f"{values['mean_delta_macro_f1']:+.4f} | "
            f"{values['mean_delta_balanced_accuracy']:+.4f} | "
            f"{values['mean_nll_reduction']:+.4f} |"
        )
    Path(args.output_markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
