#!/usr/bin/env python3
"""Summarize per-class failures and parameter/storage efficiency."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from sklearn.metrics import confusion_matrix, f1_score

EVENTS = ("hawaii-wildfire", "libya-flood", "noto-earthquake", "turkey-earthquake")
LABELS = ("intact", "damaged", "destroyed")
ROOT = Path("runs/neurips")


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def pred_path(event, method, seed=0):
    if method == "baseline": return ROOT / event / "original_eval/predictions.jsonl"
    if method == "lora": return ROOT / event / "support24_lora_eval/predictions.jsonl"
    if seed == 0: return ROOT / event / f"support24_kv_{method}_a3/eval/predictions.jsonl"
    return ROOT / event / f"support24_seed{seed}_kv_{method}_a3/eval/predictions.jsonl"


def main():
    records = []
    for event in EVENTS:
        for method, seeds in (("baseline", [0]), ("lora", [0]), ("full", range(3)), ("diagonal", range(3))):
            for seed in seeds:
                rows = read(pred_path(event, method, seed))
                y = [row["label"] for row in rows]; p = [row["prediction"] for row in rows]
                per_class = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
                records.append({"event": event, "method": method, "seed": seed,
                                "macro_f1": float(np.mean(per_class)),
                                "per_class_f1": dict(zip(LABELS, map(float, per_class))),
                                "confusion_matrix": confusion_matrix(y, p, labels=LABELS).tolist()})
    class_means = {}
    for method in ("baseline", "lora", "full", "diagonal"):
        subset = [r for r in records if r["method"] == method]
        class_means[method] = {label: float(np.mean([r["per_class_f1"][label] for r in subset])) for label in LABELS}
        class_means[method]["macro_f1"] = float(np.mean([r["macro_f1"] for r in subset]))

    example = ROOT / EVENTS[0]
    with safe_open(str(example / "support24_lora/adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        lora_params = sum(handle.get_tensor(key).numel() for key in handle.keys())
    efficiency = {
        "lora": {"trainable_parameters": lora_params,
                 "artifact_bytes_mean": float(np.mean([(ROOT/e/"support24_lora/adapter_model.safetensors").stat().st_size for e in EVENTS])),
                 "adaptation_seconds": None, "note": "support-only 3-fold CV plus selected final fit; legacy runner did not record wall time"},
    }
    for method in ("full", "diagonal"):
        dirs = [ROOT/e/(f"support24_kv_{method}_a3" if s == 0 else f"support24_seed{s}_kv_{method}_a3") for e in EVENTS for s in range(3)]
        extraction = [json.loads((d/"extraction.json").read_text()) for d in dirs]
        adaptation = [json.loads((d/"adaptation.json").read_text()) for d in dirs]
        efficiency[method] = {
            "trainable_parameters": extraction[0]["kv_ttt_scalars"],
            "artifact_bytes_mean": float(np.mean([(d/"kv_state.pt").stat().st_size for d in dirs])),
            "extraction_seconds_mean": float(np.mean([x["extraction_seconds"] for x in extraction])),
            "adaptation_seconds_mean": float(np.mean([x["adaptation_seconds"] for x in adaptation])),
        }
    result = {"class_means": class_means, "efficiency": efficiency, "records": records}
    Path("reports/failure_efficiency.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Failure and efficiency analysis", "", "## Mean per-class F1", "",
             "| Method | Intact | Damaged | Destroyed | Macro-F1 |", "|---|---:|---:|---:|---:|"]
    for method, x in class_means.items():
        lines.append(f"| {method} | {x['intact']:.4f} | {x['damaged']:.4f} | {x['destroyed']:.4f} | {x['macro_f1']:.4f} |")
    lines += ["", "KV values average three support seeds; raw VLM and LoRA use the fixed support24 baseline.", "",
              "## Efficiency", "", "| Method | Trainable scalars | Mean artifact size | Extraction | Coefficient fit |",
              "|---|---:|---:|---:|---:|"]
    for method in ("lora", "full", "diagonal"):
        x = efficiency[method]
        ext = "n/a" if "extraction_seconds_mean" not in x else f"{x['extraction_seconds_mean']:.1f}s"
        fit = "not recorded" if x.get("adaptation_seconds") is None and method == "lora" else f"{x['adaptation_seconds_mean']:.1f}s"
        lines.append(f"| {method} | {x['trainable_parameters']:,} | {x['artifact_bytes_mean']/1024:.1f} KiB | {ext} | {fit} |")
    lines += ["", "Full uses 100 trainable controller scalars and Diagonal uses 20, versus 10,092,544 LoRA parameters. KV extraction and coefficient-fit times exclude model loading and 300-query evaluation."]
    Path("reports/failure_efficiency.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__": main()
