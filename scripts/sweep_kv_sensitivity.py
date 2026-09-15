#!/usr/bin/env python3
"""One-at-a-time sensitivity sweep over KV-TTT hyperparameters.

Uses a single completed fold (source adapter already trained) and evaluates on
a label-stratified subset of the target query set so the sweep finishes in a
short wall-clock budget. Every adapted config is compared to the SAME source
baseline, so the deltas isolate one hyperparameter at a time.

Run: python3 scripts/sweep_kv_sensitivity.py [--subset-size 60]
"""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

BASE = {
    "rank": 5,
    "steps": 4,
    "alpha": 0.5,
    "lr": 0.05,
    "layers": "14 27",
}

# Order matters: the first dimensions are the ones we suspect matter most.
# rank{5..32} and steps{1..8} are already swept and are NOT sensitive (every
# config lands on macro-F1 ~= 0.3235 on the 60-sample subset). alpha showed the
# only signal at alpha=2.0 (f1 0.3235->0.3271, NLL 1.548), so push alpha far
# higher (3/5/10/100: gamma=alpha*tanh(a) then approaches the alpha cap).
# lr and the remaining alpha values resume from saved eval metrics.
SWEEP = {
    "alpha": [3.0, 5.0, 10.0, 100.0],
    "lr": [0.2, 1.0],
}


def subset_file(manifest: Path, size: int, seed: int) -> Path:
    """Label-stratified subset kept next to the original manifest so the
    relative image paths in the manifest still resolve."""
    rows = [json.loads(line) for line in open(manifest)]
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    picked: list[dict] = []
    per_label = max(1, size // len(by_label))
    for group in by_label.values():
        rng.shuffle(group)
        picked.extend(group[:per_label])
    rng.shuffle(picked)
    out = manifest.with_name(f"target_query_subset_{size}_s{seed}.jsonl")
    with open(out, "w") as fh:
        for row in picked:
            fh.write(json.dumps(row) + "\n")
    return out


def adapt(out_dir: Path, config: dict, support: Path, source_adapter: Path, crop: int) -> None:
    cmd = [
        PYTHON,
        "scripts/adapt_event_kv.py",
        "--support-manifest", str(support),
        "--source-adapter", str(source_adapter),
        "--output-dir", str(out_dir),
        "--crop-size", str(crop),
        "--rank", str(config["rank"]),
        "--alpha-max", str(config["alpha"]),
        "--steps", str(config["steps"]),
        "--learning-rate", str(config["lr"]),
        "--l2", "1e-3",
    ]
    if config["layers"]:
        cmd += ["--layers"] + config["layers"].split()
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)


def evaluate(manifest: Path, source_adapter: Path, kv_state: Path, out_dir: Path, views: int, crop: int) -> dict:
    cmd = [
        PYTHON,
        "scripts/evaluate.py",
        "--manifest", str(manifest),
        "--adapter", str(source_adapter),
        "--kv-state", str(kv_state),
        "--d4-views", str(views),
        "--crop-size", str(crop),
        "--output-dir", str(out_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return json.loads((out_dir / "metrics.json").read_text())


def summarize(results: list[dict], baseline: dict) -> str:
    """Which single-parameter movement causes the largest metric delta?"""
    base_f1, base_nll = baseline["macro_f1"], baseline["nll"]
    lines = []
    for key in SWEEP:
        deltas = []
        for row in results:
            if row["macro_f1"] is None:
                continue
            # only rows that differ from baseline in exactly this one key
            differs = False
            differs_only_here = True
            for k in ("steps", "rank", "alpha", "lr"):
                if row[k] != baseline[k]:
                    differs = True
                    if k != key:
                        differs_only_here = False
            if differs and differs_only_here:
                deltas.append((row["macro_f1"] - base_f1, row["nll"] - base_nll, row["tag"]))
        if deltas:
            max_f1_delta = max(abs(f1) for f1, _, _ in deltas)
            lines.append(f"{key}: max |macro-F1 delta| = {max_f1_delta:.4f} "
                         f"across {len(deltas)} cfg (tags: {', '.join(t for _, _, t in deltas)})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="hawaii-wildfire")
    ap.add_argument("--subset-size", type=int, default=60)
    ap.add_argument("--d4-views", type=int, default=1)
    ap.add_argument("--crop-size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    fold_dir = ROOT / "data" / "prepared" / "neurips" / args.fold
    run_dir = ROOT / "runs" / "neurips" / args.fold
    source_adapter = run_dir / "source_adapter"
    support = fold_dir / "target_support.jsonl"
    assert source_adapter.is_dir(), f"missing source adapter: {source_adapter}"
    assert (source_adapter / "adapter_config.json").exists(), "source adapter incomplete"

    subset = subset_file(fold_dir / "target_query.jsonl", args.subset_size, args.seed)

    runs_root = run_dir / "kv_sweep"
    runs_root.mkdir(parents=True, exist_ok=True)
    crop = args.crop_size

    combos: list[dict] = [dict(BASE)]
    for key, values in SWEEP.items():
        for value in values:
            combo = dict(BASE)
            combo[key] = value
            combos.append(combo)

    results: list[dict] = []
    for combo in combos:
        tag = "_".join(f"{k}-{combo[k]}" for k in ["rank", "steps", "alpha", "lr"])
        tag = tag.replace(".", "p")
        ad = runs_root / f"event_kv_{tag}"
        ev = runs_root / f"eval_{tag}"
        if (ev / "metrics.json").exists():
            existing = json.loads((ev / "metrics.json").read_text())
            adaptation = json.loads((ad / "adaptation.json").read_text())
            losses = adaptation.get("losses", [])
            results.append({
                "tag": tag, "rank": combo["rank"], "steps": combo["steps"],
                "alpha": combo["alpha"], "lr": combo["lr"],
                "loss_first": round(losses[0], 4) if losses else None,
                "loss_last": round(losses[-1], 4) if losses else None,
                "macro_f1": round(existing.get("macro_f1", -1), 4),
                "balanced_acc": round(existing.get("balanced_accuracy", -1), 4),
                "nll": round(existing.get("nll", -1), 4),
            })
            continue
        try:
            if not (ad / "kv_state.pt").exists():
                adapt(ad, combo, support, source_adapter, crop)
            adaptation = json.loads((ad / "adaptation.json").read_text())
            losses = adaptation.get("losses", [])
            eval_metrics = evaluate(subset, source_adapter, ad / "kv_state.pt", ev, args.d4_views, crop)
            row = {
                "tag": tag,
                "rank": combo["rank"],
                "steps": combo["steps"],
                "alpha": combo["alpha"],
                "lr": combo["lr"],
                "loss_first": round(losses[0], 4) if losses else None,
                "loss_last": round(losses[-1], 4) if losses else None,
                "macro_f1": round(eval_metrics.get("macro_f1", -1), 4),
                "balanced_acc": round(eval_metrics.get("balanced_accuracy", -1), 4),
                "nll": round(eval_metrics.get("nll", -1), 4),
            }
            results.append(row)
            print(f"OK {tag}: f1={row['macro_f1']:.4f} nll={row['nll']:.4f} "
                  f"loss {row['loss_first']}->{row['loss_last']}", flush=True)
        except Exception as exc:
            row = {"tag": tag, "rank": combo["rank"], "steps": combo["steps"],
                   "alpha": combo["alpha"], "lr": combo["lr"],
                   "loss_first": None, "loss_last": None,
                   "macro_f1": None, "balanced_acc": None, "nll": None}
            results.append(row)
            print(f"FAIL {tag}: {exc}", file=sys.stderr, flush=True)

    base = next(r for r in results if r["tag"] == "_".join(f"{k}-{BASE[k]}" for k in ["rank", "steps", "alpha", "lr"]).replace(".", "p"))
    line = (
        f"\nBaseline: {base['macro_f1']} macro-F1, {base['nll']} NLL\n"
        "Per-parameter max |macro-F1 delta|:\n"
        + summarize(results, base)
    )
    print(line)

    out_csv = ROOT / "reports" / f"kv_sweep_{args.fold}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()