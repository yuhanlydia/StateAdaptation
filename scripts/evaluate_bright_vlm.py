#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from eventttt.bright_vlm import load_bright_vlm, score_bright_sample
from eventttt.io import read_samples
from eventttt.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Gemma/Llama BRIGHT evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--family", choices=("phi", "gemma", "llama", "qwen3_vl", "internvl3"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=448)
    args = parser.parse_args()

    samples = read_samples(args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]
    model, processor = load_bright_vlm(args.model_id, args.family)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions.jsonl"
    rows = []
    with predictions.open("w", encoding="utf-8") as handle:
        for sample in tqdm(samples, desc=f"{args.family} BRIGHT", dynamic_ncols=True):
            row = score_bright_sample(model, processor, args.family, sample, args.crop_size)
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
    metrics = classification_metrics(
        [row["label_id"] for row in rows],
        np.asarray([row["probabilities"] for row in rows]),
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "eval_config.json").write_text(
        json.dumps({"manifest": str(Path(args.manifest).resolve()), "model_id": args.model_id,
                    "family": args.family, "crop_size": args.crop_size,
                    "count": len(rows)}, indent=2) + "\n"
    )
    print(json.dumps({"output_dir": str(output), "count": len(rows), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
