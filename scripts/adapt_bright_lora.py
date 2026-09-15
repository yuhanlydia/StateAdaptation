#!/usr/bin/env python3
"""Support-only BRIGHT LoRA-TTA pilot/evaluation for one event fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from eventttt.bright_vlm import (
    enable_bright_lora, fit_bright_lora, load_bright_vlm, score_bright_sample,
)
from eventttt.io import read_samples
from eventttt.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--family", choices=("phi", "gemma", "llama", "qwen3_vl", "internvl3"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    support = read_samples(args.support_manifest)
    query = read_samples(args.query_manifest)
    if args.query_limit is not None:
        query = query[:args.query_limit]
    model, processor = load_bright_vlm(args.model_id, args.family)
    model = enable_bright_lora(model, args.family)
    losses = fit_bright_lora(
        model, processor, args.family, support, args.crop_size,
        args.passes, args.learning_rate, args.seed, args.grad_accum_steps,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "adapter")
    try:
        processor.save_pretrained(output / "adapter")
    except AttributeError as error:
        # Phi-3.5's legacy processor lacks the modern ``chat_template``
        # attribute. The adapter weights are complete; retain the warning and
        # reuse the immutable base processor at evaluation time.
        (output / "processor_save_warning.txt").write_text(str(error) + "\n")
    rows = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample in tqdm(query, desc=f"{args.family} BRIGHT LoRA", dynamic_ncols=True):
            row = score_bright_sample(model, processor, args.family, sample, args.crop_size)
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
    metrics = classification_metrics(
        [row["label_id"] for row in rows],
        np.asarray([row["probabilities"] for row in rows]),
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "adaptation.json").write_text(json.dumps({
        "method": "bright_support_lora_tta", "model_id": args.model_id,
        "family": args.family, "support_examples": len(support),
        "query_examples": len(query), "passes": args.passes,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate, "rank": 16, "alpha": 32,
        "target_modules": ["qkv_proj", "o_proj"] if args.family == "phi" else
                          ["q_proj", "k_proj", "v_proj", "o_proj"],
        "losses": losses, "arguments": vars(args),
    }, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
