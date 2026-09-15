#!/usr/bin/env python3
"""Small GPU admission smoke for single-image candidate scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eventttt.io import read_task_samples
from eventttt.qwen import DEFAULT_MODEL, preflight
from eventttt.task_qwen import score_task_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    status = preflight(require_gpu=True)
    samples = read_task_samples(args.manifest)
    if not samples:
        raise RuntimeError("empty task manifest")
    from eventttt.qwen import load_model

    model, processor = load_model(
        args.model_id, gradient_checkpointing=False, use_lora=False
    )
    device = next(model.parameters()).device
    rows = [score_task_sample(model, processor, sample, device) for sample in samples[: args.limit]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"preflight": status, "rows": rows}, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows), "device": str(device)}))


if __name__ == "__main__":
    main()
