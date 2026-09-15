#!/usr/bin/env python3
"""Fixed-support LoRA-TTA baseline for Camelyon17/ManipBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from eventttt.io import read_task_samples
from eventttt.qwen import preflight
from eventttt.task_qwen import fit_task_lora
from eventttt.task_vlm import default_task_model, load_task_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", choices=("qwen2", "qwen3_vl", "internvl3"), default="qwen2")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.model_id is None:
        args.model_id = default_task_model(args.family)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(json.dumps(preflight(require_gpu=True), indent=2))
    samples = read_task_samples(args.support_manifest)
    # ManipBench images are larger than BRIGHT; checkpointing keeps the
    # support-only backward pass within a 24 GiB card while preserving the
    # same deterministic 448px image preprocessing used at evaluation.
    model, processor = load_task_model(args.model_id, args.family,
                                       gradient_checkpointing=True, use_lora=True)
    losses = fit_task_lora(
        model, processor, samples, passes=args.passes,
        learning_rate=args.learning_rate, seed=args.seed, family=args.family,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    (output / "adaptation.json").write_text(json.dumps({
        "method": "task_lora_tta", "support_examples": len(samples),
        "passes": args.passes, "learning_rate": args.learning_rate,
        "losses": losses, "arguments": vars(args),
    }, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "losses": losses}, indent=2))


if __name__ == "__main__":
    main()
