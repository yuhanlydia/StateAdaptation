#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventttt.gates import exclude_reserved_samples
from eventttt.io import read_samples, write_json
from eventttt.qwen import DEFAULT_MODEL, fit_steps, load_model, preflight, trainable_parameter_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-event SFT with Qwen2.5-VL-7B LoRA")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="manifest whose sample IDs are reserved from source training",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=6)
    parser.add_argument(
        "--sampling",
        choices=("class_cycle", "inverse_frequency"),
        default="class_cycle",
    )
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True), indent=2))
    samples = read_samples(args.train_manifest)
    reserved = [
        sample
        for manifest in args.exclude_manifest
        for sample in read_samples(manifest)
    ]
    samples, excluded_examples = exclude_reserved_samples(samples, reserved)
    model, processor = load_model(args.model_id)
    print(json.dumps(trainable_parameter_report(model), indent=2))
    losses = fit_steps(
        model,
        processor,
        samples,
        args.steps,
        args.learning_rate,
        args.batch_size,
        args.gradient_accumulation,
        args.crop_size,
        seed=args.seed,
        sampling=args.sampling,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    write_json(
        output / "train_summary.json",
        {
            "model_id": args.model_id,
            "examples": len(samples),
            "excluded_examples": excluded_examples,
            "steps": args.steps,
            "final_loss": losses[-1] if losses else None,
            "arguments": vars(args),
        },
    )


if __name__ == "__main__":
    main()
