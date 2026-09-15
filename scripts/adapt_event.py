#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eventttt.io import read_samples, write_json
from eventttt.metrics import classification_metrics, select_checkpoint
from eventttt.qwen import DEFAULT_MODEL, fit_steps, load_model, preflight, score_samples
from eventttt.splits import stratified_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Support-only CV and temporary per-event LoRA")
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-adapter")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--candidate-steps", nargs="+", type=int, default=[0, 4, 8, 16, 32])
    parser.add_argument(
        "--fixed-steps",
        type=int,
        help="Pre-registered duration for 1-shot support, where support CV is undefined",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=3)
    parser.add_argument(
        "--sampling",
        choices=("class_cycle", "inverse_frequency"),
        default="class_cycle",
    )
    parser.add_argument("--selection-d4-views", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True), indent=2))
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    support = read_samples(args.support_manifest)
    model, processor = load_model(
        args.model_id,
        source_adapter=args.source_adapter,
    )
    initial = {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(model).items()
    }

    def reset() -> None:
        result = set_peft_model_state_dict(model, initial)
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"Unexpected adapter keys while resetting: {result.unexpected_keys}")

    records = []
    if args.fixed_steps is None:
        for fold_index, (train, held) in enumerate(stratified_folds(support, args.folds, args.seed)):
            for steps in sorted(set(args.candidate_steps)):
                reset()
                fit_steps(
                    model,
                    processor,
                    train,
                    steps,
                    args.learning_rate,
                    args.batch_size,
                    args.gradient_accumulation,
                    args.crop_size,
                    seed=args.seed + fold_index,
                    sampling=args.sampling,
                )
                rows = score_samples(
                    model, processor, held, args.selection_d4_views, args.crop_size
                )
                metrics = classification_metrics(
                    [row["label_id"] for row in rows],
                    np.asarray([row["probabilities"] for row in rows]),
                )
                records.append({"fold": fold_index, "steps": steps, **metrics})
                print(
                    json.dumps(
                        {"fold": fold_index, "steps": steps, "macro_f1": metrics["macro_f1"]}
                    )
                )
        selected = select_checkpoint(records)
    else:
        if args.fixed_steps < 0:
            raise ValueError("--fixed-steps must be non-negative")
        selected = args.fixed_steps
    reset()
    losses = fit_steps(
        model,
        processor,
        support,
        selected,
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
    write_json(output / "support_cv.json", records)
    write_json(
        output / "selection.json",
        {
            "selected_steps": selected,
            "selection_method": "support_cv" if args.fixed_steps is None else "pre_registered",
            "support_examples": len(support),
            "final_loss": losses[-1] if losses else None,
            "arguments": vars(args),
        },
    )


if __name__ == "__main__":
    main()
