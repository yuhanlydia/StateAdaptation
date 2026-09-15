#!/usr/bin/env python3
"""Support-only hyperparameter sweep for task KV-TTT.

The query set is never read.  A stratified quarter of the support set is held
out for selecting alpha/optimizer duration, then the selected setting can be
passed to the formal runner for a fresh full-support adaptation.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from eventttt.io import read_task_samples
from eventttt.kv_ttt import build_controller_from_state, freeze_model
from eventttt.metrics import classification_metrics_nclass
from eventttt.task_kv import (
    _selected_modules, extract_task_subspace, fit_task_coefficients,
    task_visual_mask,
)
from eventttt.task_qwen import score_task_sample
from eventttt.task_vlm import default_task_model, load_task_model


def split_support(samples, seed):
    rng = np.random.default_rng(seed)
    by_label = {}
    for sample in samples:
        by_label.setdefault(sample.label_id, []).append(sample)
    train, valid = [], []
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        n_valid = max(1, len(group) // 4)
        valid.extend(group[:n_valid])
        train.extend(group[n_valid:])
    return train, valid


def score(model, processor, family, samples, mask, controller):
    device = next(model.parameters()).device
    rows = [score_task_sample(model, processor, sample, device, controller, mask, family)
            for sample in tqdm(samples, desc="Support validation", dynamic_ncols=True)]
    labels = samples[0].candidate_labels
    return classification_metrics_nclass(
        [row["label_id"] for row in rows],
        np.asarray([row["probabilities"] for row in rows]), labels,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("qwen3_vl", "internvl3"), required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dataset", choices=("camelyon17", "manipbench_q1"), required=True)
    parser.add_argument("--domain", default="hospital_2")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 27])
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 3.0, 6.0])
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--learning-rate", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()
    model_id = args.model_id or default_task_model(args.family)
    if args.dataset == "camelyon17":
        fold = Path("data/prepared/camelyon17") / f"seed_{args.seed}"
    else:
        fold = Path("data/prepared/manipbench_q1") / args.domain / f"seed_{args.seed}"
    support = read_task_samples(fold / "support.jsonl")
    train, valid = split_support(support, args.seed + 1009)
    model, processor = load_task_model(
        model_id, args.family, gradient_checkpointing=(args.family == "internvl3")
    )
    freeze_model(model)
    modules, layer_count, layers = _selected_modules(model, args.layers)
    mask = task_visual_mask(model, processor)
    bases, spectra = extract_task_subspace(
        model, processor, train, modules, mask, rank=args.rank,
        basis_mode="covariance", seed=args.seed, family=args.family,
    )
    results = []
    try:
        for alpha in args.alpha:
            for steps in args.steps:
                for learning_rate in args.learning_rate:
                    state = {"bases": bases, "rank": args.rank,
                             "alpha_max": alpha, "coefficient_mode": "full"}
                    controller = build_controller_from_state(
                        modules, state, device=next(model.parameters()).device
                    )
                    try:
                        losses = fit_task_coefficients(
                            model, processor, train, controller, mask,
                            steps=steps, learning_rate=learning_rate,
                            l2=args.l2, family=args.family,
                        )
                        metrics = score(model, processor, args.family, valid, mask, controller)
                        results.append({
                            "alpha_max": alpha, "steps": steps,
                            "learning_rate": learning_rate, "l2": args.l2,
                            "losses": losses,
                            "effective_gamma": controller.effective_gamma(),
                            "metrics": metrics,
                        })
                    finally:
                        controller.close()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected = max(results, key=lambda item: (
        item["metrics"]["macro_f1"], -item["metrics"]["nll"],
    ))
    payload = {
        "family": args.family, "model_id": model_id, "dataset": args.dataset,
        "domain": args.domain, "seed": args.seed,
        "support": len(support), "adaptation_support": len(train),
        "validation_support": len(valid), "rank": args.rank,
        "layers": layers, "num_decoder_layers": layer_count,
        "spectra": spectra, "results": results, "selected": selected,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": args.output, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
