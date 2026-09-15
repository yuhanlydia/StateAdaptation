#!/usr/bin/env python3
"""Run Frozen/LoRA/Random-KV/Ours on Guardian execution verification folds."""

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
from eventttt.task_kv import _selected_modules, extract_task_subspace, fit_task_coefficients, task_visual_mask
from eventttt.task_qwen import fit_task_lora, score_task_sample
from eventttt.task_vlm import default_task_model, load_task_model


def _metrics(rows, samples):
    return classification_metrics_nclass(
        [row["label_id"] for row in rows],
        np.asarray([row["probabilities"] for row in rows]),
        samples[0].candidate_labels,
    )


def _save(out: Path, rows, metrics, adaptation):
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "adaptation.json").write_text(json.dumps(adaptation, indent=2) + "\n")


def _score(model, processor, family, samples, controller=None, mask=None):
    device = next(model.parameters()).device
    return [score_task_sample(model, processor, sample, device, controller, mask, family)
            for sample in tqdm(samples, desc="Guardian scoring", dynamic_ncols=True)]


def run_method(method, family, model_id, support, query, out, *, rank, alpha, steps,
               learning_rate, l2, layers):
    if (out / "metrics.json").is_file():
        return json.loads((out / "metrics.json").read_text())
    model = processor = controller = mask = None
    adaptation = {"method": method, "family": family, "model_id": model_id,
                  "support_examples": len(support), "query_examples": len(query),
                  "candidate_labels": list(support[0].candidate_labels), "layers": list(layers)}
    try:
        if method == "frozen":
            model, processor = load_task_model(model_id, family)
        elif method == "lora":
            model, processor = load_task_model(model_id, family, use_lora=True, gradient_checkpointing=True)
            adaptation["losses"] = fit_task_lora(model, processor, support, passes=4,
                                                  learning_rate=2e-4, seed=0, family=family)
            adaptation.update({"rank": 16, "alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})
        elif method in {"random_kv", "ours"}:
            model, processor = load_task_model(model_id, family, gradient_checkpointing=(family == "internvl3"))
            freeze_model(model)
            modules, layer_count, selected = _selected_modules(model, list(layers))
            mask = task_visual_mask(model, processor)
            basis_mode = "random" if method == "random_kv" else "covariance"
            bases, spectra = extract_task_subspace(model, processor, support, modules, mask,
                                                    rank=rank, basis_mode=basis_mode, seed=0,
                                                    family=family)
            controller = build_controller_from_state(
                modules, {"bases": bases, "rank": rank, "alpha_max": alpha, "coefficient_mode": "full"},
                device=next(model.parameters()).device,
            )
            adaptation.update({"num_decoder_layers": layer_count, "layers": selected,
                               "rank": rank, "alpha_max": alpha, "coefficient_mode": "full",
                               "basis_mode": basis_mode, "steps": steps,
                               "learning_rate": learning_rate, "l2": l2, "spectra": spectra})
            adaptation["losses"] = fit_task_coefficients(
                model, processor, support, controller, mask, steps=steps,
                learning_rate=learning_rate, l2=l2, family=family,
            )
            adaptation["kv_scalars"] = controller.num_scalars()
        else:
            raise ValueError(method)
        rows = _score(model, processor, family, query, controller, mask)
        metrics = _metrics(rows, query)
        _save(out, rows, metrics, adaptation)
        return metrics
    finally:
        if controller is not None:
            controller.close()
        del model, processor, controller, mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("ur5fail_test", "robofail", "robovqa"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--family", choices=("qwen3_vl", "internvl3"), default="internvl3")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--manifest-root", default="data/prepared/guardian_execution")
    parser.add_argument("--run-root", default="runs/guardian_failure")
    parser.add_argument("--methods", nargs="+", choices=("frozen", "lora", "random_kv", "ours"), default=None)
    parser.add_argument("--kv-rank", type=int, default=16)
    parser.add_argument("--kv-alpha-max", type=float, default=3.0)
    parser.add_argument("--kv-steps", type=int, default=4)
    parser.add_argument("--kv-learning-rate", type=float, default=0.05)
    parser.add_argument("--kv-l2", type=float, default=1e-3)
    parser.add_argument("--kv-layers", type=int, nargs="+", default=[14, 27])
    args = parser.parse_args()
    model_id = args.model_id or default_task_model(args.family)
    fold = Path(args.manifest_root) / args.split / f"seed_{args.seed}"
    support, query = read_task_samples(fold / "support.jsonl"), read_task_samples(fold / "query.jsonl")
    out_root = Path(args.run_root) / args.family / args.split / f"seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    methods = args.methods or ["frozen", "lora", "random_kv", "ours"]
    config = {"family": args.family, "model_id": model_id, "split": args.split, "seed": args.seed,
              "support": len(support), "query": len(query), "methods": methods,
              "candidate_labels": list(support[0].candidate_labels), "kv_rank": args.kv_rank,
              "kv_alpha_max": args.kv_alpha_max, "kv_layers": args.kv_layers,
              "kv_steps": args.kv_steps, "kv_learning_rate": args.kv_learning_rate, "kv_l2": args.kv_l2,
              "view_mode": "before_after_vertical_front_view"}
    (out_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for method in methods:
        metrics = run_method(method, args.family, model_id, support, query, out_root / method,
                             rank=args.kv_rank, alpha=args.kv_alpha_max, steps=args.kv_steps,
                             learning_rate=args.kv_learning_rate, l2=args.kv_l2, layers=tuple(args.kv_layers))
        print(json.dumps({"method": method, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
