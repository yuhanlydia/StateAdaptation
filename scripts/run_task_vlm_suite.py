#!/usr/bin/env python3
"""Run one formal single-image benchmark fold for a selected VLM family.

The output is intentionally compact and resumable: predictions and metrics
are written per method, while model weights and temporary adapters stay out of
the tracked repository under ``runs/``.
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
    _selected_modules,
    extract_task_subspace,
    fit_task_coefficients,
    task_visual_mask,
)
from eventttt.task_qwen import fit_task_lora, score_task_sample
from eventttt.task_vlm import default_task_model, load_task_model


def _metrics(rows, samples):
    labels = samples[0].candidate_labels
    return classification_metrics_nclass(
        [row["label_id"] for row in rows],
        np.asarray([row["probabilities"] for row in rows]),
        labels,
    )


def _save_method(out: Path, rows, metrics, adaptation=None):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if adaptation is not None:
        (out / "adaptation.json").write_text(json.dumps(adaptation, indent=2) + "\n")


def _load(family, model_id, *, lora=False, checkpoint=False):
    model, processor = load_task_model(
        model_id, family, use_lora=lora, gradient_checkpointing=checkpoint
    )
    return model, processor


def _score(model, processor, family, samples, controller=None, mask=None):
    device = next(model.parameters()).device
    return [
        score_task_sample(model, processor, sample, device, controller, mask, family)
        for sample in tqdm(samples, desc="Task scoring", dynamic_ncols=True)
    ]


def run_method(method, family, model_id, support, query, out, *,
               kv_rank=16, kv_alpha_max=3.0, kv_steps=4,
               kv_learning_rate=0.05, kv_l2=1e-3, kv_layers=(14, 27)):
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    controller = mask = None
    model = processor = None
    adaptation = {"family": family, "model_id": model_id, "method": method,
                  "support_examples": len(support), "query_examples": len(query)}
    try:
        if method == "frozen":
            model, processor = _load(family, model_id)
        elif method == "lora":
            model, processor = _load(family, model_id, lora=True, checkpoint=True)
            adaptation["losses"] = fit_task_lora(
                model, processor, support, passes=4, learning_rate=2e-4,
                seed=0, family=family,
            )
        elif method in {"random_kv", "ours"}:
            # InternVL's Qwen2 language decoder otherwise retains enough
            # backward activations across the 32-example support pass to
            # exceed a 24 GiB card. Checkpointing is a memory-only change;
            # the frozen/KV protocol and optimizer steps are unchanged.
            model, processor = _load(
                family, model_id, checkpoint=(family == "internvl3")
            )
            freeze_model(model)
            modules, layer_count, layers = _selected_modules(model, list(kv_layers))
            mask = task_visual_mask(model, processor)
            basis_mode = "random" if method == "random_kv" else "covariance"
            bases, spectra = extract_task_subspace(
                model, processor, support, modules, mask, rank=kv_rank,
                basis_mode=basis_mode, seed=0, family=family,
            )
            state = {"bases": bases, "rank": kv_rank, "alpha_max": kv_alpha_max,
                     "coefficient_mode": "full"}
            controller = build_controller_from_state(
                modules, state, device=next(model.parameters()).device
            )
            adaptation.update({"layers": layers, "num_decoder_layers": layer_count,
                               "rank": kv_rank, "alpha_max": kv_alpha_max,
                               "coefficient_mode": "full", "basis_mode": basis_mode,
                               "spectra": spectra, "steps": kv_steps,
                               "learning_rate": kv_learning_rate, "l2": kv_l2})
            adaptation["losses"] = fit_task_coefficients(
                model, processor, support, controller, mask, steps=kv_steps,
                learning_rate=kv_learning_rate, l2=kv_l2, family=family,
            )
            adaptation["kv_scalars"] = controller.num_scalars()
            adaptation["effective_gamma"] = controller.effective_gamma()
        else:
            raise ValueError(method)
        rows = _score(model, processor, family, query, controller, mask)
        metrics = _metrics(rows, query)
        _save_method(out, rows, metrics, adaptation)
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
    parser.add_argument("--family", choices=("qwen3_vl", "internvl3"), required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dataset", choices=("camelyon17", "manipbench_q1"), required=True)
    parser.add_argument("--domain", default="hospital_2")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--run-root", default="runs/task_vlm_formal")
    parser.add_argument("--kv-rank", type=int, default=16)
    parser.add_argument("--kv-alpha-max", type=float, default=3.0)
    parser.add_argument("--kv-steps", type=int, default=4)
    parser.add_argument("--kv-learning-rate", type=float, default=0.05)
    parser.add_argument("--kv-l2", type=float, default=1e-3)
    parser.add_argument("--kv-layers", type=int, nargs="+", default=[14, 27])
    parser.add_argument("--methods", nargs="+",
                        choices=("frozen", "lora", "random_kv", "ours"),
                        default=None)
    args = parser.parse_args()
    model_id = args.model_id or default_task_model(args.family)
    if args.dataset == "camelyon17":
        fold = Path("data/prepared/camelyon17") / f"seed_{args.seed}"
    else:
        fold = Path("data/prepared/manipbench_q1") / args.domain / f"seed_{args.seed}"
    support = read_task_samples(fold / "support.jsonl")
    query = read_task_samples(fold / "query.jsonl")
    out_root = Path(args.run_root) / args.family / args.dataset / args.domain / f"seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    methods = args.methods or ["frozen", "lora", "random_kv", "ours"]
    config = {"family": args.family, "model_id": model_id, "dataset": args.dataset,
              "domain": args.domain, "seed": args.seed, "support": len(support),
              "query": len(query), "methods": methods,
              "image_max_size": 224 if args.family == "internvl3" else 448,
              "visual_tokens": 64 if args.family == "internvl3" else None,
              "lora_passes": 4, "kv_rank": args.kv_rank,
              "kv_layers": args.kv_layers, "kv_steps": args.kv_steps,
              "kv_alpha_max": args.kv_alpha_max,
              "kv_learning_rate": args.kv_learning_rate, "kv_l2": args.kv_l2}
    (out_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for method in config["methods"]:
        metrics = run_method(
            method, args.family, model_id, support, query, out_root / method,
            kv_rank=args.kv_rank, kv_alpha_max=args.kv_alpha_max,
            kv_steps=args.kv_steps, kv_learning_rate=args.kv_learning_rate,
            kv_l2=args.kv_l2, kv_layers=tuple(args.kv_layers),
        )
        print(json.dumps({"method": method, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
