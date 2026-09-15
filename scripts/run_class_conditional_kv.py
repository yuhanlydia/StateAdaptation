#!/usr/bin/env python3
"""Small class-conditional KV-TTT diagnostic for fixed-label tasks.

Each label gets a support-only KV basis and controller. At inference, each
candidate likelihood is read from the controller trained for that candidate.
No query label is used. This is an optional mixture-robust diagnostic, not the
main KV-TTT method.
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
from eventttt.qwen import product_of_experts
from eventttt.task_kv import (
    _selected_modules, extract_task_subspace, fit_task_coefficients, task_visual_mask,
)
from eventttt.task_qwen import candidate_scores
from eventttt.task_vlm import default_task_model, load_task_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("qwen3_vl", "internvl3"), default="internvl3")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--support", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 27])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    model_id = args.model_id or default_task_model(args.family)
    support, query = read_task_samples(args.support), read_task_samples(args.query)
    labels = tuple(support[0].candidate_labels)
    model, processor = load_task_model(
        model_id, args.family, gradient_checkpointing=(args.family == "internvl3")
    )
    freeze_model(model)
    modules, layer_count, selected_layers = _selected_modules(model, args.layers)
    visual_mask = task_visual_mask(model, processor)
    controllers, adaptation = {}, {}
    try:
        for label in labels:
            class_support = [sample for sample in support if sample.label == label]
            bases, spectra = extract_task_subspace(
                model, processor, class_support, modules, visual_mask,
                rank=args.rank, family=args.family,
            )
            controller = build_controller_from_state(
                modules,
                {"bases": bases, "rank": args.rank, "alpha_max": args.alpha,
                 "coefficient_mode": "full"},
                device=next(model.parameters()).device,
            )
            losses = fit_task_coefficients(
                model, processor, class_support, controller, visual_mask,
                steps=args.steps, learning_rate=args.learning_rate,
                l2=args.l2, family=args.family,
            )
            controllers[label] = controller
            adaptation[label] = {
                "support_examples": len(class_support), "losses": losses,
                "spectra": spectra, "scalars": controller.num_scalars(),
            }

        device = next(model.parameters()).device
        rows = []
        for sample in tqdm(query, desc="Class-conditional scoring", dynamic_ncols=True):
            scores = []
            for label_index, label in enumerate(labels):
                all_scores = candidate_scores(
                    model, processor, sample, device=device,
                    controller=controllers[label], visual_mask=visual_mask,
                    family=args.family,
                )
                scores.append(float(all_scores[label_index]))
            probabilities = product_of_experts([scores])
            prediction_id = int(np.argmax(probabilities))
            rows.append({
                "sample_id": sample.sample_id, "label": sample.label,
                "label_id": sample.label_id, "prediction": labels[prediction_id],
                "probabilities": probabilities.tolist(), "candidate_scores": scores,
            })
        metrics = classification_metrics_nclass(
            [row["label_id"] for row in rows],
            np.asarray([row["probabilities"] for row in rows]), labels,
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (output / "adaptation.json").write_text(json.dumps({
            "warning": "OPTIONAL CLASS-CONDITIONAL DIAGNOSTIC; not the main method",
            "model_id": model_id, "layers": selected_layers,
            "num_decoder_layers": layer_count, "rank": args.rank,
            "alpha": args.alpha, "steps": args.steps,
            "learning_rate": args.learning_rate, "l2": args.l2,
            "classes": adaptation,
        }, indent=2) + "\n")
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        print(json.dumps(metrics, indent=2))
    finally:
        for controller in controllers.values():
            controller.close()
        del model, processor, controllers
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
