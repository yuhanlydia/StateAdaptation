#!/usr/bin/env python3
"""Evaluate a single-image task with candidate likelihood scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from eventttt.io import read_task_samples
from eventttt.metrics import classification_metrics_nclass
from eventttt.qwen import preflight
from eventttt.task_vlm import default_task_model, load_task_model
from eventttt.task_kv import task_visual_mask
from eventttt.kv_ttt import build_controller_from_state, discover_language_decoder_kv, load_kv_state
from eventttt.hidden_residual import HiddenResidualController, discover_hidden_layers, load_hidden_state
from eventttt.task_qwen import score_task_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", choices=("qwen2", "qwen3_vl", "internvl3"), default="qwen2")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--kv-state", default="")
    parser.add_argument("--hidden-state", default="")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.model_id is None:
        args.model_id = default_task_model(args.family)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.kv_state and args.hidden_state:
        raise ValueError("choose at most one of --kv-state and --hidden-state")
    print(json.dumps(preflight(require_gpu=True), indent=2))
    samples = read_task_samples(args.manifest)
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("empty task manifest")
    model, processor = load_task_model(
        args.model_id, args.family, source_adapter=args.adapter or None,
        gradient_checkpointing=False, use_lora=bool(args.adapter),
    )
    device = next(model.parameters()).device
    controller = None
    visual_mask = None
    if args.kv_state:
        payload = load_kv_state(args.kv_state, device=device)
        # Adaptation jobs may have been launched with the local artifact path
        # while older states record the Hub identifier.  Accept that exact
        # basename-equivalent pair, but reject genuinely different models.
        recorded = str(payload["model_id"])
        requested = str(args.model_id)
        if recorded != requested and Path(recorded).name != Path(requested).name:
            raise ValueError("KV state model_id does not match evaluation model")
        modules, _ = discover_language_decoder_kv(model)
        selected = set(payload["layers"])
        module_slice = [(layer, kind, module) for layer, kind, module in modules if layer in selected]
        if {layer for layer, _, _ in module_slice} != selected:
            raise ValueError("KV state refers to unavailable decoder layers")
        controller = build_controller_from_state(module_slice, payload, device=device)
        visual_mask = task_visual_mask(model, processor)
    if args.hidden_state:
        payload = load_hidden_state(args.hidden_state, device=device)
        recorded = str(payload["model_id"]); requested = str(args.model_id)
        if recorded != requested and Path(recorded).name != Path(requested).name:
            raise ValueError("hidden state model_id does not match evaluation model")
        all_layers, _ = discover_hidden_layers(model)
        by_layer = {layer: module for layer, module in all_layers}
        selected = [int(layer) for layer in payload["layers"]]
        if any(layer not in by_layer for layer in selected):
            raise ValueError("hidden state refers to unavailable decoder layers")
        modules = [(layer, by_layer[layer]) for layer in selected]
        bases = {int(key): value for key, value in payload["bases"].items()}
        controller = HiddenResidualController(
            modules, bases, rank=payload["rank"], alpha_max=payload.get("alpha_max", 3.0),
            coefficient_mode=payload.get("coefficient_mode", "full"), device=device,
        )
        with torch.no_grad():
            for key, value in payload.get("coefficients_raw", {}).items():
                controller.coefficients[str(key)].copy_(value.to(device))
        visual_mask = task_visual_mask(model, processor)
    rows = [score_task_sample(model, processor, sample, device, controller, visual_mask, args.family)
            for sample in tqdm(samples, desc="Task scoring", dynamic_ncols=True)]
    labels = samples[0].candidate_labels
    if any(sample.candidate_labels != labels for sample in samples):
        raise ValueError("candidate label set changes within manifest")
    metrics = classification_metrics_nclass(
        [row["label_id"] for row in rows], np.asarray([row["probabilities"] for row in rows]), labels
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "count": len(rows), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
