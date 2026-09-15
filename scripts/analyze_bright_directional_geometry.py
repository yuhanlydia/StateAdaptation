#!/usr/bin/env python3
"""Oracle module-wise directional geometry for paired-image BRIGHT folds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from eventttt.bright_kv import post_image_mask
from eventttt.bright_vlm import bright_labeled_batch, load_bright_vlm
from eventttt.io import read_samples
from eventttt.kv_ttt import KVGradientCollector, freeze_model
from eventttt.schemas import DAMAGE_LABELS
from analyze_directional_geometry import analyze_group, empty_statistics
from diagnose_task_subspace import discover_projections


def balanced_query(samples, per_class: int):
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample.label].append(sample)
    missing = [label for label in DAMAGE_LABELS if len(by_class[label]) < per_class]
    if missing:
        raise ValueError(f"insufficient query examples for {missing}")
    return [sample for label in DAMAGE_LABELS for sample in by_class[label][:per_class]]


def collect(model, processor, samples, modules, visual_mask, family, description):
    device = next(model.parameters()).device
    statistics = empty_statistics(modules, DAMAGE_LABELS)
    collector = KVGradientCollector(modules)
    model.eval()
    model.config.use_cache = False
    model.enable_input_require_grads()
    try:
        for sample in tqdm(samples, desc=description, dynamic_ncols=True):
            batch, _ = bright_labeled_batch(processor, family, sample, crop_size=448)
            batch = {key: value.to(device) for key, value in batch.items()}
            model(**batch).loss.backward()
            for key, gradient in collector.gradients(visual_mask(batch["input_ids"])).items():
                gradient = gradient.detach().float().cpu()
                for block in (statistics["all"][key], statistics["classes"][sample.label][key]):
                    block["gradient_rows"].append(gradient)
                    block["samples"] += 1
            collector.clear()
            model.zero_grad(set_to_none=True)
    finally:
        collector.close()
        disable = getattr(model, "disable_input_require_grads", None)
        if callable(disable):
            disable()
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("qwen3_vl", "internvl3"), default="internvl3")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-per-class", type=int, default=100)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 27])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--kinds", nargs="+", default=["Q", "K", "V", "O"])
    args = parser.parse_args()

    support = read_samples(args.support)
    query = balanced_query(read_samples(args.query), args.query_per_class)
    model, processor = load_bright_vlm(args.model_id, args.family)
    freeze_model(model)
    modules = discover_projections(model, set(args.layers), set(args.kinds))
    visual_mask = post_image_mask(model, processor, args.family)
    support_stats = collect(model, processor, support, modules, visual_mask, args.family, "BRIGHT support")
    query_stats = collect(model, processor, query, modules, visual_mask, args.family, "BRIGHT query (oracle)")
    output = {
        "warning": "ORACLE DIAGNOSTIC: query labels are used",
        "family": args.family,
        "model_id": args.model_id,
        "rank": args.rank,
        "layers": args.layers,
        "kinds": args.kinds,
        "support_examples": len(support),
        "query_examples": len(query),
        "all": analyze_group(support_stats["all"], query_stats["all"], args.rank),
        "classes": {
            label: analyze_group(support_stats["classes"][label], query_stats["classes"][label], args.rank)
            for label in DAMAGE_LABELS
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
