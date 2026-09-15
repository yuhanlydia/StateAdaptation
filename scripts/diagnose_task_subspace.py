#!/usr/bin/env python3
"""Oracle and gradient-geometry diagnostics for task-level residual tuning.

This script intentionally permits query labels for mechanism analysis.  Its
outputs are oracle diagnostics and must never be reported as clean test-time
adaptation results.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from eventttt.io import read_task_samples
from eventttt.kv_ttt import KVGradientCollector, ResidualKVController, freeze_model
from eventttt.metrics import classification_metrics_nclass
from eventttt.task_kv import fit_task_coefficients, task_visual_mask
from eventttt.task_qwen import labeled_batch, score_task_sample
from eventttt.task_vlm import default_task_model, load_task_model


_PROJECTION_RE = re.compile(
    r"\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
)
_KINDS = {"q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O"}


def discover_projections(model, layers, kinds):
    found = []
    for name, module in model.named_modules():
        if any(part in name.lower() for part in ("vision_tower", "vision_model", "vision_encoder")):
            continue
        match = _PROJECTION_RE.search(name)
        if match and int(match.group(1)) in layers and _KINDS[match.group(2)] in kinds:
            found.append((int(match.group(1)), _KINDS[match.group(2)], module))
    expected = {(layer, kind) for layer in layers for kind in kinds}
    actual = {(layer, kind) for layer, kind, _ in found}
    if expected != actual:
        raise RuntimeError(f"projection discovery mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return found


def collect_covariance(model, processor, samples, modules, visual_mask, family, description):
    device = next(model.parameters()).device
    covariance = {
        (layer, kind): torch.zeros(module.out_features, module.out_features,
                                   dtype=torch.float32, device=device)
        for layer, kind, module in modules
    }
    collector = KVGradientCollector(modules)
    model.eval()
    model.config.use_cache = False
    model.enable_input_require_grads()
    try:
        for sample in tqdm(samples, desc=description, dynamic_ncols=True):
            batch, _ = labeled_batch(processor, sample, family=family)
            batch = {key: value.to(device) for key, value in batch.items()}
            model(**batch).loss.backward()
            for key, gradient in collector.gradients(visual_mask(batch["input_ids"])).items():
                gradient = gradient.float()
                covariance[key].add_(gradient.T @ gradient)
            collector.clear()
            model.zero_grad(set_to_none=True)
    finally:
        collector.close()
        disable = getattr(model, "disable_input_require_grads", None)
        if callable(disable):
            disable()
    return {key: value.cpu() for key, value in covariance.items()}


def top_bases(covariance, rank):
    bases = {}
    for key, matrix in covariance.items():
        _, vectors = torch.linalg.eigh(matrix.float())
        bases[key] = vectors[:, -rank:].contiguous()
    return bases


def overlap(support_bases, query_covariance):
    rows, captured_total, total = {}, 0.0, 0.0
    for key, basis in support_bases.items():
        covariance = query_covariance[key].float()
        captured = float(torch.trace(basis.T @ covariance @ basis))
        energy = float(torch.trace(covariance))
        rows[f"{key[0]}:{key[1]}"] = {
            "captured": captured,
            "total": energy,
            "ratio": captured / max(energy, 1e-12),
        }
        captured_total += captured
        total += energy
    return {"weighted_ratio": captured_total / max(total, 1e-12), "modules": rows}


def score(model, processor, family, samples, controller, visual_mask):
    device = next(model.parameters()).device
    rows = [score_task_sample(model, processor, sample, device, controller,
                              visual_mask, family)
            for sample in tqdm(samples, desc="Oracle scoring", dynamic_ncols=True)]
    metrics = classification_metrics_nclass(
        [row["label_id"] for row in rows],
        np.asarray([row["probabilities"] for row in rows]),
        samples[0].candidate_labels,
    )
    return metrics


def evaluate_basis(model, processor, family, support, query, modules, basis,
                   visual_mask, rank, alpha, steps, learning_rate, l2):
    controller = ResidualKVController(
        modules, basis, rank=rank, alpha_max=alpha,
        coefficient_mode="full", device=next(model.parameters()).device,
    )
    try:
        losses = fit_task_coefficients(
            model, processor, support, controller, visual_mask, steps=steps,
            learning_rate=learning_rate, l2=l2, family=family,
        )
        return {
            "metrics": score(model, processor, family, query, controller, visual_mask),
            "support_losses": losses,
            "scalars": controller.num_scalars(),
        }
    finally:
        controller.close()
        model.zero_grad(set_to_none=True)


def parse_sites(raw):
    sites = []
    for item in raw:
        kinds = tuple(dict.fromkeys(item.upper()))
        if not kinds or any(kind not in "QKVO" for kind in kinds):
            raise ValueError(f"invalid site {item!r}; use KV, Q, O, or QKVO")
        sites.append((item.upper(), kinds))
    return sites


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
    parser.add_argument("--sites", nargs="+", default=["KV", "Q", "O", "QKVO"])
    parser.add_argument("--query-gradient-limit", type=int, default=None)
    parser.add_argument("--skip-query-oracle", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    model_id = args.model_id or default_task_model(args.family)
    support = read_task_samples(args.support)
    query = read_task_samples(args.query)
    gradient_query = query[:args.query_gradient_limit] if args.query_gradient_limit else query
    sites = parse_sites(args.sites)
    all_kinds = tuple(dict.fromkeys(kind for _, kinds in sites for kind in kinds))
    model, processor = load_task_model(
        model_id, args.family, gradient_checkpointing=(args.family == "internvl3")
    )
    freeze_model(model)
    visual_mask = task_visual_mask(model, processor)
    modules = discover_projections(model, set(args.layers), set(all_kinds))
    result = {
        "warning": "ORACLE DIAGNOSTIC: query labels are used for covariance and configuration analysis",
        "family": args.family, "model_id": model_id, "layers": args.layers,
        "rank": args.rank, "alpha": args.alpha, "steps": args.steps,
        "learning_rate": args.learning_rate, "l2": args.l2,
        "support_examples": len(support), "query_examples": len(query),
        "query_gradient_examples": len(gradient_query), "sites": {},
    }
    support_covariance = collect_covariance(
        model, processor, support, modules, visual_mask, args.family, "Support gradients"
    )
    query_covariance = collect_covariance(
        model, processor, gradient_query, modules, visual_mask, args.family, "Query gradients (oracle)"
    )
    for site, kinds in sites:
        selected_modules = [row for row in modules if row[1] in kinds]
        keys = {(layer, kind) for layer, kind, _ in selected_modules}
        support_cov = {key: value for key, value in support_covariance.items() if key in keys}
        query_cov = {key: value for key, value in query_covariance.items() if key in keys}
        support_basis = top_bases(support_cov, args.rank)
        entry = {"support_to_query_overlap": overlap(support_basis, query_cov)}
        entry["support_basis"] = evaluate_basis(
            model, processor, args.family, support, query, selected_modules,
            support_basis, visual_mask, args.rank, args.alpha, args.steps,
            args.learning_rate, args.l2,
        )
        if site == "KV" and not args.skip_query_oracle:
            entry["query_basis_oracle"] = evaluate_basis(
                model, processor, args.family, support, query, selected_modules,
                top_bases(query_cov, args.rank), visual_mask, args.rank,
                args.alpha, args.steps, args.learning_rate, args.l2,
            )
        result["sites"][site] = entry
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
