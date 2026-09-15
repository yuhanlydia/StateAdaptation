#!/usr/bin/env python3
"""Module-wise, signed, and class-conditional gradient geometry diagnostics.

Query labels are intentionally used. Outputs are mechanism diagnostics only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from eventttt.io import read_task_samples
from eventttt.kv_ttt import KVGradientCollector, freeze_model
from eventttt.task_kv import task_visual_mask
from eventttt.task_qwen import labeled_batch
from eventttt.task_vlm import default_task_model, load_task_model
from diagnose_task_subspace import discover_projections, top_bases


def empty_statistics(modules, labels):
    keys = [(layer, kind) for layer, kind, _ in modules]
    def block():
        return {
            key: {
                "gradient_rows": [],
                "samples": 0,
            }
            for key in keys
        }

    return {"all": block(), "classes": {label: block() for label in labels}}


def collect_statistics(model, processor, samples, modules, visual_mask, family, description):
    device = next(model.parameters()).device
    labels = tuple(samples[0].candidate_labels)
    statistics = empty_statistics(modules, labels)
    collector = KVGradientCollector(modules)
    model.eval()
    model.config.use_cache = False
    model.enable_input_require_grads()
    try:
        for sample in tqdm(samples, desc=description, dynamic_ncols=True):
            batch, _ = labeled_batch(processor, sample, family=family)
            batch = {key: value.to(device) for key, value in batch.items()}
            model(**batch).loss.backward()
            gradients = collector.gradients(visual_mask(batch["input_ids"]))
            for key, gradient in gradients.items():
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


def cosine(left, right):
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-20:
        return None
    return float(torch.dot(left, right) / denominator)


def module_geometry(support_block, query_block, basis):
    support_rows = torch.cat(support_block["gradient_rows"], dim=0)
    query_rows = torch.cat(query_block["gradient_rows"], dim=0)
    captured = float(torch.linalg.vector_norm(query_rows @ basis).square())
    total = float(torch.linalg.vector_norm(query_rows).square())
    support_mean = support_rows.mean(dim=0)
    query_mean = query_rows.mean(dim=0)
    support_projected = basis @ (basis.T @ support_mean)
    query_projected = basis @ (basis.T @ query_mean)
    return {
        "rho": captured / max(total, 1e-20),
        "kappa": cosine(support_projected, query_projected),
        "captured_energy": captured,
        "query_energy": total,
        "support_rows": int(support_rows.shape[0]),
        "query_rows": int(query_rows.shape[0]),
        "support_samples": support_block["samples"],
        "query_samples": query_block["samples"],
    }


def aggregate(rows):
    captured = sum(row["captured_energy"] for row in rows)
    total = sum(row["query_energy"] for row in rows)
    kappas = [row["kappa"] for row in rows if row["kappa"] is not None]
    weights = [row["query_energy"] for row in rows if row["kappa"] is not None]
    return {
        "rho": captured / max(total, 1e-20),
        "energy_weighted_kappa": (
            sum(value * weight for value, weight in zip(kappas, weights)) / max(sum(weights), 1e-20)
            if kappas else None
        ),
    }


def analyze_group(support_group, query_group, rank):
    covariances = {
        key: torch.cat(block["gradient_rows"], dim=0).T @ torch.cat(block["gradient_rows"], dim=0)
        for key, block in support_group.items()
    }
    bases = top_bases(covariances, rank)
    modules = {
        f"{key[0]}:{key[1]}": module_geometry(support_group[key], query_group[key], bases[key])
        for key in sorted(support_group)
    }
    by_kind, by_layer = defaultdict(list), defaultdict(list)
    for name, row in modules.items():
        layer, kind = name.split(":")
        by_kind[kind].append(row)
        by_layer[layer].append(row)
    return {
        "aggregate": aggregate(list(modules.values())),
        "by_kind": {key: aggregate(value) for key, value in sorted(by_kind.items())},
        "by_layer": {key: aggregate(value) for key, value in sorted(by_layer.items(), key=lambda x: int(x[0]))},
        "modules": modules,
    }


def analyze_query_prior_mixture_oracle(support_stats, query_stats, labels, rank):
    """Reweight support class means by the oracle query class prior.

    The support-derived covariance/basis is unchanged.  Query labels affect
    only the mixture weights, making this a diagnostic rather than a valid TTA
    procedure.
    """
    support_group, query_group = support_stats["all"], query_stats["all"]
    query_counts = {
        label: query_stats["classes"][label][next(iter(query_group))]["samples"]
        for label in labels
    }
    query_total = sum(query_counts.values())
    priors = {label: query_counts[label] / query_total for label in labels}
    covariances = {
        key: torch.cat(block["gradient_rows"], dim=0).T @ torch.cat(block["gradient_rows"], dim=0)
        for key, block in support_group.items()
    }
    bases = top_bases(covariances, rank)
    modules = {}
    for key in sorted(support_group):
        basis = bases[key]
        support_mean = sum(
            priors[label]
            * torch.cat(support_stats["classes"][label][key]["gradient_rows"], dim=0).mean(dim=0)
            for label in labels
        )
        query_rows = torch.cat(query_group[key]["gradient_rows"], dim=0)
        query_mean = query_rows.mean(dim=0)
        captured = float(torch.linalg.vector_norm(query_rows @ basis).square())
        total = float(torch.linalg.vector_norm(query_rows).square())
        modules[f"{key[0]}:{key[1]}"] = {
            "rho": captured / max(total, 1e-20),
            "kappa": cosine(
                basis @ (basis.T @ support_mean),
                basis @ (basis.T @ query_mean),
            ),
            "captured_energy": captured,
            "query_energy": total,
        }
    by_kind, by_layer = defaultdict(list), defaultdict(list)
    for name, row in modules.items():
        layer, kind = name.split(":")
        by_kind[kind].append(row)
        by_layer[layer].append(row)
    return {
        "warning": "ORACLE: support class means use query-label class priors",
        "query_class_counts": query_counts,
        "query_class_priors": priors,
        "aggregate": aggregate(list(modules.values())),
        "by_kind": {key: aggregate(value) for key, value in sorted(by_kind.items())},
        "by_layer": {key: aggregate(value) for key, value in sorted(by_layer.items(), key=lambda x: int(x[0]))},
        "modules": modules,
    }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("qwen3_vl", "internvl3"), default="internvl3")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--support", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 27])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--kinds", nargs="+", default=["Q", "K", "V", "O"])
    args = parser.parse_args()

    model_id = args.model_id or default_task_model(args.family)
    support, query = read_task_samples(args.support), read_task_samples(args.query)
    if tuple(support[0].candidate_labels) != tuple(query[0].candidate_labels):
        raise ValueError("support/query candidate labels differ")
    model, processor = load_task_model(
        model_id, args.family, gradient_checkpointing=(args.family == "internvl3")
    )
    freeze_model(model)
    modules = discover_projections(model, set(args.layers), set(args.kinds))
    visual_mask = task_visual_mask(model, processor)
    support_stats = collect_statistics(
        model, processor, support, modules, visual_mask, args.family, "Support geometry"
    )
    query_stats = collect_statistics(
        model, processor, query, modules, visual_mask, args.family, "Query geometry (oracle)"
    )
    output = {
        "warning": "ORACLE DIAGNOSTIC: query labels are used",
        "family": args.family,
        "model_id": model_id,
        "rank": args.rank,
        "layers": args.layers,
        "kinds": args.kinds,
        "support_examples": len(support),
        "query_examples": len(query),
        "all": analyze_group(support_stats["all"], query_stats["all"], args.rank),
        "query_prior_mixture_oracle": analyze_query_prior_mixture_oracle(
            support_stats, query_stats, support[0].candidate_labels, args.rank
        ),
        "classes": {
            label: analyze_group(
                support_stats["classes"][label], query_stats["classes"][label], args.rank
            )
            for label in support[0].candidate_labels
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
