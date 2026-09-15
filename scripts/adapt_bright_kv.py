#!/usr/bin/env python3
"""Support-only Random-KV and Gradient-Covariance KV/Ours on one BRIGHT fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from eventttt.bright_kv import batch_builder, discover_bright_kv, post_image_mask
from eventttt.bright_vlm import load_bright_vlm, score_bright_sample
from eventttt.io import read_samples
from eventttt.kv_ttt import (
    ResidualKVController, default_layers, extract_kv_subspace, fit_kv_coefficients,
)
from eventttt.metrics import classification_metrics
from eventttt.phi_kv import PhiKVController, extract_phi_subspace, fit_phi_coefficients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--family", choices=("phi", "gemma", "llama", "qwen3_vl", "internvl3"), required=True)
    parser.add_argument("--method", choices=("random_kv", "gradient_cov_kv"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--coefficient-mode", choices=("full", "diagonal"), default="full")
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 27])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    support = read_samples(args.support_manifest)
    query = read_samples(args.query_manifest)
    if args.query_limit is not None:
        query = query[:args.query_limit]
    model, processor = load_bright_vlm(args.model_id, args.family)
    modules, num_layers = discover_bright_kv(model, args.family, processor)
    selected = set(args.layers)
    if min(selected) < 0 or max(selected) >= num_layers:
        raise ValueError(f"Requested layers {sorted(selected)} outside model depth {num_layers}")
    if args.family == "phi":
        modules = [item for item in modules if item[0] in selected]
    else:
        modules = [item for item in modules if item[0] in selected]
    mask = post_image_mask(model, processor, args.family)
    builder = batch_builder(processor, args.family)
    basis_mode = "random" if args.method == "random_kv" else "covariance"
    if args.family == "phi":
        bases, spectra = extract_phi_subspace(
            model, processor, support, modules, rank=args.rank,
            basis_mode=basis_mode, seed=args.seed, batch_builder=builder,
            mask_builder=mask,
        )
        controller = PhiKVController(
            modules, bases, rank=args.rank, alpha_max=args.alpha_max,
            coefficient_mode=args.coefficient_mode,
            device=next(model.parameters()).device,
        )
        losses = fit_phi_coefficients(
            model, processor, support, controller, steps=args.steps,
            learning_rate=0.05, batch_builder=builder, mask_builder=mask,
        )
    else:
        bases, spectra = extract_kv_subspace(
            model, processor, support, modules, mask, rank=args.rank,
            crop_size=args.crop_size, basis_mode=basis_mode, basis_seed=args.seed,
            batch_builder=builder,
        )
        controller = ResidualKVController(
            modules, bases, rank=args.rank, alpha_max=args.alpha_max,
            coefficient_mode=args.coefficient_mode,
            device=next(model.parameters()).device,
        )
        losses = fit_kv_coefficients(
            model, controller, processor, support, mask, steps=args.steps,
            learning_rate=0.05, crop_size=args.crop_size, batch_builder=builder,
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample in tqdm(query, desc=f"{args.family} {args.method}", dynamic_ncols=True):
            row = score_bright_sample(
                model, processor, args.family, sample, args.crop_size,
                controller=controller, mask_builder=mask,
            )
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
    metrics = classification_metrics(
        [row["label_id"] for row in rows],
        np.asarray([row["probabilities"] for row in rows]),
    )
    torch.save({
        "method": args.method, "family": args.family, "rank": args.rank,
        "layers": sorted(selected), "spectra": spectra,
        "losses": losses, "bases": {str(k): v.cpu() for k, v in bases.items()},
        "coefficients_raw": {k: v.detach().cpu() for k, v in controller.coefficients.items()},
    }, output / "kv_state.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "adaptation.json").write_text(json.dumps({
        "method": args.method, "family": args.family, "model_id": args.model_id,
        "support_examples": len(support), "query_examples": len(query),
        "rank": args.rank, "alpha_max": args.alpha_max,
        "coefficient_mode": args.coefficient_mode, "steps": args.steps,
        "learning_rate": 0.05, "l2": 1e-3, "layers": sorted(selected),
        "losses": losses, "arguments": vars(args),
    }, indent=2) + "\n")
    controller.close()
    print(json.dumps({"output_dir": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
