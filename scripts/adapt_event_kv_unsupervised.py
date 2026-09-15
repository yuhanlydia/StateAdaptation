#!/usr/bin/env python3
"""Label-free event adaptation using identity-view pseudo-label consistency."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from eventttt.io import adapter_fingerprint, model_fingerprint, read_samples, write_json
from eventttt.kv_ttt import (
    build_controller_from_state,
    build_post_image_mask_fn,
    default_layers,
    discover_language_decoder_kv,
    extract_consistency_kv_subspace,
    fit_consistency_kv_coefficients,
    freeze_model,
    image_token_id_of,
    save_kv_state,
)
from eventttt.qwen import DEFAULT_MODEL, load_model, preflight, trainable_parameter_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlabeled-manifest", required=True)
    parser.add_argument(
        "--source-adapter",
        default="",
        help="optional LoRA starting point; leave empty to adapt the raw base VLM",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--coefficient-mode", choices=("diagonal", "full"), default="diagonal")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--d4-views", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True), indent=2))
    torch.manual_seed(args.seed)
    samples = read_samples(args.unlabeled_manifest)
    if not samples:
        raise ValueError("Unlabeled adaptation manifest is empty")

    model, processor = load_model(
        args.model_id,
        source_adapter=args.source_adapter or None,
        gradient_checkpointing=False,
        use_lora=bool(args.source_adapter),
    )
    freeze_model(model)
    modules, num_layers = discover_language_decoder_kv(model)
    selected_layers = args.layers or default_layers(num_layers)
    selected = [(layer, kind, module) for layer, kind, module in modules if layer in selected_layers]
    if len(selected) != 2 * len(set(selected_layers)):
        raise ValueError(f"Could not find K/V projections for every selected layer: {selected_layers}")
    report = trainable_parameter_report(model)
    if report["trainable"] != 0:
        raise RuntimeError("VLM and source LoRA must remain frozen")
    device = next(model.parameters()).device
    build_post_mask = build_post_image_mask_fn(image_token_id_of(model))

    started = time.time()
    bases, spectra = extract_consistency_kv_subspace(
        model, processor, samples, selected, build_post_mask,
        rank=args.rank, d4_views=args.d4_views, crop_size=args.crop_size,
    )
    extraction_seconds = time.time() - started
    controller = build_controller_from_state(
        selected,
        {"bases": bases, "rank": args.rank, "alpha_max": args.alpha_max,
         "coefficient_mode": args.coefficient_mode},
        device=device,
    )
    started = time.time()
    losses = fit_consistency_kv_coefficients(
        model, controller, processor, samples, build_post_mask,
        steps=args.steps, learning_rate=args.learning_rate, l2=args.l2,
        d4_views=args.d4_views, crop_size=args.crop_size,
    )
    adaptation_seconds = time.time() - started

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "objective": "identity_view_hard_pseudolabel_distillation",
        "teacher": "frozen_source_model_identity_view_argmax",
        "student_views": "non_identity_d4_views",
        "labels_used": False,
        "unlabeled_manifest": str(Path(args.unlabeled_manifest).resolve()),
        "unlabeled_examples": len(samples),
        "d4_views": args.d4_views,
        "source_adapter": (
            str(Path(args.source_adapter).resolve()) if args.source_adapter else None
        ),
        "base_model_only": not bool(args.source_adapter),
        "adapter_sha256": (
            adapter_fingerprint(args.source_adapter) if args.source_adapter else None
        ),
        "model_sha256": model_fingerprint(args.model_id),
        "seed": args.seed,
    }
    save_kv_state(output / "kv_state.pt", controller, args.model_id, metadata=metadata)
    write_json(output / "extraction.json", {
        **metadata, "selected_layers": selected_layers, "rank": args.rank,
        "spectra": spectra, "gradient_passes": len(samples) * (args.d4_views - 1),
        "kv_ttt_scalars": controller.num_scalars(),
        "extraction_seconds": round(extraction_seconds, 2), "arguments": vars(args),
    })
    write_json(output / "adaptation.json", {
        "objective": metadata["objective"], "labels_used": False,
        "optimizer": "Adam", "steps": args.steps, "learning_rate": args.learning_rate,
        "coefficient_l2": args.l2, "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "effective_gamma": controller.effective_gamma(),
        "adaptation_seconds": round(adaptation_seconds, 2),
    })
    print(json.dumps({"output_dir": str(output), "losses": losses}, indent=2))


if __name__ == "__main__":
    main()
