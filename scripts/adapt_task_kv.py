#!/usr/bin/env python3
"""Adapt a frozen Qwen VLM with task-generic KV-TTT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eventttt.io import model_fingerprint, read_task_samples
from eventttt.kv_ttt import build_controller_from_state, freeze_model, save_kv_state
from eventttt.qwen import preflight, trainable_parameter_report
from eventttt.task_kv import extract_task_subspace, fit_task_coefficients, task_visual_mask, _selected_modules
from eventttt.task_vlm import default_task_model, load_task_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", choices=("qwen2", "qwen3_vl", "internvl3"), default="qwen2")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--basis-mode", choices=("covariance", "random"), default="covariance")
    parser.add_argument("--coefficient-mode", choices=("full", "diagonal"), default="full")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()
    if args.model_id is None:
        args.model_id = default_task_model(args.family)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(json.dumps(preflight(require_gpu=True), indent=2))
    samples = read_task_samples(args.support_manifest)
    model, processor = load_task_model(args.model_id, args.family,
                                       gradient_checkpointing=False, use_lora=False)
    freeze_model(model)
    report = trainable_parameter_report(model)
    modules, num_layers, layers = _selected_modules(model, args.layers)
    mask = task_visual_mask(model, processor)
    bases, spectra = extract_task_subspace(
        model, processor, samples, modules, mask, args.rank, args.basis_mode, args.seed,
        args.family
    )
    state = {"bases": bases, "rank": args.rank, "basis_mode": args.basis_mode,
             "alpha_max": args.alpha_max, "coefficient_mode": args.coefficient_mode}
    controller = build_controller_from_state(modules, state, device=next(model.parameters()).device)
    losses = fit_task_coefficients(model, processor, samples, controller, mask,
                                   args.steps, args.learning_rate, args.l2, args.family)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    save_kv_state(output / "kv_state.pt", controller, args.model_id, metadata={
        "base_model_only": True, "model_sha256": model_fingerprint(args.model_id),
        "dataset": samples[0].dataset, "support_manifest": str(Path(args.support_manifest).resolve()),
    })
    (output / "extraction.json").write_text(json.dumps({
        "method": "task_gradient_covariance_kv", "basis_mode": args.basis_mode,
        "support_examples": len(samples), "rank": args.rank, "layers": layers,
        "num_decoder_layers": num_layers, "spectra": spectra,
        "trainable_before_extraction": report, "arguments": vars(args),
    }, indent=2) + "\n")
    (output / "adaptation.json").write_text(json.dumps({
        "losses": losses, "steps": args.steps, "learning_rate": args.learning_rate,
        "l2": args.l2, "alpha_max": args.alpha_max,
        "coefficient_mode": args.coefficient_mode,
        "kv_scalars": controller.num_scalars(),
    }, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "losses": losses,
                      "kv_scalars": controller.num_scalars()}, indent=2))


if __name__ == "__main__":
    main()
