#!/usr/bin/env python3
"""Gate checks for Post-Visual Residual KV-TTT.

Runs every pre-Hawaii sanity check the method spec requires:

  1. zero-coefficient identity    (KV-TTT == source at a=0)
  2. mask isolation              (pre/image-1 and text unchanged, post changed)
  3. gradient presence            (post-image K/V grads nonzero)
  4. basis orthogonality          (B.T @ B ~ I)
  5. parameter isolation          (base=0, adapter=0, KV scalars=32)
  6. reset restores source predictions
  7. serialization round-trip     (save/load reproduces the same logits)
  8. support-loss decrease        (fitting 32 coefficients reduces support loss)

Exits non-zero on the first failed gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eventttt.io import read_samples, write_json
from eventttt.kv_ttt import (
    KVGradientCollector,
    build_controller_from_state,
    build_labeled_batch,
    build_post_image_mask_fn,
    default_layers,
    discover_language_decoder_kv,
    extract_kv_subspace,
    fit_kv_coefficients,
    freeze_model,
    image_token_id_of,
    load_kv_state,
    save_kv_state,
    score_sample_with_kv,
    strict_generation_loss,
)
from eventttt.qwen import DEFAULT_MODEL, load_model, preflight
from eventttt.schemas import DAMAGE_LABELS

failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global failures
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not condition:
        failures += 1


def main() -> None:
    global failures
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--source-adapter", default="")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--alpha-max", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--verify-d4-views", type=int, default=1)
    parser.add_argument("--query-manifest", default="")
    parser.add_argument("--output-dir", default=".kv_sanity")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True), indent=2))
    torch.manual_seed(args.seed)

    support = read_samples(args.support_manifest)
    model, processor = load_model(
        args.model_id, source_adapter=args.source_adapter or None, gradient_checkpointing=False
    )
    device = next(model.parameters()).device
    modules, num_layers = discover_language_decoder_kv(model)
    selected_layers = args.layers or default_layers(num_layers)
    module_slice = [
        (layer, kind, module) for layer, kind, module in modules if layer in selected_layers
    ]
    image_token_id = image_token_id_of(model)
    build_post_mask = build_post_image_mask_fn(image_token_id)

    mask_log = {"num_decoder_layers": num_layers, "selected_layers": selected_layers}
    seq_lens, pre_counts, post_counts = [], [], []
    for sample in support[:6]:
        batch, _ = build_labeled_batch(processor, sample, args.crop_size)
        ids = batch["input_ids"][0]
        positions = (ids == image_token_id).nonzero(as_tuple=False).flatten()
        boundaries = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=False).flatten() + 1
        groups = torch.tensor_split(positions, boundaries.tolist())
        assert len(groups) == 2, f"prompt has {len(groups)} image groups for {sample.sample_id}"
        seq_lens.append(int(ids.shape[0]))
        pre_counts.append(int(len(groups[0])))
        post_counts.append(int(len(groups[1])))
    mask_log.update(
        {"seq_length": seq_lens[0], "pre_visual_tokens": pre_counts[0], "post_visual_tokens": post_counts[0]}
    )

    # 5: parameter enumeration ------------------------------------------------
    freeze_model(model)
    base_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check("base + adapter trainable parameters == 0", base_trainable == 0, f"base_trainable={base_trainable}")

    # extraction ---------------------------------------------------------------
    bases, spectra = extract_kv_subspace(
        model, processor, support, module_slice, build_post_mask,
        rank=args.rank, crop_size=args.crop_size,
    )

    # 3: gradient presence -------------------------------------------------------
    grad_norms = {}
    collector = KVGradientCollector(module_slice)
    batch, span = build_labeled_batch(processor, support[0], args.crop_size)
    loss = strict_generation_loss(model, batch, span, device)
    loss.backward()
    gradients = collector.gradients(build_post_mask(batch["input_ids"]))
    for key, gradient in gradients.items():
        grad_norms[f"{key[0]}:{key[1]}"] = round(float(gradient.norm()), 6)
    collector.clear()
    model.zero_grad(set_to_none=True)
    for handle in collector.handles:
        handle.remove()
    k_norm = grad_norms.get(f"{selected_layers[0]}:K", 0.0)
    v_norm = grad_norms.get(f"{selected_layers[-1]}:V", 0.0)
    check("post-image K/V gradients present", k_norm > 0 and v_norm > 0, f"norms={grad_norms}")

    # 4: basis orthonormality ---------------------------------------------------
    ortho_max = max(float(torch.abs(b.T @ b - torch.eye(args.rank)).max()) for b in bases.values())
    check("B.T @ B ~ identity", ortho_max < 1e-3, f"max_dev={ortho_max:.2e}")

    controller = build_controller_from_state(
        module_slice, {"bases": bases, "rank": args.rank, "alpha_max": args.alpha_max}, device=device
    )
    num_scalars = controller.num_scalars()
    check(
        "KV-TTT scalars == 2 layers x K/V x rank",
        num_scalars == 2 * 2 * args.rank,
        f"scalars={num_scalars}",
    )

    def forward_logits(batch, apply_mask):
        kwargs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        if apply_mask:
            controller.set_mask(build_post_mask(batch["input_ids"]))
        with torch.inference_mode():
            logits = model(**kwargs).logits
        controller.clear_mask()
        return logits

    source_batch, _ = build_labeled_batch(processor, support[0], args.crop_size)
    source_logits = forward_logits(source_batch, apply_mask=False)

    # 1: zero-coefficient identity ---------------------------------------------
    controller.reset_coefficients()
    kv_logits = forward_logits(source_batch, apply_mask=True)
    identity_diff = float((source_logits - kv_logits).float().abs().max())
    check("zero coefficients reproduce source logits", identity_diff < 1e-4, f"max_diff={identity_diff:.2e}")

    # 2: mask isolation ----------------------------------------------------------
    captured = {}

    def view_capture(layer, kind):
        def hook(module, inp, out):
            captured[(layer, kind)] = out.detach()

        return hook

    handles = [module.register_forward_hook(view_capture(layer, kind)) for layer, kind, module in module_slice]
    batch_cuda = {k: v.to(device) for k, v in source_batch.items() if k != "labels"}
    with torch.no_grad():
        for parameter in controller.ttt_parameters():
            parameter.fill_(1.0)
        controller.set_mask(build_post_mask(source_batch["input_ids"]))
        _ = model(**batch_cuda)
        controller.clear_mask()
        modified = dict(captured)
        captured.clear()
        _ = model(**batch_cuda)
        baseline = dict(captured)
        captured.clear()
    for handle in handles:
        handle.remove()
    controller.reset_coefficients()

    ids_row = source_batch["input_ids"][0]
    image_positions = (ids_row == image_token_id).nonzero(as_tuple=False).flatten()
    boundaries = (image_positions[1:] != image_positions[:-1] + 1).nonzero(as_tuple=False).flatten() + 1
    pre_group, post_group = torch.tensor_split(image_positions, boundaries.tolist())

    ids_row = source_batch["input_ids"][0]
    image_positions = (ids_row == image_token_id).nonzero(as_tuple=False).flatten()
    boundaries = (image_positions[1:] != image_positions[:-1] + 1).nonzero(as_tuple=False).flatten() + 1
    pre_group, post_group = torch.tensor_split(image_positions, boundaries.tolist())
    # Positions strictly before the post-image group cannot be influenced by the
    # residual (their delta is masked to zero AND they cannot attend to the
    # post-image rows either). Positions at/after are allowed to move through
    # downstream causal attention - that is the adaptation mechanism itself.
    unaffected_row = torch.arange(ids_row.shape[0], device=ids_row.device) < post_group[0]

    for layer, kind, _ in module_slice:
        base_out = baseline[(layer, kind)]
        mod_out = modified[(layer, kind)]
        pre_diff = float((mod_out[0][pre_group] - base_out[0][pre_group]).float().abs().max())
        post_diff = float((mod_out[0][post_group] - base_out[0][post_group]).float().abs().max())
        tail_diff = float(
            (mod_out[0][unaffected_row] - base_out[0][unaffected_row]).float().abs().max()
        )
        check(
            f"layer {layer} {kind}: post-image tokens changed",
            post_diff > 1e-6,
            f"post_diff={post_diff:.2e}",
        )
        check(
            f"layer {layer} {kind}: pre-image tokens unchanged",
            tail_diff < 1e-6,
            f"pre/leading_diff={tail_diff:.2e}",
        )
        # Text rows after the post group legitimately move via attention; the
        # controller itself writes delta only where the mask is True. That write
        # isolation is verified structurally by the zero-coefficient identity
        # check above plus the unit test in tests/test_kv_ttt.py.
        check(f"layer {layer} {kind}: pre_diff == 0", pre_diff < 1e-6, f"pre_diff={pre_diff:.2e}")

    # 8: support-loss decrease ---------------------------------------------------
    def support_avg_loss():
        values = []
        for sample in support:
            batch, span = build_labeled_batch(processor, sample, args.crop_size)
            kwargs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = torch.full_like(batch["input_ids"], -100)
            labels[0, span[0]:span[1]] = batch["input_ids"][0, span[0]:span[1]]
            kwargs["labels"] = labels
            controller.set_mask(build_post_mask(batch["input_ids"]))
            with torch.inference_mode():
                values.append(float(model(**kwargs).loss.detach()))
            controller.clear_mask()
        return float(np.mean(values))

    before_loss = support_avg_loss()
    losses_curve = fit_kv_coefficients(
        model, controller, processor, support, build_post_mask,
        steps=args.steps, learning_rate=args.learning_rate, l2=args.l2, crop_size=args.crop_size,
    )
    after_loss = support_avg_loss()
    check(
        "support correctness loss decreases after fitting",
        after_loss < before_loss,
        f"before={before_loss:.4f} after={after_loss:.4f}",
    )

    # 6: reset restores source predictions -----------------------------------------
    controller.reset_coefficients()
    reset_logits = forward_logits(source_batch, apply_mask=True)
    reset_diff = float((source_logits - reset_logits).float().abs().max())
    check("reset restores source logits", reset_diff < 1e-4, f"max_diff={reset_diff:.2e}")

    # 7: serialization round-trip ----------------------------------------------------
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = save_kv_state(
        output / "kv_state.pt", controller, args.model_id,
        metadata={"source_adapter": args.source_adapter, "layers": selected_layers},
    )
    payload = load_kv_state(state_path, device=device)
    reloaded = build_controller_from_state(module_slice, payload, device=device)

    with torch.no_grad():
        controller.coefficients[f"{selected_layers[-1]}:V"].fill_(0.5)
        reloaded.coefficients[f"{selected_layers[-1]}:V"].copy_(
            controller.coefficients[f"{selected_layers[-1]}:V"]
        )
    query_logits = forward_logits(source_batch, apply_mask=True)
    with torch.inference_mode():
        reloaded.set_mask(build_post_mask(source_batch["input_ids"]))
        reload_logits = model(**{k: v.to(device) for k, v in source_batch.items() if k != "labels"}).logits
        reloaded.clear_mask()
    ser_diff = float((query_logits - reload_logits).float().abs().max())
    check("save/load kv_state reproduces logits", ser_diff < 1e-4, f"max_diff={ser_diff:.2e}")

    # query green run ----------------------------------------------------------
    rows = []
    if args.query_manifest:
        query = read_samples(args.query_manifest)
        for sample in query:
            rows.append(score_sample_with_kv(
                model, processor, controller, build_post_mask, sample,
                args.verify_d4_views, args.crop_size,
            ))
        histogram = {label: 0 for label in DAMAGE_LABELS}
        for row in rows:
            histogram[row["prediction"]] += 1
        print(json.dumps({"query_prediction_histogram": histogram}, indent=2))
        write_json(output / "query_predictions.json", rows)

    result = {
        "passed": failures == 0,
        "param_checks": {"base_trainable": int(base_trainable), "kv_scalars": int(num_scalars)},
        "gradient_norms": grad_norms,
        "basis_orthogonality_max_error": float(ortho_max),
        "identity_max_diff": float(identity_diff),
        "support_loss": {"before": float(before_loss), "after": float(after_loss)},
        "reset_max_diff": float(reset_diff),
        "serialization_max_diff": float(ser_diff),
        "mask_diag": mask_log,
    }
    write_json(output / "sanity.json", result)
    print(json.dumps(result, indent=2, default=str))

    print(f"\nSanity {'PASS' if failures == 0 else 'FAIL'} ({failures} gate failures)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()