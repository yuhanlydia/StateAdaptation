#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from eventttt.io import (
    build_eval_config,
    iter_jsonl,
    read_samples,
    write_json,
)
from eventttt.kv_ttt import (
    build_controller_from_state,
    build_post_image_mask_fn,
    discover_language_decoder_kv,
    image_token_id_of,
    load_kv_state,
    score_sample_with_kv,
)
from eventttt.metrics import metrics_by_event
from eventttt.qwen import DEFAULT_MODEL, load_model, preflight, score_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an adapter with D4 product-of-experts")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter", default="", help="LoRA adapter path; leave empty for base model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--no-lora", action="store_true", help="Evaluate raw base model without any LoRA")
    parser.add_argument("--d4-views", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument(
        "--kv-state",
        default="",
        help="Path to a KV-TTT kv_state.pt; apply the learned post-image residual K/V controller",
    )
    args = parser.parse_args()

    print(json.dumps(preflight(require_gpu=True), indent=2))
    samples = read_samples(args.manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config = build_eval_config(
        model_id=args.model_id,
        adapter=args.adapter or None,
        kv_state=args.kv_state or None,
        manifest=args.manifest,
        d4_views=args.d4_views,
        crop_size=args.crop_size,
        no_lora=args.no_lora,
    )

    predictions = output / "predictions.jsonl"
    completed_rows = list(iter_jsonl(predictions)) if predictions.exists() else []
    if completed_rows:
        config_path = output / "eval_config.json"
        if not config_path.exists():
            raise RuntimeError(
                f"{config_path} is missing but {predictions} exists; refusing to resume "
                "predictions with unknown provenance"
            )
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(
                "Refusing to resume: evaluation configuration changed. "
                "Previous eval_config.json does not match this invocation.\n"
                f"previous: {json.dumps(previous, indent=2, sort_keys=True)}\n"
                f"current:  {json.dumps(config, indent=2, sort_keys=True)}"
            )
        print(f"resume: evaluation config matches {config_path}")
    write_json(output / "eval_config.json", config)
    completed_ids = {row["sample_id"] for row in completed_rows}
    if len(completed_ids) != len(completed_rows):
        raise ValueError(f"Duplicate sample IDs in partial predictions: {predictions}")
    pending = [sample for sample in samples if sample.sample_id not in completed_ids]
    if completed_rows:
        print(f"resume: {len(completed_rows)} predictions complete, {len(pending)} remaining")
    model, processor = load_model(
        args.model_id,
        source_adapter=args.adapter or None,
        gradient_checkpointing=False,
        use_lora=not args.no_lora,
    )

    controller = None
    build_post_mask = None
    if args.kv_state:
        payload = load_kv_state(args.kv_state, device=next(model.parameters()).device)
        if payload["model_id"] != args.model_id:
            raise ValueError(
                f"kv-state model_id {payload['model_id']} does not match --model-id {args.model_id}"
            )
        metadata = payload.get("metadata") or {}
        if "adapter_sha256" not in metadata:
            raise ValueError(
                f"kv-state {args.kv_state} carries no adapter_sha256 binding; "
                "refuse to apply it to an unchecked model"
            )
        bound_model = metadata.get("model_sha256")
        if not bound_model or bound_model != config["model_sha256"]:
            raise ValueError(
                "Refusing to apply kv-state: its base model fingerprint does not "
                "match the evaluation model"
            )
        bound_adapter = metadata["adapter_sha256"]
        if bound_adapter is None:
            if args.adapter or not args.no_lora or not metadata.get("base_model_only"):
                raise ValueError(
                    "Raw-VLM kv-state requires --no-lora and no --adapter"
                )
        elif not args.adapter or bound_adapter != config["adapter_sha256"]:
            raise ValueError(
                "Refusing to apply kv-state: its source adapter fingerprint "
                f"{bound_adapter} does not match --adapter {config['adapter_sha256']}"
            )
        modules, _ = discover_language_decoder_kv(model)
        selected = set(payload["layers"])
        module_slice = [(layer, kind, module) for layer, kind, module in modules if layer in selected]
        missing = selected - {layer for layer, _, _ in module_slice}
        if missing:
            raise ValueError(f"kv-state layers missing from loaded model: {sorted(missing)}")
        controller = build_controller_from_state(
            module_slice, payload, device=next(model.parameters()).device
        )
        build_post_mask = build_post_image_mask_fn(image_token_id_of(model))
        print(f"kv-state: rank={payload['rank']} layers={payload['layers']}")

    scorer = (
        lambda sample: score_sample_with_kv(
            model, processor, controller, build_post_mask, sample, args.d4_views, args.crop_size
        )
    ) if controller else (lambda sample: score_sample(model, processor, sample, args.d4_views, args.crop_size))

    with predictions.open("a", encoding="utf-8", buffering=1) as handle:
        for sample in tqdm(pending, desc="Scoring", dynamic_ncols=True):
            row = scorer(sample)
            handle.write(json.dumps(row) + "\n")
    rows = list(iter_jsonl(predictions))
    if len(rows) != len(samples):
        raise RuntimeError(f"Expected {len(samples)} predictions, found {len(rows)}")
    write_json(output / "metrics.json", metrics_by_event(rows))


if __name__ == "__main__":
    main()
