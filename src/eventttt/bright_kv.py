"""BRIGHT adapters for the shared Gradient-Covariance KV controller.

The controller itself remains the tested Qwen implementation; this module
only supplies the paired-image batch and family-specific post-image mask for
Gemma 3 and LLaVA/Llama. Phi's fused qkv path remains in ``phi_kv.py``.
"""
from __future__ import annotations

from typing import Callable

import torch

from .bright_vlm import bright_labeled_batch
from .kv_ttt import build_post_image_mask_fn, discover_language_decoder_kv


def batch_builder(processor, model_family: str) -> Callable:
    def build(proc, sample, crop_size=448):
        return bright_labeled_batch(proc, model_family, sample, crop_size)
    return build


def image_token_id(model, processor) -> int:
    config = getattr(model, "config", None)
    for name in ("image_token_id", "image_token_index"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    for name in ("image_token_id", "image_token_index"):
        value = getattr(processor, name, None)
        if value is not None:
            return int(value)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        token = "<IMG_CONTEXT>" if getattr(processor, "image_token_id", None) is not None else "<image>"
        value = tokenizer.convert_tokens_to_ids(token)
        if value is not None and int(value) >= 0:
            return int(value)
    raise RuntimeError("Could not determine the BRIGHT image token id")


def post_image_mask(model, processor, model_family: str) -> Callable:
    if model_family == "phi":
        def phi_mask(input_ids: torch.Tensor) -> torch.Tensor:
            mask = input_ids < 0
            if not bool(mask.any()):
                raise ValueError("Phi BRIGHT batch has no visual token IDs")
            # Phi's negative visual IDs form pre and post image groups.
            result = torch.zeros_like(mask)
            for row, values in enumerate(mask):
                positions = values.nonzero(as_tuple=False).flatten()
                boundaries = (positions[1:] != positions[:-1] + 1).nonzero(
                    as_tuple=False).flatten() + 1
                groups = list(torch.tensor_split(positions, boundaries.tolist()))
                if len(groups) != 2:
                    raise ValueError(f"Expected two Phi image groups, found {len(groups)}")
                result[row, groups[1]] = True
            return result
        return phi_mask
    return build_post_image_mask_fn(image_token_id(model, processor))


def discover_bright_kv(model, model_family: str, processor):
    if model_family == "phi":
        from .phi_kv import discover_phi_kv
        return discover_phi_kv(model)
    return discover_language_decoder_kv(model)
