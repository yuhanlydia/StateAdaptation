"""Task-generic correctness-gradient KV-TTT for single-image examples."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .kv_ttt import (
    KVGradientCollector, ResidualKVController, build_controller_from_state,
    discover_language_decoder_kv, freeze_model, image_token_id_of,
    save_kv_state,
)
from .schemas import TaskSample
from .task_qwen import labeled_batch
from .visual_masks import build_visual_mask_fn


def task_visual_mask(model, processor=None):
    token_id = getattr(processor, "image_token_id", None)
    if token_id is None:
        token_id = image_token_id_of(model)
    return build_visual_mask_fn(int(token_id), "all_visual", expected_groups=1)


def _selected_modules(model, layers=None):
    modules, count = discover_language_decoder_kv(model)
    selected = layers if layers is not None else [count // 2, count - 1]
    missing = set(selected) - {layer for layer, _, _ in modules}
    if missing:
        raise ValueError(f"requested layers missing: {sorted(missing)}")
    return [(layer, kind, module) for layer, kind, module in modules if layer in set(selected)], count, selected


def extract_task_subspace(model, processor, samples: Sequence[TaskSample], modules,
                          visual_mask, rank=16, basis_mode="covariance", seed=0,
                          family: str = "qwen2"):
    if basis_mode not in {"covariance", "random"}:
        raise ValueError("task backend currently supports covariance or random basis")
    freeze_model(model)
    model.eval()
    model.config.use_cache = False
    model.enable_input_require_grads()
    device = next(model.parameters()).device
    dims = {(layer, kind): module.out_features for layer, kind, module in modules}
    if basis_mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bases = {key: torch.linalg.qr(torch.randn(dim, rank, generator=generator))[0][:, :rank].contiguous()
                 for key, dim in dims.items()}
        return bases, {str(key): [] for key in dims}
    covariance = {key: torch.zeros(dim, dim, dtype=torch.float32, device=device)
                  for key, dim in dims.items()}
    collector = KVGradientCollector(modules)
    try:
        for sample in tqdm(samples, desc="Task KV gradients", dynamic_ncols=True):
            batch, _ = labeled_batch(processor, sample, family=family)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            grads = collector.gradients(visual_mask(batch["input_ids"]))
            for key, grad in grads.items():
                grad = grad.float()
                covariance[key].add_(grad.t() @ grad)
            collector.clear()
            model.zero_grad(set_to_none=True)
    finally:
        collector.close()
        disable = getattr(model, "disable_input_require_grads", None)
        if callable(disable):
            disable()
    bases, spectra = {}, {}
    for key, matrix in covariance.items():
        # CPU eigendecomposition is slower but avoids intermittent cuSolver
        # handle failures after long mixed-precision evaluation batches.
        values, vectors = torch.linalg.eigh(matrix.detach().cpu())
        values, order = torch.sort(values, descending=True)
        bases[key] = vectors[:, order[:rank]].detach().cpu().contiguous()
        spectra[str(key)] = [float(value) for value in values[: rank * 3]]
    return bases, spectra


def fit_task_coefficients(model, processor, samples, controller, visual_mask,
                          steps=4, learning_rate=0.05, l2=1e-3, family: str = "qwen2"):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(controller.ttt_parameters(), lr=learning_rate)
    losses = []
    model.eval()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for sample in samples:
            batch, _ = labeled_batch(processor, sample, family=family)
            batch = {key: value.to(device) for key, value in batch.items()}
            controller.set_mask(visual_mask(batch["input_ids"]))
            loss = model(**batch).loss
            controller.clear_mask()
            loss.backward()
            total += float(loss.detach())
        penalty = l2 * sum(parameter.pow(2).sum() for parameter in controller.ttt_parameters())
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(controller.ttt_parameters(), 1.0)
        optimizer.step()
        losses.append(total / len(samples) + float(penalty.detach()))
    return losses
