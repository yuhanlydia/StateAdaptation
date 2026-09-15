"""Hidden-state residual baseline for single-image candidate-label tasks.

The controller is deliberately parallel to the KV controller: the source VLM
is frozen, a correctness-gradient covariance basis is extracted at two
language-decoder hidden layers, and only a bounded low-rank residual is tuned
on the support set.  Visual-token masking prevents text-state updates.
"""

from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .kv_ttt import freeze_model
from .schemas import TaskSample
from .task_qwen import labeled_batch

_LAYER_RE = re.compile(r"\.layers\.(\d+)$")


def discover_hidden_layers(model: nn.Module):
    found = []
    for name, module in model.named_modules():
        match = _LAYER_RE.search(name)
        if match is not None and hasattr(module, "self_attn"):
            found.append((int(match.group(1)), module))
    if not found:
        raise RuntimeError("No language decoder layers were discovered")
    # PEFT wrappers can expose aliases; retain one module per layer id.
    unique = {}
    for layer, module in found:
        unique.setdefault(layer, module)
    return [(layer, unique[layer]) for layer in sorted(unique)], max(unique) + 1


class HiddenResidualController(nn.Module):
    def __init__(self, modules, bases, rank=16, alpha_max=3.0,
                 coefficient_mode="full", device=None):
        super().__init__()
        if coefficient_mode not in {"full", "diagonal"}:
            raise ValueError("coefficient_mode must be full or diagonal")
        self.modules = list(modules)
        self.rank = int(rank)
        self.alpha_max = float(alpha_max)
        self.coefficient_mode = coefficient_mode
        self._device = device or next(iter(bases.values())).device
        self.bases = {int(k): v.float().to(self._device) for k, v in bases.items()}
        self.coefficients = nn.ParameterDict()
        for layer, _ in self.modules:
            shape = (rank, rank) if coefficient_mode == "full" else (rank,)
            self.coefficients[str(layer)] = nn.Parameter(torch.zeros(shape, device=self._device))
        self._active_mask = None
        self._hooks = [m.register_forward_hook(self._make_hook(layer)) for layer, m in self.modules]

    def _make_hook(self, layer):
        def hook(module, inp, output):
            mask = self._active_mask
            if mask is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            basis = self.bases[layer].to(device=hidden.device, dtype=hidden.dtype)
            raw = self.coefficients[str(layer)]
            low = hidden @ basis
            if self.coefficient_mode == "diagonal":
                mixed = low * (self.alpha_max * torch.tanh(raw)).to(hidden.dtype)
            else:
                mixing = (self.alpha_max * raw / (1.0 + torch.linalg.vector_norm(raw))).to(hidden.dtype)
                mixed = low @ mixing
            delta = mixed @ basis.transpose(-1, -2)
            updated = hidden + mask.to(hidden.dtype).unsqueeze(-1) * delta
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated
        return hook

    def set_mask(self, mask):
        self._active_mask = mask.to(self._device)

    def clear_mask(self):
        self._active_mask = None

    def close(self):
        self.clear_mask()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def ttt_parameters(self):
        return list(self.coefficients.parameters())

    def num_scalars(self):
        return sum(p.numel() for p in self.ttt_parameters())


def extract_hidden_subspace(model, processor, samples: Sequence[TaskSample], modules,
                            visual_mask, rank=16, basis_mode="covariance", seed=0):
    if basis_mode not in {"covariance", "random"}:
        raise ValueError("basis_mode must be covariance or random")
    freeze_model(model)
    model.eval()
    model.config.use_cache = False
    model.enable_input_require_grads()
    device = next(model.parameters()).device
    dims = {layer: module.self_attn.q_proj.in_features for layer, module in modules}
    if basis_mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bases = {layer: torch.linalg.qr(torch.randn(dim, rank, generator=generator))[0][:, :rank].contiguous()
                 for layer, dim in dims.items()}
        return bases, {str(layer): [] for layer in dims}
    covariance = {layer: torch.zeros(dim, dim, dtype=torch.float32, device=device) for layer, dim in dims.items()}
    saved = {}
    handles = []
    def make_capture(layer):
        def capture(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden.retain_grad()
            saved[layer] = hidden
        return capture
    for layer, module in modules:
        handles.append(module.register_forward_hook(make_capture(layer)))
    try:
        for sample in tqdm(samples, desc="Hidden gradients", dynamic_ncols=True):
            saved.clear()
            batch, _ = labeled_batch(processor, sample)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            mask = visual_mask(batch["input_ids"])
            for layer in dims:
                grad = saved[layer].grad[mask].float()
                covariance[layer].add_(grad.transpose(0, 1) @ grad)
            model.zero_grad(set_to_none=True)
    finally:
        for handle in handles:
            handle.remove()
        disable = getattr(model, "disable_input_require_grads", None)
        if callable(disable):
            disable()
    bases, spectra = {}, {}
    for layer, matrix in covariance.items():
        values, vectors = torch.linalg.eigh(matrix.detach().cpu())
        values, order = torch.sort(values, descending=True)
        bases[layer] = vectors[:, order[:rank]].detach().cpu().contiguous()
        spectra[str(layer)] = [float(v) for v in values[:rank * 3]]
    return bases, spectra


def fit_hidden_coefficients(model, processor, samples, controller, visual_mask,
                            steps=4, learning_rate=0.05, l2=1e-3):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(controller.ttt_parameters(), lr=learning_rate)
    losses = []
    model.eval()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for sample in samples:
            batch, _ = labeled_batch(processor, sample)
            batch = {key: value.to(device) for key, value in batch.items()}
            controller.set_mask(visual_mask(batch["input_ids"]))
            loss = model(**batch).loss
            controller.clear_mask()
            loss.backward()
            total += float(loss.detach())
        penalty = l2 * sum(p.pow(2).sum() for p in controller.ttt_parameters())
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(controller.ttt_parameters(), 1.0)
        optimizer.step()
        losses.append(total / len(samples) + float(penalty.detach()))
    return losses


def save_hidden_state(path, controller, model_id, metadata=None):
    payload = {
        "version": 1, "method": "hidden_residual", "model_id": model_id,
        "rank": controller.rank, "layers": [layer for layer, _ in controller.modules],
        "alpha_max": controller.alpha_max, "coefficient_mode": controller.coefficient_mode,
        "bases": {str(k): v.cpu() for k, v in controller.bases.items()},
        "coefficients_raw": {k: v.detach().cpu() for k, v in controller.coefficients.items()},
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_hidden_state(path, device=None):
    return torch.load(path, map_location=device or "cpu")
