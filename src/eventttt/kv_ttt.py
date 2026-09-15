"""Post-Visual Residual KV-Test-Time-Tuning.

An independent, experimental adaptation path for a frozen source-trained
Qwen2.5-VL model. It does not update VLM, LoRA, text tokens or the pre-event
image tokens. Given a tiny labelled support set it:

  * discovers the language-decoder ``k_proj``/``v_proj`` outputs;
  * isolates the *post-event* image token positions;
  * extracts a correctness-gradient KV subspace ``B`` (top eigenvectors of the
    per-kind covariance ``C = sum_i G_i^T G_i``, equivalent to the SVD right
    singular directions of the stacked ``G``);
  * learns a tiny raw coefficient vector ``a`` (2 layers x K/V x rank = 32);
  * applies the residual ``Z' = Z + M ⊙ (Z B) diag(alpha·tanh(a)) B^T``.

Everything else stays frozen; this file never changes the original LoRA
EventTune path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .aggregation import product_of_experts
from .prompts import messages
from .qwen import load_sample_pair
from .schemas import DAMAGE_LABELS, Sample
from .vision import d4_pair

KV_TYPE_K = "K"
KV_TYPE_V = "V"

_KV_MODULE_RE = re.compile(
    r"\.layers\.(\d+)\.self_attn\.(k_proj|v_proj)$"
)


def _require_packages():
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    return {
        "process_vision_info": process_vision_info,
        "AutoProcessor": AutoProcessor,
        "Model": Qwen2_5_VLForConditionalGeneration,
    }


def _find_last_subsequence(sequence: Sequence[int], query: Sequence[int]) -> int:
    for start in range(len(sequence) - len(query), -1, -1):
        if list(sequence[start : start + len(query)]) == list(query):
            return start
    raise ValueError(f"Answer token sequence {list(query)} not found at end of chat tokens")


def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_processor(model_id: str):
    return _require_packages()["AutoProcessor"].from_pretrained(model_id)


def discover_language_decoder_kv(
    model: nn.Module,
) -> tuple[list[tuple[int, str, nn.Module]], int]:
    """Return ``[(layer_id, kind, module)]`` for every K/V projection in the
    language decoder plus the number of decoder layers. Uses regex discovery so
    PEFT-style wrappers do not require hard-coded nesting."""
    found: list[tuple[int, str, nn.Module]] = []
    for name, module in model.named_modules():
        # Multimodal backbones often expose a second ``layers.*.self_attn``
        # tree in the vision tower. KV-TTT is defined on the language decoder
        # only; selecting a vision projection would silently give a dimension
        # mismatch (and violate the post-image language residual contract).
        if any(part in name.lower() for part in ("vision_tower", "vision_model", "vision_encoder")):
            continue
        match = _KV_MODULE_RE.search(name)
        if match is None:
            continue
        found.append((int(match.group(1)), "K" if match.group(2) == "k_proj" else "V", module))
    if not found:
        raise RuntimeError(
            "No language_decoder k_proj/v_proj modules matched regex "
            f"{_KV_MODULE_RE.pattern}. First modules: "
            + ", ".join(name for name, _ in list(model.named_modules())[:10])
        )
    num_layers = max(layer_id for layer_id, _, _ in found) + 1
    return found, num_layers


def default_layers(num_layers: int) -> list[int]:
    """Middle + final decoder layer."""
    return [num_layers // 2, num_layers - 1]


def image_token_id_of(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    token_id = getattr(config, "image_token_id", None)
    if token_id is None:
        raise RuntimeError("Model config has no image_token_id")
    return int(token_id)


def build_post_image_mask_fn(image_token_id: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Factory producing the post-event image mask builder.

    Every EventTune prompt has exactly two images, so positions of
    ``image_token_id`` form exactly two contiguous groups per row: the pre-event
    image first, the post-event image second. Fails closed otherwise."""

    def build(input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, T] but had shape {list(input_ids.shape)}")
        mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        for row_index in range(input_ids.shape[0]):
            positions = (input_ids[row_index] == image_token_id).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                raise ValueError(
                    f"Row {row_index} has no image-pad tokens (image_token_id={image_token_id})"
                )
            boundaries = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=False).flatten() + 1
            groups = list(torch.tensor_split(positions, boundaries.tolist()))
            # Some processors concatenate adjacent ``<image>`` placeholders
            # into one expanded run. The two crops are equal-sized, so the
            # midpoint is an unambiguous pre/post boundary in that case.
            if len(groups) == 1 and positions.numel() >= 4 and positions.numel() % 2 == 0:
                midpoint = positions.numel() // 2
                groups = [positions[:midpoint], positions[midpoint:]]
            if len(groups) != 2:
                raise ValueError(
                    f"Row {row_index} expected exactly two image groups (pre, post), found {len(groups)}"
                )
            mask[row_index, groups[1]] = True
        return mask

    return build


def build_labeled_batch(processor, sample: Sample, crop_size: int = 448):
    """Build the paired-image prompt (pre image, post image, question, assistant
    label) for one ``sample``. Returns ``(batch, span)`` where ``span`` is the
    ``[start, end)`` token range of the class label."""
    process_vision_info = _require_packages()["process_vision_info"]
    pre, post = load_sample_pair(sample, crop_size)
    chat = messages(sample, True, pre, post)
    text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info([chat])
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    answer_ids = processor.tokenizer(sample.label, add_special_tokens=False)["input_ids"]
    ids = batch["input_ids"][0].tolist()
    start = _find_last_subsequence(ids, answer_ids)
    return batch, (start, start + len(answer_ids))


def strict_generation_loss(
    model: nn.Module, batch: dict[str, torch.Tensor], span: tuple[int, int], device
) -> torch.Tensor:
    """Correctness loss limited to the assistant class-label token span."""
    labels = torch.full_like(batch["input_ids"], -100)
    labels[0, span[0] : span[1]] = batch["input_ids"][0, span[0] : span[1]]
    inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
    inputs["labels"] = labels.to(device)
    return model(**inputs).loss


def consistency_js_loss(log_scores: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon disagreement between categorical view predictions.

    ``log_scores`` has shape ``[views, classes]`` and may contain unnormalised
    sequence log-likelihoods. The result is zero exactly when all view
    distributions agree. No target label or class-prior assumption is used.
    """
    if log_scores.ndim != 2 or log_scores.shape[0] < 2:
        raise ValueError("log_scores must have shape [at least 2 views, classes]")
    log_probabilities = torch.log_softmax(log_scores.float(), dim=-1)
    probabilities = log_probabilities.exp()
    mean_probability = probabilities.mean(dim=0)
    return (
        probabilities
        * (log_probabilities - mean_probability.clamp_min(1e-8).log())
    ).sum(dim=-1).mean()


def _candidate_log_scores_with_grad(
    model,
    processor,
    sample: Sample,
    pre,
    post,
    device,
    build_post_mask: Callable,
    controller=None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Differentiably score all class strings for one image view."""
    from .qwen import _candidate_batch

    scores = []
    masks = []
    for label in DAMAGE_LABELS:
        batch, spans = _candidate_batch(
            processor, sample, pre, post, candidate_labels=(label,)
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        mask = build_post_mask(batch["input_ids"])
        masks.append(mask)
        if controller is not None:
            controller.set_mask(mask)
        logits = model(**batch).logits.float()
        if controller is not None:
            controller.clear_mask()
        start, end = spans[0]
        targets = batch["input_ids"][0, start:end]
        token_log_probabilities = torch.log_softmax(
            logits[0, start - 1 : end - 1], dim=-1
        )
        scores.append(
            token_log_probabilities.gather(-1, targets.unsqueeze(-1)).sum()
        )
    return torch.stack(scores), masks


def sample_consistency_loss(
    model,
    processor,
    sample: Sample,
    build_post_mask: Callable,
    d4_views: int = 2,
    crop_size: int = 448,
    controller=None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return label-free D4 consistency loss and forward-aligned masks."""
    if d4_views < 2:
        raise ValueError("Unsupervised consistency adaptation requires at least 2 views")
    device = next(model.parameters()).device
    pre, post = load_sample_pair(sample, crop_size)
    view_scores = []
    masks = []
    for pre_view, post_view, _ in d4_pair(pre, post, d4_views):
        scores, view_masks = _candidate_log_scores_with_grad(
            model, processor, sample, pre_view, post_view, device, build_post_mask,
            controller=controller,
        )
        view_scores.append(scores)
        masks.extend(view_masks)
    return consistency_js_loss(torch.stack(view_scores)), masks


class KVGradientCollector:
    """Registers forward hooks that retain the gradient of hooked projections."""

    def __init__(self, modules: Sequence[tuple[int, str, nn.Module]]):
        self.modules = list(modules)
        self.outputs: dict[tuple[int, str], torch.Tensor] = {}
        self.history: dict[tuple[int, str], list[torch.Tensor]] = {
            (layer_id, kind): [] for layer_id, kind, _ in self.modules
        }
        self.handles = [
            module.register_forward_hook(self._hook(layer_id, kind))
            for layer_id, kind, module in self.modules
        ]

    def _hook(self, layer_id: int, kind: str):
        def forward_hook(module, inp, out):
            if out.requires_grad:
                out.retain_grad()
            self.outputs[(layer_id, kind)] = out
            self.history[(layer_id, kind)].append(out)

        return forward_hook

    def gradients(self, post_mask: torch.Tensor) -> dict[tuple[int, str], torch.Tensor]:
        result = {}
        for key, out in self.outputs.items():
            grad = out.grad
            if grad is None:
                raise RuntimeError(f"No gradient stored for {key}")
            result[key] = grad[post_mask]
        return result

    def clear(self) -> None:
        self.outputs.clear()
        for values in self.history.values():
            values.clear()

    def history_gradients(
        self, post_masks: Sequence[torch.Tensor]
    ) -> dict[tuple[int, str], torch.Tensor]:
        """Return post-image gradients from a sequence of forward passes.

        ``post_masks`` must be in the same order as the forwards. This is used
        by consistency TTT, where one loss spans several D4 views and candidate
        label scoring passes.
        """
        result = {}
        for key, outputs in self.history.items():
            if len(outputs) != len(post_masks):
                raise RuntimeError(
                    f"Forward/mask count mismatch for {key}: "
                    f"{len(outputs)} outputs vs {len(post_masks)} masks"
                )
            rows = []
            for output, mask in zip(outputs, post_masks):
                if output.grad is None:
                    raise RuntimeError(f"No gradient stored for {key}")
                rows.append(output.grad[mask.to(output.grad.device)])
            result[key] = torch.cat(rows, dim=0)
        return result

    def close(self) -> None:
        """Unregister the forward hooks and drop cached activations."""
        self.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _disable_input_require_grads(model: nn.Module) -> None:
    """Best-effort removal of the ``enable_input_require_grads`` hook."""
    disable = getattr(model, "disable_input_require_grads", None)
    if callable(disable):
        try:
            disable()
        except (AttributeError, TypeError):
            pass


def extract_kv_subspace(
    model,
    processor,
    support: Sequence[Sample],
    modules: Sequence[tuple[int, str, nn.Module]],
    build_post_mask: Callable,
    rank: int = 8,
    crop_size: int = 448,
    progress: bool = True,
    basis_mode: str = "covariance",
    basis_seed: int = 0,
    batch_builder: Callable | None = None,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, list[float]]]:
    """Correctness-gradient KV subspace extraction over the support set.

    Per support example runs one forward/backward, keeps only the post-image
    K/V gradients and accumulates the FP32 covariance ``C += G^T G``. Returns
    ``(bases, spectra)`` with bases the top-``rank`` eigenvectors of each
    covariance (same as the SVD right singular directions)."""
    if basis_mode not in {"covariance", "centered_covariance", "mean_gradient", "random"}:
        raise ValueError(f"unknown basis_mode: {basis_mode}")
    if basis_mode == "mean_gradient" and rank != 1:
        raise ValueError("mean_gradient basis is intrinsically rank 1")
    freeze_model(model)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.enable_input_require_grads()

    device = next(model.parameters()).device
    batch_builder = batch_builder or build_labeled_batch
    dims = {(layer_id, kind): module.out_features for layer_id, kind, module in modules}
    if basis_mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(basis_seed)
        bases = {}
        for key, dim in dims.items():
            basis, _ = torch.linalg.qr(torch.randn(dim, rank, generator=generator))
            bases[key] = basis[:, :rank].contiguous()
        return bases, {str(key): [] for key in dims}
    covariance = {
        key: torch.zeros(dim, dim, dtype=torch.float32, device=device)
        for key, dim in dims.items()
    }
    gradient_sum = {
        key: torch.zeros(dim, dtype=torch.float32, device=device)
        for key, dim in dims.items()
    }
    gradient_count = {key: 0 for key in dims}
    collector = KVGradientCollector(modules)
    samples = list(support)
    try:
        for sample in tqdm(samples, desc="KV gradients", dynamic_ncols=True, disable=not progress):
            batch, span = batch_builder(processor, sample, crop_size)
            loss = strict_generation_loss(model, batch, span, device)
            loss.backward()
            post_mask = build_post_mask(batch["input_ids"])
            gradients = collector.gradients(post_mask)
            for key, gradient in gradients.items():
                gradient = gradient.float()
                covariance[key].add_(gradient.t() @ gradient)
                gradient_sum[key].add_(gradient.sum(dim=0))
                gradient_count[key] += gradient.shape[0]
            collector.clear()
            model.zero_grad(set_to_none=True)
            del loss, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        collector.close()
        _disable_input_require_grads(model)

    bases: dict[tuple[int, str], torch.Tensor] = {}
    spectra: dict[str, list[float]] = {}
    for key, covariance_matrix in covariance.items():
        if basis_mode == "mean_gradient":
            direction = gradient_sum[key]
            norm = torch.linalg.vector_norm(direction)
            if norm <= 0:
                raise RuntimeError(f"zero mean gradient for {key}")
            bases[key] = (direction / norm).unsqueeze(1).contiguous().detach().cpu()
            spectra[str(key)] = [float(norm)]
            continue
        if basis_mode == "centered_covariance":
            count = gradient_count[key]
            if count <= 0:
                raise RuntimeError(f"no gradients collected for {key}")
            covariance_matrix = covariance_matrix - torch.outer(
                gradient_sum[key], gradient_sum[key]
            ) / count
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix.to(torch.float32))
        values, order = torch.sort(eigenvalues, descending=True)
        bases[key] = eigenvectors[:, order[:rank]].contiguous().detach().cpu()
        spectra[str(key)] = [float(value) for value in values[: rank * 3]]
    return bases, spectra


def extract_consistency_kv_subspace(
    model,
    processor,
    samples: Sequence[Sample],
    modules: Sequence[tuple[int, str, nn.Module]],
    build_post_mask: Callable,
    rank: int = 8,
    d4_views: int = 2,
    crop_size: int = 448,
    progress: bool = True,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[str, list[float]]]:
    """Extract a label-free KV subspace from D4 pseudo-label consistency.

    The frozen source model predicts an identity-view teacher label. Each
    non-identity D4 view is then differentiated toward that same label. This
    keeps only one 7B backward graph alive at a time on a 24 GiB GPU.
    """
    if d4_views < 2:
        raise ValueError("Unsupervised pseudo-label adaptation requires at least 2 views")
    if not samples:
        raise ValueError("Cannot extract a subspace from an empty unlabeled set")
    freeze_model(model)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    model.enable_input_require_grads()

    device = next(model.parameters()).device
    dims = {(layer_id, kind): module.out_features for layer_id, kind, module in modules}
    covariance = {
        key: torch.zeros(dim, dim, dtype=torch.float32, device=device)
        for key, dim in dims.items()
    }
    collector = KVGradientCollector(modules)
    try:
        from .qwen import _candidate_batch, score_sample

        for sample in tqdm(
            list(samples), desc="Consistency KV gradients", dynamic_ncols=True,
            disable=not progress,
        ):
            teacher_label = score_sample(
                model, processor, sample, d4_views=1, crop_size=crop_size
            )["prediction"]
            pre, post = load_sample_pair(sample, crop_size)
            views = list(d4_pair(pre, post, d4_views))[1:]
            for pre_view, post_view, _ in views:
                batch, spans = _candidate_batch(
                    processor, sample, pre_view, post_view,
                    candidate_labels=(teacher_label,),
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                mask = build_post_mask(batch["input_ids"])
                logits = model(**batch).logits.float()
                start, end = spans[0]
                targets = batch["input_ids"][0, start:end]
                loss = -torch.log_softmax(
                    logits[0, start - 1 : end - 1], dim=-1
                ).gather(-1, targets.unsqueeze(-1)).mean()
                loss.backward()
                gradients = collector.gradients(mask)
                for key, gradient in gradients.items():
                    gradient = gradient.float()
                    covariance[key].add_(gradient.t() @ gradient)
                collector.clear()
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        collector.close()
        _disable_input_require_grads(model)

    bases = {}
    spectra = {}
    for key, covariance_matrix in covariance.items():
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix.float())
        values, order = torch.sort(eigenvalues, descending=True)
        bases[key] = eigenvectors[:, order[:rank]].contiguous().detach().cpu()
        spectra[str(key)] = [float(value) for value in values[: rank * 3]]
    return bases, spectra


class ResidualKVController(nn.Module):
    """Apply a bounded residual inside the gradient-derived subspace.

    ``diagonal`` preserves the original method,
    ``Z' = Z + M ⊙ (Z B) diag(gamma) B^T``. ``full`` learns a dense
    rank-by-rank mixing matrix, ``Z' = Z + M ⊙ (Z B) A B^T``.  In full
    mode ``A = alpha_max * R / (1 + ||R||_F)`` so its Frobenius (and therefore
    spectral) norm is strictly below ``alpha_max``.  This permits rotations
    between discovered directions without giving up a global update bound.

    With every raw coefficient at zero the transform is the identity, so the
    controller leaves the source model numerically unchanged by default."""

    def __init__(
        self,
        modules: Sequence[tuple[int, str, nn.Module]],
        bases: dict[tuple[int, str], torch.Tensor],
        rank: int = 8,
        alpha_max: float = 0.5,
        coefficient_mode: str = "diagonal",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.modules = list(modules)
        layers = sorted({layer_id for layer_id, _, _ in self.modules})
        self.layers = layers
        self.rank = int(rank)
        self.alpha_max = float(alpha_max)
        if coefficient_mode not in {"diagonal", "full"}:
            raise ValueError(f"Unknown coefficient_mode: {coefficient_mode}")
        self.coefficient_mode = coefficient_mode
        self._device = device or next(iter(bases.values())).device
        coefficients = nn.ParameterDict()
        for layer_id, kind, _ in self.modules:
            shape = (rank,) if coefficient_mode == "diagonal" else (rank, rank)
            coefficients[f"{layer_id}:{kind}"] = nn.Parameter(
                torch.zeros(shape, dtype=torch.float32, device=self._device)
            )
        self.coefficients = coefficients
        self.bases = {
            key: tensor.float().to(self._device) for key, tensor in bases.items()
        }
        self._active_mask: torch.Tensor | None = None
        self._hooks = [
            module.register_forward_hook(self._make_hook(layer_id, kind))
            for layer_id, kind, module in self.modules
        ]

    def _make_hook(self, layer_id: int, kind: str):
        def hook(module, inp, output):
            mask = self._active_mask
            if mask is None:
                return output
            basis = self.bases[(layer_id, kind)].to(dtype=output.dtype, device=output.device)
            raw = self.coefficients[f"{layer_id}:{kind}"]
            low = output @ basis
            if self.coefficient_mode == "diagonal":
                gamma = (self.alpha_max * torch.tanh(raw)).to(
                    dtype=output.dtype, device=output.device
                )
                mixed = low * gamma
            else:
                # ||A||_2 <= ||A||_F < alpha_max, including during training.
                scale = self.alpha_max / (1.0 + torch.linalg.vector_norm(raw))
                mixing = (scale * raw).to(dtype=output.dtype, device=output.device)
                mixed = low @ mixing
            delta = mixed @ basis.transpose(-1, -2)
            return output + mask.to(dtype=output.dtype, device=output.device).unsqueeze(-1) * delta

        return hook

    def set_mask(self, mask: torch.Tensor) -> None:
        self._active_mask = mask.to(self._device)

    def clear_mask(self) -> None:
        self._active_mask = None

    def close(self) -> None:
        """Unregister the forward hooks and drop the active mask."""
        self.clear_mask()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def reset_coefficients(self) -> None:
        with torch.no_grad():
            for parameter in self.coefficients.values():
                parameter.zero_()

    def ttt_parameters(self) -> list[nn.Parameter]:
        return list(self.coefficients.parameters())

    def num_scalars(self) -> int:
        return sum(parameter.numel() for parameter in self.coefficients.values())

    def effective_gamma(self) -> dict[str, float]:
        with torch.no_grad():
            result = {}
            for layer_id, kind, _ in self.modules:
                raw = self.coefficients[f"{layer_id}:{kind}"]
                if self.coefficient_mode == "diagonal":
                    value = torch.abs(self.alpha_max * torch.tanh(raw)).max()
                else:
                    mixing = self.alpha_max * raw / (1.0 + torch.linalg.vector_norm(raw))
                    value = torch.linalg.matrix_norm(mixing, ord=2)
                result[f"{layer_id}:{kind}"] = float(value)
            return result


def build_controller_from_state(
    modules: Sequence[tuple[int, str, nn.Module]],
    payload: dict,
    device=None,
) -> ResidualKVController:
    def normalize(key):
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], int):
            return (key[0], key[1])
        parts = key.split(":")
        return (int(parts[0]), parts[1])

    bases = {}
    for key, tensor in payload["bases"].items():
        bases[normalize(key)] = tensor
    controller = ResidualKVController(
        modules,
        bases,
        rank=payload["rank"],
        alpha_max=payload.get("alpha_max", 0.5),
        coefficient_mode=payload.get("coefficient_mode", "diagonal"),
        device=device,
    )
    raw = payload.get("coefficients_raw") or {}
    with torch.no_grad():
        for key, tensor in raw.items():
            layer_id, kind = normalize(key)
            controller.coefficients[f"{layer_id}:{kind}"].copy_(tensor)
    return controller


def fit_kv_coefficients(
    model,
    controller: ResidualKVController,
    processor,
    samples: Sequence[Sample],
    build_post_mask: Callable,
    steps: int = 4,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    max_grad_norm: float = 1.0,
    crop_size: int = 448,
    progress: bool = True,
    batch_builder: Callable | None = None,
) -> list[float]:
    """Optimize only the KV coefficients over the whole support set (no random
    sampling). One optimizer update accumulates every support example."""
    freeze_model(model)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False

    device = next(model.parameters()).device
    batch_builder = batch_builder or build_labeled_batch
    samples = list(samples)
    optimizer = torch.optim.Adam(controller.ttt_parameters(), lr=learning_rate)
    losses: list[float] = []
    step_range = range(steps)
    if progress:
        step_range = tqdm(step_range, desc="KV-TTT updates", dynamic_ncols=True)
    for _ in step_range:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for sample in samples:
            batch, span = batch_builder(processor, sample, crop_size)
            batch = {key: value.to(device) for key, value in batch.items()}
            controller.set_mask(build_post_mask(batch["input_ids"]))
            loss = strict_generation_loss(model, batch, span, device)
            (loss / len(samples)).backward()
            accumulated += float(loss.detach() / len(samples))
            controller.clear_mask()
            del batch, loss
        penalty = l2 * sum(parameter.pow(2).sum() for parameter in controller.ttt_parameters())
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(controller.ttt_parameters(), max_grad_norm)
        optimizer.step()
        losses.append(accumulated + float(penalty.detach()))
    return losses


def fit_consistency_kv_coefficients(
    model,
    controller: ResidualKVController,
    processor,
    samples: Sequence[Sample],
    build_post_mask: Callable,
    steps: int = 4,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    max_grad_norm: float = 1.0,
    d4_views: int = 2,
    crop_size: int = 448,
    progress: bool = True,
) -> list[float]:
    """Fit only KV coefficients toward source-model D4-consistent pseudo labels."""
    freeze_model(model)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    device_samples = list(samples)
    if not device_samples:
        raise ValueError("Cannot adapt on an empty unlabeled set")
    if d4_views < 2:
        raise ValueError("Unsupervised pseudo-label adaptation requires at least 2 views")
    from .qwen import _candidate_batch, score_sample

    teacher_labels = [
        score_sample(model, processor, sample, d4_views=1, crop_size=crop_size)["prediction"]
        for sample in device_samples
    ]
    optimizer = torch.optim.Adam(controller.ttt_parameters(), lr=learning_rate)
    losses = []
    step_range = tqdm(
        range(steps), desc="Unsupervised KV-TTT updates", dynamic_ncols=True,
        disable=not progress,
    )
    for _ in step_range:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        denominator = len(device_samples) * (d4_views - 1)
        for sample, teacher_label in zip(device_samples, teacher_labels):
            pre, post = load_sample_pair(sample, crop_size)
            for pre_view, post_view, _ in list(d4_pair(pre, post, d4_views))[1:]:
                batch, spans = _candidate_batch(
                    processor, sample, pre_view, post_view,
                    candidate_labels=(teacher_label,),
                )
                batch = {key: value.to(next(model.parameters()).device) for key, value in batch.items()}
                controller.set_mask(build_post_mask(batch["input_ids"]))
                logits = model(**batch).logits.float()
                controller.clear_mask()
                start, end = spans[0]
                targets = batch["input_ids"][0, start:end]
                loss = -torch.log_softmax(
                    logits[0, start - 1 : end - 1], dim=-1
                ).gather(-1, targets.unsqueeze(-1)).mean()
                (loss / denominator).backward()
                accumulated += float(loss.detach() / denominator)
        penalty = l2 * sum(p.pow(2).sum() for p in controller.ttt_parameters())
        penalty.backward()
        torch.nn.utils.clip_grad_norm_(controller.ttt_parameters(), max_grad_norm)
        optimizer.step()
        losses.append(accumulated + float(penalty.detach()))
    return losses


def save_kv_state(
    path: str | Path,
    controller: ResidualKVController,
    model_id: str,
    metadata: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model_id": model_id,
        "rank": controller.rank,
        "layers": controller.layers,
        "alpha_max": controller.alpha_max,
        "coefficient_mode": controller.coefficient_mode,
        "bases": {
            f"{layer_id}:{kind}": controller.bases[(layer_id, kind)].detach().cpu()
            for (layer_id, kind) in controller.bases
        },
        "coefficients_raw": {
            key: parameter.detach().cpu() for key, parameter in controller.coefficients.items()
        },
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_kv_state(path: str | Path, device=None) -> dict:
    payload = torch.load(path, map_location=device or "cpu")
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported kv-state version {payload.get('version')} in {path}")
    return payload


def score_sample_with_kv(
    model,
    processor,
    controller: ResidualKVController,
    build_post_mask: Callable,
    sample: Sample,
    d4_views: int = 8,
    crop_size: int = 448,
) -> dict:
    from .qwen import _candidate_batch

    model.eval()
    pre, post = load_sample_pair(sample, crop_size)
    device = next(model.parameters()).device
    view_scores = []
    with torch.inference_mode():
        for pre_view, post_view, _ in d4_pair(pre, post, d4_views):
            candidate_scores = []
            for label in DAMAGE_LABELS:
                batch, spans = _candidate_batch(
                    processor, sample, pre_view, post_view, candidate_labels=(label,)
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                controller.set_mask(build_post_mask(batch["input_ids"]))
                logits = model(**batch).logits.float()
                controller.clear_mask()
                start, valid = spans[0]
                targets = batch["input_ids"][0, start:valid]
                token_log_probs = torch.log_softmax(logits[0, start - 1 : valid - 1], dim=-1)
                score = token_log_probs.gather(-1, targets.unsqueeze(-1)).sum()
                candidate_scores.append(float(score))
                del batch, logits, token_log_probs
            view_scores.append(candidate_scores)
    mean_log_score = np.mean(np.asarray(view_scores, dtype=np.float64), axis=0)
    probabilities = product_of_experts(view_scores)
    return {
        "sample_id": sample.sample_id,
        "event_id": sample.event_id,
        "tile_id": sample.tile_id,
        "label": sample.label,
        "label_id": sample.label_id,
        "prediction": DAMAGE_LABELS[int(probabilities.argmax())],
        "probabilities": probabilities.tolist(),
        "mean_log_scores": mean_log_score.tolist(),
        "view_log_scores": np.asarray(view_scores).tolist(),
        "view_disagreement": float(np.var(np.asarray(view_scores), axis=0).mean()),
    }
