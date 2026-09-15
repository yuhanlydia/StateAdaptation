"""Qwen backend for single-image candidate-label tasks.

This module deliberately keeps task examples separate from the historical
paired-image BRIGHT ``Sample`` path while sharing model loading and KV module
discovery. Candidate likelihood scoring is used for every task, so free-form
generation never determines the primary prediction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .aggregation import product_of_experts
from .qwen import _find_last_subsequence, _require_training_packages, load_model
from .schemas import TaskSample


def task_messages(sample: TaskSample, answer: str | None = None, image=None) -> list[dict]:
    result = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "Answer the visual question using exactly one candidate label "
                "and no explanation."
            )}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image if image is not None else sample.image},
                {"type": "text", "text": sample.question},
            ],
        },
    ]
    if answer is not None:
        result.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return result


def load_task_image(sample: TaskSample, max_size: int = 448):
    """Deterministic bounded resize for non-BRIGHT images.

    ManipBench images are substantially larger than the BRIGHT crop budget;
    leaving them at native resolution changes the effective visual-token budget
    and makes the candidate scorer needlessly slow. Aspect ratio is preserved.
    """
    image = Image.open(sample.image).convert("RGB")
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _batch_for_candidates(processor, sample: TaskSample, image, labels: Sequence[str],
                          family: str = "qwen2"):
    image = image if image is not None else load_task_image(sample)
    if family == "internvl3":
        texts = [processor.build_task_prompt(sample, label) for label in labels]
        # InternVL pairs one image tensor with each prompt row; repeat the
        # same task image for the candidate batch (one row per label).
        batch = processor(text=texts, images=[image] * len(texts), padding=True,
                          return_tensors="pt")
        spans = []
        for row, label in enumerate(labels):
            answer_ids = processor.tokenizer(
                f"Answer: {label}", add_special_tokens=False
            )["input_ids"]
            start = _find_last_subsequence(batch["input_ids"][row].tolist(), answer_ids)
            spans.append((start, start + len(answer_ids)))
        return batch, spans
    process_vision_info = _require_training_packages()["process_vision_info"]
    chats = [task_messages(sample, label, image) for label in labels]
    texts = [processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
             for chat in chats]
    images, videos = process_vision_info(chats)
    batch = processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt")
    spans = []
    for row, label in enumerate(labels):
        answer_ids = processor.tokenizer(label, add_special_tokens=False)["input_ids"]
        start = _find_last_subsequence(batch["input_ids"][row].tolist(), answer_ids)
        spans.append((start, start + len(answer_ids)))
    return batch, spans


def labeled_batch(processor, sample: TaskSample, image=None, family: str = "qwen2"):
    batch, spans = _batch_for_candidates(processor, sample, image, (sample.label,), family)
    labels = torch.full_like(batch["input_ids"], -100)
    start, end = spans[0]
    labels[0, start:end] = batch["input_ids"][0, start:end]
    batch["labels"] = labels
    return batch, (start, end)


def candidate_scores(model, processor, sample: TaskSample, image=None, device=None,
                     controller=None, visual_mask=None, family: str = "qwen2"):
    model.eval()
    device = device or next(model.parameters()).device
    # Score all candidate completions in one padded forward.  This keeps the
    # task protocol identical (one image and one candidate per row) while
    # avoiding a separate 7B forward for every label.
    image = image if image is not None else load_task_image(sample)
    labels = tuple(sample.candidate_labels)
    batch, spans = _batch_for_candidates(processor, sample, image, labels, family)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        if controller is not None:
            if visual_mask is None:
                raise ValueError("visual_mask is required when a controller is active")
            controller.set_mask(visual_mask(batch["input_ids"]))
        try:
            logits = model(**batch).logits.float()
        finally:
            if controller is not None:
                controller.clear_mask()
    scores = []
    for row, (start, end) in enumerate(spans):
        targets = batch["input_ids"][row, start:end]
        log_probs = torch.log_softmax(logits[row, start - 1:end - 1], dim=-1)
        scores.append(float(log_probs.gather(-1, targets.unsqueeze(-1)).sum()))
    return np.asarray(scores, dtype=np.float64)


def score_task_sample(model, processor, sample: TaskSample, device=None,
                      controller=None, visual_mask=None, family: str = "qwen2") -> dict:
    scores = candidate_scores(model, processor, sample, device=device,
                              controller=controller, visual_mask=visual_mask, family=family)
    probabilities = product_of_experts([scores.tolist()])
    prediction_id = int(np.argmax(probabilities))
    return {
        "sample_id": sample.sample_id,
        "domain_id": sample.domain_id,
        "group_id": sample.group_id,
        "label": sample.label,
        "label_id": sample.label_id,
        "prediction": sample.candidate_labels[prediction_id],
        "probabilities": probabilities.tolist(),
        "mean_log_scores": scores.tolist(),
    }


def fit_task_lora(
    model, processor, samples: Sequence[TaskSample],
    passes: int = 4, learning_rate: float = 2e-4,
    seed: int = 0, max_grad_norm: float = 1.0, family: str = "qwen2",
) -> list[float]:
    """Fixed-duration support-only LoRA fit for single-image tasks."""
    if passes < 0:
        raise ValueError("passes must be non-negative")
    if not samples or passes == 0:
        return []
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    model.train()
    losses = []
    for _ in range(passes):
        order = torch.randperm(len(samples), generator=generator).tolist()
        total = 0.0
        for index in order:
            batch, _ = labeled_batch(processor, samples[index], family=family)
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(total / len(samples))
    return losses
