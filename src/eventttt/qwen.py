from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence
import warnings
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from tqdm.auto import tqdm

from .aggregation import product_of_experts
from .prompts import messages
from .schemas import DAMAGE_LABELS, Sample
from .vision import crop_pair, d4_pair, load_image


_LOCAL_DEFAULT_MODEL = Path("artifacts/models/Qwen2.5-VL-7B-Instruct")
DEFAULT_MODEL = os.environ.get(
    "EVENTTUNE_QWEN_MODEL",
    str(_LOCAL_DEFAULT_MODEL) if _LOCAL_DEFAULT_MODEL.is_dir() else "Qwen/Qwen2.5-VL-7B-Instruct",
)


def _require_training_packages():
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
        )
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements.txt before model training (transformers, peft, "
            "qwen-vl-utils, torchvision)."
        ) from exc
    return {
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "process_vision_info": process_vision_info,
        "AutoProcessor": AutoProcessor,
        "Model": Qwen2_5_VLForConditionalGeneration,
    }


def preflight(require_gpu: bool = True) -> dict:
    packages = _require_training_packages()

    status = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count(),
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "gpu_memory_gib": [
            round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2)
            for index in range(torch.cuda.device_count())
        ],
        "model_class": packages["Model"].__name__,
    }
    if require_gpu and not status["cuda_available"]:
        raise RuntimeError("CUDA is required for Qwen2.5-VL-7B training but is not visible")
    if require_gpu and not status["bf16_supported"]:
        raise RuntimeError("The standard LoRA path requires a CUDA GPU with bfloat16 support")
    return status


def load_model(
    model_id: str = DEFAULT_MODEL,
    source_adapter: str | None = None,
    gradient_checkpointing: bool = True,
    use_lora: bool = True,
):
    packages = _require_training_packages()
    processor = packages["AutoProcessor"].from_pretrained(model_id)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    model = packages["Model"].from_pretrained(model_id, **kwargs)
    if source_adapter:
        if not use_lora:
            raise ValueError("source_adapter requires LoRA to be enabled")
        model = packages["PeftModel"].from_pretrained(model, source_adapter, is_trainable=True)
    elif use_lora:
        config = packages["LoraConfig"](
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = packages["get_peft_model"](model, config)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model, processor


def trainable_parameter_report(model) -> dict:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable": trainable, "total": total, "fraction": trainable / total}


def load_sample_pair(sample: Sample, size: int = 448):
    return crop_pair(
        load_image(sample.pre_image),
        load_image(sample.post_image),
        sample.bbox_xyxy,
        size=size,
    )


class SampleDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def class_balanced_weights(samples: Sequence[Sample]) -> torch.Tensor:
    if not samples:
        raise ValueError("Cannot sample from an empty dataset")
    counts = Counter(sample.label for sample in samples)
    return torch.tensor(
        [1.0 / counts[sample.label] for sample in samples], dtype=torch.double
    )


class ClassCycleSampler(Sampler[int]):
    """Yield one randomly selected example per represented class each cycle."""

    def __init__(self, samples: Sequence[Sample], generator: torch.Generator):
        if not samples:
            raise ValueError("Cannot sample from an empty dataset")
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            groups.setdefault(sample.label, []).append(index)
        self.groups = {label: groups[label] for label in sorted(groups)}
        self.generator = generator
        self.num_samples = math.ceil(len(samples) / len(groups)) * len(groups)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        queues: dict[str, list[int]] = {label: [] for label in self.groups}

        def draw(label: str) -> int:
            if not queues[label]:
                order = torch.randperm(
                    len(self.groups[label]), generator=self.generator
                ).tolist()
                queues[label] = [self.groups[label][index] for index in order]
            return queues[label].pop()

        labels = list(self.groups)
        produced = 0
        while produced < self.num_samples:
            order = torch.randperm(len(labels), generator=self.generator).tolist()
            for label_index in order:
                if produced >= self.num_samples:
                    break
                yield draw(labels[label_index])
                produced += 1


def _find_last_subsequence(sequence: Sequence[int], query: Sequence[int]) -> int:
    for start in range(len(sequence) - len(query), -1, -1):
        if list(sequence[start : start + len(query)]) == list(query):
            return start
    raise ValueError(f"Answer token sequence {list(query)} not found at end of chat tokens")


@dataclass
class SFTCollator:
    processor: object
    crop_size: int = 448

    def __call__(self, samples: Sequence[Sample]) -> dict[str, torch.Tensor]:
        process_vision_info = _require_training_packages()["process_vision_info"]
        batch_messages = []
        for sample in samples:
            pre, post = load_sample_pair(sample, self.crop_size)
            batch_messages.append(messages(sample, True, pre, post))
        texts = [
            self.processor.apply_chat_template(value, tokenize=False, add_generation_prompt=False)
            for value in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(batch["input_ids"], -100)
        for row_index, sample in enumerate(samples):
            answer_ids = self.processor.tokenizer(
                sample.label, add_special_tokens=False
            )["input_ids"]
            ids = batch["input_ids"][row_index].tolist()
            start = _find_last_subsequence(ids, answer_ids)
            end = start + len(answer_ids)
            labels[row_index, start:end] = batch["input_ids"][row_index, start:end]
        batch["labels"] = labels
        return batch


def fit_steps(
    model,
    processor,
    samples: Sequence[Sample],
    steps: int,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation: int = 4,
    crop_size: int = 448,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    sampling: str = "class_cycle",
) -> list[float]:
    if steps <= 0:
        return []
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    if sampling == "class_cycle":
        sampler = ClassCycleSampler(samples, generator)
        represented_classes = len({sample.label for sample in samples})
        examples_per_update = batch_size * gradient_accumulation
        if examples_per_update % represented_classes:
            warnings.warn(
                "class_cycle crosses optimizer-update boundaries because "
                f"batch_size * gradient_accumulation ({examples_per_update}) is not "
                f"divisible by represented classes ({represented_classes})"
            )
    elif sampling == "inverse_frequency":
        sampler = WeightedRandomSampler(
            class_balanced_weights(samples),
            num_samples=len(samples),
            replacement=True,
            generator=generator,
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")
    loader = DataLoader(
        SampleDataset(samples),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=SFTCollator(processor, crop_size),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    def infinite_batches():
        while True:
            yield from loader

    iterator = iter(infinite_batches())
    progress = tqdm(range(steps), desc="LoRA updates", dynamic_ncols=True)
    for update in progress:
        accumulated = 0.0
        for _ in range(gradient_accumulation):
            batch = next(iterator)
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / gradient_accumulation
            loss.backward()
            accumulated += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(accumulated)
        progress.set_postfix(loss=f"{accumulated:.4f}")
    return losses


def _candidate_batch(processor, sample: Sample, pre, post, candidate_labels=DAMAGE_LABELS):
    process_vision_info = _require_training_packages()["process_vision_info"]
    variants = []
    starts = []
    for label in candidate_labels:
        variant = Sample.from_dict({**sample.to_dict(), "label": label, "label_id": DAMAGE_LABELS.index(label)})
        chat = messages(variant, True, pre, post)
        text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        variants.append((chat, text, label))
    chats = [item[0] for item in variants]
    image_inputs, video_inputs = process_vision_info(chats)
    batch = processor(
        text=[item[1] for item in variants],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    for row_index, (_, _, label) in enumerate(variants):
        answer_ids = processor.tokenizer(label, add_special_tokens=False)["input_ids"]
        start = _find_last_subsequence(batch["input_ids"][row_index].tolist(), answer_ids)
        starts.append((start, start + len(answer_ids)))
    return batch, starts


@torch.inference_mode()
def score_sample(model, processor, sample: Sample, d4_views: int = 1, crop_size: int = 448) -> dict:
    model.eval()
    pre, post = load_sample_pair(sample, crop_size)
    view_scores = []
    device = next(model.parameters()).device
    for pre_view, post_view, _ in d4_pair(pre, post, d4_views):
        candidate_scores = []
        # Score labels sequentially. Batching all candidates triples the paired
        # image tensors and exceeds 24 GB during BF16 7B evaluation.
        for label in DAMAGE_LABELS:
            batch, spans = _candidate_batch(
                processor, sample, pre_view, post_view, candidate_labels=(label,)
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.float()
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


def score_samples(model, processor, samples: Iterable[Sample], d4_views: int = 1, crop_size: int = 448):
    rows = list(samples)
    return [
        score_sample(model, processor, sample, d4_views, crop_size)
        for sample in tqdm(rows, desc="Scoring", dynamic_ncols=True)
    ]
