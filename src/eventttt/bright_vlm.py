"""Shared frozen BRIGHT scorer for the supported multimodal model families.

The model families use different multimodal chat conventions, but both are
evaluated with the same paired pre/post crop and candidate-label likelihood
protocol.  This module intentionally does not add a generation fallback.
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import torch
from PIL import Image

from .aggregation import product_of_experts
from .prompts import SYSTEM_PROMPT, messages, question_for
from .qwen import _find_last_subsequence
from .schemas import DAMAGE_LABELS, Sample
from .vision import crop_pair, load_image


Family = Literal["phi", "gemma", "llama", "qwen3_vl", "internvl3"]


class _InternVLProcessor:
    """Small processor adapter for InternVL's remote-code model.

    InternVL3 exposes a tokenizer and model-specific image-token convention,
    but no ``AutoProcessor``.  This wrapper keeps the rest of the BRIGHT
    pipeline identical to the other families.
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.num_image_token = int(model.num_image_token)
        self.image_token = "<IMG_CONTEXT>"
        self.image_token_id = int(tokenizer.convert_tokens_to_ids(self.image_token))
        self.pixel_dtype = next(model.vision_model.parameters()).dtype
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

    def build_prompt(self, sample: Sample, label: str) -> str:
        self._task_mode = False
        template = copy.deepcopy(self.model.conv_template)
        template.system_message = SYSTEM_PROMPT
        question = (
            "<image>\n<image>\n"
            f"{question_for(sample)}"
        )
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], label)
        prompt = template.get_prompt()
        image_tokens = "<img>" + self.image_token * self.num_image_token + "</img>"
        prompt = prompt.replace("<image>", image_tokens, 1)
        prompt = prompt.replace("<image>", image_tokens, 1)
        return prompt

    def build_task_prompt(self, sample, label: str) -> str:
        """Build the one-image candidate prompt used by task benchmarks."""
        self._task_mode = True
        template = copy.deepcopy(self.model.conv_template)
        template.system_message = (
            "Answer the visual question using exactly the format "
            "'Answer: <candidate label>' and no explanation."
        )
        template.append_message(template.roles[0], f"<image>\n{sample.question}")
        # InternVL's instruction-tuned chat head is trained to emit the
        # benchmark's explicit answer prefix.  Omitting it makes the raw
        # single-letter tokens (A/B/C/D) collapse to an A prior on ManipBench.
        template.append_message(template.roles[1], f"Answer: {label}")
        prompt = template.get_prompt()
        # A 224px task image produces one quarter of the native 448px
        # InternVL visual tokens. This keeps the long ManipBench question
        # sequence within a 24GB card; BRIGHT continues to use 448px/256
        # tokens through build_prompt above.
        task_tokens = max(1, self.num_image_token // 4)
        image_tokens = "<img>" + self.image_token * task_tokens + "</img>"
        return prompt.replace("<image>", image_tokens, 1)

    def _image_tensor(self, image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        size = 224 if getattr(self, "_task_mode", False) else 448
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        values = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        return ((values - self.mean) / self.std).to(self.pixel_dtype)

    def __call__(self, text, images=None, return_tensors="pt", **kwargs):
        if images is None or len(images) < 1:
            raise ValueError("InternVL batches require at least one image")
        prompts = text if isinstance(text, list) else [text]
        encoded = self.tokenizer(prompts, return_tensors=return_tensors, padding=True)
        encoded["pixel_values"] = torch.stack([self._image_tensor(image) for image in images])
        encoded["image_flags"] = torch.ones((len(images), 1), dtype=torch.long)
        self.model.img_context_token_id = self.image_token_id
        return encoded

    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)


def load_bright_vlm(model_id: str, family: Family):
    """Load one of the supported checkpoints for frozen scoring."""
    if family == "internvl3":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )
        # InternVL3-8B fits frozen inference and checkpointed LoRA on a 24GB
        # RTX 3090 in bf16.  Keeping the full model on one device avoids the
        # meta-parameter gradient problem caused by CPU offload during TTA.
        model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            use_flash_attn=False,
        ).eval()
        # PEFT forwards an unused ``inputs_embeds`` keyword.  InternVL's
        # remote-code forward constructs its own image-conditioned embeddings,
        # so accept and ignore that keyword while retaining input_ids.
        original_forward = model.forward

        def _forward_compat(*args, inputs_embeds=None, **kwargs):
            # The remote wrapper always materializes full-vocabulary logits
            # before applying its masked support loss.  For KV-TTT/LoRA we
            # only need logits immediately before the non-ignored answer
            # tokens; selecting those positions avoids a 24GB-card OOM while
            # preserving the exact token cross-entropy.
            labels = kwargs.get("labels")
            if labels is None:
                return original_forward(*args, **kwargs)
            from transformers.modeling_outputs import CausalLMOutputWithPast

            pixel_values = kwargs["pixel_values"]
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            image_flags = kwargs.get("image_flags").squeeze(-1)
            input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
            vit_embeds = model.extract_feature(pixel_values)
            vit_embeds = vit_embeds[image_flags == 1]
            batch_size, seq_len, hidden = input_embeds.shape
            flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden)
            flat_ids = input_ids.reshape(batch_size * seq_len)
            selected = flat_ids == model.img_context_token_id
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, hidden)
            input_embeds = flat_embeds.reshape(batch_size, seq_len, hidden)

            target_positions = (labels != -100).nonzero(as_tuple=False)
            if target_positions.numel() == 0:
                raise ValueError("InternVL training batch has no non-ignored labels")
            # Task support batches are one row; preserve the general row
            # index for a clear failure if that invariant changes.
            if target_positions[:, 0].unique().numel() != 1:
                raise ValueError("InternVL compact training path expects one row")
            target = target_positions[:, 1]
            keep = target - 1
            if torch.any(keep < 0):
                raise ValueError("answer label begins at position zero")
            outputs = model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=kwargs.get("output_attentions"),
                output_hidden_states=kwargs.get("output_hidden_states"),
                return_dict=True,
                logits_to_keep=keep,
            )
            logits = outputs.logits[0]
            targets = labels[0, target].to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets)
            return CausalLMOutputWithPast(
                loss=loss, logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )
        model.forward = _forward_compat
        processor = _InternVLProcessor(tokenizer, model)
        model.img_context_token_id = processor.image_token_id
        # The remote-code wrapper delegates generation to a Qwen2 language
        # model whose own config controls KV-cache allocation. Disable both
        # configs; setting only the outer InternVL config leaves a large cache
        # alive during support-set backward passes.
        model.config.use_cache = False
        language_model = getattr(model, "language_model", None)
        if language_model is not None:
            language_model.config.use_cache = False
        return model, processor

    from transformers import AutoProcessor

    if family == "phi":
        from .task_phi import load_phi

        return load_phi(model_id, efficient_attention=True)

    processor = AutoProcessor.from_pretrained(model_id)
    if family == "qwen3_vl":
        # Qwen3-VL is exposed by the current Transformers integration rather
        # than AutoModelForImageTextToText in older releases.
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
        return model, processor
    from transformers import AutoModelForImageTextToText
    if family == "llama":
        # Older LLaVA checkpoints omit these processor fields even though the
        # CLIP vision config has a 14px patch and a CLS token.
        vision = getattr(processor.image_processor, "patch_size", None)
        processor.patch_size = vision or 14
        processor.num_additional_image_tokens = 1
        processor.vision_feature_select_strategy = "default"
        dtype = torch.float16
    elif family == "gemma":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown BRIGHT VLM family: {family}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, device_map="auto"
    )
    return model, processor


def _variant(sample: Sample, label: str) -> Sample:
    return Sample.from_dict(
        {**sample.to_dict(), "label": label, "label_id": DAMAGE_LABELS.index(label)}
    )


def _llama_text(sample: Sample, label: str) -> str:
    # The public checkpoint is an LLaVA wrapper around Llama 3 and has no chat
    # template in its processor.  Keep the paired-image order explicit.
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "<image><image>\n"
        f"{SYSTEM_PROMPT} {question_for(sample)}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{label}"
    )


def _phi_text(sample: Sample, label: str) -> str:
    return (
        "<|user|>\n<|image_1|>\n<|image_2|>\n"
        f"{SYSTEM_PROMPT} {question_for(sample)}"
        f"<|end|>\n<|assistant|>\n{label}<|end|>\n"
    )


def _inputs_for_candidate(
    processor, model_family: Family, sample: Sample, label: str, pre, post
):
    if model_family in ("gemma", "qwen3_vl"):
        text = processor.apply_chat_template(
            messages(_variant(sample, label), True, pre, post),
            tokenize=False,
            add_generation_prompt=False,
        )
    elif model_family == "llama":
        text = _llama_text(sample, label)
    elif model_family == "internvl3":
        text = processor.build_prompt(_variant(sample, label), label)
    else:
        text = _phi_text(sample, label)
    text_arg = text if model_family == "phi" else [text]
    batch = processor(text=text_arg, images=[pre, post], return_tensors="pt")
    answer_ids = processor.tokenizer(label, add_special_tokens=False)["input_ids"]
    start = _find_last_subsequence(batch["input_ids"][0].tolist(), answer_ids)
    return batch, (start, start + len(answer_ids))


def bright_labeled_batch(processor, model_family: Family, sample: Sample, crop_size: int = 448):
    """Build a BRIGHT pair with loss restricted to the true class tokens."""
    pre, post = crop_pair(
        load_image(sample.pre_image), load_image(sample.post_image),
        sample.bbox_xyxy, size=crop_size,
    )
    batch, (start, end) = _inputs_for_candidate(
        processor, model_family, sample, sample.label, pre, post
    )
    labels = torch.full_like(batch["input_ids"], -100)
    labels[0, start:end] = batch["input_ids"][0, start:end]
    batch["labels"] = labels
    return batch, (start, end)


def enable_bright_lora(model, model_family: Family, rank: int = 16,
                       alpha: int = 32, dropout: float = 0.05):
    """Attach the pre-registered LoRA-TTA modules for a BRIGHT pilot."""
    from peft import LoraConfig, get_peft_model

    targets = ["qkv_proj", "o_proj"] if model_family == "phi" else [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ]
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    adapted = get_peft_model(model, config)
    checkpoint = getattr(adapted, "gradient_checkpointing_enable", None)
    if callable(checkpoint):
        checkpoint()
    enable = getattr(adapted, "enable_input_require_grads", None)
    if callable(enable):
        enable()
    return adapted


def fit_bright_lora(model, processor, model_family: Family, samples,
                    crop_size: int = 448, passes: int = 1,
                    learning_rate: float = 2e-4, seed: int = 0,
                    grad_accum_steps: int = 3):
    """Support-only supervised LoRA fit; query samples never enter this loop."""
    if passes < 0:
        raise ValueError("passes must be non-negative")
    if not samples or passes == 0:
        return []
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    generator = torch.Generator().manual_seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    device = next(model.parameters()).device
    model.train()
    losses = []
    for _ in range(passes):
        total = 0.0
        order = torch.randperm(len(samples), generator=generator).tolist()
        optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(order, start=1):
            batch, _ = bright_labeled_batch(
                processor, model_family, samples[index], crop_size
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / grad_accum_steps).backward()
            if position % grad_accum_steps == 0 or position == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
        losses.append(total / len(samples))
    return losses


@torch.inference_mode()
def score_bright_sample(
    model,
    processor,
    model_family: Family,
    sample: Sample,
    crop_size: int = 448,
    controller=None,
    mask_builder=None,
) -> dict:
    """Score one BRIGHT pair by normalized candidate completion likelihood."""
    model.eval()
    pre, post = crop_pair(
        load_image(sample.pre_image),
        load_image(sample.post_image),
        sample.bbox_xyxy,
        size=crop_size,
    )
    device = next(model.parameters()).device
    scores = []
    for label in DAMAGE_LABELS:
        batch, (start, end) = _inputs_for_candidate(
            processor, model_family, sample, label, pre, post
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        if controller is not None:
            if mask_builder is None:
                raise ValueError("mask_builder is required with a KV controller")
            controller.set_mask(mask_builder(batch["input_ids"]))
        try:
            logits = model(**batch).logits.float()
        finally:
            if controller is not None:
                controller.clear_mask()
        targets = batch["input_ids"][0, start:end]
        token_log_probs = torch.log_softmax(logits[0, start - 1 : end - 1], dim=-1)
        scores.append(float(token_log_probs.gather(-1, targets.unsqueeze(-1)).sum()))
        del batch, logits, token_log_probs
    probabilities = product_of_experts([scores])
    prediction_id = int(np.argmax(probabilities))
    return {
        "sample_id": sample.sample_id,
        "event_id": sample.event_id,
        "tile_id": sample.tile_id,
        "label": sample.label,
        "label_id": sample.label_id,
        "prediction": DAMAGE_LABELS[prediction_id],
        "probabilities": probabilities.tolist(),
        "mean_log_scores": scores,
    }
