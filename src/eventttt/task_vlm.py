"""Model loading glue for the single-image task benchmarks."""

from __future__ import annotations

from .bright_vlm import enable_bright_lora, load_bright_vlm
from .qwen import DEFAULT_MODEL, load_model

DEFAULT_QWEN3_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_INTERNVL3_MODEL = "OpenGVLab/InternVL3-8B-Instruct"


def load_task_model(model_id: str, family: str, *, use_lora: bool = False,
                    gradient_checkpointing: bool = False, source_adapter: str | None = None):
    if family == "qwen2":
        return load_model(model_id, source_adapter=source_adapter,
                          gradient_checkpointing=gradient_checkpointing,
                          use_lora=use_lora)
    if family not in {"qwen3_vl", "internvl3"}:
        raise ValueError(f"Unknown task model family: {family}")
    model, processor = load_bright_vlm(model_id, family)
    if source_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, source_adapter, is_trainable=True)
    elif use_lora:
        model = enable_bright_lora(model, family)
    if gradient_checkpointing and not use_lora:
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
        req = getattr(model, "enable_input_require_grads", None)
        if callable(req):
            req()
        model.config.use_cache = False
    return model, processor


def default_task_model(family: str) -> str:
    if family == "qwen2":
        return DEFAULT_MODEL
    if family == "qwen3_vl":
        return DEFAULT_QWEN3_MODEL
    if family == "internvl3":
        return DEFAULT_INTERNVL3_MODEL
    raise ValueError(f"Unknown task model family: {family}")
