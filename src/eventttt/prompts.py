from __future__ import annotations

from .schemas import DAMAGE_LABELS, Sample


SYSTEM_PROMPT = (
    "You assess building damage from paired remote-sensing crops. "
    "The first image is pre-event optical imagery. The second is post-event imagery, "
    "which may be SAR. Return exactly one label and no explanation."
)


def question_for(sample: Sample | None = None) -> str:
    if sample is not None and sample.question:
        return sample.question
    labels = ", ".join(DAMAGE_LABELS)
    return (
        "Compare the same central building before and after the disaster. "
        f"Classify its damage severity as exactly one of: {labels}."
    )


def user_content(sample: Sample, pre_image=None, post_image=None) -> list[dict]:
    return [
        {"type": "image", "image": pre_image if pre_image is not None else sample.pre_image},
        {"type": "image", "image": post_image if post_image is not None else sample.post_image},
        {"type": "text", "text": question_for(sample)},
    ]


def messages(sample: Sample, include_answer: bool, pre_image=None, post_image=None) -> list[dict]:
    result = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content(sample, pre_image, post_image)},
    ]
    if include_answer:
        result.append({"role": "assistant", "content": [{"type": "text", "text": sample.label}]})
    return result
