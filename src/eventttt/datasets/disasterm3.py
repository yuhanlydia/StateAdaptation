from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from eventttt.io import iter_jsonl
from eventttt.schemas import Sample

from .common import event_and_tile, unified_label


def _records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("data", value.get("records", [value]))
    yield from value


def _assistant_answer(row: dict) -> str | None:
    for key in ("answer", "label", "response", "output"):
        if row.get(key) is not None:
            return str(row[key])
    for message in reversed(row.get("messages", row.get("conversations", []))):
        role = message.get("role", message.get("from", ""))
        if role in ("assistant", "gpt"):
            return str(message.get("content", message.get("value", "")))
    return None


def _images(row: dict) -> tuple[str, str] | None:
    if row.get("pre_image") and row.get("post_image"):
        return str(row["pre_image"]), str(row["post_image"])
    images = row.get("images", row.get("image", []))
    if isinstance(images, str):
        images = [images]
    if len(images) >= 2:
        return str(images[0]), str(images[1])
    return None


def adapt_disasterm3(root: str | Path, annotations: str | Path) -> list[Sample]:
    """Normalize damage-severity rows from the released DisasterM3 instruction files.

    Non-severity tasks and answers that cannot be mapped to the three shared labels are
    deliberately skipped. The original question is retained for optional auxiliary SFT.
    """
    dataset_root = Path(root).resolve()
    annotation_path = Path(annotations).resolve()
    samples: list[Sample] = []
    for index, row in enumerate(_records(annotation_path)):
        pair = _images(row)
        answer = _assistant_answer(row)
        if pair is None or answer is None:
            continue
        try:
            label, label_id = unified_label(answer)
        except ValueError:
            continue
        pre, post = (Path(value) for value in pair)
        pre = pre if pre.is_absolute() else dataset_root / pre
        post = post if post.is_absolute() else dataset_root / post
        event = str(row.get("event_id", row.get("event", row.get("disaster", ""))))
        tile = str(row.get("tile_id", row.get("id", Path(post).stem)))
        inferred_event, inferred_tile = event_and_tile(Path(post).stem)
        event = event or inferred_event
        tile = tile or inferred_tile
        sample_id = str(row.get("id", f"disasterm3-{index:08d}"))
        samples.append(
            Sample(
                sample_id=sample_id,
                event_id=event,
                tile_id=tile,
                pre_image=str(pre.resolve()),
                post_image=str(post.resolve()),
                label=label,
                label_id=label_id,
                dataset="disasterm3",
                question=row.get("question", row.get("instruction")),
                metadata={"source_annotation": str(annotation_path)},
            )
        )
    if not samples:
        raise ValueError(
            "No three-class paired-image severity examples were found. "
            "Check the DisasterM3 release schema and the selected subset."
        )
    return samples
