from __future__ import annotations

import json
import re
from pathlib import Path

from eventttt.schemas import Sample
from eventttt.vision import materialize_pair

from .common import event_and_tile, unified_label


NUMBER = r"[-+]?(?:\d*\.\d+|\d+)"


def _wkt_bbox(wkt: str) -> tuple[float, float, float, float]:
    values = [float(value) for value in re.findall(NUMBER, wkt)]
    if len(values) < 4 or len(values) % 2:
        raise ValueError(f"Invalid polygon WKT: {wkt[:80]!r}")
    xs, ys = values[0::2], values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def _image_for_label(label_path: Path) -> Path:
    path_text = str(label_path)
    candidates = [
        Path(path_text.replace("/labels/", "/images/")).with_suffix(".png"),
        Path(path_text.replace("/labels/", "/images/")).with_suffix(".tif"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find image paired with {label_path}")


def adapt_xbd(
    root: str | Path,
    output_crops: str | Path | None = None,
    crop_size: int = 448,
    crop_margin: float = 1.5,
) -> list[Sample]:
    """Adapt the standard xBD/xView2 labels/images directory structure."""
    dataset_root = Path(root).resolve()
    post_labels = sorted(dataset_root.rglob("*_post_disaster.json"))
    if not post_labels:
        raise FileNotFoundError(f"No *_post_disaster.json found under {dataset_root}")
    samples: list[Sample] = []
    for post_label in post_labels:
        pre_label = Path(str(post_label).replace("_post_disaster.json", "_pre_disaster.json"))
        post_image = _image_for_label(post_label)
        pre_image = _image_for_label(pre_label)
        tile_stem = post_label.stem.replace("_post_disaster", "")
        event, tile_id = event_and_tile(tile_stem)
        payload = json.loads(post_label.read_text(encoding="utf-8"))
        features = payload.get("features", {}).get("xy", payload.get("features", []))
        for index, feature in enumerate(features):
            properties = feature.get("properties", {})
            raw_label = properties.get("subtype", properties.get("damage"))
            if raw_label in (None, "un-classified"):
                continue
            label, label_id = unified_label(raw_label)
            wkt = feature.get("wkt", feature.get("geometry", {}).get("wkt"))
            if not wkt:
                continue
            bbox = _wkt_bbox(wkt)
            uid = properties.get("uid", f"{index:05d}")
            sample_id = f"xbd-{tile_id}-{uid}"
            pre_value, post_value = str(pre_image), str(post_image)
            stored_bbox = bbox
            if output_crops is not None:
                pre_value, post_value = materialize_pair(
                    pre_image,
                    post_image,
                    bbox,
                    Path(output_crops) / event,
                    sample_id,
                    margin=crop_margin,
                    size=crop_size,
                )
                stored_bbox = None
            samples.append(
                Sample(
                    sample_id=sample_id,
                    event_id=event,
                    tile_id=tile_id,
                    pre_image=pre_value,
                    post_image=post_value,
                    label=label,
                    label_id=label_id,
                    dataset="xbd",
                    bbox_xyxy=stored_bbox,
                    metadata={"source_label": str(raw_label), "uid": uid},
                )
            )
    return samples
