from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage

from eventttt.schemas import Sample
from eventttt.vision import load_image, materialize_pair

from .common import event_and_tile, unified_label


def _phase_directories(root: Path, phase: str) -> list[Path]:
    standard = root / f"{phase}-event"
    directories = [standard] if standard.exists() else []
    directories.extend(path for path in sorted(root.glob(f"{phase}-event*")) if path not in directories)
    return directories


def _target_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.int16)


def _instances(target: np.ndarray, min_pixels: int) -> Iterable[tuple[int, tuple[int, int, int, int], int]]:
    structure = np.ones((3, 3), dtype=np.uint8)
    for raw_label in (1, 2, 3):
        components, count = ndimage.label(target == raw_label, structure=structure)
        objects = ndimage.find_objects(components)
        for component_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            pixels = int(np.count_nonzero(components[slices] == component_id))
            if pixels < min_pixels:
                continue
            ys, xs = slices
            yield raw_label, (xs.start, ys.start, xs.stop, ys.stop), pixels


def adapt_bright_rasters(
    root: str | Path,
    output_crops: str | Path | None = None,
    crop_size: int = 448,
    crop_margin: float = 1.5,
    min_pixels: int = 6,
) -> list[Sample]:
    """Convert the official BRIGHT semantic rasters into building-component crops.

    Connected components are an approximation when adjacent buildings touch. Prefer
    ``adapt_bright_coco`` when CVPRW 2026 instance annotations are available.
    """
    dataset_root = Path(root).resolve()
    targets = sorted((dataset_root / "target").glob("*_building_damage.tif"))
    if not targets:
        raise FileNotFoundError(f"No *_building_damage.tif under {dataset_root / 'target'}")

    samples: list[Sample] = []
    for target_path in targets:
        tile = re.sub(r"_building_damage$", "", target_path.stem)
        event, tile_id = event_and_tile(tile)
        pre_candidates = [directory / f"{tile}_pre_disaster.tif" for directory in _phase_directories(dataset_root, "pre")]
        pre = next((path for path in pre_candidates if path.exists()), pre_candidates[0] if pre_candidates else dataset_root / "pre-event" / f"{tile}_pre_disaster.tif")
        post = dataset_root / "post-event" / f"{tile}_post_disaster.tif"
        if not pre.exists() or not post.exists():
            raise FileNotFoundError(f"Missing pair for {tile}: pre={pre.exists()}, post={post.exists()}")
        for index, (raw_label, bbox, pixels) in enumerate(_instances(_target_array(target_path), min_pixels)):
            label, label_id = unified_label(raw_label)
            sample_id = f"bright-{tile_id}-{index:05d}"
            pre_value, post_value = str(pre), str(post)
            stored_bbox = tuple(float(value) for value in bbox)
            if output_crops is not None:
                pre_value, post_value = materialize_pair(
                    pre,
                    post,
                    stored_bbox,
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
                    dataset="bright-raster",
                    bbox_xyxy=stored_bbox,
                    metadata={"target_path": str(target_path), "component_pixels": pixels},
                )
            )
    return samples


def _find_pair(root: Path, info: dict, phase: str) -> Path:
    explicit_keys = (f"{phase}_image", f"{phase}_file_name", f"{phase}_path")
    for key in explicit_keys:
        if info.get(key):
            path = Path(info[key])
            path = path if path.is_absolute() else root / path
            if path.exists():
                return path.resolve()

    name = Path(info.get("file_name", info.get("name", info.get("sample_id", "")))).name
    base = re.sub(r"_(pre|post)_disaster(?=\.[^.]+$)", "", name)
    base = Path(base).stem
    pattern = f"{base}_{phase}_disaster.*"
    dirs = [*_phase_directories(root, phase), root]
    matches = [candidate for directory in dirs for candidate in directory.glob(pattern)]
    if not matches:
        matches = list(root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {phase} image for {name!r}, found {matches}")
    return matches[0].resolve()


def _coco_documents(path: Path):
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No JSON annotations found under {path}")
    for file_path in files:
        yield file_path, json.loads(file_path.read_text(encoding="utf-8-sig"))


def adapt_bright_coco(
    root: str | Path,
    annotations: str | Path,
    output_crops: str | Path | None = None,
    crop_size: int = 448,
    crop_margin: float = 1.5,
) -> list[Sample]:
    """Adapt COCO-style BRIGHT instance labels to the EventTTT manifest."""
    dataset_root = Path(root).resolve()
    samples: list[Sample] = []
    for annotation_file, payload in _coco_documents(Path(annotations).resolve()):
        images = {int(row["id"]): row for row in payload["images"]}
        categories = {
            int(row["id"]): row.get("name", row["id"]) for row in payload["categories"]
        }
        pairs = {
            image_id: (
                _find_pair(dataset_root, info, "pre"),
                _find_pair(dataset_root, info, "post"),
            )
            for image_id, info in images.items()
        }
        for annotation in payload["annotations"]:
            image_id = int(annotation["image_id"])
            info = images[image_id]
            pre, post = pairs[image_id]
            tile_stem = str(info.get("sample_id", re.sub(r"_(pre|post)_disaster$", "", post.stem)))
            event, tile_id = event_and_tile(tile_stem)
            label, label_id = unified_label(categories[int(annotation["category_id"])])
            x, y, width, height = (float(v) for v in annotation["bbox"])
            bbox = (x, y, x + width, y + height)
            sample_id = f"bright-instance-{tile_id}-{annotation['id']}"
            pre_value, post_value = str(pre), str(post)
            stored_bbox = bbox
            if output_crops is not None:
                pre_value, post_value = materialize_pair(
                    pre,
                    post,
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
                    dataset="bright-instance",
                    bbox_xyxy=stored_bbox,
                    metadata={
                        "annotation_id": annotation["id"],
                        "annotation_file": str(annotation_file),
                        "iscrowd": annotation.get("iscrowd", 0),
                    },
                )
            )
    return samples
