from __future__ import annotations

import re
from pathlib import Path

from eventttt.schemas import LABEL_TO_ID


EVENT_TILE = re.compile(r"^(?P<event>.+?)_(?P<tile>\d+)(?:_(?:pre|post)_disaster)?$")


def event_and_tile(stem: str) -> tuple[str, str]:
    cleaned = re.sub(r"_(pre|post)_disaster$|_building_damage$", "", stem)
    match = EVENT_TILE.match(cleaned)
    if not match:
        return cleaned, cleaned
    return match.group("event"), cleaned


def unified_label(value: str | int) -> tuple[str, int]:
    normalized = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    mapping = {
        "0": "intact",
        "1": "intact",
        "2": "damaged",
        "3": "destroyed",
        "no-damage": "intact",
        "undamaged": "intact",
        "intact": "intact",
        "minor-damage": "damaged",
        "major-damage": "damaged",
        "damaged": "damaged",
        "destroyed": "destroyed",
    }
    if normalized not in mapping:
        raise ValueError(f"Cannot map damage label {value!r}")
    label = mapping[normalized]
    return label, LABEL_TO_ID[label]


def paired_path(root: Path, tile: str, phase: str, suffix: str | None = None) -> Path:
    if suffix is None:
        suffix = f"_{phase}_disaster.tif"
    return root / ("pre-event" if phase == "pre" else "post-event") / f"{tile}{suffix}"
