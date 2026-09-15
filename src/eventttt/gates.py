from __future__ import annotations

from collections import defaultdict
from random import Random
from typing import Iterable, Sequence

from .schemas import Sample


def exclude_reserved_samples(
    samples: Sequence[Sample], reserved: Sequence[Sample]
) -> tuple[list[Sample], int]:
    """Remove a reserved gate from training and reject mismatched manifests."""
    reserved_ids = {sample.sample_id for sample in reserved}
    sample_ids = {sample.sample_id for sample in samples}
    unknown_ids = reserved_ids - sample_ids
    if unknown_ids:
        examples = ", ".join(sorted(unknown_ids)[:5])
        raise ValueError(
            f"Exclusion manifests contain {len(unknown_ids)} sample IDs outside "
            f"the training manifest (examples: {examples})"
        )
    remaining = [sample for sample in samples if sample.sample_id not in reserved_ids]
    if not remaining:
        raise ValueError("No source examples remain after applying exclusions")
    return remaining, len(reserved_ids)


def select_event_label_gate(
    samples: Sequence[Sample], per_event_label: int, seed: int
) -> list[Sample]:
    if per_event_label <= 0:
        raise ValueError("per_event_label must be positive")
    groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.event_id, sample.label)].append(sample)
    if not groups:
        raise ValueError("Cannot create a source gate from an empty manifest")
    rng = Random(seed)
    selected = []
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) < per_event_label:
            raise ValueError(
                f"Need {per_event_label} examples for event/label {key}, found {len(rows)}"
            )
        selected.extend(rng.sample(rows, per_event_label))
    rng.shuffle(selected)
    return selected


def assess_source_gate(
    metrics: dict,
    predictions: Iterable[dict],
    minimum_macro_f1: float,
    minimum_predicted_classes: int,
) -> dict:
    rows = list(predictions)
    if int(metrics["count"]) != len(rows):
        raise ValueError(
            f"Metrics count {metrics['count']} does not match {len(rows)} predictions"
        )
    predicted_classes = sorted({str(row["prediction"]) for row in rows})
    macro_f1 = float(metrics["macro_f1"])
    passed = (
        macro_f1 >= minimum_macro_f1
        and len(predicted_classes) >= minimum_predicted_classes
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "count": len(rows),
        "macro_f1": macro_f1,
        "minimum_macro_f1": minimum_macro_f1,
        "predicted_classes": predicted_classes,
        "predicted_class_count": len(predicted_classes),
        "minimum_predicted_classes": minimum_predicted_classes,
    }
