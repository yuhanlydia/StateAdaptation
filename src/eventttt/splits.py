from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random
from typing import Iterable

from .schemas import DAMAGE_LABELS, Sample


@dataclass(frozen=True)
class EventSplit:
    target_event: str
    source: tuple[Sample, ...]
    support: tuple[Sample, ...]
    query: tuple[Sample, ...]


def _balanced_support(samples: list[Sample], shots_per_class: int, rng: Random) -> list[Sample]:
    by_label: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)
    chosen: list[Sample] = []
    for label in DAMAGE_LABELS:
        candidates = by_label[label]
        rng.shuffle(candidates)
        if len(candidates) < shots_per_class:
            raise ValueError(
                f"Need {shots_per_class} {label} examples, found {len(candidates)}"
            )
        chosen.extend(candidates[:shots_per_class])
    rng.shuffle(chosen)
    return chosen


def make_event_split(
    samples: Iterable[Sample],
    target_event: str,
    seed: int,
    shots_per_class: int | None = 4,
    support_budget: int | None = None,
) -> EventSplit:
    """Leave-one-event-out split with strict support/query tile separation."""
    all_samples = list(samples)
    source = [sample for sample in all_samples if sample.event_id != target_event]
    target = [sample for sample in all_samples if sample.event_id == target_event]
    if not source or not target:
        raise ValueError(f"Both source and target must be non-empty for event {target_event!r}")
    if (shots_per_class is None) == (support_budget is None):
        raise ValueError("Set exactly one of shots_per_class or support_budget")

    rng = Random(seed)
    tiles = list({sample.tile_id for sample in target})
    rng.shuffle(tiles)
    support_pool: list[Sample] = []
    required = support_budget or (shots_per_class or 0) * len(DAMAGE_LABELS)
    for tile in tiles:
        support_pool.extend(sample for sample in target if sample.tile_id == tile)
        if len(support_pool) < required:
            continue
        if shots_per_class is None:
            break
        counts = Counter(sample.label for sample in support_pool)
        if all(counts[label] >= shots_per_class for label in DAMAGE_LABELS):
            break

    if shots_per_class is not None:
        support = _balanced_support(support_pool, shots_per_class, rng)
    else:
        rng.shuffle(support_pool)
        if len(support_pool) < int(support_budget):
            raise ValueError(f"Target event has fewer than {support_budget} support candidates")
        support = support_pool[: int(support_budget)]
    support_tiles = {sample.tile_id for sample in support_pool}
    query = [sample for sample in target if sample.tile_id not in support_tiles]
    if not query:
        raise ValueError("No query examples remain after tile-disjoint support selection")
    return EventSplit(target_event, tuple(source), tuple(support), tuple(query))


def stratified_folds(samples: Iterable[Sample], folds: int, seed: int) -> list[tuple[list[Sample], list[Sample]]]:
    """Small-data stratified folds used only inside target support."""
    rows = list(samples)
    by_label: dict[str, list[Sample]] = defaultdict(list)
    for row in rows:
        by_label[row.label].append(row)
    minimum = min(len(group) for group in by_label.values())
    if folds < 2 or minimum < 2:
        raise ValueError("At least two examples per represented class are needed for support CV")
    folds = min(folds, minimum)
    rng = Random(seed)
    buckets: list[list[Sample]] = [[] for _ in range(folds)]
    for group in by_label.values():
        rng.shuffle(group)
        for index, sample in enumerate(group):
            buckets[index % folds].append(sample)
    result = []
    for held_index in range(folds):
        held = buckets[held_index]
        train = [sample for index, bucket in enumerate(buckets) if index != held_index for sample in bucket]
        result.append((train, held))
    return result
