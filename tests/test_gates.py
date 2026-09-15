from collections import Counter

import pytest

from eventttt.gates import (
    assess_source_gate,
    exclude_reserved_samples,
    select_event_label_gate,
)
from eventttt.schemas import Sample


def sample(index, event, label, label_id):
    return Sample(
        sample_id=str(index),
        event_id=event,
        tile_id=str(index),
        pre_image="pre.png",
        post_image="post.png",
        label=label,
        label_id=label_id,
        dataset="test",
    )


def test_source_gate_selection_is_balanced_and_reproducible():
    rows = []
    index = 0
    for event in ("a", "b"):
        for label_id, label in enumerate(("intact", "damaged", "destroyed")):
            for _ in range(3):
                rows.append(sample(index, event, label, label_id))
                index += 1

    first = select_event_label_gate(rows, per_event_label=2, seed=7)
    second = select_event_label_gate(rows, per_event_label=2, seed=7)

    assert [row.sample_id for row in first] == [row.sample_id for row in second]
    assert Counter((row.event_id, row.label) for row in first) == {
        (event, label): 2
        for event in ("a", "b")
        for label in ("intact", "damaged", "destroyed")
    }


def test_source_gate_requires_enough_examples_per_group():
    with pytest.raises(ValueError, match="Need 2 examples"):
        select_event_label_gate([sample(0, "a", "intact", 0)], 2, 0)


def test_reserved_gate_is_excluded_and_must_match_training_manifest():
    rows = [sample(0, "a", "intact", 0), sample(1, "a", "damaged", 1)]
    remaining, excluded = exclude_reserved_samples(rows, [rows[1]])

    assert [row.sample_id for row in remaining] == ["0"]
    assert excluded == 1
    with pytest.raises(ValueError, match="outside the training manifest"):
        exclude_reserved_samples(rows, [sample(2, "b", "destroyed", 2)])


def test_source_gate_assessment_checks_score_and_class_diversity():
    metrics = {"count": 3, "macro_f1": 0.4}
    diverse = [{"prediction": value} for value in ("intact", "damaged", "intact")]
    collapsed = [{"prediction": "intact"} for _ in range(3)]

    assert assess_source_gate(metrics, diverse, 0.2, 2)["passed"] is True
    assert assess_source_gate(metrics, collapsed, 0.2, 2)["passed"] is False
    assert assess_source_gate({**metrics, "macro_f1": 0.1}, diverse, 0.2, 2)[
        "passed"
    ] is False
