import numpy as np
import torch

from eventttt.qwen import ClassCycleSampler, class_balanced_weights
from eventttt.schemas import Sample


def sample(index, label, label_id):
    return Sample(
        sample_id=str(index),
        event_id="event",
        tile_id=str(index),
        pre_image="pre.png",
        post_image="post.png",
        label=label,
        label_id=label_id,
        dataset="test",
    )


def test_class_balanced_weights_equalize_total_class_mass():
    rows = [sample(0, "intact", 0)]
    rows += [sample(index, "damaged", 1) for index in range(1, 4)]
    rows += [sample(index, "destroyed", 2) for index in range(4, 9)]

    weights = class_balanced_weights(rows).numpy()
    totals = [
        weights[[row.label_id == label_id for row in rows]].sum()
        for label_id in range(3)
    ]
    assert np.allclose(totals, [1.0, 1.0, 1.0])


def test_class_cycle_sampler_balances_every_three_examples():
    rows = [sample(0, "intact", 0)]
    rows += [sample(index, "damaged", 1) for index in range(1, 4)]
    rows += [sample(index, "destroyed", 2) for index in range(4, 8)]
    sampler = ClassCycleSampler(rows, torch.Generator().manual_seed(7))

    indices = list(sampler)

    assert len(indices) == 9
    for start in range(0, len(indices), 3):
        assert {rows[index].label for index in indices[start : start + 3]} == {
            "intact",
            "damaged",
            "destroyed",
        }


def test_class_cycle_sampler_is_seed_reproducible():
    rows = [sample(index, "intact", 0) for index in range(3)]
    rows += [sample(index, "damaged", 1) for index in range(3, 6)]
    rows += [sample(index, "destroyed", 2) for index in range(6, 9)]

    first = list(ClassCycleSampler(rows, torch.Generator().manual_seed(11)))
    second = list(ClassCycleSampler(rows, torch.Generator().manual_seed(11)))

    assert first == second
