from eventttt.schemas import DAMAGE_LABELS, Sample
from eventttt.splits import make_event_split, stratified_folds


def rows():
    result = []
    for event in ("a", "b"):
        for tile_index in range(5):
            for label_id, label in enumerate(DAMAGE_LABELS):
                result.append(
                    Sample(
                        sample_id=f"{event}-{tile_index}-{label}",
                        event_id=event,
                        tile_id=f"{event}-{tile_index}",
                        pre_image="pre.png",
                        post_image="post.png",
                        label=label,
                        label_id=label_id,
                        dataset="test",
                    )
                )
    return result


def test_event_split_has_no_event_or_tile_leakage():
    split = make_event_split(rows(), "b", seed=3, shots_per_class=1, support_budget=None)
    assert {row.event_id for row in split.source} == {"a"}
    assert {row.event_id for row in split.support} == {"b"}
    assert {row.tile_id for row in split.support}.isdisjoint(
        {row.tile_id for row in split.query}
    )
    assert {row.label for row in split.support} == set(DAMAGE_LABELS)


def test_support_cv_only_partitions_support():
    support = [row for row in rows() if row.event_id == "b"]
    folds = stratified_folds(support, 3, seed=0)
    assert len(folds) == 3
    for train, held in folds:
        assert {row.sample_id for row in train}.isdisjoint({row.sample_id for row in held})
        assert len(train) + len(held) == len(support)
