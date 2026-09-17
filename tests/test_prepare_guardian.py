import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "prepare_guardian_failure.py"
    spec = importlib.util.spec_from_file_location("prepare_guardian_failure", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_group_disjoint_support_excludes_entire_selected_groups():
    module = _module()
    rows = [
        {"sample_id": "a0", "group_id": "a", "label": "success"},
        {"sample_id": "a1", "group_id": "a", "label": "failure"},
        {"sample_id": "b0", "group_id": "b", "label": "success"},
        {"sample_id": "b1", "group_id": "b", "label": "failure"},
        {"sample_id": "c0", "group_id": "c", "label": "success"},
        {"sample_id": "c1", "group_id": "c", "label": "failure"},
    ]

    support, query = module.group_disjoint_split(rows, support_per_class=1, seed=0)

    assert {row["label"] for row in support} == {"success", "failure"}
    assert {row["group_id"] for row in support}.isdisjoint(
        {row["group_id"] for row in query}
    )
