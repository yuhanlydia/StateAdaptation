"""Contracts for query-blind, patient-grouped H2 support folds."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aperture_debug.folds import DEBUG_PROTOCOL, build_folds
from aperture_table.protocol import PROTOCOL, _save_rows, read_roles, write_locked
from vigor_handoff.protocol import digest, file_hash, resolved_manifest


LABELS = ["negative", "positive"]


def _row(sample_id, group_id, label_id, root):
    return {
        "sample_id": sample_id,
        "domain_id": "hospital_2",
        "group_id": group_id,
        "image": str((root / f"{sample_id}.png").resolve()),
        "label_id": label_id,
        "label": LABELS[label_id],
        "candidate_labels": LABELS,
        "metadata": {"pixel_sha256": f"pixel-{sample_id}"},
    }


def _make_job(root, group_specs, fit_groups):
    rows = []
    for group, counts in group_specs:
        for label_id, count in enumerate(counts):
            for offset in range(count):
                rows.append(_row(f"{group}-{label_id}-{offset}", group, label_id, root))
    fit = [row for row in rows if row["group_id"] in fit_groups]
    calibration = [row for row in rows if row["group_id"] not in fit_groups]
    query = [_row(f"query-{label}-{offset}", f"query-patient-{label}-{offset}", label, root)
             for label in range(2) for offset in range(100)]
    prepared = root / "prepared" / "h2" / "seed_0"
    query_path = prepared.parent / "query.jsonl"
    _save_rows(prepared / "fit.jsonl", fit)
    _save_rows(prepared / "calibration.jsonl", calibration)
    _save_rows(query_path, query)
    resolved = {
        "support": resolved_manifest(prepared / "fit.jsonl"),
        "calibration": resolved_manifest(prepared / "calibration.jsonl"),
        "query": resolved_manifest(query_path),
    }
    record = {
        "protocol": PROTOCOL,
        "domain": "h2",
        "support_seed": 0,
        "content": {key: digest(value) for key, value in resolved.items()},
    }
    write_locked(prepared / "split.json", record)
    return {
        "protocol": PROTOCOL,
        "domain_key": "h2",
        "support_seed": 0,
        "classes": 2,
        "support": str(prepared / "fit.jsonl"),
        "calibration": str(prepared / "calibration.jsonl"),
        "query": str(query_path),
        "split_record": str(prepared / "split.json"),
        "fit_count": len(fit),
        "cal_count": len(calibration),
        "query_count": 200,
    }


@pytest.fixture
def original_job(tmp_path):
    # 12 fit + 4 calibration; each of the six patients contains both classes.
    specs = [("p0", (2, 1)), ("p1", (1, 2)), ("p2", (2, 1)),
             ("p3", (1, 2)), ("p4", (1, 1)), ("p5", (1, 1))]
    return _make_job(tmp_path, specs, {"p0", "p1", "p2", "p3"})


def _fold_job(original, override):
    job = deepcopy(original)
    job.update(override)
    return job


def test_group_isolation_class_coverage_and_read_roles_compatibility(tmp_path, original_job):
    overrides = build_folds(original_job, tmp_path / "cv")
    assert len(overrides) == 3
    original_hash = file_hash(original_job["split_record"])
    for index, override in enumerate(overrides):
        assert override["fold_index"] == index
        assert override["debug_group_cv_protocol"] == DEBUG_PROTOCOL
        assert override["original_split_hash"] == original_hash
        assert "query" not in override
        roles = read_roles(_fold_job(original_job, override))
        train_groups = {row["group_id"] for row in roles["support"]}
        val_groups = {row["group_id"] for row in roles["calibration"]}
        assert train_groups.isdisjoint(val_groups)
        assert set(Counter(row["label_id"] for row in roles["support"])) == {0, 1}
        assert set(Counter(row["label_id"] for row in roles["calibration"])) == {0, 1}
        record = json.loads(Path(override["split_record"]).read_text())
        assert record["protocol"] == PROTOCOL
        assert record["debug_protocol"] == DEBUG_PROTOCOL
        assert record["query_used_for_selection"] is False
        assert record["original_split_hash"] == original_hash


def test_every_support_sample_and_patient_is_validation_exactly_once(tmp_path, original_job):
    overrides = build_folds(original_job, tmp_path / "cv")
    validation = [row for override in overrides
                  for row in resolved_manifest(override["calibration"])]
    original = read_roles(original_job)
    support = original["support"] + original["calibration"]
    assert Counter(row["sample_id"] for row in validation) == Counter(
        {row["sample_id"]: 1 for row in support})
    assert Counter(row["group_id"] for override in overrides
                   for row in resolved_manifest(override["calibration"])) == Counter(
        row["group_id"] for row in support)


def test_assignment_is_deterministic_and_locked_files_are_not_overwritten(tmp_path, original_job):
    first = build_folds(original_job, tmp_path / "cv-a")
    again = build_folds(original_job, tmp_path / "cv-a")
    second = build_folds(original_job, tmp_path / "cv-b")
    ids = lambda folds: [[row["sample_id"] for row in resolved_manifest(f["calibration"])]
                         for f in folds]
    assert ids(first) == ids(again) == ids(second)
    path = Path(first[0]["support"])
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="prepared manifest changed"):
        build_folds(original_job, tmp_path / "cv-a")


def test_rejects_fewer_than_three_patient_groups_before_writing(tmp_path):
    job = _make_job(tmp_path / "source", [("fit", (6, 6)), ("cal", (2, 2))], {"fit"})
    output = tmp_path / "cv"
    with pytest.raises(ValueError, match="at least 3 patients"):
        build_folds(job, output)
    assert not output.exists()


def test_rejects_when_all_classes_cannot_appear_in_three_validation_folds(tmp_path):
    # Only p0 and p3 carry class 1, although each original role is class-balanced.
    specs = [("p0", (0, 6)), ("p1", (3, 0)), ("p2", (3, 0)), ("p3", (2, 2))]
    job = _make_job(tmp_path / "source", specs, {"p0", "p1", "p2"})
    output = tmp_path / "cv"
    with pytest.raises(ValueError, match="impossible"):
        build_folds(job, output)
    assert not output.exists()
