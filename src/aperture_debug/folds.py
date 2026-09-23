"""Deterministic, patient-grouped support cross-validation for H2 diagnostics."""

from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path

from aperture_table.protocol import PROTOCOL, _save_rows, read_roles, write_locked
from vigor_handoff.protocol import digest, file_hash


DEBUG_PROTOCOL = "aperture_debug_group_cv_v1"
FOLD_COUNT = 3


def _counts(rows, classes):
    counts = Counter(row["label_id"] for row in rows)
    return tuple(counts.get(label, 0) for label in range(classes))


def _balanced(counts):
    return bool(counts) and min(counts) > 0 and len(set(counts)) == 1


def _assignment(rows, classes):
    groups = sorted({str(row["group_id"]) for row in rows})
    if len(groups) < FOLD_COUNT:
        raise ValueError(f"grouped 3-fold CV requires at least 3 patients; found {len(groups)}")
    by_group = {group: [row for row in rows if str(row["group_id"]) == group]
                for group in groups}
    total = _counts(rows, classes)
    candidates = []
    for assignment in product(range(FOLD_COUNT), repeat=len(groups)):
        if set(assignment) != set(range(FOLD_COUNT)):
            continue
        validation = [[row for group, fold in zip(groups, assignment) if fold == index
                       for row in by_group[group]] for index in range(FOLD_COUNT)]
        val_counts = [_counts(fold, classes) for fold in validation]
        train_counts = [tuple(total[label] - count[label] for label in range(classes))
                        for count in val_counts]
        # read_roles' sealed table contract requires equal per-class budgets in each role.
        if not all(_balanced(counts) for counts in val_counts + train_counts):
            continue
        class_spread = sum(max(count[label] for count in val_counts) -
                           min(count[label] for count in val_counts)
                           for label in range(classes))
        class_deviation = sum(abs(FOLD_COUNT * count[label] - total[label])
                              for count in val_counts for label in range(classes))
        sizes = [len(fold) for fold in validation]
        score = (class_spread, class_deviation, max(sizes) - min(sizes),
                 sum(abs(FOLD_COUNT * size - len(rows)) for size in sizes), assignment)
        candidates.append((score, assignment))
    if not candidates:
        raise ValueError("class-balanced grouped 3-fold CV is impossible for these patients")
    return groups, min(candidates)[1]


def build_folds(original_job, output_dir):
    """Write three locked fold manifests and return read_roles-compatible overrides.

    Only the original 16 support examples are repartitioned. The sealed 200-example
    query manifest is read by ``read_roles`` for integrity validation and is never
    used by the fold assignment or copied into the debug output.
    """
    if (original_job.get("protocol") != PROTOCOL or original_job.get("domain_key") != "h2"
            or original_job.get("classes") != 2 or original_job.get("fit_count") != 12
            or original_job.get("cal_count") != 4 or original_job.get("query_count") != 200):
        raise ValueError("build_folds requires an original sealed H2 table job")

    roles = read_roles(original_job)
    support = sorted((deepcopy(row) for row in roles["support"] + roles["calibration"]),
                     key=lambda row: row["sample_id"])
    if len(support) != 16 or len({row["sample_id"] for row in support}) != 16:
        raise ValueError("expected 16 distinct original H2 support examples")
    if any(not row.get("group_id") for row in support):
        raise ValueError("missing patient group_id")

    groups, assignment = _assignment(support, original_job["classes"])
    fold_of = dict(zip(groups, assignment))
    original_split = Path(original_job["split_record"]).resolve()
    original_hash = file_hash(original_split)
    output_dir = Path(output_dir).resolve()
    overrides = []

    # Complete feasibility checks precede all writes, preventing partial fold sets.
    partitions = []
    for index in range(FOLD_COUNT):
        validation = [row for row in support if fold_of[str(row["group_id"])] == index]
        training = [row for row in support if fold_of[str(row["group_id"])] != index]
        partitions.append((training, validation))

    for index, (training, validation) in enumerate(partitions):
        folder = output_dir / f"fold_{index}"
        support_path = folder / "fit.jsonl"
        calibration_path = folder / "calibration.jsonl"
        split_path = folder / "split.json"
        _save_rows(support_path, training)
        _save_rows(calibration_path, validation)
        record = {
            "protocol": PROTOCOL,
            "debug_protocol": DEBUG_PROTOCOL,
            "debug_group_cv": True,
            "domain": "h2",
            "support_seed": original_job["support_seed"],
            "fold_index": index,
            "fold_count": FOLD_COUNT,
            "original_split_record": str(original_split),
            "original_split_hash": original_hash,
            "query_used_for_selection": False,
            "content": {
                "support": digest(training),
                "calibration": digest(validation),
                "query": digest(roles["query"]),
            },
            "per_class_counts": {
                "support": list(_counts(training, original_job["classes"])),
                "calibration": list(_counts(validation, original_job["classes"])),
                "query": list(_counts(roles["query"], original_job["classes"])),
            },
            "fit_groups": sorted({str(row["group_id"]) for row in training}),
            "calibration_groups": sorted({str(row["group_id"]) for row in validation}),
            "query_count": len(roles["query"]),
            "query_ids": [row["sample_id"] for row in roles["query"]],
        }
        write_locked(split_path, record)
        override = {
            "support": str(support_path),
            "calibration": str(calibration_path),
            "split_record": str(split_path),
            "fit_count": len(training),
            "cal_count": len(validation),
            "debug_group_cv": True,
            "debug_group_cv_protocol": DEBUG_PROTOCOL,
            "fold_index": index,
            "fold_count": FOLD_COUNT,
            "original_split_hash": original_hash,
        }
        # Prove each emitted override remains accepted by the unchanged table validator.
        fold_job = deepcopy(original_job)
        fold_job.update(override)
        read_roles(fold_job)
        overrides.append(override)
    return overrides
