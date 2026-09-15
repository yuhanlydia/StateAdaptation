#!/usr/bin/env python3
"""Fail-closed audit for balanced support24 inference-suite inputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eventttt.io import iter_jsonl, read_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-root", default="data/prepared/neurips")
    parser.add_argument("--runs-root", default="runs/neurips")
    parser.add_argument("--events", nargs="*")
    parser.add_argument("--output", default="reports/balanced_inference_asset_audit.json")
    args = parser.parse_args()

    prep_root, runs_root = Path(args.prep_root), Path(args.runs_root)
    events = args.events or sorted(path.name for path in prep_root.iterdir() if path.is_dir())
    rows, errors = [], []
    for event in events:
        support_path = prep_root / event / "target_support.jsonl"
        query_path = prep_root / event / "target_query.jsonl"
        original_path = runs_root / event / "original_eval" / "predictions.jsonl"
        missing = [str(path) for path in (support_path, query_path, original_path) if not path.is_file()]
        if missing:
            errors.extend(f"{event}: missing {path}" for path in missing)
            rows.append({"event": event, "ready": False, "missing": missing})
            continue

        support, query = read_samples(support_path), read_samples(query_path)
        original = list(iter_jsonl(original_path))
        support_ids = [sample.sample_id for sample in support]
        query_ids = [sample.sample_id for sample in query]
        original_ids = [row["sample_id"] for row in original]
        event_errors = []
        if len(support) != 24:
            event_errors.append(f"support count is {len(support)}, expected 24")
        support_labels = Counter(sample.label for sample in support)
        if sorted(support_labels.values()) != [8, 8, 8]:
            event_errors.append(f"support labels are not 8/class: {dict(support_labels)}")
        if len(set(support_ids)) != len(support_ids):
            event_errors.append("duplicate support sample IDs")
        if len(set(query_ids)) != len(query_ids):
            event_errors.append("duplicate query sample IDs")
        if set(support_ids) & set(query_ids):
            event_errors.append("support/query sample-ID overlap")
        if len(set(original_ids)) != len(original_ids):
            event_errors.append("duplicate original prediction sample IDs")
        if set(query_ids) != set(original_ids):
            event_errors.append(
                f"query/original ID mismatch: query={len(set(query_ids))}, "
                f"original={len(set(original_ids))}"
            )
        missing_images = sorted({
            path for sample in support + query
            for path in (sample.pre_image, sample.post_image) if not Path(path).is_file()
        })
        if missing_images:
            event_errors.append(f"{len(missing_images)} image paths are missing")
        errors.extend(f"{event}: {message}" for message in event_errors)
        rows.append({
            "event": event,
            "ready": not event_errors,
            "support_count": len(support),
            "support_labels": dict(sorted(support_labels.items())),
            "query_count": len(query),
            "query_labels": dict(sorted(Counter(sample.label for sample in query).items())),
            "original_count": len(original),
            "missing_images": len(missing_images),
            "errors": event_errors,
        })

    report = {
        "contract": "same support24, balanced target_query, existing raw-VLM original IDs",
        "events_requested": len(events),
        "events_ready": sum(row["ready"] for row in rows),
        "ready": bool(events) and not errors,
        "events": rows,
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("events_requested", "events_ready", "ready")}, indent=2))
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
