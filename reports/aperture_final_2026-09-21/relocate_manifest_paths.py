#!/usr/bin/env python3
"""Relocate the released prepared JSONL manifests after extracting the data archive."""

import argparse
import json
from pathlib import Path


ORIGINAL_ROOT = Path(
    "/home/root123/.local/share/stateadaptation/aperture-final-round-20260921"
)


def rewrite(value, original_root: str, extracted_root: Path):
    if isinstance(value, str) and value.startswith(original_root + "/data/"):
        relative_path = value[len(original_root) + 1 :]
        relocated = extracted_root / relative_path
        if not relocated.is_file():
            raise FileNotFoundError(relocated)
        return str(relocated), 1
    if isinstance(value, dict):
        result = {}
        changes = 0
        for key, item in value.items():
            result[key], count = rewrite(item, original_root, extracted_root)
            changes += count
        return result, changes
    if isinstance(value, list):
        result = []
        changes = 0
        for item in value:
            changed, count = rewrite(item, original_root, extracted_root)
            result.append(changed)
            changes += count
        return result, changes
    return value, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extracted_root", type=Path, help="directory containing data/")
    args = parser.parse_args()
    root = args.extracted_root.resolve()
    prepared = root / "data" / "prepared"
    if not prepared.is_dir():
        parser.error(f"Missing {prepared}; extract the release data archive first")
    files = 0
    paths = 0
    for manifest in sorted(prepared.rglob("*.jsonl")):
        output = []
        changes = 0
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                value, count = rewrite(json.loads(line), str(ORIGINAL_ROOT), root)
                output.append(json.dumps(value, ensure_ascii=False) + "\n")
                changes += count
        if changes:
            manifest.write_text("".join(output), encoding="utf-8")
            files += 1
            paths += changes
    print(f"Relocated {paths} paths in {files} JSONL files under {root}")


if __name__ == "__main__":
    main()
