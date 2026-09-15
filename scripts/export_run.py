#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventttt.artifacts import export_run


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage one completed EventTune run for private Hub upload"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="relative path below artifacts/model, for example runs/event/seed_0/run-v1",
    )
    args = parser.parse_args()
    if args.destination.is_absolute() or ".." in args.destination.parts:
        parser.error("--destination must be a safe relative path below artifacts/model")
    destination = ROOT / "artifacts" / "model" / args.destination
    manifest = export_run(args.run_dir, destination)
    print(json.dumps({"destination": str(destination), **manifest}, indent=2))


if __name__ == "__main__":
    main()
