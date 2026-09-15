#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from eventttt.io import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an EventTune JSONL manifest")
    parser.add_argument("manifest")
    parser.add_argument("--skip-paths", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_manifest(args.manifest, check_paths=not args.skip_paths),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

