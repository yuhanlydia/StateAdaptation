#!/usr/bin/env python3
from __future__ import annotations

import json

from eventttt.qwen import preflight


if __name__ == "__main__":
    print(json.dumps(preflight(require_gpu=True), indent=2))
