#!/usr/bin/env python3
"""Run the fixed final Aperture evaluation; no GPU is used by plan/prepare/check."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from aperture_final.cli import main
if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'BLOCKED: {exc}',file=sys.stderr)
        raise SystemExit(2)
