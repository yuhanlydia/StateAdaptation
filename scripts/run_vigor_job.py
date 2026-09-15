#!/usr/bin/env python3
"""Internal single-job entry point; run_vigor_suite.py is the public launcher."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vigor_handoff.worker import execute_job
p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
job=json.loads(Path(a.job).read_text())
if job.get('engine')=='legacy_directional_geometry':
    from vigor_handoff.geometry_runner import run_geometry
    result=run_geometry(job)
else:
    result=execute_job(job)
print(json.dumps(result,indent=2))
