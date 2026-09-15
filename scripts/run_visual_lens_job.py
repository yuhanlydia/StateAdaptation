#!/usr/bin/env python3
"""One P0 job; CUDA visibility is assigned by run_visual_lens.py."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    job=json.loads(Path(a.job).read_text())
    from vigor_handoff.protocol import JobStore
    from vigor_handoff.worker import fingerprint_assets
    gate=job.get('required_audit')
    if gate:
        state=JobStore(gate['output'],fingerprint_assets(gate))
        if not state.complete():raise RuntimeError(f"required P0-C gate incomplete or changed: {gate['output']}")
    if job['engine']=='audit':
        from visual_lens.audit import execute_audit
        result=execute_audit(job)
    elif job['engine']=='evidence':
        from visual_lens.evidence import execute_evidence
        result=execute_evidence(job)
    else:
        from vigor_handoff.worker import execute_job
        result=execute_job(job)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
