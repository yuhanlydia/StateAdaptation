"""Provenance/resume wrapper for the existing query-label geometry analysis."""
import json
from pathlib import Path
import subprocess
import sys
from .protocol import JobStore,preflight_job
from .worker import fingerprint_assets


def run_geometry(job):
    if job['evidence']!='oracle_diagnostic' or job['kind']!='single':
        raise ValueError('legacy directional engine requires a single-image oracle diagnostic')
    preflight_job(job);store=JobStore(job['output'],fingerprint_assets(job))
    if store.complete():return {'status':'skipped','job_id':job['job_id']}
    with store.lock():
        out=store.path/'geometry.json'
        cmd=[sys.executable,'scripts/analyze_directional_geometry.py',
             '--family',job['family'],'--model-id',job['model_path'],
             '--support',job['support'],'--query',job['query'],'--output',str(out),
             '--rank',str(job['rank']),'--layers',*[str(v) for v in job['layers']],
             '--kinds','Q','K','V','O']
        subprocess.run(cmd,check=True)
        result=json.loads(out.read_text())
        if 'ORACLE' not in result.get('warning',''):
            raise ValueError('missing oracle disclosure in diagnostic result')
        store.finish({'evidence':'oracle_diagnostic','job_id':job['job_id'],
                      'diagnostic':'modulewise_energy_weighted_directional_geometry'},['geometry.json'])
    return {'status':'done','job_id':job['job_id']}
