#!/usr/bin/env python3
"""Build a fixed full matrix, preflight it, and optionally execute/resume on GPUs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vigor_handoff.protocol import expand_plan,preflight_job,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/vigor_iclr2027.json')
    p.add_argument('--phases',nargs='+',choices=['main','ablation','diagnostic','support','matched_lora'],default=['main','ablation'])
    p.add_argument('--plan-dir',default='runs/vigor_handoff_v1/plan')
    p.add_argument('--execute',action='store_true',help='without this, no GPU job is launched')
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--gpus',default='0',help='comma-separated visible CUDA IDs or UUIDs, one worker per GPU')
    p.add_argument('--families',nargs='*');p.add_argument('--domains',nargs='*')
    p.add_argument('--arms',nargs='*');p.add_argument('--seeds',nargs='*',type=int)
    p.add_argument('--allow-oracle',action='store_true')
    p.add_argument('--keep-going',action='store_true',help='record failures and continue other independent jobs')
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text());jobs=expand_plan(config,args.phases)
    if args.families:jobs=[j for j in jobs if j['family'] in args.families]
    if args.domains:jobs=[j for j in jobs if j['domain'] in args.domains]
    if args.arms:jobs=[j for j in jobs if j['arm'] in args.arms]
    if args.seeds is not None:jobs=[j for j in jobs if j['support_seed'] in args.seeds]
    if not jobs:p.error('selection contains no jobs')
    if args.execute and 'diagnostic' in args.phases and not args.allow_oracle:
        p.error('oracle diagnostics require --allow-oracle and stay outside primary tables')
    root=Path(args.plan_dir);root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'plan.json',{'config':args.config,'jobs':jobs})
    for j in jobs:atomic_json(root/f"{j['job_id']}.json",j)
    print(f'Fixed plan: {len(jobs)} jobs; saved to {root}/plan.json')
    for phase in args.phases:print(f"  {phase}: {sum(j['phase']==phase for j in jobs)}")
    print('No outcome-dependent pruning or query-based model selection is implemented.')
    if args.preflight or args.execute:
        failures=[];validated=set()
        for j in jobs:
            key=(j['model_path'],j['support'],j['query'])
            if key in validated:continue
            try:preflight_job(j);validated.add(key)
            except Exception as exc:failures.append({'job':j['job_id'],'error':str(exc)})
        atomic_json(root/'preflight.json',{'validated':len(validated),'failures':failures})
        if failures:
            for f in failures[:20]:print(f"BLOCKED {f['job']}: {f['error']}",file=sys.stderr)
            return 2
    if not args.execute:
        print('Dry run only. Add --preflight to check local assets; --execute to launch.')
        return 0
    gpus=[v.strip() for v in args.gpus.split(',') if v.strip()]
    if not gpus or len(set(gpus))!=len(gpus):p.error('GPU list must be nonempty and unique')
    def run_partition(gpu,partition):
        results=[]
        for j in partition:
            log=root/f"{j['job_id']}.log"
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONHASHSEED':'0','PYTHONUNBUFFERED':'1'}
            with log.open('a') as f:
                result=subprocess.run([sys.executable,'scripts/run_vigor_job.py','--job',str(root/f"{j['job_id']}.json")],env=env,stdout=f,stderr=subprocess.STDOUT)
            status={'job_id':j['job_id'],'returncode':result.returncode,'log':str(log),'gpu':gpu}
            atomic_json(root/f"{j['job_id']}.exit.json",status);results.append(status)
            print(f"{'OK' if result.returncode==0 else 'FAILED'} {j['job_id']}",flush=True)
            if result.returncode and not args.keep_going:break
        return results
    results=[]
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures=[executor.submit(run_partition,gpu,jobs[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for future in futures:results.extend(future.result())
    atomic_json(root/'execution.json',results)
    success=len(results)==len(jobs) and all(r['returncode']==0 for r in results)
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
