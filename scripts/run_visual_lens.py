#!/usr/bin/env python3
"""P0 stage orchestrator. Planning/checking never load a model or use a GPU."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,os,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from visual_lens.plan import STAGES,build_plan
from vigor_handoff.protocol import atomic_json,preflight_job


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','check','run'])
    p.add_argument('--config',default='configs/visual_lens_p0.json')
    p.add_argument('--stages',nargs='+',choices=list(STAGES),default=list(STAGES))
    p.add_argument('--gpus',default='0')
    p.add_argument('--families',nargs='+');p.add_argument('--domains',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());all_jobs=build_plan(cfg)
    jobs=[j for j in all_jobs if j['stage'] in a.stages]
    if a.families:jobs=[j for j in jobs if j['family'] in a.families]
    if a.domains:jobs=[j for j in jobs if j['domain'] in a.domains]
    if a.seeds is not None:jobs=[j for j in jobs if j['support_seed'] in a.seeds]
    if not jobs:p.error('no jobs match the selection')
    # Add required support-only gates even if a domain/seed filter omits their source fold.
    if 'audit' in a.stages:
        extra={j['required_audit']['job_id']:j['required_audit'] for j in jobs if j.get('required_audit')}
        have={j['job_id'] for j in jobs};jobs=[g for k,g in extra.items() if k not in have]+jobs
    root=Path(cfg['run_root']);plan=root/'plan';plan.mkdir(parents=True,exist_ok=True)
    # Keep the full canonical matrix stable across partial invocations.
    atomic_json(root/'plan.json',{'config':cfg,'jobs':all_jobs})
    atomic_json(plan/'selection.json',{'jobs':jobs})
    for j in all_jobs:atomic_json(plan/f"{j['job_id']}.json",j)
    print(f"Selected {len(jobs)}/{len(all_jobs)} jobs; canonical plan: {root/'plan.json'}")
    for stage in STAGES:print(f"  {stage}: {sum(j['stage']==stage for j in jobs)}")
    if a.action=='plan':return 0
    unique={};failures=[]
    for j in jobs:
        key=(j['model_path'],j['support'],j['query'])
        if key in unique:continue
        try:unique[key]=preflight_job(j)
        except Exception as exc:failures.append({'job_id':j['job_id'],'error':str(exc)})
    atomic_json(plan/'preflight.json',{'passed':len(unique),'failures':failures})
    if failures:
        for f in failures:print('BLOCKED',f['job_id'],f['error'])
        return 2
    if a.action=='check':return 0
    gpus=[x.strip() for x in a.gpus.split(',') if x.strip()]
    if not gpus or len(gpus)!=len(set(gpus)):p.error('unique nonempty GPU IDs required')
    for stage in STAGES:
        block=[j for j in jobs if j['stage']==stage]
        if not block:continue
        def partition(gpu,part):
            exits=[]
            for j in part:
                log=plan/f"{j['job_id']}.log"
                env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONHASHSEED':'0','PYTHONUNBUFFERED':'1'}
                with log.open('a') as f:
                    r=subprocess.run([sys.executable,'scripts/run_visual_lens_job.py','--job',str(plan/f"{j['job_id']}.json")],env=env,stdout=f,stderr=subprocess.STDOUT)
                state={'job_id':j['job_id'],'returncode':r.returncode,'log':str(log)}
                atomic_json(plan/f"{j['job_id']}.exit.json",state);exits.append(state)
                print(('OK ' if r.returncode==0 else 'FAILED ')+j['job_id'],flush=True)
                if r.returncode:break
            return exits
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures=[executor.submit(partition,gpu,block[i::len(gpus)]) for i,gpu in enumerate(gpus)]
            exits=[r for f in futures for r in f.result()]
        atomic_json(plan/f'{stage}_execution.json',exits)
        if len(exits)!=len(block) or any(r['returncode'] for r in exits):
            print(f'{stage} failed; downstream stages were not launched.');return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
