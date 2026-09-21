"""Fixed medical experiment: patient-disjoint data, audits, fits, sealed evaluation."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from .data import prepare, locked_json
from .protocol import make_jobs, check_job, source_signature
from .worker import seal_choices, check_repeatability, require_repeatability
from .reporting import export
from vigor_handoff.protocol import atomic_json


def gates_for(jobs,root):
    result={}
    for j in jobs:
        if j['phase']=='curve' or j['arm']!='aperture' or j['model_name'] in result:continue
        a=deepcopy(j);a.update(arm='audit',job_id='audit-'+j['model_name'],output=str(Path(root)/'audits'/j['model_name']))
        duplicate=deepcopy(j);duplicate['output']=str(Path(root)/'repeat_fits'/j['model_name']);duplicate['job_id']='repeat-'+j['model_name']
        result[j['model_name']]={'audit':a,'reference':j,'repeat':duplicate}
    return result


def run_tasks(tasks,gpus,script,logdir):
    stop=threading.Event();guard=threading.Lock();records=[];logdir=Path(logdir);logdir.mkdir(parents=True,exist_ok=True)
    def worker(gpu,part):
        for task in part:
            if stop.is_set():break
            tag=task['job']['job_id']+'--'+task['stage'];inp=logdir/(tag+'.job.json');atomic_json(inp,task)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONHASHSEED':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1',
                 'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            log=logdir/(tag+'.log')
            with log.open('a') as f:
                proc=subprocess.run([sys.executable,str(script),'--internal-job',str(inp)],env=env,stdout=f,stderr=subprocess.STDOUT)
            r={'job_id':task['job']['job_id'],'stage':task['stage'],'returncode':proc.returncode,'log':str(log),'gpu':gpu}
            atomic_json(logdir/(tag+'.exit.json'),r)
            with guard:records.append(r)
            print(('OK ' if proc.returncode==0 else 'FAILED ')+tag,flush=True)
            if proc.returncode:stop.set();break
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        fs=[ex.submit(worker,g,tasks[i::len(gpus)]) for i,g in enumerate(gpus)]
        for f in fs:f.result()
    if len(records)!=len(tasks) or any(r['returncode'] for r in records):
        raise RuntimeError('stage failed; later stages stopped. Inspect logs. No data/protocol fallback permitted.')


def internal(path):
    t=json.loads(Path(path).read_text());j=t['job'];stage=t['stage']
    from .worker import configure_determinism, execute_fit, execute_evaluate
    configure_determinism(j)
    if stage=='audit':
        from visual_lens.audit import execute_audit
        return execute_audit(j)
    if stage=='fit':return execute_fit(j,t['audit_folder'])
    if stage=='evaluate':return execute_evaluate(j,t['selection_path'])
    raise ValueError('unknown internal stage')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',choices=['plan','prepare','check','run','summarize'])
    p.add_argument('--config',default='configs/aperture_medical_multimodel.json');p.add_argument('--internal-job')
    p.add_argument('--phases',nargs='+',choices=['main','replication','curve'],default=['main','replication'])
    p.add_argument('--hospitals',type=int,nargs='+');p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--gpus',default='0');p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args(argv)
    if a.internal_job:print(internal(a.internal_job));return 0
    if not a.action:p.error('action required')
    cfg=json.loads(Path(a.config).read_text());all_jobs=make_jobs(cfg);root=Path(cfg['run_root'])
    if a.hospitals and not set(a.hospitals)<=set(cfg['hospitals']):p.error('hospital outside fixed plan')
    if a.seeds and not set(a.seeds)<=set(cfg['support_seeds']):p.error('seed outside fixed plan')
    jobs=[j for j in all_jobs if j['phase'] in a.phases and (not a.hospitals or j['hospital'] in a.hospitals)
          and (a.seeds is None or j['support_seed'] in a.seeds)]
    if not jobs:p.error('empty selection')
    locked_json(root/'plan.json',{'config':cfg,'jobs':all_jobs,'source_signature':source_signature()})
    print(f'Selected {len(jobs)} of {len(all_jobs)} planned states. One dataset; five hospitals. H2 previously explored.')
    if a.action=='plan':
        print('Primary Qwen: 75; InternVL3 replication: 60; optional 16/64-label curves: 120.')
        print(f"Plan locked at {root/'plan.json'}");return 0
    if a.action=='prepare':
        print(prepare(cfg));return 0
    if a.action=='summarize':
        print(export(all_jobs,root,a.phases,a.allow_partial));return 0
    gates=gates_for(all_jobs,root);needed={j['model_name']:gates[j['model_name']] for j in jobs}
    checked={j['unit_id']:j for j in jobs}
    for g in needed.values():checked.setdefault(g['reference']['unit_id'],g['reference'])
    failures=[]
    for j in checked.values():
        try:check_job(j,assets=True)
        except Exception as exc:failures.append({'unit':j['unit_id'],'error':str(exc)})
    atomic_json(root/'preflight.json',{'units':len(checked),'failures':failures})
    if failures:
        for f in failures:print('BLOCKED',f['unit'],f['error'])
        return 2
    if a.action=='check':print('Patient/slide/content/budget/model checks passed. GPU integration remains to be run.');return 0
    gpus=[g.strip() for g in a.gpus.split(',') if g.strip()]
    if not gpus or len(gpus)!=len(set(gpus)):p.error('distinct GPU IDs required')
    script=Path(__file__).resolve().parents[2]/'scripts/run_aperture_medical_multimodel.py'
    run_tasks([{'stage':'audit','job':g['audit']} for g in needed.values()],gpus,script,root/'logs')
    # Repeat the same fit in an independent fresh process, using no query forward.
    for i,g in enumerate(needed.values()):
        gate_fits=[{'stage':'fit','job':g[role],'audit_folder':g['audit']['output']}
                   for role in ('reference','repeat')]
        # A repeatability pair must use the SAME device, in separate processes.
        run_tasks(gate_fits,[gpus[i%len(gpus)]],script,root/'logs')
    for g in needed.values():
        j=g['reference'];check_repeatability(Path(j['output'])/'fit',Path(g['repeat']['output'])/'fit',j['repeat_gate'],j['repeat_score_atol'])
    # All gate checks occur before the expensive matrix or any query evaluation.
    run_tasks([{'stage':'fit','job':j,'audit_folder':needed[j['model_name']]['audit']['output']} for j in jobs],gpus,script,root/'logs')
    byunit={}
    for j in jobs:byunit.setdefault(j['unit_id'],[]).append(j)
    for u,js in byunit.items():seal_choices(js,root/'selections'/(u+'.json'))
    run_tasks([{'stage':'evaluate','job':j,'selection_path':str(root/'selections'/(j['unit_id']+'.json'))} for j in jobs],gpus,script,root/'logs')
    full=[j for j in all_jobs if j['phase'] in a.phases]
    if len(jobs)==len(full):print(export(all_jobs,root,a.phases))
    else:print('Selected units complete. Paper export waits for all predeclared states in these phases.')
    return 0
