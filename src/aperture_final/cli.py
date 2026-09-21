"""Final-round orchestration. No model import for plan/prepare/check/export."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import threading
from .protocol import make_jobs, prepare_job, check_job, write_locked, source_signature
from .worker import seal_choices
from .reporting import export
from vigor_handoff.protocol import atomic_json


def audits_for(jobs, root):
    result={}
    for j in jobs:
        if j['kind'] in result:continue
        a=deepcopy(j);a.update(arm='audit',job_id='audit-'+j['kind'],output=str(Path(root)/'audits'/j['kind']))
        result[j['kind']]=a
    return result


def run_partitioned(tasks, gpus, script, logdir):
    """At most one model process per GPU; any failed task stops new launches."""
    stop=threading.Event();logdir=Path(logdir);logdir.mkdir(parents=True,exist_ok=True)
    records=[];guard=threading.Lock()
    def partition(gpu, part):
        for task in part:
            if stop.is_set():break
            name=task['job']['job_id']+'--'+task['stage']
            inp=logdir/(name+'.job.json');atomic_json(inp,task)
            log=logdir/(name+'.log')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONHASHSEED':'0','PYTHONUNBUFFERED':'1',
                 'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            with log.open('a') as f:
                r=subprocess.run([sys.executable,str(script),'--internal-job',str(inp)],env=env,stdout=f,stderr=subprocess.STDOUT)
            state={'job_id':task['job']['job_id'],'stage':task['stage'],'returncode':r.returncode,'gpu':gpu,'log':str(log)}
            atomic_json(logdir/(name+'.exit.json'),state)
            with guard:records.append(state)
            print(('OK ' if r.returncode==0 else 'FAILED ')+name,flush=True)
            if r.returncode:stop.set();break
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures=[ex.submit(partition,gpu,tasks[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in futures:f.result()
    if len(records)!=len(tasks) or any(r['returncode'] for r in records):
        raise RuntimeError('stage failed; later stages have not been launched; inspect logs/exit records')
    return records


def internal(path):
    task=json.loads(Path(path).read_text());j=task['job'];stage=task['stage']
    if stage=='audit':
        from visual_lens.audit import execute_audit
        return execute_audit(j)
    if stage=='fit':
        from .worker import execute_fit
        return execute_fit(j,task['audit_folder'])
    if stage=='evaluate':
        from .worker import execute_evaluate
        return execute_evaluate(j,task['selection_path'])
    raise ValueError('invalid internal stage')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',choices=['plan','prepare','check','run','summarize'])
    p.add_argument('--internal-job')
    p.add_argument('--config',default='configs/aperture_final_round.json')
    p.add_argument('--domains',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--gpus',default='0')
    p.add_argument('--stages',nargs='+',choices=['audit','fit','evaluate'],default=['audit','fit','evaluate'])
    p.add_argument('--allow-partial',action='store_true',help='export coverage only, never partial paper tables')
    a=p.parse_args(argv)
    if a.internal_job:
        print(internal(a.internal_job));return 0
    if not a.action:p.error('action is required')
    config=json.loads(Path(a.config).read_text());all_jobs=make_jobs(config)
    jobs=[j for j in all_jobs if (not a.domains or j['domain'] in a.domains) and
         (a.seeds is None or j['support_seed'] in a.seeds)]
    if not jobs:p.error('empty job selection')
    root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
    # A filtered invocation cannot silently redefine the canonical matrix.
    write_locked(root/'plan.json',{'config':config,'jobs':all_jobs,'source_signature':source_signature()})
    print(f'{len(jobs)}/{len(all_jobs)} states; {len({j["unit_id"] for j in jobs})} comparison units.')
    print('Different from P0: fit/calibration split WITHIN the original labeled support. No new query tuning.')
    if a.action=='plan':
        print(f"Full plan: {root/'plan.json'}; non-frozen fits: {sum(j['arm']!='frozen' for j in all_jobs)}")
        return 0
    audits=audits_for(all_jobs,root)
    required={j['kind']:audits[j['kind']] for j in jobs}
    units={j['unit_id']:j for j in jobs}
    # The fixed audit can use another already declared support seed, never a query.
    for gate in required.values():units.setdefault(gate['unit_id'],gate)
    if a.action=='prepare':
        for j in units.values():prepare_job(j)
        print(f'Prepared {len(units)} fixed fit/calibration/query splits; original manifests untouched.')
        return 0
    if a.action=='summarize':
        print(export(all_jobs,root,a.allow_partial));return 0
    failures=[]
    for j in units.values():
        try:check_job(j,assets=True)
        except Exception as exc:failures.append({'unit':j['unit_id'],'error':str(exc)})
    atomic_json(root/'preflight.json',{'checked_units':len(units),'failures':failures})
    if failures:
        for f in failures:print('BLOCKED',f['unit'],f['error'])
        return 2
    if a.action=='check':
        print('Data/model paths, label budgets and query-group separation checked. GPU fit is not yet certified.')
        return 0
    gpus=[x.strip() for x in a.gpus.split(',') if x.strip()]
    if not gpus or len(gpus)!=len(set(gpus)):p.error('provide distinct nonempty GPU IDs')
    script=Path(__file__).resolve().parents[2]/'scripts/run_aperture_final.py'
    if 'audit' in a.stages:
        tasks=[{'job':j,'stage':'audit'} for j in required.values()]
        run_partitioned(tasks,gpus,script,root/'logs')
    if 'fit' in a.stages:
        tasks=[{'job':j,'stage':'fit','audit_folder':required[j['kind']]['output']} for j in jobs]
        run_partitioned(tasks,gpus,script,root/'logs')
    if 'evaluate' in a.stages:
        byunit={}
        for j in jobs:byunit.setdefault(j['unit_id'],[]).append(j)
        for unit,unit_jobs in byunit.items():seal_choices(unit_jobs,root/'selections'/(unit+'.json'))
        tasks=[{'job':j,'stage':'evaluate','selection_path':str(root/'selections'/(j['unit_id']+'.json'))} for j in jobs]
        run_partitioned(tasks,gpus,script,root/'logs')
    if len(jobs)==len(all_jobs) and 'evaluate' in a.stages:print(export(all_jobs,root))
    else:print('Selected stages finished. Full-paper export requires all planned states.')
    return 0
