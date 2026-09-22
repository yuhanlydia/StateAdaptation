"""Single-process-per-GPU scheduling for the fixed manuscript table."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from .protocol import make_jobs,MODELS,DOMAINS,write_locked,source_signature,read_roles


def parser():
    p=argparse.ArgumentParser(description='Fixed 64-state Aperture table; no automatic benchmark expansion.')
    p.add_argument('--config',default='configs/aperture_table_v1.json')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('plan');q=sub.add_parser('prepare');q.add_argument('--reuse-source-only',action='store_true')
    sub.add_parser('check')
    for name in ('audit','run'):
        q=sub.add_parser(name);q.add_argument('--gpus',default='0')
        q.add_argument('--models',nargs='+',choices=list(MODELS))
        q.add_argument('--datasets',nargs='+',choices=list(DOMAINS))
        q.add_argument('--seeds',nargs='+',type=int,choices=[0,1])
    sub.add_parser('summarize')
    q=sub.add_parser('fill-main');q.add_argument('source');q.add_argument('destination')
    q=sub.add_parser('_worker');q.add_argument('--action',required=True,choices=['audit','fit','evaluate'])
    q.add_argument('--job',required=True);q.add_argument('--selection')
    return p


def lock_plan(config):
    jobs=make_jobs(config)
    value={'config':config,'source':source_signature(),'jobs':jobs,'states':64,'trainable_fits':48,'query_labels_used_for_selection':False}
    write_locked(Path(config['run_root'])/'plan.json',value)
    return value


def select_jobs(jobs,models=None,datasets=None,seeds=None):
    if models and not set(models)<=set(MODELS):raise ValueError('unknown model filter')
    if datasets and not set(datasets)<=set(DOMAINS):raise ValueError('unknown dataset filter')
    if seeds and not set(seeds)<={0,1}:raise ValueError('unknown support seed')
    out=[j for j in jobs if (not models or j['model_key'] in models) and (not datasets or j['domain_key'] in datasets) and (not seeds or j['support_seed'] in seeds)]
    if not out:raise ValueError('empty execution selection')
    return out


def child(config_path,root,job,action,gpu,selection=None):
    name=job['job_id']+'--'+action
    filename=Path(root)/'queue'/(name+'.json');write_locked(filename,job)
    logs=Path(root)/'logs';logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    cmd=[sys.executable,'-m','aperture_table.cli','--config',str(Path(config_path).resolve()),'_worker','--action',action,'--job',str(filename.resolve())]
    if selection:cmd+=['--selection',str(Path(selection).resolve())]
    with (logs/(name+'.log')).open('a') as stream:
        proc=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'{name} failed; inspect {logs/(name+".log")}; no tolerance/model fallback')


def gpu_batches(tasks,gpus,fn):
    if not gpus or len(set(gpus))!=len(gpus):raise ValueError('unique GPU identifiers required')
    # Buckets are serial on each assigned GPU. A stage barrier precedes query evaluation.
    buckets=[tasks[i::len(gpus)] for i in range(len(gpus))]
    def serial(gpu,bucket):
        import fcntl
        lock=Path('/tmp')/f'aperture-table-gpu-{os.getuid()}-{gpu}.lock'
        with lock.open('w') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError(f'GPU {gpu} already assigned to another table process')
            for task in bucket:fn(task,gpu)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(serial,g,b) for g,b in zip(gpus,buckets)]
        for f in futures:f.result()


def run(config,args):
    from .worker import seal_unit,compare_repeat
    all_jobs=lock_plan(config)['jobs'];jobs=select_jobs(all_jobs,args.models,args.datasets,args.seeds)
    root=Path(config['run_root']);gpus=args.gpus.split(',')
    if any(not x.isdigit() for x in gpus):raise ValueError('use comma-separated numeric local GPU IDs')
    models=list(dict.fromkeys(j['model_key'] for j in jobs))
    # H2 audit is always needed for the fixed reference repeat, even in a Path-only run.
    audit_jobs=[next(j for j in all_jobs if j['model_key']==m and j['domain_key']==d and j['support_seed']==0 and j['arm']=='aperture')
                for m in models for d in dict.fromkeys(['h2']+[j['domain_key'] for j in jobs if j['model_key']==m])]
    gpu_batches(audit_jobs,gpus,lambda j,g:child(args.config,root,j,'audit',g))
    # Actual support-only latency observations, before any new query scoring.
    forecast=[]
    for j in audit_jobs:
        r=json.loads((root/'audits'/j['model_key']/j['domain_key']/'audit.json').read_text())
        n=sum(x['query_count'] for x in jobs if x['model_key']==j['model_key'] and x['domain_key']==j['domain_key'])
        forecast.append({'model':j['model_key'],'dataset':j['domain_key'],'query_count':n,
                         'support_seconds_per_query':r['seconds_per_query_support_estimate'],
                         'estimated_query_gpu_hours':n*r['seconds_per_query_support_estimate']/3600})
    from vigor_handoff.protocol import atomic_json
    atomic_json(root/'timing_forecast.json',{'estimates':forecast,'source':'support-only timing; excludes fitting/loading/preparation; not a runtime guarantee'})
    print(json.dumps({'estimated_query_gpu_hours':sum(r['estimated_query_gpu_hours'] for r in forecast),'note':'fit/loading/audits additional'},indent=2),flush=True)
    if args.command=='audit':return
    def repeat_model(m,g):
        reference=next(j for j in all_jobs if j['model_key']==m and j['domain_key']=='h2' and j['support_seed']==0 and j['arm']=='aperture')
        replica=deepcopy(reference);replica['job_id']+='--repeat';replica['output']=str(root/'repeat'/m/'state')
        child(args.config,root,reference,'fit',g);child(args.config,root,replica,'fit',g)
        compare_repeat(reference,replica['output'],root/'repeat'/m/'gate.json')
    gpu_batches(models,gpus,repeat_model)
    gpu_batches(jobs,gpus,lambda j,g:child(args.config,root,j,'fit',g))
    units={}
    for j in jobs:units.setdefault(j['unit_id'],[]).append(j)
    for unit,group in units.items():seal_unit(group,root/'selections'/(unit+'.json'))
    gpu_batches(jobs,gpus,lambda j,g:child(args.config,root,j,'evaluate',g,root/'selections'/(j['unit_id']+'.json')))
    print('Selected units finished. Full paper export still requires all 64 registered states.')


def main(argv=None):
    args=parser().parse_args(argv)
    config=json.loads(Path(args.config).read_text());root=Path(config['run_root'])
    if args.command=='plan':
        p=lock_plan(config);print(json.dumps({k:p[k] for k in ('states','trainable_fits','source')},indent=2))
    elif args.command=='prepare':
        from .protocol import prepare
        lock_plan(config);prepare(config,prepare_source=not args.reuse_source_only)
        print('Shared source-derived fit/calibration/query partitions locked.')
    elif args.command=='check':
        from .worker import check_assets
        jobs=lock_plan(config)['jobs'];seen=set()
        for j in jobs:
            key=(j['domain_key'],j['support_seed'])
            if key not in seen:read_roles(j,images=True);seen.add(key)
        for m in MODELS:check_assets(next(j for j in jobs if j['model_key']==m),root/'assets',refresh=True)
        print('All four local models and the four dataset/seed partitions checked; real GPU audits still required.')
    elif args.command in ('run','audit'):run(config,args)
    elif args.command=='summarize':
        from .reporting import export
        result=export(config);print(f"Exported {len(result['cells'])} measured table cells.")
    elif args.command=='fill-main':
        from .reporting import fill_main
        print('Filled cells:',fill_main(args.source,args.destination,root/'paper_exports'))
    elif args.command=='_worker':
        job=json.loads(Path(args.job).read_text())
        if args.action=='audit':
            from .audit import execute_audit
            print(execute_audit(job,root))
        elif args.action=='fit':
            from .worker import execute_fit
            print(execute_fit(job,root))
        else:
            from .worker import execute_evaluate
            print(execute_evaluate(job,root,args.selection))


if __name__=='__main__':main()
