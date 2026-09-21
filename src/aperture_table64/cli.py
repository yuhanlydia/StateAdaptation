"""One process per GPU; technical gates precede fit/selection/query/export."""
from __future__ import annotations
import argparse
from copy import deepcopy
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from vigor_handoff.protocol import atomic_json, digest, file_hash, read_jsonl, JobStore
from aperture_final.protocol import write_locked
from .protocol import make_jobs, prepare, check_job, source_signature, MODEL_SPECS, DOMAINS


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/aperture_table64.json')
    subs=p.add_subparsers(dest='command',required=True)
    for cmd in ('plan','prepare','check'):subs.add_parser(cmd)
    run=subs.add_parser('run');run.add_argument('--gpus',default='0')
    run.add_argument('--models',nargs='+',choices=list(MODEL_SPECS));run.add_argument('--domains',nargs='+',choices=list(DOMAINS))
    run.add_argument('--seeds',nargs='+',type=int,choices=[0,1])
    exp=subs.add_parser('summarize');exp.add_argument('--manuscript')
    work=subs.add_parser('_worker');work.add_argument('stage',choices=['audit','fit','evaluate']);work.add_argument('task')
    return p


def plan(config):
    jobs=make_jobs(config)
    value={'protocol':config['protocol'],'config':config,'source_signature':source_signature(),
           'states':64,'trainable_fits':48,'jobs':jobs}
    write_locked(Path(config['run_root'])/'plan.json',value)
    return value


def select_jobs(jobs,args):
    return [j for j in jobs if (not args.models or j['model_key'] in args.models)
            and (not args.domains or j['domain'] in args.domains)
            and (not args.seeds or j['support_seed'] in args.seeds)]


def audit_path(root,job):return Path(root)/'audits'/f'{job["model_key"]}--{job["domain"]}'


def _task(root,stage,job,**kw):
    task={'stage':stage,'job':job,**{k:str(v) for k,v in kw.items()}}
    path=Path(root)/'tasks'/f'{stage}-{job["job_id"]}.json';write_locked(path,task)
    return path


def _call(config,stage,task,gpu):
    root=Path(config['run_root']);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONHASHSEED='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    src=Path(__file__).resolve().parents[1];env['PYTHONPATH']=str(src)+os.pathsep+env.get('PYTHONPATH','')
    with (logs/(task.stem+'.log')).open('w') as out:
        r=subprocess.run([sys.executable,'-m','aperture_table64.cli','_worker',stage,str(task)],
                         env=env,stdout=out,stderr=subprocess.STDOUT)
    if r.returncode:
        atomic_json(root/'LAST_FAILURE.json',{'stage':stage,'task':str(task),'gpu':str(gpu),'exit_code':r.returncode})
        raise RuntimeError(f'{stage} failed; inspect {logs/(task.stem+".log")}. No protocol fallback or later stage was launched.')


def _parallel(config,gpus,stage,tasks):
    q=queue.Queue();[q.put(t) for t in tasks];stop=threading.Event()
    def consume(gpu):
        while not stop.is_set():
            try:t=q.get_nowait()
            except queue.Empty:return
            try:_call(config,stage,t,gpu)
            except BaseException:stop.set();raise
            finally:q.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(consume,g) for g in gpus]
        for f in futures:f.result()


def run(config,args):
    from .worker import verified_record,compare_repeats,seal_choices
    alljobs=plan(config)['jobs'];selected=select_jobs(alljobs,args)
    root=Path(config['run_root']);gpus=args.gpus.split(',')
    if not gpus or any(not x.isdigit() for x in gpus) or len(set(gpus))!=len(gpus):raise ValueError('GPU IDs must be unique comma-separated integers')
    # All referenced assets, including support-only repeat units, are checked first.
    modelkeys=[m for m in MODEL_SPECS if any(j['model_key']==m for j in selected)]
    reference=[j for j in alljobs if j['model_key'] in modelkeys and j['domain']=='hospital_2' and j['support_seed']==0 and j['arm'] in ('lora1','aperture')]
    reps={}
    for j in selected+reference:reps.setdefault((j['model_key'],j['domain']),next(x for x in alljobs if x['model_key']==j['model_key'] and x['domain']==j['domain'] and x['support_seed']==0 and x['arm']=='frozen'))
    for j in reps.values():check_job(j)
    audits=[_task(root,'audit',j,audit_path=audit_path(root,j)) for j in reps.values()]
    _parallel(config,gpus,'audit',audits)
    # Reference and independent repeats are sequential fresh processes on SAME GPU.
    for mi,m in enumerate(modelkeys):
        records=[]
        for j in [x for x in reference if x['model_key']==m]:
            ap=audit_path(root,j);_call(config,'fit',_task(root,'fit',j,audit_path=ap),gpus[mi%len(gpus)])
            repeated=deepcopy(j);repeated['job_id']+='--repeat';repeated['output']=str(root/'repeat_fits'/repeated['job_id'])
            _call(config,'fit',_task(root,'fit',repeated,audit_path=ap),gpus[mi%len(gpus)])
            a=Path(j['output'])/'fit';b=Path(repeated['output'])/'fit'
            da,db=verified_record(a),verified_record(b)
            cmp=compare_repeats(read_jsonl(a/'calibration_scores.jsonl'),read_jsonl(b/'calibration_scores.jsonl'),.005)
            records.append({'arm':j['arm'],'reference':str(a),'repeat':str(b),
                            'reference_seal':digest(da),'repeat_seal':digest(db),'comparison':cmp})
        s=JobStore(root/'repeat_checks'/m,{'source':source_signature(),'model_key':m,'records':records})
        with s.lock():
            atomic_json(s.path/'repeat.json',{'passed':True,'gpu':gpus[mi%len(gpus)],'records':records});s.finish({'passed':True},['repeat.json'])
    tasks=[_task(root,'fit',j,audit_path=audit_path(root,j)) for j in selected]
    _parallel(config,gpus,'fit',tasks)
    selection=root/'selections'/('units-'+digest(sorted(j['unit_id'] for j in selected))[:20]+'.json')
    seal_choices(selected,selection)
    _parallel(config,gpus,'evaluate',[_task(root,'evaluate',j,selection=selection) for j in selected])
    print(f'Completed selected {len(selected)} states. summarize still requires all 64 declared states.')


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.command=='_worker':
        from .worker import execute_audit,execute_fit,execute_evaluate
        t=json.loads(Path(args.task).read_text())
        if t['stage']!=args.stage:raise ValueError('worker task/stage mismatch')
        if args.stage=='audit':execute_audit(t['job'],t['audit_path'])
        elif args.stage=='fit':execute_fit(t['job'],t['audit_path'])
        else:execute_evaluate(t['job'],t['selection'])
        return
    c=json.loads(Path(args.config).read_text());p=plan(c)
    if args.command=='plan':print(json.dumps({'states':p['states'],'trainable_fits':p['trainable_fits'],'run_root':c['run_root'],'status':'planned; not GPU results'},indent=2))
    elif args.command=='prepare':print(f'Prepared {prepare(c)} planned states; model/data checks remain separate.')
    elif args.command=='check':
        for j in p['jobs']:check_job(j)
        print('All manifests, image hashes, and local snapshot files passed preflight. GPU audits are still required.')
    elif args.command=='run':run(c,args)
    else:
        from .export import export_results
        print(export_results(c,args.manuscript))

if __name__=='__main__':main()
