"""Prepare, validate and queue a bounded, support-only hyperparameter search."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from aperture_table.protocol import read_roles, write_locked
from aperture_table.worker import check_assets, verified_record
from vigor_handoff.protocol import atomic_json
from .protocol import build, read_plan


def child(root, action, job, gpu):
    root=Path(root);queue=root/'queue';queue.mkdir(exist_ok=True)
    path=queue/(job['job_id']+'.json');write_locked(path,job)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    with (logs/(job['job_id']+'--'+action+'.log')).open('a') as out:
        result=subprocess.run([sys.executable,'-m','aperture_tuning.cli','--root',str(root),
                               '_worker','--action',action,'--job',str(path)],env=env,stdout=out,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"{action} failed: {job['job_id']}; see logs; no automatic parameter fallback")


def free_gpus():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    return [index.strip() for line in rows.splitlines() for index,uuid,free in [line.split(',')]
            if float(free.strip())>=23000 and uuid.strip() not in busy]


def stage(root, name, tasks, gpus, callback, wait_hours):
    root=Path(root);pending=list(tasks);mutex=threading.Lock();stop=threading.Event();errors=[]
    started=time.time();last_notice=[0.]
    def worker(gpu):
        while not stop.is_set():
            with mutex:
                if not pending:return
                eligible=any(t.get('required_gpu') in (None,gpu) for t in pending)
            if not eligible:
                if time.time()-started>wait_hours*3600:
                    with mutex:errors.append('required GPU unavailable before deadline');stop.set()
                    return
                stop.wait(5);continue
            if time.time()-started>wait_hours*3600:
                with mutex:errors.append(f'{name}: GPU wait deadline exceeded');stop.set()
                return
            if gpu not in free_gpus():
                with mutex:
                    if time.time()-last_notice[0]>60:
                        atomic_json(root/'status.json',{'phase':name,'status':'waiting_for_available_gpu',
                            'remaining_tasks':len(pending),'utc':time.time()})
                        print(f'{name}: waiting for free GPU, {len(pending)} tasks queued',flush=True)
                        last_notice[0]=time.time()
                stop.wait(10);continue
            lock=Path('/tmp')/f'aperture-table-gpu-{os.getuid()}-{gpu}.lock'
            with lock.open('w') as handle:
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    stop.wait(5);continue
                if gpu not in free_gpus():continue
                with mutex:
                    # A resumed unit must keep its original GPU for the same-device repeat gate.
                    available=[i for i,t in enumerate(pending) if t.get('required_gpu') in (None,gpu)]
                    if not available:
                        continue
                    task=pending.pop(available[0])
                    with (root/'assignments.jsonl').open('a') as f:
                        f.write(json.dumps({'phase':name,'task':task['id'],'gpu':gpu,'event':'start','utc':time.time()})+'\n')
                    atomic_json(root/'status.json',{'phase':name,'status':'running','task':task['id'],'gpu':gpu,'utc':time.time()})
                try:callback(task,gpu)
                except BaseException as error:
                    with mutex:
                        errors.append(repr(error));stop.set()
                        atomic_json(root/'failure.json',{'phase':name,'task':task['id'],'gpu':gpu,'error':repr(error),'utc':time.time()})
                    return
                with mutex:
                    with (root/'assignments.jsonl').open('a') as f:
                        f.write(json.dumps({'phase':name,'task':task['id'],'gpu':gpu,'event':'done','utc':time.time()})+'\n')
    def guarded(gpu):
        try:worker(gpu)
        except BaseException as error:
            with mutex:errors.append(repr(error));stop.set()
    threads=[threading.Thread(target=guarded,args=(g,)) for g in gpus]
    for t in threads:t.start()
    for t in threads:t.join()
    if errors:raise RuntimeError(errors)


def run_unit(root, plan, unit, gpu):
    from .worker import seal_selection, repeat_job
    from aperture_table.worker import compare_repeat
    root=Path(root)
    write_locked(root/'devices'/f'{unit}.json',{'gpu':gpu})
    for job in [j for j in plan['jobs'] if j['unit_id']==unit]:
        child(root,'fit',job,gpu)
    for job in seal_selection(root,unit):
        replica=repeat_job(job);child(root,'fit',replica,gpu)
        compare_repeat(job,replica['output'],root/'repeat_gates'/f"{job['job_id']}.json")


def check(root):
    root=Path(root);plan=read_plan(root);seen=set()
    for job in plan['jobs']:
        unit=job['unit_id']
        if unit not in seen:read_roles(job,images=True);seen.add(unit)
    for key in ('q25','q34','q38','g34'):
        check_assets(next(j for j in plan['jobs'] if j['model_key']==key),root/'assets',refresh=True)
    atomic_json(root/'CHECKED.json',{'tuning_source':plan['tuning_source'],'units':16,'models':4})


def run(root,gpus,wait_hours):
    from .worker import seal_global, summarize, unit_complete
    root=Path(root);plan=read_plan(root)
    checked=json.loads((root/'CHECKED.json').read_text())
    if checked['tuning_source']!=plan['tuning_source']:raise ValueError('check must pass before GPU execution')
    # A process-level lock prevents two dispatchers from claiming the same queued states.
    with (root/'DISPATCH.lock').open('w') as dispatch:
        fcntl.flock(dispatch,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audits=[]
        for j in plan['audits']:
            folder=Path(j['context'])/'audits'/j['model_key']/j['domain_key']
            if (folder/'DONE.json').exists():verified_record(folder)
            else:audits.append({'id':j['job_id'],'job':j})
        stage(root,'audits',audits,gpus,lambda t,g:child(root,'audit',t['job'],g),wait_hours)
        tasks=[]
        for unit in sorted({j['unit_id'] for j in plan['jobs']}):
            if unit_complete(root,unit):continue
            device=root/'devices'/f'{unit}.json'
            tasks.append({'id':unit,'required_gpu':json.loads(device.read_text())['gpu'] if device.exists() else None})
        stage(root,'support_search',tasks,gpus,lambda t,g:run_unit(root,plan,t['id'],g),wait_hours)
        chosen=seal_global(root)
        tasks=[]
        for j in plan['jobs']:
            if j['job_id'] not in chosen:continue
            folder=Path(j['output'])/'evaluation'
            if (folder/'DONE.json').exists():verified_record(folder)
            else:tasks.append({'id':j['job_id'],'job':j})
        stage(root,'selected_query',tasks,gpus,lambda t,g:child(root,'evaluate',t['job'],g),wait_hours)
        summarize(root)
        atomic_json(root/'status.json',{'status':'complete','candidate_fits':192,'selected_evaluations':32,'utc':time.time()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--base-config',required=True);p.add_argument('--asset-manifests')
    sub.add_parser('check');sub.add_parser('summarize')
    p=sub.add_parser('run');p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--wait-hours',type=float,default=24.)
    p=sub.add_parser('_worker');p.add_argument('--job',required=True);p.add_argument('--action',choices=['audit','fit','evaluate'],required=True)
    a=parser.parse_args();root=Path(a.root).resolve()
    if a.command=='prepare':
        import shutil
        base=json.loads(Path(a.base_config).read_text());plan=build(base,root)
        root.mkdir(parents=True,exist_ok=True)
        for directory in ('assets','selections','repeat_gates','devices'): (root/directory).mkdir(exist_ok=True)
        write_locked(root/'plan.json',plan)
        if a.asset_manifests:
            for source in Path(a.asset_manifests).glob('*.json'):
                dest=root/'assets'/source.name
                if not dest.exists():shutil.copyfile(source,dest)
        for job in plan['audits']:
            context=Path(job['context']);context.mkdir(parents=True,exist_ok=True)
            if not (context/'assets').exists():(context/'assets').symlink_to(root/'assets',target_is_directory=True)
        print(json.dumps({'fits':192,'selected_query_states':32,'audits':len(plan['audits']),'root':str(root)}))
    elif a.command=='check':check(root);print('All model bytes and support/query partitions verified.')
    elif a.command=='run':
        gpus=a.gpus.split(',')
        if not gpus or len(set(gpus))!=len(gpus) or any(not g.isdigit() for g in gpus) or a.wait_hours<=0:parser.error('invalid GPU list/wait budget')
        run(root,gpus,a.wait_hours)
    elif a.command=='summarize':
        from .worker import summarize
        summarize(root)
    else:
        plan=read_plan(root);job=json.loads(Path(a.job).read_text())
        from .worker import repeat_job
        registry=plan['jobs']+[repeat_job(j) for j in plan['jobs']]
        if job not in registry:raise ValueError('worker job is outside the locked registry')
        if a.action=='audit':
            from aperture_table.audit import execute_audit
            execute_audit(job,job['context'])
        elif a.action=='fit':
            from aperture_table.worker import execute_fit
            execute_fit(job,job['context'])
        else:
            from .worker import evaluate
            evaluate(root,job)


if __name__=='__main__':main()
