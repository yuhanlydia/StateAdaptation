"""Queue targeted debugging behind the existing experiment without disrupting it."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from aperture_table.protocol import write_locked, read_roles
from aperture_table.worker import check_assets, verified_record, compare_repeat
from aperture_tuning.cli import stage
from vigor_handoff.protocol import atomic_json
from .protocol import build, read_plan, repeat_job
from .worker import select, seal_global, evaluate, summarize


def child(root,action,job,gpu):
    root=Path(root);path=root/'queue'/(job['job_id']+'.json');write_locked(path,job)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    with (logs/(job['job_id']+'--'+action+'.log')).open('a') as out:
        result=subprocess.run([sys.executable,'-m','aperture_debug.cli','--root',str(root),'worker','--action',action,'--job',str(path)],env=env,stdout=out,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{action} failed for {job["job_id"]}; inspect log before recovery')


def run_unit(root,plan,unit,gpu):
    root=Path(root);write_locked(root/'devices'/f'{unit}.json',{'gpu':gpu})
    for job in [j for j in plan['jobs'] if j['unit_id']==unit]:child(root,'fit',job,gpu)
    for job in select(root,unit):
        child(root,'fit',job,gpu);replica=repeat_job(job);child(root,'fit',replica,gpu)
        compare_repeat(job,replica['output'],root/'repeat_gates'/f"{job['job_id']}.json")


def complete_unit(root,unit):
    try:
        winners=select(root,unit)
        for j in winners:
            p=Path(root)/'repeat_gates'/f"{j['job_id']}.json"
            if not p.exists():return False
            gate=json.loads(p.read_text())
            if not gate.get('passed'):raise ValueError('failed existing repeat')
            for output,key in ((j['output'],'first_seal'),(repeat_job(j)['output'],'repeat_seal')):
                from vigor_handoff.protocol import file_hash
                fit=Path(output)/'fit';verified_record(fit)
                if file_hash(fit/'DONE.json')!=gate[key]:raise ValueError('repeat artifacts changed')
        return True
    except FileNotFoundError:return False


def run(root,previous,gpus):
    root=Path(root);plan=read_plan(root)
    if json.loads((root/'CHECKED.json').read_text())['debug_source']!=plan['debug_source']:raise ValueError('preflight required')
    with (root/'DISPATCH.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.time()+24*3600
        while previous:
            p=Path(previous)/'pipeline_status.json'
            value=json.loads(p.read_text()) if p.exists() else {}
            if value.get('status')=='uploaded':break
            if value.get('status')=='failed':raise RuntimeError('previous campaign failed; diagnose before starting new stage')
            if time.time()>deadline:raise TimeoutError('previous campaign wait exceeded 24h')
            atomic_json(root/'status.json',{'phase':'waiting_previous_campaign','status':'waiting','utc':time.time()})
            print('Waiting for previous campaign to finish and upload',flush=True);time.sleep(30)
        audits=[]
        for j in plan['audits']:
            folder=Path(j['context'])/'audits'/j['model_key']/j['domain_key']
            if (folder/'DONE.json').exists():verified_record(folder)
            else:audits.append({'id':j['job_id'],'job':j})
        stage(root,'audits',audits,gpus,lambda t,g:child(root,'audit',t['job'],g),24)
        tasks=[]
        for unit in sorted(plan['units']):
            if complete_unit(root,unit):continue
            p=root/'devices'/f'{unit}.json'
            tasks.append({'id':unit,'required_gpu':json.loads(p.read_text())['gpu'] if p.exists() else None})
        stage(root,'support_debug_search',tasks,gpus,lambda t,g:run_unit(root,plan,t['id'],g),24)
        selected=seal_global(root);tasks=[]
        for j in plan['finals']:
            if j['job_id'] not in selected:continue
            p=Path(j['output'])/'evaluation'
            if (p/'DONE.json').exists():verified_record(p)
            else:tasks.append({'id':j['job_id'],'job':j})
        stage(root,'selected_query',tasks,gpus,lambda t,g:child(root,'evaluate',t['job'],g),24)
        summarize(root);atomic_json(root/'status.json',{'status':'complete','utc':time.time(),'search_fits':len(plan['jobs']),'selected_evaluations':len(selected)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--base-config',required=True);a.add_argument('--design',required=True);a.add_argument('--asset-manifests',required=True)
    s.add_parser('check');s.add_parser('summarize')
    a=s.add_parser('run');a.add_argument('--previous');a.add_argument('--gpus',default='0,1,2,3')
    a=s.add_parser('worker');a.add_argument('--action',choices=['audit','fit','evaluate'],required=True);a.add_argument('--job',required=True)
    a=p.parse_args();root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare':
        plan=build(json.loads(Path(a.base_config).read_text()),json.loads(Path(a.design).read_text()),root);write_locked(root/'plan.json',plan)
        (root/'assets').mkdir(exist_ok=True)
        for source in Path(a.asset_manifests).glob('*.json'):
            target=root/'assets'/source.name
            if not target.exists():shutil.copyfile(source,target)
        for j in plan['audits']:
            context=Path(j['context']);context.mkdir(parents=True,exist_ok=True)
            if not (context/'assets').exists():(context/'assets').symlink_to(root/'assets',target_is_directory=True)
        print(json.dumps({'search_fits':len(plan['jobs']),'selected_final_fits':2*len(plan['units']),'repeats':2*len(plan['units']),'audits':len(plan['audits'])}))
    elif a.command=='check':
        plan=read_plan(root);seen=set()
        for j in plan['jobs']+plan['finals']:
            key=(j['support'],j['calibration'],j['query'])
            if key not in seen:read_roles(j,images=True);seen.add(key)
        for key in {j['model_key'] for j in plan['jobs']}:check_assets(next(j for j in plan['jobs'] if j['model_key']==key),root/'assets',refresh=True)
        atomic_json(root/'CHECKED.json',{'debug_source':plan['debug_source'],'partitions':len(seen)});print('Model bytes and all fold manifests verified.')
    elif a.command=='run':
        gpus=a.gpus.split(',')
        if len(set(gpus))!=len(gpus) or not all(g.isdigit() for g in gpus):p.error('invalid GPU list')
        try:run(root,a.previous,gpus)
        except BaseException as e:atomic_json(root/'failure.json',{'error':repr(e),'utc':time.time()});raise
    elif a.command=='summarize':summarize(root)
    else:
        plan=read_plan(root);job=json.loads(Path(a.job).read_text())
        if job not in plan['jobs']+plan['finals']+[repeat_job(j) for j in plan['finals']]:raise ValueError('job outside locked registry')
        if a.action=='audit':
            if job not in plan['audits']:raise ValueError('unregistered audit')
            from aperture_table.audit import execute_audit
            execute_audit(job,job['context'])
        elif a.action=='fit':
            if job['role']!='search':
                selected=select(root,job['unit_id'])
                if job not in selected+[repeat_job(j) for j in selected]:raise ValueError('unselected final fit')
            from aperture_table.worker import execute_fit
            execute_fit(job,job['context'])
        else:evaluate(root,job)

if __name__=='__main__':main()
