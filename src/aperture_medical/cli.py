"""Medical run CLI. Every GPU process is isolated; no outcome-based expansion."""
from copy import deepcopy
from pathlib import Path
import argparse,json,os
from .protocol import make_jobs,prepare,check_job,write_locked,source_signature
from .worker import seal_choices,verified_record
from .numeric import replay_difference
from aperture_final.cli import run_partitioned
from vigor_handoff.protocol import atomic_json,read_jsonl,file_hash


def audit_jobs(jobs,root):
    result={}
    for j in jobs:
        if j['dataset'] in result:continue
        a=deepcopy(j);a.update(arm='audit',job_id='audit-'+j['dataset'],output=str(Path(root)/'audits'/j['dataset']))
        result[j['dataset']]=a
    return result


def internal(path):
    from .worker import execute_fit,execute_evaluate,_gpu_ready
    t=json.loads(Path(path).read_text());j=t['job'];stage=t['stage'];_gpu_ready()
    if stage=='audit':
        from visual_lens.audit import execute_audit
        return execute_audit(j)
    if stage=='fit':return execute_fit(j,t['audit_folder'])
    if stage=='evaluate':return execute_evaluate(j,t['selection_path'])
    raise ValueError('unknown internal action')


def fit_replay_gates(jobs,root,gates,gpus,script):
    """Repeat representative Aperture AND LoRA fits before the full matrix.

    Only support-calibration scores are compared. Failure does not choose the
    better fitting run; it stops the campaign and keeps both diagnostic records.
    """
    reps={}
    for j in jobs:
        if j['arm'] in ('aperture','lora4'):reps.setdefault((j['dataset'],j['arm']),j)
    tasks=[];pairs=[]
    for key,j in reps.items():
        clone=deepcopy(j);clone['job_id']+='--fit-replay';clone['output']=str(Path(root)/'fit_replays'/clone['job_id'])
        pairs.append((j,clone))
        for x in (j,clone):tasks.append(dict(job=x,stage='fit',audit_folder=gates[x['dataset']]['output']))
    run_partitioned(tasks,gpus,script,Path(root)/'logs')
    records=[]
    for j,clone in pairs:
        a=Path(j['output'])/'fit';b=Path(clone['output'])/'fit'
        verified_record(a);verified_record(b)
        ra=read_jsonl(a/'calibration_predictions.jsonl');rb=read_jsonl(b/'calibration_predictions.jsonl')
        if [r['sample_id'] for r in ra]!=[r['sample_id'] for r in rb]:raise ValueError('fit replay IDs differ')
        check=replay_difference([r['mean_log_scores'] for r in ra],[r['mean_log_scores'] for r in rb])
        record=dict(unit=j['unit_id'],arm=j['arm'],canonical_seal=file_hash(a/'DONE.json'),
                    replay_seal=file_hash(b/'DONE.json'),canonical_folder=str(a),replay_folder=str(b),
                    source_signature=source_signature(),**check)
        write_locked(Path(root)/'fit_replays'/(j['dataset']+'--'+j['arm']+'.json'),record)
        records.append(record)
    atomic_json(Path(root)/'fit_replay_checks.json',dict(checks=records,passed=all(r['passed'] for r in records),query_used=False))
    if not all(r['passed'] for r in records):raise RuntimeError('support-only repeat-fit gate failed; inspect numerical reproducibility, do not tune on query')


def validate_replay_gates(jobs,root):
    for dataset in {j['dataset'] for j in jobs}:
        for arm in ('aperture','lora4'):
            p=Path(root)/'fit_replays'/(dataset+'--'+arm+'.json')
            record=json.loads(p.read_text())
            if not record['passed'] or record['source_signature']!=source_signature():
                raise ValueError('repeat-fit check missing/stale/failed')
            for prefix in ('canonical','replay'):
                folder=Path(record[prefix+'_folder']);verified_record(folder)
                if record[prefix+'_seal']!=file_hash(folder/'DONE.json'):
                    raise ValueError('repeat-fit state has changed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',nargs='?',choices=['plan','prepare','check','run','summarize'])
    p.add_argument('--config',default='configs/aperture_medical_v1.json');p.add_argument('--internal-job')
    p.add_argument('--domains',nargs='+');p.add_argument('--budgets',nargs='+',type=int)
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--gpus',default='0')
    p.add_argument('--stages',nargs='+',choices=['audit','fit','evaluate'],default=['audit','fit','evaluate'])
    a=p.parse_args(argv)
    if a.internal_job:print(internal(a.internal_job));return 0
    if not a.action:p.error('action required')
    config=json.loads(Path(a.config).read_text());all_jobs=make_jobs(config)
    jobs=[j for j in all_jobs if (not a.domains or j['domain'] in a.domains) and
          (not a.budgets or j['total_label_budget'] in a.budgets) and
          (a.seeds is None or j['support_seed'] in a.seeds)]
    if not jobs:p.error('empty declared subset')
    if a.domains and set(a.domains)-{j['domain'] for j in all_jobs}:p.error('unknown domain')
    root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
    write_locked(root/'plan.json',dict(config=config,jobs=all_jobs,source_signature=source_signature()))
    print(f'{len(jobs)}/{len(all_jobs)} medical states; same query across budgets and seeds.',flush=True)
    if a.action=='plan':return 0
    if a.action=='prepare':prepare(config,a.domains);return 0
    if a.action=='summarize':
        from .reporting import export
        # Always require full protocol, independent of CLI execution filters.
        print(export(all_jobs,root));return 0
    gates=audit_jobs(all_jobs,root);required={j['dataset']:gates[j['dataset']] for j in jobs}
    units={j['unit_id']:j for j in jobs}
    for g in required.values():units.setdefault(g['unit_id'],g)
    failures=[]
    for j in units.values():
        try:check_job(j)
        except Exception as e:failures.append(dict(unit=j['unit_id'],error=str(e)))
    atomic_json(root/'preflight.json',dict(failures=failures,units=len(units)))
    if failures:
        for f in failures:print('BLOCKED',f)
        return 2
    if a.action=='check':print('Patient/content separation and label budgets checked. GPU gates still required.');return 0
    gpus=[g.strip() for g in a.gpus.split(',') if g.strip()]
    if not gpus or len(gpus)!=len(set(gpus)):p.error('GPU IDs must be distinct')
    script=Path(__file__).resolve().parents[2]/'scripts/run_aperture_medical.py'
    if 'audit' in a.stages:
        run_partitioned([dict(job=j,stage='audit') for j in required.values()],gpus,script,root/'logs')
    if 'fit' in a.stages:
        # Fixed representatives selected from the COMPLETE plan, not query scores.
        representatives=[j for j in all_jobs if j['dataset'] in required]
        fit_replay_gates(representatives,root,required,gpus,script)
        run_partitioned([dict(job=j,stage='fit',audit_folder=required[j['dataset']]['output']) for j in jobs],gpus,script,root/'logs')
    if 'evaluate' in a.stages:
        validate_replay_gates(jobs,root)
        byunit={}
        for j in jobs:byunit.setdefault(j['unit_id'],[]).append(j)
        for unit,jj in byunit.items():seal_choices(jj,root/'selections'/(unit+'.json'))
        run_partitioned([dict(job=j,stage='evaluate',selection_path=str(root/'selections'/(j['unit_id']+'.json'))) for j in jobs],gpus,script,root/'logs')
    if len(jobs)==len(all_jobs) and 'evaluate' in a.stages:
        from .reporting import export
        print(export(all_jobs,root))
    else:print('Selected execution completed; full exports require every prespecified state.')
    return 0
