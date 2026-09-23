"""Select only out-of-fold support predictions, seal repeats, then evaluate."""
import json
from pathlib import Path
import numpy as np
from aperture_final.numeric import metrics
from aperture_table.protocol import read_roles, write_locked
from aperture_table.worker import verified_record, validate_rows, _scores, identity, deterministic, score_all
from vigor_handoff.protocol import read_jsonl, file_hash, atomic_json, JobStore
from .protocol import read_plan, repeat_job


def choose(records):
    if not records or len({r['candidate'] for r in records})!=len(records):raise ValueError('missing/duplicate candidates')
    if any(r.get('split')!='support_validation' or not np.isfinite([r['f1'],r['nll']]).all() or not 0<=r['f1']<=1 or r['nll']<0 for r in records):raise ValueError('invalid support selection input')
    return min(records,key=lambda r:(-r['f1'],r['nll'],r['candidate']))['candidate']


def select(root,unit):
    root=Path(root);plan=read_plan(root); records=[];seals={};expected_validation_ids=None
    for method in ('aperture','lora'):
        candidates=sorted({j['candidate'] for j in plan['jobs'] if j['unit_id']==unit and j['method']==method})
        for candidate in candidates:
            jobs=sorted([j for j in plan['jobs'] if (j['unit_id'],j['method'],j['candidate'])==(unit,method,candidate)],key=lambda j:j['fold'])
            if [j['fold'] for j in jobs]!=list(range(plan['units'][unit]['fold_count'])):raise ValueError('missing folds')
            pooled=[];fold_metrics=[]
            for job in jobs:
                folder=Path(job['output'])/'fit';verified_record(folder)
                if json.loads((folder/'identity.json').read_text())['identity']['job']!=job:raise ValueError('wrong fit identity')
                rows=read_jsonl(folder/'calibration_predictions.jsonl');validate_rows(rows,read_roles(job)['calibration'])
                pooled.extend(rows);fold_metrics.append(metrics(*_scores(rows)))
                seals[job['job_id']]=file_hash(folder/'DONE.json')
            if len({r['sample_id'] for r in pooled})!=len(pooled):raise ValueError('overlapping validation folds')
            validation_ids={r['sample_id'] for r in pooled}
            if expected_validation_ids is None:expected_validation_ids=validation_ids
            elif validation_ids!=expected_validation_ids:raise ValueError('candidate validation populations differ')
            result=metrics(*_scores(pooled))
            records.append({'candidate':candidate,'method':method,'split':'support_validation','f1':result['macro_f1'],'nll':result['nll'],'pooled_metrics':result,'fold_metrics':fold_metrics})
    winners={m:choose([r for r in records if r['method']==m]) for m in ('aperture','lora')}
    value={'unit':unit,'debug_source':plan['debug_source'],'winners':winners,'records':records,'fit_seals':seals,'query_used':False}
    write_locked(root/'selections'/f'{unit}.json',value)
    selected=[j for j in plan['finals'] if j['unit_id']==unit and j['candidate']==winners[j['method']]]
    if len(selected)!=2 or {j['method'] for j in selected}!={'aperture','lora'}:raise ValueError('missing/duplicate final winner')
    return selected


def seal_global(root):
    root=Path(root);plan=read_plan(root);files={};chosen=[]
    for unit in sorted(plan['units']):
        winners=select(root,unit)
        path=root/'selections'/f'{unit}.json';files[str(path.relative_to(root))]=file_hash(path)
        for job in winners:
            gatepath=root/'repeat_gates'/f"{job['job_id']}.json"
            gate=json.loads(gatepath.read_text())
            if not gate.get('passed'):raise ValueError('repeat gate failed')
            for output,key in ((job['output'],'first_seal'),(repeat_job(job)['output'],'repeat_seal')):
                fit=Path(output)/'fit';verified_record(fit)
                if file_hash(fit/'DONE.json')!=gate[key]:raise ValueError('repeat changed')
            files[str(gatepath.relative_to(root))]=file_hash(gatepath);chosen.append(job['job_id'])
    write_locked(root/'SELECTION_COMPLETE.json',{'debug_source':plan['debug_source'],'files':files,'selected':chosen,'query_used':False})
    return chosen


def evaluate(root,job):
    from aperture_table.backend import loaded
    root=Path(root);plan=read_plan(root)
    barrier=json.loads((root/'SELECTION_COMPLETE.json').read_text())
    if barrier['debug_source']!=plan['debug_source'] or job['job_id'] not in barrier['selected']:raise ValueError('unselected query access')
    for path,sha in barrier['files'].items():
        if file_hash(root/path)!=sha:raise ValueError('selection barrier changed')
    # Revalidate all candidate and winner seals before each query access.
    seal_global(root)
    fit=Path(job['output'])/'fit';verified_record(fit)
    now=identity(job,root/'assets');saved=json.loads((fit/'identity.json').read_text())['identity']
    if any(saved.get(k)!=v for k,v in now.items()):raise ValueError('fit identity changed')
    roles=read_roles(job,images=True);folder=Path(job['output'])/'evaluation'
    store=JobStore(folder,{'identity':now,'fit_seal':file_hash(fit/'DONE.json'),'barrier':file_hash(root/'SELECTION_COMPLETE.json')})
    if store.complete():return
    deterministic(job['optimization_seed'])
    with store.lock(),loaded(job,fit) as (_,backend,controller):
        cal=read_jsonl(fit/'calibration_predictions.jsonl');replay=score_all(backend,roles['calibration'],controller)
        if [r['prompt_contract_hash'] for r in cal]!=[r['prompt_contract_hash'] for r in replay]:raise ValueError('prompt changed')
        delta=float(np.max(np.abs(_scores(cal)[0]-_scores(replay)[0])))
        if delta>.005:raise ValueError('state replay failed')
        # Frozen-state train classification closes the old pre-update/online-loss instrumentation gap.
        support=score_all(backend,roles['support'],controller,folder/'support_predictions.jsonl')
        support_report=metrics(*_scores(support))
        atomic_json(folder/'support_diagnostics.json',{'metrics':support_report,'predicted_class_count':len({r['prediction'] for r in support}),'scope':'final saved state, original fit support only'})
        rows=score_all(backend,roles['query'],controller,folder/'predictions.jsonl')
        temperature=json.loads((fit/'calibration.json').read_text())['temperature']['temperature']
        value={'job_id':job['job_id'],'model_key':job['model_key'],'domain_key':job['domain_key'],'support_seed':job['support_seed'],'method':job['method'],'candidate':job['candidate'],'raw':metrics(*_scores(rows)),'temperature':metrics(*_scores(rows),temperature=temperature),'saved_state_replay_max_abs':delta,'query_status':plan['query_status'],'query_used_for_selection':False,'debug_source':plan['debug_source']}
        atomic_json(folder/'evaluation.json',value)
        store.finish({'job_id':job['job_id']},['predictions.jsonl','evaluation.json','support_predictions.jsonl','support_diagnostics.json'])


def summarize(root):
    root=Path(root);plan=read_plan(root);selected=seal_global(root);records=[]
    for job in plan['finals']:
        if job['job_id'] not in selected:continue
        folder=Path(job['output'])/'evaluation';verified_record(folder)
        value=json.loads((folder/'evaluation.json').read_text())
        rows=read_jsonl(folder/'predictions.jsonl');validate_rows(rows,read_roles(job)['query'])
        if value['debug_source']!=plan['debug_source']:raise ValueError('evaluation source changed')
        temperature=json.loads((Path(job['output'])/'fit/calibration.json').read_text())['temperature']['temperature']
        for variant,t in [('raw',1.),('temperature',temperature)]:
            result=metrics(*_scores(rows),temperature=t)
            for k in ('macro_f1','accuracy','nll','count','ece','brier'):
                if not np.isclose(result[k],value[variant][k],rtol=1e-10,atol=1e-10):raise ValueError('metric recomputation mismatch')
        records.append(value)
    if len(records)!=2*len(plan['units']):raise ValueError('incomplete final results')
    groups=[]
    for target in plan['design']['targets']:
        for method in ('aperture','lora'):
            pair=[r for r in records if (r['model_key'],r['domain_key'],r['method'])==(target['model'],target['domain'],method)]
            if sorted(r['support_seed'] for r in pair)!=[0,1]:raise ValueError('missing seed')
            groups.append({'model':target['model'],'domain':target['domain'],'method':method,'f1_mean':float(np.mean([r['raw']['macro_f1'] for r in pair])),'f1_sd':float(np.std([r['raw']['macro_f1'] for r in pair],ddof=1))})
    atomic_json(root/'summary.json',{'protocol':plan['protocol'],'query_status':plan['query_status'],'records':records,'groups':groups})
