from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from aperture_final.numeric import metrics
from aperture_table.worker import verified_record, score_all, _scores, identity, deterministic, validate_rows
from aperture_table.protocol import read_roles, write_locked
from vigor_handoff.protocol import JobStore, atomic_json, digest, file_hash, read_jsonl
from .protocol import select_rows, read_plan


def seal_selection(root, unit):
    root=Path(root); plan=read_plan(root)
    jobs=[j for j in plan['jobs'] if j['unit_id']==unit]
    records=[]; seals={}
    for job in jobs:
        folder=Path(job['output'])/'fit'; verified_record(folder)
        saved=json.loads((folder/'identity.json').read_text())['identity']
        if saved['job']!=job:raise ValueError('fit job identity mismatch')
        rows=read_jsonl(folder/'calibration_predictions.jsonl')
        validate_rows(rows,read_roles(job)['calibration'])
        result=metrics(*_scores(rows))
        records.append({'method':job['method'],'candidate':job['candidate'],'split':'calibration',
                        'macro_f1':result['macro_f1'],'nll':result['nll'],'job_id':job['job_id']})
        seals[job['job_id']]=file_hash(folder/'DONE.json')
    winners={method:select_rows([r for r in records if r['method']==method]) for method in ('aperture','lora')}
    value={'unit_id':unit,'tuning_source':plan['tuning_source'],'query_used_for_selection':False,
           'rule':plan['selection_rule'],'candidates':records,'winners':winners,'fit_seals':seals}
    write_locked(root/'selections'/f'{unit}.json',value)
    return [j for j in jobs if winners[j['method']]==j['candidate']]


def repeat_job(job):
    replica=deepcopy(job)
    replica['job_id']+='--repeat'
    replica['output']=str(Path(job['output']).parent/(job['job_id']+'--repeat'))
    return replica


def verify_unit(root, unit):
    root=Path(root);winners=seal_selection(root,unit)
    for job in winners:
        gatepath=root/'repeat_gates'/f"{job['job_id']}.json"
        gate=json.loads(gatepath.read_text())
        if not gate.get('passed'):raise ValueError('selected fit repeat failed')
        for output,key in ((job['output'],'first_seal'),(repeat_job(job)['output'],'repeat_seal')):
            folder=Path(output)/'fit';verified_record(folder)
            if file_hash(folder/'DONE.json')!=gate[key]:raise ValueError('repeat seal changed')
    return winners


def unit_complete(root, unit):
    root=Path(root);plan=read_plan(root)
    jobs=[j for j in plan['jobs'] if j['unit_id']==unit]
    if any(not (Path(j['output'])/'fit/DONE.json').exists() for j in jobs):return False
    if not (root/'selections'/f'{unit}.json').exists():return False
    winners=seal_selection(root,unit)
    if any(not (root/'repeat_gates'/f"{j['job_id']}.json").exists() for j in winners):return False
    verify_unit(root,unit)
    return True


def seal_global(root):
    root=Path(root);plan=read_plan(root);manifest={};chosen=[]
    for unit in sorted({j['unit_id'] for j in plan['jobs']}):
        winners=verify_unit(root,unit)
        manifest[f'selections/{unit}.json']=file_hash(root/'selections'/f'{unit}.json')
        for job in winners:
            gatepath=root/'repeat_gates'/f"{job['job_id']}.json"
            manifest[str(gatepath.relative_to(root))]=file_hash(gatepath)
            chosen.append(job['job_id'])
    value={'tuning_source':plan['tuning_source'],'files':manifest,'selected_jobs':chosen,
           'query_used_for_selection':False,'selection_complete_before_query':True}
    write_locked(root/'SELECTION_COMPLETE.json',value)
    return chosen


def evaluate(root, job):
    import torch
    from aperture_table.backend import loaded
    root=Path(root);plan=read_plan(root)
    barrier=json.loads((root/'SELECTION_COMPLETE.json').read_text())
    if barrier['tuning_source']!=plan['tuning_source'] or job['job_id'] not in barrier['selected_jobs']:
        raise ValueError('query access requires a sealed selected winner')
    for rel,sha in barrier['files'].items():
        if file_hash(root/rel)!=sha:raise ValueError('global selection changed')
    selection=json.loads((root/'selections'/f"{job['unit_id']}.json").read_text())
    for candidate in [j for j in plan['jobs'] if j['unit_id']==job['unit_id']]:
        folder=Path(candidate['output'])/'fit';verified_record(folder)
        if file_hash(folder/'DONE.json')!=selection['fit_seals'][candidate['job_id']]:
            raise ValueError('selection candidate artifact changed')
    deterministic(job['optimization_seed']);roles=read_roles(job,images=True)
    fit=Path(job['output'])/'fit';verified_record(fit)
    now=identity(job,root/'assets')
    saved=json.loads((fit/'identity.json').read_text())['identity']
    if any(saved.get(k)!=v for k,v in now.items()):raise ValueError('fit source/inputs changed')
    folder=Path(job['output'])/'evaluation'
    store=JobStore(folder,{'identity':now,'fit_seal':file_hash(fit/'DONE.json'),
                          'selection_barrier':file_hash(root/'SELECTION_COMPLETE.json')})
    if store.complete():return
    if (folder/'DONE.json').exists():raise ValueError('corrupt evaluation')
    with store.lock(),loaded(job,fit) as (_,backend,controller):
        original=read_jsonl(fit/'calibration_predictions.jsonl')
        replay=score_all(backend,roles['calibration'],controller)
        if [r['prompt_contract_hash'] for r in original]!=[r['prompt_contract_hash'] for r in replay]:
            raise ValueError('calibration prompt changed on restore')
        delta=float(np.max(np.abs(_scores(original)[0]-_scores(replay)[0])))
        if delta>.005:raise ValueError('saved-state calibration replay failed')
        rows=score_all(backend,roles['query'],controller,folder/'predictions.jsonl')
        cal=json.loads((fit/'calibration.json').read_text())
        result={'job_id':job['job_id'],'unit_id':job['unit_id'],'model_key':job['model_key'],
                'domain_key':job['domain_key'],'support_seed':job['support_seed'],'method':job['method'],
                'candidate':job['candidate'],'raw':metrics(*_scores(rows)),
                'temperature':metrics(*_scores(rows),temperature=cal['temperature']['temperature']),
                'saved_state_replay_max_abs':delta,'query_status':plan['query_status'],
                'query_used_for_selection':False,'tuning_source':plan['tuning_source']}
        atomic_json(folder/'evaluation.json',result)
        store.finish({'job_id':job['job_id']},['predictions.jsonl','evaluation.json'])


def summarize(root):
    root=Path(root);plan=read_plan(root)
    chosen=seal_global(root);records=[]
    for job in plan['jobs']:
        if job['job_id'] not in chosen:continue
        folder=Path(job['output'])/'evaluation';verified_record(folder)
        value=json.loads((folder/'evaluation.json').read_text())
        if value['tuning_source']!=plan['tuning_source']:raise ValueError('evaluation source changed')
        rows=read_jsonl(folder/'predictions.jsonl');validate_rows(rows,read_roles(job)['query'])
        cal=json.loads((Path(job['output'])/'fit/calibration.json').read_text())
        for variant,t in (('raw',1.),('temperature',cal['temperature']['temperature'])):
            recomputed=metrics(*_scores(rows),temperature=t)
            for k in ('macro_f1','nll','accuracy','ece','brier','count'):
                if not np.isclose(recomputed[k],value[variant][k],rtol=1e-10,atol=1e-10):raise ValueError('metric mismatch')
        records.append(value)
    assert len(records)==32
    groups=[]
    for model in ('q25','q34','q38','g34'):
        for domain in ('h2','path'):
            for method in ('aperture','lora'):
                pair=[r for r in records if (r['model_key'],r['domain_key'],r['method'])==(model,domain,method)]
                assert sorted(r['support_seed'] for r in pair)==[0,1]
                groups.append({'model':model,'domain':domain,'method':method,
                               'f1_mean':float(np.mean([r['raw']['macro_f1'] for r in pair])),
                               'f1_sd':float(np.std([r['raw']['macro_f1'] for r in pair],ddof=1)),
                               'nll_mean':float(np.mean([r['raw']['nll'] for r in pair])),
                               'selected_candidates':[r['candidate'] for r in pair]})
    atomic_json(root/'summary.json',{'protocol':plan['protocol'],'query_status':plan['query_status'],
                                   'completed_selected_states':32,'groups':groups,'records':records})
