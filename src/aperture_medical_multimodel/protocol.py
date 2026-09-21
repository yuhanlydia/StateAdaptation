"""Fixed multicenter state matrix and content validation, without model imports."""
from copy import deepcopy
import json
from pathlib import Path
from .data import validate_unit
from vigor_handoff.protocol import digest, file_hash, resolved_manifest

ARMS=('frozen','lora1','lora4','aperture')
BASE_COMMIT='355c0b103ecc5130a31aaa93a8705828fe7027cd'


def source_signature():
    src=Path(__file__).resolve().parents[1]
    paths=[]
    for pkg in ('aperture_medical_multimodel','aperture_final','vigor_handoff','visual_lens','eventttt'):
        paths.extend((p.relative_to(src).as_posix(),file_hash(p)) for p in sorted((src/pkg).glob('*.py')))
    return digest(paths)


def make_jobs(config):
    c=deepcopy(config)
    if c.get('protocol')!='aperture_medical_multimodel_v1':raise ValueError('unsupported protocol')
    hs=c['hospitals'];ss=c['support_seeds'];budgets=[c['primary_support']]+c.get('curve_supports',[])
    if not hs or len(hs)!=len(set(hs)) or any(h not in range(5) for h in hs):raise ValueError('distinct official hospital IDs 0..4 required')
    if not ss or len(ss)!=len(set(ss)) or any(not isinstance(s,int) or s<0 for s in ss):raise ValueError('distinct nonnegative support seeds required')
    if any(not isinstance(b,int) or b<8 or b%8 for b in budgets) or len(budgets)!=len(set(budgets)):raise ValueError('distinct budgets divisible by 8 required')
    if c['query_count']<1:raise ValueError('positive query size required')
    models=c['models']
    if not models or len({m['name'] for m in models})!=len(models) or models[0]['family']!='qwen2':raise ValueError('primary model must be qwen2; unique model names')
    if any(m['family'] not in ('qwen2','internvl3') for m in models):raise ValueError('unvalidated medical model family')
    out=Path(c['run_root']);prepared=Path(c['prepared_root']);jobs=[]
    if not out.name.startswith('aperture_medical_multimodel_') or not prepared.name.startswith('aperture_medical_multimodel_'):raise ValueError('use new aperture_medical_multimodel_ versioned roots')
    for mi,m in enumerate(models):
        for budget in (budgets if mi==0 else [c['primary_support']]):
            phase='curve' if budget!=c['primary_support'] else 'main' if mi==0 else 'replication'
            arms=ARMS+('random',) if phase=='main' else ARMS
            for h in hs:
                for seed in ss:
                    unit=f'{m["name"]}--h{h}--n{budget}--s{seed}'
                    data=prepared/f'hospital_{h}'/f'support_{budget}'/f'seed_{seed}'
                    common={'unit_id':unit,'phase':phase,'domain':f'hospital_{h}','hospital':h,'dataset':'camelyon17',
                        'kind':'single','family':m['family'],'model_name':m['name'],'model_path':m['path'],'model_id':m['id'],
                        'support_seed':seed,'total_support':budget,'support':str(data/'fit.jsonl'),
                        'calibration':str(data/'calibration.jsonl'),'query':str(data/'query.jsonl'),
                        'split_record':str(data/'split.json'),'data_root':c['data_root'],
                        'group_key':'group_id','classes':2,'query_count':c['query_count'],
                        'support_per_class':3*budget//8,'calibration_per_class':budget//8,
                        'protocol_version':c['protocol'],'source_base_commit':BASE_COMMIT,
                        'target_split_status':'prospective_patient_disjoint; hospital2 previously explored; not official WILDS DG',
                        'optimization_seed':0,'basis_seed':0,'basis_mode':'covariance','basis_source':'support',
                        'rank':16,'layers':[14,27],'sites':['K','V'],'coefficient_mode':'full',
                        'alpha':3.,'steps':4,'lr':.05,'l2':.001,'mask':'all_visual','crop_size':m['image_size'],
                        'loss_span':'full','loss_reduction':'sum','lora_rank':16,'lora_alpha':32,'lora_dropout':.05,
                        'lora_lr':2e-4,'lora_accumulation':1,'evidence':'prospective_multicenter_medical',
                        'temperature_penalty':.01,'bias_l2':.001,'probe_count':60,'probe_seed':271022,
                        'corruptions':[],'dose_scales':[],'timing_samples':12 if seed==ss[0] else 0,
                        'timing_repeats':3,'audit_per_class':1,'audit_abs_tol':.005,'audit_relative_tol':.05,
                        'strict_determinism':True,'repeat_score_atol':.005,
                        'repeat_gate':str(out/'repeatability'/(m['name']+'.json'))}
                    for arm in arms:
                        j={**common,'arm':arm,'required_arms':list(arms),'lora_passes':4 if arm=='lora4' else 1}
                        j['job_id']=f'{unit}--{arm}';j['output']=str(out/'states'/j['job_id']);jobs.append(j)
    if len(jobs)!=len({j['job_id'] for j in jobs}):raise ValueError('duplicate jobs')
    return jobs


def check_job(job,assets=True):
    paths={'fit':job['support'],'calibration':job['calibration'],'query':job['query']}
    u={k:resolved_manifest(p) for k,p in paths.items()}
    audit=validate_unit(u,job['total_support'],job['query_count'])
    rec=json.loads(Path(job['split_record']).read_text())
    if rec['hospital']!=job['hospital'] or rec['budget']!=job['total_support'] or rec['seed']!=job['support_seed']:raise ValueError('split identity differs')
    if file_hash(Path(job['data_root'])/'metadata.csv')!=rec['metadata_sha256']:raise ValueError('official metadata changed')
    for name,p in paths.items():
        if rec['manifests'][name]!=file_hash(p):raise ValueError(f'{name} content changed after sealing')
    if digest([r['sample_id'] for r in u['query']])!=rec['query_ids_sha256']:raise ValueError('query order changed')
    for rows in u.values():
        if any(r['metadata']['hospital']!=job['hospital'] for r in rows):raise ValueError('mixed hospitals')
    if assets:
        for rows in u.values():
            for r in rows:
                if not Path(r['image']).is_file():raise FileNotFoundError(r['image'])
        p=Path(job['model_path'])
        if not (p/'config.json').is_file() or not (list(p.glob('*.safetensors')) or list(p.glob('*.bin'))):raise FileNotFoundError(f'missing model: {p}')
    return audit
