"""Fixed final-round plan and within-support calibration split (CPU only)."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from vigor_handoff.protocol import atomic_json, digest, file_hash, resolved_manifest, validate_split

ARMS = ('frozen','lora1','lora4','aperture')
STAGES = ('audit','fit','select','evaluate','summarize')
BASE_COMMIT = '4465c460350ddf2ce3c8e65f5b2cdb247ac7497d'


def write_locked(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'locked file changed: {path}; use a new versioned root')
        return
    atomic_json(path,value)


def _save_rows(path, rows):
    path=Path(path); text=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows)
    if path.exists() and path.read_text()!=text:
        raise ValueError(f'locked manifest changed: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')


def split_support(rows, calibration_per_class=2, seed=270921):
    if not rows or calibration_per_class < 1:raise ValueError('invalid support split')
    ids=[str(r['sample_id']) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate support sample')
    images=[]
    for r in rows:
        # BRIGHT stores a whole tile plus a per-building bbox: the bbox is part
        # of the actual visual input, not just the shared source filename.
        bbox=tuple(r.get('bbox_xyxy') or ())
        images.extend((k,r[k],bbox) for k in ('image','pre_image','post_image') if r.get(k))
    if len(images)!=len(set(images)):raise ValueError('duplicate image/crop within support')
    def key(r):return hashlib.sha256(f'{seed}|{r["sample_id"]}'.encode()).hexdigest()
    fit=[];cal=[]
    for c in sorted(set(r['label_id'] for r in rows)):
        group=sorted((deepcopy(r) for r in rows if r['label_id']==c),key=key)
        if len(group)<=calibration_per_class:raise ValueError('no fitting labels remain in class')
        cal.extend(group[:calibration_per_class]);fit.extend(group[calibration_per_class:])
    return sorted(fit,key=lambda r:r['sample_id']),sorted(cal,key=lambda r:r['sample_id'])


def probe_ids(query_rows, count=60, seed=270921):
    if count < 1:raise ValueError('probe count must be positive')
    ids=[r['sample_id'] for r in query_rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate query ID')
    return sorted(ids,key=lambda s:hashlib.sha256(f'{seed}|{s}'.encode()).hexdigest())[:count]


def make_jobs(config):
    cfg=deepcopy(config)
    if cfg.get('protocol')!='aperture_final_v1':raise ValueError('unsupported protocol')
    root=Path(cfg['run_root']);prepared=Path(cfg['prepared_root'])
    if root.name in ('visual_lens_p0_v1','visual_lens_p0_v2','oral','bright_uniform'):
        raise ValueError('use a new final-round output root, not a historical root')
    domains=cfg['domains'];names=[d['name'] for d in domains]
    if not domains or len(names)!=len(set(names)) or any(not re.fullmatch('[A-Za-z0-9_-]+',n) for n in names):
        raise ValueError('domain names must be unique safe slugs')
    seeds=cfg['support_seeds']
    if not seeds or len(seeds)!=len(set(seeds)) or any(not isinstance(s,int) or s<0 for s in seeds):
        raise ValueError('distinct nonnegative integer support seeds required')
    cpc=cfg.get('calibration_per_class',2)
    jobs=[]
    for d in domains:
        if d['kind'] not in ('single','paired') or d['support_per_class']<=cpc:
            raise ValueError('invalid domain/support recipe')
        for seed in seeds:
            unit=f'{d["name"]}--s{seed}'
            data=prepared/d['name']/f'seed_{seed}'
            common={'unit_id':unit,'domain':d['name'],'dataset':d.get('dataset',d['name']),
                'kind':d['kind'],'family':'qwen2','model_path':cfg['model_path'],
                'model_id':'Qwen/Qwen2.5-VL-7B-Instruct','support_seed':seed,
                'source_support':d['support'].format(seed=seed),'source_query':d['query'].format(seed=seed),
                'support':str(data/'fit.jsonl'),'calibration':str(data/'calibration.jsonl'),
                'query':str(data/'query.jsonl'),'split_record':str(data/'split.json'),
                'group_key':d['group_key'],'query_count':d['query_count'],'classes':d['classes'],
                'original_support_per_class':d['support_per_class'],'support_per_class':d['support_per_class']-cpc,
                'calibration_per_class':cpc,'split_seed':cfg.get('split_seed',270921)+seed,
                'target_split_status':'new_within_support_calibration; previously_inspected_query',
                'protocol_version':'aperture_final_v1','source_base_commit':BASE_COMMIT,
                'optimization_seed':0,'basis_seed':0,'basis_mode':'covariance','basis_source':'support',
                'rank':16,'layers':[14,27],'sites':['K','V'],'coefficient_mode':'full',
                'alpha':3.,'steps':4,'lr':.05,'l2':.001,
                'mask':'post' if d['kind']=='paired' else 'all_visual','crop_size':448,
                'loss_span':'full','loss_reduction':'mean' if d['kind']=='paired' else 'sum',
                'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,
                'lora_lr':1e-4 if d['kind']=='paired' else 2e-4,
                'lora_accumulation':3 if d['kind']=='paired' else 1,
                'evidence':'held_out_support_calibration_final_round',
                'temperature_penalty':cfg.get('temperature_penalty',.01),'bias_l2':.001,
                'probe_count':cfg.get('probe_count',60),'probe_seed':270922,
                'corruptions':cfg.get('corruptions',['gaussian4','gaussian8','jpeg70','jpeg40'])
                     if d.get('corruption',False) and seed==0 else [],
                'dose_scales':[0.,.5,1.] if seed==0 else [],
                'timing_samples':cfg.get('timing_samples',12) if seed==0 else 0,
                'timing_repeats':3,'audit_per_class':1,'audit_abs_tol':.005,'audit_relative_tol':.05}
            for arm in ARMS:
                j=deepcopy(common);j['arm']=arm;j['lora_passes']=4 if arm=='lora4' else 1
                j['job_id']=f'{unit}--{arm}';j['output']=str(root/'states'/j['job_id'])
                jobs.append(j)
    return jobs


def prepare_job(job):
    support=resolved_manifest(job['source_support']);query=resolved_manifest(job['source_query'])
    audit=validate_split(support,query,job['group_key'])
    if len(support)!=job['original_support_per_class']*job['classes'] or len(query)!=job['query_count']:
        raise ValueError(f'original data budget differs: {job["unit_id"]}')
    if set(audit['support_counts'].values())!={job['original_support_per_class']}:
        raise ValueError('original support must be class-balanced')
    fit,cal=split_support(support,job['calibration_per_class'],job['split_seed'])
    # Query groups are disjoint from ALL support. Calibration is sample-held-out;
    # shared support tiles/patients are recorded rather than called independent.
    fg={str(r[job['group_key']]) for r in fit};cg={str(r[job['group_key']]) for r in cal}
    record={'unit_id':job['unit_id'],'total_labels':len(support),'fit_labels':len(fit),
        'calibration_labels':len(cal),'query_count':len(query),'query_status':job['target_split_status'],
        'query_ids_sha256':digest([r['sample_id'] for r in query]),
        'fit_ids':[r['sample_id'] for r in fit],'calibration_ids':[r['sample_id'] for r in cal],
        'source_support_sha256':file_hash(job['source_support']),'source_query_sha256':file_hash(job['source_query']),
        'fit_calibration_shared_groups':sorted(fg&cg),
        'calibration_split_level':'sample/crop; not guaranteed group-disjoint',
        'split_seed':job['split_seed']}
    # Check lock BEFORE writing any prepared manifest.
    if Path(job['split_record']).exists() and json.loads(Path(job['split_record']).read_text())!=record:
        raise ValueError('source/split changed; use a new prepared root')
    for name,rs in [('support',fit),('calibration',cal),('query',query)]:_save_rows(job[name],rs)
    write_locked(job['split_record'],record)
    return record


def check_job(job, assets=True):
    fit=resolved_manifest(job['support']);cal=resolved_manifest(job['calibration']);query=resolved_manifest(job['query'])
    validate_split(fit+cal,query,job['group_key'])
    ids=lambda rs:{r['sample_id'] for r in rs}
    if ids(fit)&ids(cal):raise ValueError('fit/calibration overlap')
    for rs,count in [(fit,job['support_per_class']),(cal,job['calibration_per_class'])]:
        counts=Counter(r['label_id'] for r in rs)
        if len(counts)!=job['classes'] or set(counts.values())!={count}:raise ValueError('prepared class budget mismatch')
    if len(query)!=job['query_count']:raise ValueError('query count mismatch')
    record=json.loads(Path(job['split_record']).read_text())
    if [r['sample_id'] for r in fit]!=record['fit_ids'] or [r['sample_id'] for r in cal]!=record['calibration_ids']:
        raise ValueError('prepared IDs differ from split record')
    if digest([r['sample_id'] for r in query])!=record['query_ids_sha256']:raise ValueError('query IDs/order changed')
    for key in ('support','query'):
        if file_hash(job['source_'+key])!=record['source_'+key+'_sha256']:
            raise ValueError('source manifest changed after the split was locked')
    source_support=resolved_manifest(job['source_support'])
    expected_fit,expected_cal=split_support(source_support,job['calibration_per_class'],job['split_seed'])
    if fit!=expected_fit or cal!=expected_cal or query!=resolved_manifest(job['source_query']):
        raise ValueError('prepared content differs from its locked source/split')
    if assets:
        for rs in (fit,cal,query):
            for r in rs:
                for k in ('image','pre_image','post_image'):
                    if r.get(k) and not Path(r[k]).is_file():raise FileNotFoundError(r[k])
        p=Path(job['model_path'])
        if not (p/'config.json').is_file() or not (list(p.glob('*.safetensors')) or list(p.glob('*.bin'))):
            raise FileNotFoundError(f'missing local model snapshot: {p}')
    return record


def source_signature():
    root=Path(__file__).resolve().parent
    return digest([(p.name,file_hash(p)) for p in sorted(root.glob('*.py'))])
