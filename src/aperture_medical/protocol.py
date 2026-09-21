"""Group-isolated medical protocols; fixed query, nested support budgets."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from vigor_handoff.protocol import digest,file_hash,atomic_json,resolved_manifest,validate_split
from aperture_final.protocol import write_locked,probe_ids,_save_rows
from .data import population,pixel_hash

ARMS=('frozen','lora1','lora4','random','aperture')


def order(values, seed, field):
    return sorted(values,key=lambda r:hashlib.sha256(f'{seed}|{r[field]}'.encode()).hexdigest())


def partition(rows, seed):
    if not rows or any(not r.get('group_id') for r in rows):raise ValueError('missing group identity')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('duplicate sample IDs')
    groups=order([{'g':g} for g in sorted({r['group_id'] for r in rows})],seed,'g')
    if len(groups)<3:raise ValueError('need at least three independent groups')
    nq=max(1,int(.4*len(groups)));nc=max(1,int(.2*len(groups)))
    query={r['g'] for r in groups[:nq]};cal={r['g'] for r in groups[nq:nq+nc]}
    return tuple(sorted([deepcopy(r) for r in rows if f(r['group_id'])],key=lambda r:r['sample_id'])
        for f in [lambda g:g not in query|cal,lambda g:g in cal,lambda g:g in query])


def select_balanced(rows, per_class, seed, classes=None):
    if not isinstance(per_class,int) or per_class<1:raise ValueError('invalid sample budget')
    ids=range(classes) if classes is not None else sorted({r['label_id'] for r in rows})
    chosen=[]
    for c in ids:
        rr=order([r for r in rows if r['label_id']==c],seed,'sample_id')
        # Round-robin groups before adding a second patch from the same patient.
        buckets={}
        for r in rr:buckets.setdefault(r['group_id'],[]).append(r)
        gs=order([{'g':g} for g in buckets],seed,'g');balanced=[]
        for k in range(max([len(v) for v in buckets.values()]+[0])):
            balanced.extend(buckets[g['g']][k] for g in gs if len(buckets[g['g']])>k)
            if len(balanced)>=per_class:break
        if len(balanced)<per_class:raise ValueError(f'class {c}: insufficient eligible samples')
        chosen.extend(deepcopy(balanced[:per_class]))
    return sorted(chosen,key=lambda r:r['sample_id'])


def validate_triplet(fit,cal,query):
    validate_split(fit,cal,'group_id');validate_split(fit+cal,query,'group_id')
    hashes=[]
    for rs in (fit,cal,query):
        hs=[r.get('metadata',{}).get('pixel_sha256') for r in rs]
        if all(hs):
            if len(hs)!=len(set(hs)):raise ValueError('duplicate image content within partition')
            hashes.append(set(hs))
        else:hashes.append(set())
    if any(hashes[i]&hashes[j] for i in range(3) for j in range(i)):
        raise ValueError('image content overlap')


def source_signature():
    src=Path(__file__).resolve().parents[1]
    files=[]
    for name in ['aperture_medical','aperture_final','eventttt','vigor_handoff','visual_lens']:
        files.extend((src/name).glob('*.py'))
    return digest([(str(p.relative_to(src)),file_hash(p)) for p in sorted(files)])


def make_jobs(config):
    c=deepcopy(config)
    if c.get('protocol')!='aperture_medical_v1':raise ValueError('unknown medical protocol')
    seeds=c['support_seeds'];budgets=c['budgets_per_class'];cal=c['calibration_per_class']
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s)!=int or s<0 for s in seeds):raise ValueError('invalid seeds')
    if not budgets or len(set(budgets))!=len(budgets) or any(type(b)!=int or b<=cal for b in budgets):raise ValueError('invalid budgets')
    ds=c['domain_specs']
    if not ds or len({d['name'] for d in ds})!=len(ds):raise ValueError('duplicate/empty domains')
    root=Path(c['run_root']);prepared=Path(c['prepared_root'])
    if root.name in ('oral','aperture_final_v1','visual_lens_p0_v1'):raise ValueError('historical root forbidden')
    jobs=[]
    for d in ds:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',d['name']):raise ValueError('unsafe domain')
        if d['classes']<2 or d['query_per_class']<1:raise ValueError('invalid query budget')
        for b in budgets:
            for seed in seeds:
                unit=f'{d["name"]}--n{b*d["classes"]}--s{seed}';folder=prepared/d['name']/f'n{b*d["classes"]}'/f'seed_{seed}'
                j=dict(unit_id=unit,domain=d['name'],dataset=d['dataset'],kind='single',family='qwen2',
                    model_path=c['model_path'],model_id='Qwen/Qwen2.5-VL-7B-Instruct',support_seed=seed,
                    support=str(folder/'fit.jsonl'),calibration=str(folder/'calibration.jsonl'),
                    query=str(prepared/d['name']/'query.jsonl'),split_record=str(folder/'split.json'),
                    group_key='group_id',query_count=d['query_per_class']*d['classes'],classes=d['classes'],
                    total_label_budget=b*d['classes'],support_per_class=b-cal,calibration_per_class=cal,
                    original_support_per_class=b,split_seed=c['split_seed'],isolation=d['isolation'],
                    protocol_version='aperture_medical_v1',optimization_seed=0,basis_seed=0,
                    basis_mode='covariance',basis_source='support',rank=16,layers=[14,27],sites=['K','V'],
                    coefficient_mode='full',alpha=3.,steps=4,lr=.05,l2=.001,mask='all_visual',crop_size=448,
                    loss_span='full',loss_reduction='sum',lora_rank=16,lora_alpha=32,lora_dropout=.05,
                    lora_lr=2e-4,lora_accumulation=1,evidence='medical_prospective_fixed_protocol',
                    temperature_penalty=.01,bias_l2=.001,probe_count=12,probe_seed=310928,
                    corruptions=[],dose_scales=[],timing_samples=8 if seed==0 else 0,timing_repeats=3,
                    audit_per_class=1,audit_abs_tol=.005,audit_relative_tol=.05,
                    target_split_status='H2_previously_examined_anchor' if d['name']=='hospital_2' else 'new_prespecified_target',
                    spec=d)
                for arm in ARMS:
                    x=deepcopy(j);x.update(arm=arm,lora_passes=4 if arm=='lora4' else 1,
                        basis_mode='random' if arm=='random' else 'covariance',job_id=f'{unit}--{arm}')
                    x['output']=str(root/'states'/x['job_id']);jobs.append(x)
    return jobs


def prepare(config, domains=None):
    jobs=make_jobs(config);prepared=Path(config['prepared_root'])
    for spec in config['domain_specs']:
        if domains and spec['name'] not in domains:continue
        domain=spec['name'];cache=prepared/domain
        rows,source=population(spec,cache)
        fpool,cpool,qpool=partition(rows,config['split_seed'])
        query=select_balanced(qpool,spec['query_per_class'],config['split_seed']+1,spec['classes'])
        units={j['unit_id']:j for j in jobs if j['domain']==domain};selected={}
        for unit,j in units.items():
            fit=select_balanced(fpool,j['support_per_class'],1000+j['support_seed'],j['classes'])
            cal=select_balanced(cpool,j['calibration_per_class'],2000+j['support_seed'],j['classes'])
            selected[unit]=(fit,cal)
        unique={r['sample_id']:r for rs in [query]+[x for pair in selected.values() for x in pair] for r in rs}
        for r in unique.values():
            if not Path(r['image']).is_file():raise FileNotFoundError(r['image'])
            r['metadata']['pixel_sha256']=pixel_hash(r['image'])
        # Copy hashes into duplicate per-budget objects as well.
        for rs in [query]+[x for pair in selected.values() for x in pair]:
            for r in rs:r['metadata']['pixel_sha256']=unique[r['sample_id']]['metadata']['pixel_sha256']
        for unit,j in units.items():
            fit,cal=selected[unit];validate_triplet(fit,cal,query)
            record={'unit_id':unit,'source':source,'total_labels':len(fit)+len(cal),
                'fit_ids':[r['sample_id'] for r in fit],'calibration_ids':[r['sample_id'] for r in cal],
                'query_ids_sha256':digest([r['sample_id'] for r in query]),'isolation':j['isolation'],
                'query_status':j['target_split_status'],'fit_groups':sorted({r['group_id'] for r in fit}),
                'calibration_groups':sorted({r['group_id'] for r in cal}),
                'query_groups':sorted({r['group_id'] for r in query}),
                'content':{name:digest(rr) for name,rr in [('support',fit),('calibration',cal),('query',query)]}}
            write_locked(j['split_record'],record)
            for name,rr in [('support',fit),('calibration',cal),('query',query)]:_save_rows(j[name],rr)
    return len(jobs)


def check_job(job, assets=True):
    rs={n:resolved_manifest(job[n]) for n in ('support','calibration','query')}
    validate_triplet(rs['support'],rs['calibration'],rs['query'])
    rec=json.loads(Path(job['split_record']).read_text())
    if rec['unit_id']!=job['unit_id']:raise ValueError('split/job identity mismatch')
    for name,rows in rs.items():
        if digest(rows)!=rec['content'][name]:raise ValueError('locked prepared content changed')
        n=job['query_count']//job['classes'] if name=='query' else job['calibration_per_class'] if name=='calibration' else job['support_per_class']
        if Counter(r['label_id'] for r in rows)!=Counter({i:n for i in range(job['classes'])}):raise ValueError('class budget mismatch')
        if assets:
            for r in rows:
                if pixel_hash(r['image'])!=r['metadata']['pixel_sha256']:raise ValueError('source image changed')
    if assets:
        root=Path(job['model_path'])
        if not (root/'config.json').is_file() or not list(root.glob('*.safetensors')):raise FileNotFoundError('local BF16 model missing')
    return rec
