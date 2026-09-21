"""Official CAMELYON17 metadata, deterministic patient-disjoint partitions.

No image decoding, model prediction, or query-performance-based split selection.
Hospital indices come from metadata (0..4), not directory or paper numbering.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
from vigor_handoff.protocol import atomic_json, digest, file_hash

LABELS = ('normal', 'tumor')
QUESTION = 'Does the central region contain tumor tissue? Answer exactly normal or tumor.'


def order_key(value, seed):
    return hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()


def locked_jsonl(path, rows):
    p=Path(path); text=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows)
    if p.exists():
        if p.read_text()!=text:raise ValueError(f'locked manifest changed: {p}')
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(p)


def locked_json(path, obj):
    p=Path(path)
    if p.exists():
        if json.loads(p.read_text())!=obj:raise ValueError(f'locked record changed: {p}')
        return
    atomic_json(p,obj)


def load_metadata(root):
    root=Path(root).resolve(); p=root/'metadata.csv'
    if not p.is_file():raise FileNotFoundError(f'{p}: supply the COMPLETE official CAMELYON17-WILDS extraction, not a hospital_2 subset')
    rows=[];seen=set();patient_centers={};slide_patients={}
    with p.open(newline='') as f:
        reader=csv.DictReader(f)
        required={'patient','node','x_coord','y_coord','center','slide','tumor'}
        if not required.issubset(reader.fieldnames or []):raise ValueError(f'official metadata columns missing: {required-set(reader.fieldnames or [])}')
        for a in reader:
            hospital=int(a['center']);y=int(a['tumor']);slide=int(a['slide'])
            patient=str(a['patient']).strip();node=int(a['node']);x=int(a['x_coord']);v=int(a['y_coord'])
            if hospital not in range(5) or y not in (0,1) or not patient or node<0 or x<0 or v<0:
                raise ValueError('invalid hospital/label/patient/coordinate')
            if patient in patient_centers and patient_centers[patient]!=hospital:raise ValueError('patient assigned to multiple hospitals')
            if slide in slide_patients and slide_patients[slide]!=patient:raise ValueError('slide assigned to multiple patients')
            patient_centers[patient]=hospital;slide_patients[slide]=patient
            sid=f'cam17-p{patient}-n{node}-x{x}-y{v}'
            if sid in seen:raise ValueError('duplicate official patch')
            seen.add(sid)
            image=root/'patches'/f'patient_{patient}_node_{node}'/f'patch_patient_{patient}_node_{node}_x_{x}_y_{v}.png'
            rows.append(dict(sample_id=sid,domain_id=f'hospital_{hospital}',group_id=f'patient_{patient}',
                image=str(image),label=LABELS[y],label_id=y,candidate_labels=list(LABELS),
                question=QUESTION,dataset='camelyon17-wilds',metadata={'patient':patient,'slide':slide,
                'hospital':hospital,'node':node,'x_coord':x,'y_coord':v,'official_split':a.get('split')}))
    if not rows:raise ValueError('empty official metadata')
    return rows


def partition_patients(rows, seed, pool_per_class=64):
    """40% query patients, 20% calibration, remainder fitting (at least 2 each).

    Use label COUNTS only to ensure feasible stratified pools. Lexicographic
    hash-ordered combinations give a reproducible partition. Never consult model
    scores. All budgets and seeds share the same pool-level patient assignment.
    """
    if not rows or pool_per_class<1:raise ValueError('empty cohort or invalid pool budget')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('duplicate metadata ID')
    groups=defaultdict(list)
    for r in rows:groups[r['group_id']].append(r)
    patients=sorted(groups,key=lambda s:order_key(s,seed));n=len(patients)
    if n<6:raise ValueError('at least six independent patients per hospital are required; do not replace by slides')
    nq=max(2,int(round(.4*n)));nc=max(2,int(round(.2*n)))
    if n-nq-nc<2:raise ValueError('insufficient fitting patients')
    counts={p:Counter(r['label_id'] for r in groups[p]) for p in patients}
    def sufficient(ps):return all(sum(counts[p][c] for p in ps)>=pool_per_class for c in (0,1))
    tested=0
    for query in combinations(patients,nq):
        if not sufficient(query):continue
        other=[p for p in patients if p not in query]
        for cal in combinations(other,nc):
            tested+=1
            if tested>200000:raise ValueError('patient stratification search budget exceeded; inspect metadata, do not tune on query')
            fit=[p for p in other if p not in cal]
            if sufficient(cal) and sufficient(fit):
                return {name:sorted((r for p in ps for r in groups[p]),key=lambda r:r['sample_id'])
                        for name,ps in [('fit',fit),('calibration',cal),('query',query)]}
    raise ValueError('no feasible patient-disjoint class-covered split; do not silently relax patient isolation')


def _round_robin(rows, count, seed):
    groups=defaultdict(list)
    for r in rows:groups[r['group_id']].append(r)
    keys=sorted(groups,key=lambda g:order_key(g,seed))
    buckets={g:sorted(groups[g],key=lambda r:order_key(r['sample_id'],seed)) for g in keys}
    chosen=[];i=0
    while len(chosen)<count:
        made=False
        for g in keys:
            if i<len(buckets[g]):
                chosen.append(deepcopy(buckets[g][i]));made=True
                if len(chosen)==count:break
        if not made:raise ValueError(f'insufficient patches: requested {count}')
        i+=1
    return chosen


def select_balanced(rows, per_class, seed):
    if not isinstance(per_class,int) or per_class<1:raise ValueError('positive class budget required')
    selected=[]
    for c in (0,1):selected+=_round_robin([r for r in rows if r['label_id']==c],per_class,seed+c*100003)
    return sorted(selected,key=lambda r:r['sample_id'])


def select_query(rows, count, seed):
    if count<1:raise ValueError('positive query count required')
    # Equal-patient sampling is deliberate; class balance is NOT forced.
    return sorted(_round_robin(rows,count,seed),key=lambda r:r['sample_id'])


def build_unit(pools, budget, seed, query_count, split_seed):
    if budget<8 or budget%8:raise ValueError('total label budgets must be multiples of eight')
    unit={'fit':select_balanced(pools['fit'],3*budget//8,seed),
          'calibration':select_balanced(pools['calibration'],budget//8,seed+913),
          'query':select_query(pools['query'],query_count,split_seed)}
    validate_unit(unit,budget,query_count)
    return unit


def validate_unit(unit, budget, query_count):
    if set(unit)!={'fit','calibration','query'}:raise ValueError('three partitions required')
    patient_sets=[];slides=[];ids=[];images=[]
    for name,rs in unit.items():
        if not rs:raise ValueError('empty partition')
        if len(rs)!=len({r['sample_id'] for r in rs}):raise ValueError('duplicate patch ID')
        for r in rs:
            if r['label_id'] not in (0,1) or r['label']!=LABELS[r['label_id']] or tuple(r['candidate_labels'])!=LABELS:
                raise ValueError('candidate/label contract mismatch')
            if not r['metadata'].get('patient'):raise ValueError('missing patient ID')
        patient_sets.append({r['metadata']['patient'] for r in rs})
        slides.append({r['metadata']['slide'] for r in rs});ids.append({r['sample_id'] for r in rs});images.append({r['image'] for r in rs})
    for sets,name in [(patient_sets,'patient'),(slides,'slide'),(ids,'sample'),(images,'image')]:
        if any(a&b for i,a in enumerate(sets) for b in sets[i+1:]):raise ValueError(f'{name} overlap across fit/calibration/query')
    for name,n in [('fit',3*budget//8),('calibration',budget//8)]:
        if Counter(r['label_id'] for r in unit[name])!={0:n,1:n}:raise ValueError('class budget mismatch')
    if len(unit['query'])!=query_count or {r['label_id'] for r in unit['query']}!={0,1}:raise ValueError('query count/class coverage mismatch')
    return {'patients':{k:len({r['metadata']['patient'] for r in rs}) for k,rs in unit.items()},
            'slides':{k:len({r['metadata']['slide'] for r in rs}) for k,rs in unit.items()},
            'counts':{k:dict(Counter(r['label'] for r in rs)) for k,rs in unit.items()}}


def prepare(config):
    root=Path(config['prepared_root']);rows=load_metadata(config['data_root'])
    metadata_hash=file_hash(Path(config['data_root'])/'metadata.csv')
    specs={int(h):[r for r in rows if r['metadata']['hospital']==h] for h in config['hospitals']}
    budgets=sorted(set([config['primary_support']]+config.get('curve_supports',[])))
    lock={'protocol':config['protocol'],'data_root':str(Path(config['data_root']).resolve()),
          'metadata_sha256':metadata_hash,'hospitals':config['hospitals'],'support_seeds':config['support_seeds'],
          'budgets':budgets,'query_count':config['query_count'],'split_seed':config['split_seed'],
          'query_sampling':'patient-interleaved; no forced class balance','calibration_split':'patient-disjoint'}
    locked_json(root/'cohort_lock.json',lock)
    for h,cohort in specs.items():
        pools=partition_patients(cohort,config['split_seed']+h, max(32,max(budgets)//2))
        patients={name:sorted({r['metadata']['patient'] for r in rs}) for name,rs in pools.items()}
        locked_json(root/f'hospital_{h}'/'patient_pools.json',patients)
        for seed in config['support_seeds']:
            for budget in budgets:
                u=build_unit(pools,budget,seed,config['query_count'],config['split_seed']+h)
                out=root/f'hospital_{h}'/f'support_{budget}'/f'seed_{seed}'
                for name,rs in u.items():locked_jsonl(out/f'{name}.jsonl',rs)
                record={'metadata_sha256':metadata_hash,'hospital':h,'budget':budget,'seed':seed,
                        'query_ids_sha256':digest([r['sample_id'] for r in u['query']]),
                        'manifests':{name:file_hash(out/f'{name}.jsonl') for name in u},
                        **validate_unit(u,budget,config['query_count'])}
                locked_json(out/'split.json',record)
    return lock
