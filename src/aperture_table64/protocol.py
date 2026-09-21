"""Locked Table-1 population, native model registry, and immutable job identities."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re
from vigor_handoff.protocol import digest, file_hash, atomic_json, resolved_manifest, validate_split
from aperture_final.protocol import write_locked, _save_rows

MODEL_SPECS = {
 'q25': {'id':'Qwen/Qwen2.5-VL-7B-Instruct','family':'qwen2','model_type':'qwen2_5_vl','size':'7B'},
 'q34': {'id':'Qwen/Qwen3-VL-4B-Instruct','family':'qwen3_vl','model_type':'qwen3_vl','size':'4B'},
 'q38': {'id':'Qwen/Qwen3-VL-8B-Instruct','family':'qwen3_vl','model_type':'qwen3_vl','size':'8B'},
 'g34': {'id':'google/gemma-3-4b-it','family':'gemma3','model_type':'gemma3','size':'4B'},
}
ARMS=('frozen','lora1','lora4','aperture')
DOMAINS=('hospital_2','pathmnist_crc')
PROTOCOL='aperture_table64_v1'


def make_jobs(config):
    c=deepcopy(config)
    required={'protocol','run_root','prepared_root','support_seeds','arms','models','camelyon_root','pathmnist_npz'}
    if set(c)!=required:raise ValueError('unknown or missing config field; hyperparameters are fixed in this protocol')
    if c.get('protocol') != PROTOCOL or c.get('support_seeds') != [0,1] or c.get('arms') != list(ARMS):
        raise ValueError('this registered table requires the fixed protocol, two seeds, and all four arms')
    if set(c.get('models',{})) != set(MODEL_SPECS): raise ValueError('all four registered checkpoints are required')
    root=Path(c['run_root']);prepared=Path(c['prepared_root'])
    if not root.name.startswith('aperture_table64_') or not prepared.name.startswith('aperture_table64_'):
        raise ValueError('independent aperture_table64_ output/prepared roots required')
    jobs=[]
    for key,spec in MODEL_SPECS.items():
        for d in DOMAINS:
            classes=2 if d=='hospital_2' else 9
            for seed in (0,1):
                f=prepared/d/f'seed_{seed}'
                for arm in ARMS:
                    unit=f'{key}--{d}--s{seed}';jid=f'{unit}--{arm}'
                    jobs.append(dict(protocol=PROTOCOL,model_key=key,model_id=spec['id'],model_type=spec['model_type'],
                        family=spec['family'],model_path=c['models'][key],domain=d,support_seed=seed,arm=arm,
                        unit_id=unit,job_id=jid,output=str(root/'states'/jid),
                        support=str(f/'fit.jsonl'),calibration=str(f/'calibration.jsonl'),query=str(prepared/d/'query.jsonl'),
                        split_record=str(f/'split.json'),group_key='group_id',classes=classes,
                        support_count=6*classes,calibration_count=2*classes,query_count=200 if classes==2 else 180,
                        optimization_seed=0,rank=16,layers=[14,27],sites=['K','V'],alpha=3.,steps=4,lr=.05,l2=.001,
                        loss_reduction='sum',lora_rank=16,lora_alpha=32,lora_dropout=.05,lora_lr=2e-4,
                        lora_accumulation=1,lora_passes=4 if arm=='lora4' else 1,
                        temperature_penalty=.01,bias_l2=.001,score_tolerance=.005))
    return jobs


def validate_counts(rows, expected, classes):
    if len(rows)!=expected or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('wrong count or duplicate sample IDs')
    if Counter(r['label_id'] for r in rows)!=Counter({k:expected//classes for k in range(classes)}):
        raise ValueError('class budget mismatch')
    candidates=rows[0]['candidate_labels'] if rows and 'candidate_labels' in rows[0] else None
    if candidates is not None:
        if len(candidates)!=classes or len(set(candidates))!=classes:raise ValueError('invalid candidate list')
        if any(r['candidate_labels']!=candidates or r['label']!=candidates[r['label_id']] for r in rows):
            raise ValueError('label/index/candidate contract mismatch')


def subset_query(rows, per_class, classes, seed):
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('duplicate parent query IDs')
    result=[]
    for c in range(classes):
        eligible=[r for r in rows if r['label_id']==c]
        eligible.sort(key=lambda r:hashlib.sha256(f'{seed}|{r["sample_id"]}'.encode()).hexdigest())
        if len(eligible)<per_class:raise ValueError('insufficient fixed-query population')
        result.extend(deepcopy(eligible[:per_class]))
    return sorted(result,key=lambda r:r['sample_id'])


def validate_triplet(fit,cal,query):
    validate_split(fit,cal,'group_id');validate_split(fit+cal,query,'group_id')
    hashes=[]
    for rows in (fit,cal,query):
        h=[r.get('metadata',{}).get('pixel_sha256') for r in rows]
        if not all(h) or len(h)!=len(set(h)):raise ValueError('missing or duplicate image hashes')
        hashes.append(set(h))
    if any(hashes[i]&hashes[j] for i in range(3) for j in range(i)):
        raise ValueError('image content overlaps between roles')


def prepare(config):
    # Reuse the published official-data readers and EXACT medical_v1 patient roles.
    # The two-hospital/multimodel package uses a different protocol; do not import it.
    from aperture_medical.data import population, pixel_hash
    from aperture_medical.protocol import partition, select_balanced
    jobs=make_jobs(config);prepared=Path(config['prepared_root'])
    for domain in DOMAINS:
        cls=2 if domain=='hospital_2' else 9
        spec={'name':domain,'classes':cls,'dataset':'camelyon17-wilds' if cls==2 else 'pathmnist',
              'source_root':config['camelyon_root'] if cls==2 else config['pathmnist_npz']}
        rows,source=population(spec,prepared/domain)
        fp,cp,qp=partition(rows,310927)
        parent=select_balanced(qp,250 if cls==2 else 100,310928,cls)
        query=subset_query(parent,100 if cls==2 else 20,cls,310929)
        subsets={}
        for seed in (0,1):
            subsets[seed]=(select_balanced(fp,6,1000+seed,cls), select_balanced(cp,2,2000+seed,cls))
        unique={r['sample_id']:r for rs in [query]+[x for pair in subsets.values() for x in pair] for r in rs}
        image_hashes={key:pixel_hash(row['image']) for key,row in unique.items()}
        for rs in [query]+[x for pair in subsets.values() for x in pair]:
            for row in rs:row.setdefault('metadata',{})['pixel_sha256']=image_hashes[row['sample_id']]
        for seed,(fit,cal) in subsets.items():
            validate_triplet(fit,cal,query)
            j=next(j for j in jobs if j['domain']==domain and j['support_seed']==seed)
            for name,rs in [('support',fit),('calibration',cal),('query',query)]:
                validate_counts(rs,j[{'support':'support_count','calibration':'calibration_count','query':'query_count'}[name]],cls)
                _save_rows(j[name],rs)
            record={'protocol':PROTOCOL,'domain':domain,'support_seed':seed,'source':source,
              'parent_query_ids_sha256':digest([r['sample_id'] for r in parent]),'subset_hash_seed':310929,
              'role_split_seed':310927,'content':{n:digest(rs) for n,rs in [('support',fit),('calibration',cal),('query',query)]},
              'isolation':'patient' if cls==2 else 'image_content_only; patient identities unavailable',
              'groups':{n:sorted({r['group_id'] for r in rs}) for n,rs in [('support',fit),('calibration',cal),('query',query)]}}
            write_locked(j['split_record'],record)
    return len(jobs)


def check_job(job, images=True, model=True):
    rows={k:resolved_manifest(job[k]) for k in ('support','calibration','query')}
    rec=json.loads(Path(job['split_record']).read_text())
    if rec['protocol']!=PROTOCOL or rec['domain']!=job['domain'] or rec['support_seed']!=job['support_seed']:
        raise ValueError('wrong split protocol/domain/seed')
    validate_triplet(rows['support'],rows['calibration'],rows['query'])
    for name,rs in rows.items():
        validate_counts(rs,job[name+'_count' if name!='support' else 'support_count'],job['classes'])
        if digest(rs)!=rec['content'][name]:raise ValueError('sealed manifest changed')
    if images:
        from aperture_medical.data import pixel_hash
        for r in {r['sample_id']:r for rs in rows.values() for r in rs}.values():
            if pixel_hash(r['image'])!=r['metadata']['pixel_sha256']:raise ValueError('image content changed')
    if model:check_snapshot(job)
    return rec


def check_snapshot(job):
    root=Path(job['model_path'])
    cfg=json.loads((root/'config.json').read_text())
    if cfg.get('model_type')!=job['model_type'] or cfg.get('quantization_config'):
        raise ValueError('wrong model family or quantized snapshot')
    text=cfg.get('text_config',cfg)
    if int(text.get('num_hidden_layers',0))<=27:raise ValueError('checkpoint lacks the fixed intervention layers')
    index=root/'model.safetensors.index.json'
    if index.is_file():files=sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:files=['model.safetensors']
    if any(not (root/f).is_file() or (root/f).stat().st_size==0 for f in files):
        raise FileNotFoundError('missing complete local safetensors snapshot')
    if not (root/'tokenizer_config.json').is_file():raise FileNotFoundError('tokenizer/chat-template metadata missing')
    return files


def source_signature():
    src=Path(__file__).resolve().parents[1]
    paths=[]
    for name in ('aperture_table64','aperture_medical','aperture_final','vigor_handoff','visual_lens','eventttt'):
        paths.extend((src/name).glob('*.py'))
    return digest([(str(p.relative_to(src)),file_hash(p)) for p in sorted(paths)])


def assets_identity(job):
    rec=check_job(job)
    root=Path(job['model_path']).resolve();weights=check_snapshot(job)
    # Include tokenizer models, native chat templates and processors, not only weights.
    extras=[p.name for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.jinja','.model','.txt')]
    return {'job':job,'source_signature':source_signature(),'split':rec,
            'model_root':str(root),'model_content':[(f,cached_hash(root/f)) for f in sorted(set(weights+extras))]}


def cached_hash(path):
    """Content hash cached by resolved path/stat; cache lives outside the snapshot."""
    import tempfile
    path=Path(path).resolve();stat=path.stat()
    key=digest([str(path),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns])
    cache=Path(tempfile.gettempdir())/'aperture_table64_hashes';cache.mkdir(exist_ok=True)
    f=cache/(key+'.json')
    if f.is_file():
        h=json.loads(f.read_text())['sha256']
        if re.fullmatch('[0-9a-f]{64}',h):return h
    h=file_hash(path);atomic_json(f,{'sha256':h});return h
