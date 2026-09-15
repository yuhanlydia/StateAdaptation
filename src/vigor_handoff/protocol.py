"""Planning, provenance, leakage checks and safe completion (no GPU imports)."""
from __future__ import annotations
from contextlib import contextmanager
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

BASE_COMMIT='3150cae1eecf2ee82e508d32fe7f2e80913f8149'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def read_jsonl(path,repair_tail=False):
    path=Path(path); data=path.read_bytes();lines=data.splitlines(keepends=True)
    rows=[];valid_end=0
    for n,line in enumerate(lines):
        if not line.strip():valid_end+=len(line);continue
        try:row=json.loads(line)
        except (ValueError,UnicodeDecodeError) as exc:
            if repair_tail and n==len(lines)-1 and not line.endswith(b'\n'):
                with path.open('r+b') as f:f.truncate(valid_end)
                break
            raise ValueError(f'invalid JSONL {path}:{n+1}') from exc
        if not isinstance(row,dict):raise ValueError('JSONL rows must be objects')
        rows.append(row);valid_end+=len(line)
    return rows


def validate_split(support,query,group_key=None):
    if not support or not query:raise ValueError('support/query cannot be empty')
    for name,rows in [('support',support),('query',query)]:
        ids=[r['sample_id'] for r in rows]
        if len(set(ids))!=len(ids):raise ValueError(f'duplicate {name} sample IDs')
    if {r['sample_id'] for r in support}&{r['sample_id'] for r in query}:
        raise ValueError('support/query sample overlap')
    group_key=group_key or ('tile_id' if 'tile_id' in support[0] else 'group_id')
    if any(not r.get(group_key) for r in support+query):
        raise ValueError(f'missing group identity {group_key}; never substitute sample ID')
    groups=lambda rows:{(r.get('event_id',r.get('domain_id','')),str(r[group_key])) for r in rows}
    if groups(support)&groups(query):raise ValueError('support/query group overlap')
    images=lambda rows:{r[k] for r in rows for k in ('image','pre_image','post_image') if r.get(k)}
    if images(support)&images(query):raise ValueError('support/query image overlap')
    labels=tuple(support[0].get('candidate_labels',('intact','damaged','destroyed')))
    for r in support+query:
        if tuple(r.get('candidate_labels',labels))!=labels:raise ValueError('candidate order mismatch')
        if r.get('label')!=labels[r['label_id']]:raise ValueError('label/index mismatch')
    if set(r['label_id'] for r in support)!=set(range(len(labels))):
        raise ValueError('support does not cover every class')
    return {'support':len(support),'query':len(query),'classes':list(labels),
            'support_counts':dict(Counter(r['label'] for r in support)),
            'query_counts':dict(Counter(r['label'] for r in query)),'group_key':group_key}


def validate_predictions(predictions,query,complete=True):
    expected={r['sample_id']:r for r in query};seen=set()
    for p in predictions:
        sid=p['sample_id']
        if sid in seen or sid not in expected:raise ValueError('prediction coverage/duplicate error')
        seen.add(sid)
        if p['label_id']!=expected[sid]['label_id']:raise ValueError('prediction label mismatch')
        probs=p['probabilities']
        n=len(expected[sid].get('candidate_labels',('intact','damaged','destroyed')))
        if len(probs)!=n or any(not math.isfinite(v) or v<0 or v>1 for v in probs) or abs(sum(probs)-1)>1e-4:
            raise ValueError('invalid prediction probabilities')
    if complete and seen!=set(expected):raise ValueError('incomplete prediction coverage')


class JobStore:
    def __init__(self,path,identity):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.identity=identity;self.fingerprint=digest(identity)
        config=self.path/'identity.json'
        if config.exists():
            old=json.loads(config.read_text())
            if old.get('fingerprint')!=self.fingerprint:
                raise ValueError(f'fingerprint mismatch in {self.path}; use a new run root')
        else:atomic_json(config,{'fingerprint':self.fingerprint,'identity':identity})

    @contextmanager
    def lock(self):
        p=self.path/'RUNNING.lock'
        try:fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as exc:raise RuntimeError(f'active/stale lock at {p}; inspect PID before removal') from exc
        try:
            os.write(fd,json.dumps({'pid':os.getpid(),'host':socket.gethostname(),'time':time.time()}).encode());os.close(fd)
            yield
        finally:p.unlink(missing_ok=True)

    def complete(self):
        try:
            done=json.loads((self.path/'DONE.json').read_text())
            return done['fingerprint']==self.fingerprint and all(
                (self.path/name).is_file() and file_hash(self.path/name)==h
                for name,h in done['artifacts'].items())
        except (OSError,ValueError,KeyError):return False

    def finish(self,metrics,extra_files=()):
        # Reject NaN/Inf at JSON serialization rather than declaring success.
        atomic_json(self.path/'metrics.json',metrics)
        names=['metrics.json']+list(extra_files)
        atomic_json(self.path/'DONE.json',{'fingerprint':self.fingerprint,
                    'artifacts':{name:file_hash(self.path/name) for name in names},'completed_utc':time.time()})


def resolved_manifest(path):
    path=Path(path).resolve();rows=read_jsonl(path)
    for row in rows:
        for key in ('image','pre_image','post_image','mask_path'):
            if row.get(key):
                p=Path(row[key]);row[key]=str(p if p.is_absolute() else (path.parent/p).resolve())
    return rows


def preflight_job(job,check_images=True):
    support=resolved_manifest(job['support']);query=resolved_manifest(job['query'])
    audit=validate_split(support,query,job.get('group_key'))
    spc=job.get('support_per_class')
    if spc and any(v!=spc for v in audit['support_counts'].values()):
        raise ValueError(f"support budget mismatch: {audit['support_counts']}, expected {spc}/class")
    qc=job.get('query_count')
    if qc and audit['query']!=qc:raise ValueError(f"query count {audit['query']} != {qc}")
    if job.get('query_per_class') and any(v!=job['query_per_class'] for v in audit['query_counts'].values()):
        raise ValueError('query class-count mismatch')
    if check_images:
        paths={r[k] for r in support+query for k in ('image','pre_image','post_image') if r.get(k)}
        missing=[p for p in paths if not Path(p).is_file()]
        if missing:raise FileNotFoundError(f'{len(missing)} missing images; first: {missing[:3]}')
    model=Path(job['model_path'])
    if not model.is_dir() or not (model/'config.json').is_file():
        raise FileNotFoundError(f'local model snapshot not found: {model}; set models[].path in config')
    if not list(model.glob('*.safetensors')) and not list(model.glob('*.bin')):
        raise FileNotFoundError(f'no model weights at {model}')
    return audit


def expand_plan(config,phases):
    """No result-dependent decisions; all variants are specified before execution."""
    result=[]
    valid={'main','ablation','diagnostic','matched_lora','support'}
    if set(phases)-valid:raise ValueError(f'unknown phases {set(phases)-valid}')
    for model in config['models']:
        for d in config['datasets']:
            if model['family'] not in d.get('families',[m['family'] for m in config['models']]):continue
            for seed in d.get('seeds',[0]):
                kind=d['kind']; name=d['name']
                base={'dataset':name,'domain':d['domain'],'kind':kind,'family':model['family'],
                      'model_path':model['path'],'model_id':model['id'],
                      'support_seed':seed,'optimization_seed':0,'basis_seed':0,
                      'support':d['support'].format(seed=seed),'query':d['query'].format(seed=seed),
                      'group_key':d.get('group_key','tile_id' if kind=='paired' else 'group_id'),
                      'support_per_class':d.get('support_per_class'),'query_count':d.get('query_count'),
                      'query_per_class':d.get('query_per_class'),
                      'mask':'post' if kind=='paired' else 'all_visual',
                      'rank':16,'alpha':3.,'steps':4,'lr':.05,'l2':.001,
                      'layers':[14,27],'sites':['K','V'],'coefficient_mode':'full',
                      'basis_mode':'covariance','basis_source':'support','crop_size':448,
                      'loss_reduction':'mean' if kind=='paired' else 'sum',
                      'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,
                      'lora_lr':1e-4 if model['family']=='qwen2' and kind=='paired' else 2e-4,
                      'lora_passes':1 if kind=='paired' else 4,
                      'lora_accumulation':3 if kind=='paired' else 1,
                      'protocol_version':'vigor_handoff_v1','target_split_status':d.get('status','previously_inspected')}
                base.update(d.get('overrides',{}))
                def add(phase,arm,**changes):
                    j=deepcopy(base);j.update(changes)
                    j.update(phase=phase,arm=arm,evidence=('oracle_diagnostic' if phase=='diagnostic'
                               else 'primary_reproduction' if phase=='main' else 'supplementary_reproduction'))
                    ident=digest(j)[:12]
                    slug=f"{model['family']}--{name}--{d['domain']}--s{seed}--{arm}--{ident}"
                    j['job_id']=slug;j['output']=str(Path(config.get('run_root','runs/vigor_handoff_v1'))/phase/slug)
                    result.append(j)
                if 'main' in phases:
                    for arm in ['frozen','lora','random_kv','ours']:
                        add('main',arm,basis_mode='random' if arm=='random_kv' else 'covariance')
                allowed=model['family'] in config.get('ablation_families',['qwen2'])
                allowed=allowed and name in config.get('ablation_datasets',['bright'])
                allowed=allowed and seed in config.get('ablation_seeds',[0])
                if 'ablation' in phases and allowed:
                    for mode in ['centered','activation_pca']:
                        add('ablation',f'basis_{mode}',basis_mode=mode)
                    add('ablation','basis_mean_rank1',basis_mode='mean',rank=1)
                    add('ablation','covariance_rank1',rank=1)
                    add('ablation','random_rank1',basis_mode='random',rank=1)
                    for bs in [1,2]:add('ablation',f'random_basis_seed{bs}',basis_mode='random',basis_seed=bs)
                    add('ablation','basis_shuffled',shuffle_labels=True)
                    add('ablation','diagonal',coefficient_mode='diagonal')
                    add('ablation','zero',steps=0)
                    add('ablation','hard_projection',steps=0,hard_projection=True)
                    for rank in [5,8,32]:add('ablation',f'rank{rank}',rank=rank)
                    for alpha in [.5,1.,2.]:add('ablation',f'alpha{alpha:g}',alpha=alpha)
                    for steps in [1,2,8]:add('ablation',f'updates{steps}',steps=steps)
                    for sites in [['K'],['V']]:add('ablation',f'site_{sites[0]}',sites=sites)
                    for layers in [[14],[27]]:add('ablation',f'layer{layers[0]}',layers=layers)
                    for mask in (['pre','all_visual','text'] if kind=='paired' else ['text']):
                        add('ablation',f'mask_{mask}',mask=mask)
                    add('ablation','hidden_same_rank',sites=['H'])
                    add('ablation','no_l2',l2=0.)
                    add('ablation','loss_scale_control',loss_reduction='sum' if base['loss_reduction']=='mean' else 'mean')
                    add('ablation','lora_support_passes4',lora_passes=4)
                if 'matched_lora' in phases and seed!=0 and kind=='paired':
                    add('matched_lora','lora')
                if 'support' in phases and allowed:
                    for n in config.get('support_ablation_per_class',[4,16]):
                        # A larger budget MUST have its own prespecified manifest; no invention.
                        for pair in d.get('support_ablation',[]):
                            if pair['per_class']==n:
                                add('support',f'support{n*3}',support=pair['support'].format(seed=seed),
                                    query=pair['query'].format(seed=seed),support_per_class=n)
                diag_allowed=(model['family'] in config.get('diagnostic_families',['internvl3'])
                              and d.get('diagnostics',False))
                if 'diagnostic' in phases and diag_allowed:
                    add('diagnostic','directional_geometry',engine='legacy_directional_geometry')
                    add('diagnostic','query_basis',basis_source='query')
                    for site in ['Q','O','QKVO']:
                        add('diagnostic',f'site_{site}',sites=list(site))
    if len({j['job_id'] for j in result})!=len(result):raise ValueError('duplicate job IDs')
    return result
