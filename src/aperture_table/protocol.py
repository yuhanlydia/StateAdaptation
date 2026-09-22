"""Fixed 64-state design; derived query lists never alter the original medical study."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from vigor_handoff.protocol import digest, file_hash, atomic_json, resolved_manifest, validate_split

MODELS = {
 'q25': ('Qwen/Qwen2.5-VL-7B-Instruct','qwen2_5_vl','Qwen2_5_VLForConditionalGeneration'),
 'q34': ('Qwen/Qwen3-VL-4B-Instruct','qwen3_vl','Qwen3VLForConditionalGeneration'),
 'q38': ('Qwen/Qwen3-VL-8B-Instruct','qwen3_vl','Qwen3VLForConditionalGeneration'),
 'g34': ('google/gemma-3-4b-it','gemma3','Gemma3ForConditionalGeneration'),
}
DOMAINS = {'h2': ('hospital_2',2,200), 'path': ('pathmnist_crc',9,180)}
ARMS = ('frozen','lora1','lora4','aperture')
PROTOCOL = 'aperture_table_v1'


def default_config():
 return {'protocol':PROTOCOL,'run_root':'runs/aperture_table_v1',
         'prepared_root':'data/prepared/aperture_table_v1',
         'source_config':'configs/aperture_medical_v1.json',
         'support_seeds':[0,1], 'models':[
          {'key':k,'id':v[0],'family':v[1],'path':'artifacts/models/'+v[0].split('/')[-1]}
          for k,v in MODELS.items()]}


def write_locked(path,value):
 p=Path(path)
 if p.exists():
  if json.loads(p.read_text()) != value:raise ValueError(f'locked content changed: {p}; use a new versioned root')
 else:atomic_json(p,value)


def source_signature():
 root=Path(__file__).resolve().parents[1]
 files=[]
 for pkg in ('aperture_table','aperture_medical','aperture_final','vigor_handoff','visual_lens','eventttt'):
  files+=list((root/pkg).glob('*.py'))
 return digest([(str(p.relative_to(root)),file_hash(p)) for p in sorted(files)])


def expected_cells():
 return {f'{m}-{d}-{a}-{metric}' for m in MODELS for d in DOMAINS
         for a in (*ARMS,'bias') for metric in ('f1','nll','nllt')
         if not (a=='bias' and metric=='nllt')}


def make_jobs(config):
 c=deepcopy(config)
 if c.get('protocol')!=PROTOCOL or c.get('support_seeds')!=[0,1] or any(type(x)!=int for x in c.get('support_seeds',[])):raise ValueError('the table requires fixed two-support-seed protocol')
 if [m['key'] for m in c['models']]!=list(MODELS):raise ValueError('all four prescribed backbones are required')
 if Path(c['run_root']).name!=PROTOCOL or Path(c['prepared_root']).name!=PROTOCOL:raise ValueError('use isolated aperture_table_v1 roots')
 jobs=[]
 for model in c['models']:
  key=model['key'];ident,family,_=MODELS[key]
  if model['id']!=ident or model['family']!=family:raise ValueError('checkpoint/family replacement is not allowed')
  for dk,(domain,classes,nq) in DOMAINS.items():
   for seed in c['support_seeds']:
    unit=f'{key}--{dk}--s{seed}';folder=Path(c['prepared_root'])/dk/f'seed_{seed}'
    for arm in ARMS:
     jid=f'{unit}--{arm}'
     jobs.append(dict(protocol=PROTOCOL,job_id=jid,unit_id=unit,model_key=key,model_id=ident,
        family=family,model_path=str(Path(model['path']).resolve()),domain_key=dk,domain=domain,
        classes=classes,query_count=nq,fit_count=classes*6,cal_count=classes*2,
        total_labels=classes*8,support_seed=seed,arm=arm,
        support=str(folder/'fit.jsonl'),calibration=str(folder/'calibration.jsonl'),
        query=str(Path(c['prepared_root'])/dk/'query.jsonl'),split_record=str(folder/'split.json'),
        output=str(Path(c['run_root'])/'states'/jid),rank=16,layers=[14,27],alpha=3.,steps=4,
        lr=.05,l2=.001,loss_reduction='sum',lora_rank=16,lora_alpha=32,lora_dropout=.05,
        lora_lr=2e-4,lora_accumulation=1,lora_passes=4 if arm=='lora4' else 1,optimization_seed=0,
        temperature_penalty=.01,bias_l2=.001,image_size=896 if family=='gemma3' else 448,
        attention='sdpa_math',weight_dtype='bfloat16'))
 return jobs


def select_query(rows,classes,per_class):
 out=[]
 if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('duplicate source query IDs')
 for c in range(classes):
  eligible=sorted([r for r in rows if r['label_id']==c],key=lambda r:hashlib.sha256(('aperture-table-query-v1|'+r['sample_id']).encode()).hexdigest())
  if len(eligible)<per_class:raise ValueError('insufficient locked source query pool')
  out+=deepcopy(eligible[:per_class])
 return sorted(out,key=lambda r:r['sample_id'])


def validate_triplet(fit,cal,query,classes):
 validate_split(fit,cal,'group_id');validate_split(fit+cal,query,'group_id')
 seen=[]
 for rows in (fit,cal,query):
  if any(len(r['candidate_labels'])!=classes for r in rows):raise ValueError('wrong candidate count')
  h=[r.get('metadata',{}).get('pixel_sha256') for r in rows]
  if not all(h) or len(h)!=len(set(h)):raise ValueError('missing or duplicate image-content identity')
  seen.append(set(h))
 if any(seen[i]&seen[j] for i in range(3) for j in range(i)):raise ValueError('image-content leakage across roles')


def _save_rows(path,rows):
 p=Path(path);text=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows)
 if p.exists():
  if p.read_text()!=text:raise ValueError(f'prepared manifest changed: {p}')
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)


def prepare(config, *, prepare_source=True):
 make_jobs(config)  # Fail invalid registries before touching data.
 source=json.loads(Path(config['source_config']).read_text())
 if source.get('protocol')!='aperture_medical_v1' or source['split_seed']!=310927 or source['calibration_per_class']!=2:
  raise ValueError('expected the original medical_v1 partition contract')
 if prepare_source:
  from aperture_medical.protocol import prepare as source_prepare
  source_prepare(source,domains=[x[0] for x in DOMAINS.values()])
 # Do not rerun any historical model states. Only reuse the prescribed source partitions.
 for dk,(name,nc,nq) in DOMAINS.items():
  source_query=Path(source['prepared_root'])/name/'query.jsonl'
  full_query=resolved_manifest(source_query);query=select_query(full_query,nc,nq//nc)
  for seed in [0,1]:
   sf=Path(source['prepared_root'])/name/f'n{nc*8}'/f'seed_{seed}'
   fit=resolved_manifest(sf/'fit.jsonl');cal=resolved_manifest(sf/'calibration.jsonl')
   if Counter(r['label_id'] for r in fit)!=Counter({i:6 for i in range(nc)}):raise ValueError('source fit budget mismatch')
   if Counter(r['label_id'] for r in cal)!=Counter({i:2 for i in range(nc)}):raise ValueError('source calibration budget mismatch')
   validate_triplet(fit,cal,query,nc)
   folder=Path(config['prepared_root'])/dk/f'seed_{seed}'
   record={'protocol':PROTOCOL,'domain':dk,'support_seed':seed,
           'source_files':{str(p.resolve()):file_hash(p) for p in [sf/'fit.jsonl',sf/'calibration.jsonl',source_query,sf/'split.json']},
           'content':{k:digest(v) for k,v in [('support',fit),('calibration',cal),('query',query)]},
           'isolation':'patient' if dk=='h2' else 'image_content_only',
           'query_count':nq,'query_ids':[r['sample_id'] for r in query]}
   write_locked(folder/'split.json',record)
   for p,rs in [(folder/'fit.jsonl',fit),(folder/'calibration.jsonl',cal),(folder.parent/'query.jsonl',query)]:_save_rows(p,rs)
 return make_jobs(config)


def read_roles(job,images=False):
 from PIL import Image
 import numpy as np
 record=json.loads(Path(job['split_record']).read_text())
 if record['protocol']!=PROTOCOL or record['domain']!=job['domain_key'] or record['support_seed']!=job['support_seed']:
  raise ValueError('wrong split record')
 rows={key:resolved_manifest(job[key]) for key in ('support','calibration','query')}
 validate_triplet(rows['support'],rows['calibration'],rows['query'],job['classes'])
 for key,count in [('support',job['fit_count']),('calibration',job['cal_count']),('query',job['query_count'])]:
  rs=rows[key]
  if len(rs)!=count or digest(rs)!=record['content'][key]:raise ValueError('prepared rows/count changed')
  if Counter(r['label_id'] for r in rs)!=Counter({i:count//job['classes'] for i in range(job['classes'])}):raise ValueError('class budget mismatch')
  if images:
   for r in rs:
    with Image.open(r['image']) as im:a=np.asarray(im.convert('RGB'))
    h=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
    if h!=r['metadata']['pixel_sha256']:raise ValueError('image bytes changed')
 return rows
