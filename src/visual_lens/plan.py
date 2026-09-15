"""Fixed, outcome-independent P0 matrix with explicit dependency gates."""
from copy import deepcopy
from pathlib import Path
from vigor_handoff.protocol import expand_plan,digest

STAGES=('audit','matched','objective','evidence')


def build_plan(config):
    datasets=[]
    for e in config['bright_events']:
        path=str(Path(config['bright_root'])/e/'seed_{seed}')
        datasets.append({'name':'bright','domain':e,'kind':'paired','support':path+'/support.jsonl',
          'query':path+'/query.jsonl','seeds':config['bright_seeds'],'families':config['bright_families'],
          'group_key':'tile_id','support_per_class':8,'query_count':300,'query_per_class':100})
    family=config['medical_family']
    cp=str(Path(config['camelyon_root'])/'seed_{seed}')
    datasets.append({'name':'camelyon17','domain':'hospital_2','kind':'single','support':cp+'/support.jsonl',
       'query':cp+'/query.jsonl','seeds':[0],'families':[family],'group_key':'group_id',
       'support_per_class':8,'query_count':300})
    gp=str(Path(config['guardian_root'])/'robofail'/'seed_{seed}')
    guardian={'name':'guardian','domain':'robofail','kind':'single','support':gp+'/support.jsonl',
       'query':gp+'/query.jsonl','seeds':[0],'families':[family],'group_key':'group_id',
       'support_per_class':8,'overrides':{'steps':8,'lr':.01}}
    temp={'models':config['models'],'datasets':datasets+[guardian],'run_root':config['run_root']}
    base_jobs=expand_plan(temp,['main'])
    jobs=[]
    def emit(base,stage,arm=None,**changes):
        j=deepcopy(base);j.pop('job_id',None);j.pop('output',None)
        j.update(changes);j['stage']=stage;j['arm']=arm or base['arm']
        j['protocol_version']='visual_lens_p0_v1';j.setdefault('loss_span','full')
        j['target_split_status']=config['target_status']
        j['phase']='main' if stage=='matched' else stage
        j['evidence']='primary_reproduction' if stage=='matched' else 'support_only_audit' if stage=='audit' else 'supplementary_reproduction'
        slug=f"{j['family']}--{j['domain']}--s{j['support_seed']}--{j['arm']}--{digest(j)[:12]}"
        j['job_id']=slug;j['output']=str(Path(config['run_root'])/stage/slug)
        jobs.append(j);return j
    # Gate each implementation family, plus both distinct task loss recipes.
    gates={}
    for b in base_jobs:
        if b['arm']!='ours' or b['support_seed']!=0:continue
        if b['dataset']=='bright' and b['domain']!=config['bright_events'][0]:continue
        gate=emit(b,'audit','gradient_path_audit',engine='audit',
                  audit_per_class=config['audit_per_class'],audit_abs_tol=config['audit_abs_tol'],
                  audit_relative_tol=config['audit_relative_tol'])
        gates[(b['family'],b['domain'])]=gate
    def gate_for(b):
        return gates.get((b['family'],b['domain'])) or next(g for (f,d),g in gates.items() if f==b['family'])
    matched=[]
    for b in base_jobs:
        if b['dataset']=='guardian':continue
        if b['dataset']=='camelyon17' and b['arm']=='random_kv':continue
        matched.append(emit(b,'matched',engine='model',required_audit=gate_for(b)))
        if b['dataset']=='bright' and b['arm']=='lora':
            matched.append(emit(b,'matched','lora_passes4',engine='model',lora_passes=4,required_audit=gate_for(b)))
    for b in base_jobs:
        if b['arm']!='ours' or b['dataset']=='bright':continue
        for reduction in ('sum','mean'):
            for span in ('full','label'):
                emit(b,'objective',f'lens_{reduction}_{span}',engine='model',
                     loss_reduction=reduction,loss_span=span,required_audit=gate_for(b))
    for b in matched:
        if b['support_seed'] not in config['evidence_seeds']:continue
        if b['dataset']=='bright':
            if b['domain'] not in config['evidence_events'] or b['arm'] not in ('frozen','lora_passes4','ours'):continue
            conditions=['real','shuffle','neutral','shuffle_post','neutral_post']
        else:
            if b['arm'] not in ('frozen','lora','ours'):continue
            conditions=['real','shuffle','neutral']
        emit(b,'evidence',engine='evidence',base_job=b,conditions=conditions,
             shuffle_seed=config['shuffle_seed'],bias_l2=config['bias_l2'])
    if len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('duplicate P0 jobs')
    return jobs
