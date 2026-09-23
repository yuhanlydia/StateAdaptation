"""Targeted support-only debugging, separate from the sealed table and first search."""
from copy import deepcopy
import json
from pathlib import Path
from aperture_table.protocol import make_jobs, source_signature, write_locked
from vigor_handoff.protocol import digest, file_hash

PROTOCOL = 'aperture_debug_v2'

def signature():
    return digest({'table':source_signature(), 'debug':[(p.name,file_hash(p)) for p in sorted(Path(__file__).parent.glob('*.py'))]})

def build(base, design, root):
    from .folds import build_folds
    root=Path(root).resolve(); jobs=[]; finals=[]; audits={}; units={}
    sig=signature()
    refs={j['unit_id']:j for j in make_jobs(base) if j['arm']=='aperture'}
    for target in design['targets']:
        for seed in (0,1):
            unit=f"{target['model']}--{target['domain']}--s{seed}"
            ref=refs[unit]
            fold_overrides=build_folds(ref,root/'folds'/unit) if target['domain']=='h2' else [{}]
            units[unit]={'fold_count':len(fold_overrides),'selection':'pooled out-of-fold Macro-F1 then NLL' if len(fold_overrides)>1 else 'calibration Macro-F1 then NLL'}
            for method, options in target['candidates'].items():
                if method not in ('aperture','lora') or not options:raise ValueError('invalid method registry')
                if len({x['id'] for x in options})!=len(options):raise ValueError('duplicate candidate')
                for option in options:
                    allowed={'layers','alpha','lr','steps','l2','rank'} if method=='aperture' else {'lora_lr','lora_passes'}
                    if set(option['parameters'])-allowed:raise ValueError('only registered hyperparameters may be overridden')
                    template=deepcopy(ref);template.update(option['parameters'])
                    template.update(arm='aperture' if method=='aperture' else 'lora4',method=method,candidate=option['id'],debug_protocol=PROTOCOL,debug_source=sig)
                    arch={k:template[k] for k in ('model_key','domain_key','layers','rank','alpha','family','image_size','attention','weight_dtype')}
                    template['context']=str(root/'contexts'/digest(arch)[:16])
                    for fold,overrides in enumerate(fold_overrides):
                        job={**template,**overrides,'fold':fold,'role':'search'}
                        job['job_id']=f"{unit}--{method}--{option['id']}--fold{fold}"
                        job['output']=str(root/'search'/job['job_id'])
                        jobs.append(job);audits.setdefault(job['context'],job)
                    final={**template,'role':'final'}
                    final['job_id']=f"{unit}--{method}--{option['id']}--final"
                    final['output']=str(root/'final'/final['job_id']);finals.append(final)
    if len(units)!=2*len(design['targets']):raise ValueError('duplicate target')
    return {'protocol':PROTOCOL,'debug_source':sig,'design':design,'units':units,'jobs':jobs,'finals':finals,'audits':list(audits.values()),'query_status':'Previously inspected query; targeted adaptive exploratory debugging, not independent test evidence.'}

def read_plan(root):
    plan=json.loads((Path(root)/'plan.json').read_text())
    if plan['debug_source']!=signature():raise ValueError('debug source changed; use a new versioned run')
    return plan

def repeat_job(job):
    result=deepcopy(job);result['job_id']+='--repeat';result['output']+='--repeat';result['role']=job['role']
    return result
