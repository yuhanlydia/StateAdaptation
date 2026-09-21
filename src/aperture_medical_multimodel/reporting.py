"""Medical metrics and complete-only, protocol-stratified paper exports."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, average_precision_score
from aperture_final.numeric import metrics as candidate_metrics
from vigor_handoff.protocol import atomic_json, read_jsonl, resolved_manifest, file_hash

METRICS=('nll','macro_f1','balanced_accuracy','auroc','average_precision','patient_macro_nll','brier','ece')
LABELS={'frozen':'Frozen','lora1':'LoRA-1pass','lora4':'LoRA-4passes','aperture':'Aperture','random':'Random visual'}


def medical_metrics(scores, labels, patient_ids, temperature=1., bias=None):
    m=candidate_metrics(scores,labels,temperature,bias=bias)
    s=np.asarray(scores,dtype=float);y=np.asarray(labels,dtype=int);g=np.asarray(patient_ids,dtype=str)
    if s.shape[1]!=2 or g.shape!=y.shape or any(not x for x in g):raise ValueError('binary task and one patient ID per patch required')
    if bias is not None:s=s+np.asarray(bias)
    margin=(s[:,1]-s[:,0])/temperature
    z=s/temperature;logp=z-logsumexp(z,axis=1,keepdims=True);loss=-logp[np.arange(len(y)),y]
    m.update(auroc=float(roc_auc_score(y,margin)) if len(set(y))==2 else None,
             average_precision=float(average_precision_score(y,margin)) if len(set(y))==2 else None,
             patient_macro_nll=float(np.mean([loss[g==p].mean() for p in set(g)])),
             patient_count=len(set(g)))
    return m


def aggregate(rows):
    groups=defaultdict(list);seen=set()
    for r in rows:
        key=tuple(r[k] for k in ('family','budget','hospital','method'));ident=key+(r['seed'],)
        if ident in seen:raise ValueError('duplicate seed result')
        seen.add(ident);groups[key].append(r)
    result=[]
    for key,rs in sorted(groups.items()):
        a=dict(zip(('family','budget','hospital','method'),key));a['seeds']=len(rs)
        for name in METRICS:
            v=[r.get(name) for r in rs]
            if any(x is None for x in v):a[name+'_mean']=None;a[name+'_sd']=None
            else:
                if not np.isfinite(v).all():raise ValueError('nonfinite metrics')
                a[name+'_mean']=float(np.mean(v));a[name+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
        result.append(a)
    return result


def tables(aggr,family,budget,metric):
    a=[r for r in aggr if r['family']==family and r['budget']==budget]
    hs=sorted({r['hospital'] for r in a});methods=sorted({r['method'] for r in a})
    text=['% Same fixed label budget. Mean +/- sample SD over support seeds.',
          r'\begin{tabular}{l'+('r'*(len(hs)+1))+'}',r'\toprule',
          'Method & '+' & '.join(f'H{h}' for h in hs)+r' & Center mean \\',r'\midrule']
    for method in methods:
        byh={r['hospital']:r for r in a if r['method']==method};values=[];cols=[]
        for h in hs:
            r=byh.get(h);v=r.get(metric+'_mean') if r else None;sd=r.get(metric+'_sd') if r else None
            cols.append('---' if v is None else f'${v:.3f}'+(f'_{{\\pm {sd:.3f}}}' if sd is not None else '')+'$')
            if v is not None:values.append(v)
        mean=f'{np.mean(values):.3f}' if len(values)==len(hs) else '---'
        text.append(method.replace('_',r'\_')+' & '+' & '.join(cols)+' & '+mean+r' \\')
    text.extend([r'\bottomrule',r'\end{tabular}'])
    return '\n'.join(text)+'\n'


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('');return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def export(jobs,root,phases,allow_partial=False):
    from .worker import verified_record
    jobs=[j for j in jobs if j['phase'] in phases]
    if not jobs:raise ValueError('no planned states in selected phases')
    out=Path(root)/'paper_exports'/('-'.join(sorted(phases)));out.mkdir(parents=True,exist_ok=True)
    missing=[];records={};qhash={};groups=defaultdict(dict);resources=[]
    for j in jobs:
        try:
            fit=Path(j['output'])/'fit';ev=Path(j['output'])/'evaluation'
            verified_record(fit);verified_record(ev)
            r=json.loads((ev/'evaluation.json').read_text())
            if any(r[k]!=j[k] for k in ('job_id','unit_id','arm')):raise ValueError('result/job identity differs')
            groups[j['unit_id']][j['arm']]=(j,r)
            preds=read_jsonl(ev/'clean.jsonl');manifest=resolved_manifest(j['query'])
            if [x['sample_id'] for x in preds]!=[x['sample_id'] for x in manifest]:raise ValueError('query ordering mismatch')
            records[j['job_id']]=(j,r,preds,manifest)
        except (OSError,ValueError,KeyError) as exc:missing.append({'job':j['job_id'],'error':str(exc)})
    coverage={'planned':len(jobs),'completed':len(records),'missing':missing,'collection_complete':not missing,'paper_ready':False}
    atomic_json(out/'coverage.json',coverage)
    if missing:
        if not allow_partial:raise RuntimeError(f'{len(missing)} missing/invalid states; no paper tables exported')
        return coverage
    rows=[];calibrations=[]
    for j,r,preds,manifest in records.values():
        ids=[x['sample_id'] for x in manifest];key=j['hospital']
        if key in qhash and qhash[key]!=ids:raise ValueError('query differs across budgets/backbones/seeds')
        qhash[key]=ids
        cal=json.loads((Path(j['output'])/'fit'/'calibration.json').read_text())
        s=[x['mean_log_scores'] for x in preds];y=[x['label_id'] for x in preds];g=[x['metadata']['patient'] for x in manifest]
        base={'family':j['family'],'budget':j['total_support'],'hospital':j['hospital'],'seed':j['support_seed']}
        for variant,temp,bias in [('raw',1.,None),('temperature',cal['temperature']['temperature'],None)]+([('bias',1.,cal['bias']['values'])] if j['arm']=='frozen' else []):
            name=LABELS[j['arm']]+(' + T' if variant=='temperature' else ' + all-support bias' if variant=='bias' else '')
            m=medical_metrics(s,y,g,temp,bias)
            if abs(m['nll']-r['clean'][{'raw':'raw','temperature':'temperature','bias':'candidate_bias'}[variant]]['nll'])>1e-8:raise ValueError('metric recomputation mismatch')
            rows.append({**base,'method':name,**{k:m[k] for k in METRICS},'query_count':len(y),'patients':len(set(g))})
        calibrations.append({**base,'arm':j['arm'],'temperature':cal['temperature']['temperature'],
                             'calibration_nll':cal['raw_calibration_nll'],'calibration_count':cal['temperature']['calibration_count']})
        st=json.loads((Path(j['output'])/'fit'/'metrics.json').read_text())
        resources.append({**base,'arm':j['arm'],**{k:st.get(k) for k in ('optimized_scalars','basis_scalars','serialized_state_bytes','fitting_visits','optimizer_updates','fit_wall_seconds','extraction_seconds')}})
    for unit,arms in groups.items():
        first=next(iter(arms.values()))[0]
        if set(arms)!=set(first['required_arms']):raise ValueError('incomplete comparison')
        for field,method in [('selected_lora_raw','LoRA-selected'),('selected_lora_temperature','LoRA-selected + T')]:
            chosen=arms['aperture'][1][field];j,r,pred,manifest=records[arms[chosen][0]['job_id']]
            base={'family':j['family'],'budget':j['total_support'],'hospital':j['hospital'],'seed':j['support_seed']}
            target=LABELS[chosen]+(' + T' if field.endswith('temperature') else '')
            matching=[x for x in rows if all(x[k]==v for k,v in base.items()) and x['method']==target]
            if len(matching)!=1:raise ValueError('selected LoRA row missing')
            rows.append({**matching[0],'method':method,'selected_arm':chosen})
    a=aggregate(rows);csv_write(out/'per_seed.csv',rows);csv_write(out/'mean_sd.csv',a)
    csv_write(out/'resources.csv',resources);csv_write(out/'calibration.csv',calibrations)
    for family,budget in sorted({(r['family'],r['budget']) for r in a}):
        for metric in METRICS:(out/f'{family}_n{budget}_{metric}.tex').write_text(tables(a,family,budget,metric))
    coverage['paper_ready']=True
    atomic_json(out/'coverage.json',coverage)
    atomic_json(out/'results.json',{'coverage':coverage,'rows':rows,'aggregate':a,
         'scope':'CAMELYON17-WILDS, 5 hospitals, patient-disjoint adaptation. Not official domain generalization or clinical diagnosis.'})
    return coverage
