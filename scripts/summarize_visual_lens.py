#!/usr/bin/env python3
"""Summarize sealed P0 artifacts and paired visual-dependence contrasts."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vigor_handoff.protocol import JobStore,atomic_json,read_jsonl
from vigor_handoff.statistics import paired_cluster_bootstrap
from visual_lens.controls import paired_did


def resolve_paths(config, plan=None, output=None):
    cfg=json.loads(Path(config).read_text())
    root=Path(cfg['run_root'])
    return Path(plan) if plan else root/'plan.json', Path(output) if output else root/'summary'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/visual_lens_p0.json')
    p.add_argument('--plan')
    p.add_argument('--output')
    p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    if a.bootstrap<1:p.error('--bootstrap must be positive')
    plan,out=resolve_paths(a.config,a.plan,a.output)
    jobs=json.loads(plan.read_text())['jobs'];out.mkdir(parents=True,exist_ok=True)
    complete=[];missing=[];errors=[]
    for job in jobs:
        root=Path(job['output']);identity=root/'identity.json'
        if not identity.exists():missing.append(job['job_id']);continue
        data=json.loads(identity.read_text());store=JobStore(root,data['identity'])
        if not store.complete():missing.append(job['job_id']);continue
        complete.append((job,json.loads((root/'metrics.json').read_text())))
    table=['# Visual Lens P0 results','',f'Completed {len(complete)}/{len(jobs)} planned jobs. Missing results are not zero.','',
           'Primary reproduction and objective controls are separated. Mean ± sample SD is support-seed variation, not a confidence interval.','',
           '| Stage | Backbone | Domain | Method | Seeds | Macro-F1 | NLL |','|---|---|---|---|---|---:|---:|']
    groups=defaultdict(list)
    for j,m in complete:
        if j['stage'] in ('matched','objective'):groups[(j['stage'],j['family'],j['domain'],j['arm'])].append((j,m))
    for key,items in sorted(groups.items()):
        def fmt(name):
            v=np.array([m[name] for j,m in items]);return f'{v.mean():.4f}'+(f' ± {v.std(ddof=1):.4f}' if len(v)>1 else '')
        table.append('| '+' | '.join(key)+f" | {','.join(str(j['support_seed']) for j,m in items)} | {fmt('macro_f1')} | {fmt('nll')} |")
    (out/'results.md').write_text('\n'.join(table)+'\n')
    matched={(j['family'],j['domain'],j['support_seed'],j['arm']):j for j,m in complete if j['stage']=='matched'}
    paired=[]
    for key,j in matched.items():
        if j['arm']!='ours':continue
        for arm in ('frozen','lora','lora_passes4','random_kv'):
            ref=matched.get((*key[:-1],arm))
            if ref is None:continue
            try:
                left=read_jsonl(Path(ref['output'])/'predictions.jsonl');right=read_jsonl(Path(j['output'])/'predictions.jsonl')
                stat=paired_cluster_bootstrap(left,right,len(left[0]['probabilities']),a.bootstrap)
                paired.append({'family':key[0],'domain':key[1],'support_seed':key[2],'comparison':'lens-minus-'+arm,**stat})
            except ValueError as exc:errors.append({'comparison':key,'error':str(exc)})
    ev={(j['family'],j['domain'],j['support_seed'],j['base_job']['arm']):j for j,m in complete if j['stage']=='evidence'}
    effects=[]
    for key,j in ev.items():
        frozen=ev.get((*key[:-1],'frozen'))
        if frozen is None:continue
        jp=Path(j['output']);fp=Path(frozen['output'])
        fr=read_jsonl(fp/'real/predictions.jsonl')
        mr=read_jsonl(jp/'real/predictions.jsonl')
        for condition in j['conditions']:
            if condition=='real':continue
            fc=read_jsonl(fp/condition/'predictions.jsonl')
            mc=read_jsonl(jp/condition/'predictions.jsonl')
            if key[-1]!='frozen':
                effects.append({'family':key[0],'domain':key[1],'support_seed':key[2],'method':key[-1],
                               'control':condition,**paired_did(fr,mr,fc,mc,a.bootstrap)})
            else:
                br=read_jsonl(fp/'real/bias_predictions.jsonl');bc=read_jsonl(fp/condition/'bias_predictions.jsonl')
                effects.append({'family':key[0],'domain':key[1],'support_seed':key[2],'method':'candidate_bias',
                               'control':condition,**paired_did(fr,br,fc,bc,a.bootstrap)})
    atomic_json(out/'summary.json',{'completed':len(complete),'expected':len(jobs),'missing':missing,
                'paired_main_intervals':paired,'visual_dependence':effects,'errors':errors,
                'uncertainty':'Query-cluster intervals conditional on each fitted support state; seeds are not pooled as independent query observations.'})
    e=['# Visual dependence','', '| Backbone/domain | Method | Control | D macro-F1 | Paired 95% CI |', '|---|---|---|---:|---|']
    for row in effects:
        stat=row['macro_f1'];e.append(f"| {row['family']}/{row['domain']} | {row['method']} | {row['control']} | {stat['difference_in_differences']:+.4f} | {stat['ci95']} |")
    (out/'visual_dependence.md').write_text('\n'.join(e)+'\n')
    (out/'coverage.md').write_text(f'# Coverage\n\nComplete {len(complete)}/{len(jobs)}\n\n'+'\n'.join('- MISSING '+s for s in missing)+'\n')
    print(f'Summary written to {out}; {len(complete)}/{len(jobs)} sealed jobs. No fabricated results.')
if __name__=='__main__':main()
