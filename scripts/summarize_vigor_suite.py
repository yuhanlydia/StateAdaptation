#!/usr/bin/env python3
"""Summarize only sealed, validated jobs; oracle rows never enter the main table."""
from pathlib import Path
import argparse
import json
from collections import defaultdict
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from vigor_handoff.protocol import JobStore,atomic_json,read_jsonl
from vigor_handoff.statistics import paired_cluster_bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',default='runs/vigor_handoff_v1/plan/plan.json')
    p.add_argument('--output',default='runs/vigor_handoff_v1/summary')
    p.add_argument('--bootstrap',type=int,default=0,help='optional paired cluster draws within each support seed')
    a=p.parse_args();jobs=json.loads(Path(a.plan).read_text())['jobs'];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];missing=[];pairs=[]
    for j in jobs:
        path=Path(j['output']);identity_path=path/'identity.json'
        if not identity_path.exists():missing.append(j['job_id']);continue
        saved=json.loads(identity_path.read_text())
        store=JobStore(path,saved['identity'])
        if not store.complete():missing.append(j['job_id']);continue
        metrics=json.loads((path/'metrics.json').read_text())
        rows.append({**j,'metrics':metrics})
    groups=defaultdict(list)
    for r in rows:groups[(r['phase'],r['family'],r['dataset'],r['domain'],r['arm'])].append(r)
    for phase in sorted(set(j['phase'] for j in jobs)):
        text=['# '+phase.replace('_',' ').title()+' results','',
              'Only sealed complete jobs are included. Missing jobs are not treated as zeros.',
              'Mean ± sample SD describes support seeds; it is not a confidence interval.','',
              '| Backbone | Dataset/domain | Arm | Seeds | Macro-F1 | NLL |','|---|---|---|---|---|---|']
        for key,rs in sorted(groups.items()):
            ph,family,dataset,domain,arm=key
            if ph!=phase:continue
            def fmt(metric):
                vals=[r['metrics'].get(metric) for r in rs]
                vals=[v for v in vals if isinstance(v,(float,int))]
                if not vals:return 'not available'
                return f'{np.mean(vals):.4f}'+(f' ± {np.std(vals,ddof=1):.4f}' if len(vals)>1 else ' (one seed)')
            text.append(f"| {family} | {dataset}/{domain} | {arm} | {','.join(str(r['support_seed']) for r in rs)} | {fmt('macro_f1')} | {fmt('nll')} |")
        (out/f'{phase}.md').write_text('\n'.join(text)+'\n')
    if a.bootstrap:
        indexed={(r['family'],r['dataset'],r['domain'],r['support_seed'],r['arm']):r for r in rows if r['phase']=='main'}
        for key,r in indexed.items():
            if key[-1]!='ours':continue
            for base in ('frozen','lora','random_kv'):
                other=indexed.get((*key[:-1],base))
                if other is None:continue
                # Same split files are mandatory, not just matching reported sample count.
                if r['support']!=other['support'] or r['query']!=other['query']:continue
                pred1=read_jsonl(Path(other['output'])/'predictions.jsonl');pred2=read_jsonl(Path(r['output'])/'predictions.jsonl')
                nclass=len(pred1[0]['probabilities'])
                try:stats=paired_cluster_bootstrap(pred1,pred2,nclass,a.bootstrap)
                except ValueError as exc:stats={'not_estimated':str(exc)}
                pairs.append({'family':key[0],'dataset':key[1],'domain':key[2],'support_seed':key[3],'comparison':f'ours-minus-{base}',**stats})
    atomic_json(out/'summary.json',{'completed':len(rows),'expected':len(jobs),'missing':missing,'rows':rows,'paired_cluster_intervals':pairs})
    (out/'coverage.md').write_text(f'# Coverage\n\nCompleted: {len(rows)}/{len(jobs)}.\n\n'+'\n'.join('- MISSING '+m for m in missing)+'\n')
    print(f'Wrote {len(rows)}/{len(jobs)} verified rows to {out}; oracle diagnostics remain separate.')

if __name__=='__main__':main()
