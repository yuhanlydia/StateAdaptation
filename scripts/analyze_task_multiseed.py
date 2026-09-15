#!/usr/bin/env python3
"""Summarize paired task predictions across fixed-query support seeds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-template',help='path containing {seed}'); p.add_argument('--run',action='append',help='explicit seed=run_dir (repeatable)'); p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]); p.add_argument('--output',required=True); a=p.parse_args()
 if not a.run_template and not a.run: p.error('one of --run-template or --run is required')
 paths=[]
 if a.run:
  for item in a.run:
   seed,path=item.split('=',1); paths.append((int(seed),Path(path)))
 else:
  paths=[(seed,Path(a.run_template.format(seed=seed))) for seed in a.seeds]
 rows=[]
 for seed,root in paths:
  metrics=json.loads((root/'metrics.json').read_text()); preds=[json.loads(x) for x in (root/'predictions.jsonl').read_text().splitlines()]
  rows.append((seed,metrics,preds))
 summary={}
 for key in ('macro_f1','balanced_accuracy','nll','brier','ece'):
  values=np.asarray([m[key] for _,m,_ in rows],dtype=float); summary[key]={'values':values.tolist(),'mean':float(values.mean()),'std':float(values.std(ddof=1)) if len(values)>1 else 0.0}
 base=rows[0][2]; ids=[r['sample_id'] for r in base]; identical=True
 for _,_,preds in rows[1:]: identical &= [r['sample_id'] for r in preds]==ids
 paired=[]
 if identical:
  y=np.asarray([r['label_id'] for r in base])
  base_correct=np.asarray([r['prediction']==r['label'] for r in base],dtype=float)
  for seed,_,preds in rows[1:]:
   other=np.asarray([r['prediction']==r['label'] for r in preds],dtype=float)
   diff=other-base_correct
   # exact sign-flip permutation on paired correctness differences
   observed=abs(float(diff.mean())); nonzero=diff[diff!=0]
   if len(nonzero):
    vals=np.asarray([abs(float(np.mean(nonzero*(np.random.default_rng(1729+i).integers(0,2,len(nonzero))*2-1)))) for i in range(4096)])
    p=float((np.sum(vals>=observed)+1)/(len(vals)+1))
   else: p=1.0
   paired.append({'seed':seed,'accuracy_difference_vs_first':float(diff.mean()),'sign_flip_p':p})
 result={'seeds':[seed for seed,_,_ in rows],'n':len(base),'query_ids_identical':bool(identical),'summary':summary,'paired_correctness':paired}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
