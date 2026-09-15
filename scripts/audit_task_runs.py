#!/usr/bin/env python3
"""Audit and annotate task run artifacts without touching model predictions."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import torch
from eventttt.io import manifest_fingerprint, model_fingerprint
from eventttt.schemas import TaskSample
from eventttt.io import read_task_samples

def env():
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: commit='unknown'
    return {'git_commit':commit,'torch':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),
            'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',default='runs/oral'); p.add_argument('--manifest-root',default='data/prepared'); p.add_argument('--model-id',default='artifacts/models/Qwen2.5-VL-7B-Instruct'); p.add_argument('--write',action='store_true'); a=p.parse_args()
 root=Path(a.root); result={'checked':0,'passed':0,'failed':[]}
 for pred in sorted(root.glob('**/predictions.jsonl')):
  parts=pred.parts; out=pred.parent
  try:
   family=parts[parts.index('oral')+1]; dataset=parts[parts.index(family)+1]; seedpart=next(x for x in parts if x.startswith('seed_')); seed=int(seedpart.split('_')[1])
   if dataset=='camelyon17': manifest=Path(a.manifest_root)/'camelyon17'/f'seed_{seed}'/'query.jsonl'
   elif dataset=='manipbench_q1': domain=parts[parts.index(dataset)+1]; manifest=Path(a.manifest_root)/'manipbench_q1'/domain/f'seed_{seed}'/'query.jsonl'
   else: continue
   expected=read_task_samples(manifest); rows=[json.loads(x) for x in pred.read_text().splitlines() if x.strip()]
   errors=[]
   if len(rows)!=len(expected): errors.append(f'count {len(rows)} != {len(expected)}')
   if [r.get('sample_id') for r in rows] != [s.sample_id for s in expected]: errors.append('query order or IDs differ')
   for i,r in enumerate(rows):
    probs=r.get('probabilities',[])
    if abs(sum(probs)-1)>1e-5: errors.append(f'row {i} probabilities do not sum to one'); break
    if r.get('prediction') not in expected[i].candidate_labels: errors.append(f'row {i} invalid candidate'); break
   result['checked']+=1
   if errors: result['failed'].append({'run':str(out),'errors':errors}); continue
   result['passed']+=1
   metadata={'model_id':a.model_id,'model_sha256':model_fingerprint(a.model_id),'manifest':str(manifest.resolve()),'manifest_sha256':manifest_fingerprint(manifest),'dataset':dataset,'seed':seed,'arm':out.name,'family':family,'environment':env()}
   if a.write:
    (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'environment.json').write_text(json.dumps(env(),indent=2)+'\n')
    (out/'query_manifest.sha256').write_text(metadata['manifest_sha256']+'\n')
    (out/'model.sha256').write_text(metadata['model_sha256']+'\n')
  except Exception as exc:
   result['failed'].append({'run':str(out),'errors':[repr(exc)]})
 Path('reports/task_run_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
 raise SystemExit(1 if result['failed'] else 0)
if __name__=='__main__': main()
