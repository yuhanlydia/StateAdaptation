"""Synthetic 64-state records exercise the real seal/selection/export path, not GPU performance."""
import copy
import json
from pathlib import Path
import numpy as np
from aperture_final.numeric import metrics,probabilities
from aperture_table.protocol import default_config,make_jobs,MODELS,DOMAINS,source_signature
from aperture_table.worker import seal_unit
from aperture_table.reporting import export,fill_main
from vigor_handoff.protocol import JobStore,atomic_json,digest,file_hash


def test_exact_worker_record_layout_to_complete_table(tmp_path,monkeypatch):
 c=default_config();root=tmp_path/'aperture_table_v1';c['run_root']=str(root)
 jobs=make_jobs(c);query={}
 for dk,(_,nc,nq) in DOMAINS.items():
  query[dk]=[{'sample_id':f'{dk}-{i}','label_id':i%nc,'candidate_labels':list(map(str,range(nc)))} for i in range(nq)]
 monkeypatch.setattr('aperture_table.reporting.read_roles',lambda j:{'query':query[j['domain_key']]})
 for m in MODELS:
  for dk in DOMAINS:
   p=root/'audits'/m/dk;s=JobStore(p,{'source':source_signature()});atomic_json(p/'audit.json',{'passed':True});s.finish({'passed':True},['audit.json'])
 for j in jobs:
  p=Path(j['output'])/'fit';s=JobStore(p,{'job':j})
  cal={'raw_nll':1.,'temperature':{'temperature':2.,'calibration_nll':.8}}
  if j['arm']=='frozen':cal['bias']={'values':[0.]*j['classes']}
  atomic_json(p/'calibration.json',cal);atomic_json(p/'fitting.json',{'optimized_scalars':1024})
  s.finish({'ok':True},['calibration.json','fitting.json'])
 units={}
 for j in jobs:units.setdefault(j['unit_id'],[]).append(j)
 for u,jj in units.items():seal_unit(jj,root/'selections'/(u+'.json'))
 for m in MODELS:
  j=next(j for j in jobs if j['model_key']==m and j['domain_key']=='h2' and j['support_seed']==0 and j['arm']=='aperture')
  p=root/'repeat'/m/'state'/'fit';s=JobStore(p,{'repeat':True});s.finish({'ok':True})
  atomic_json(root/'repeat'/m/'gate.json',{'passed':True,'source':source_signature(),
     'first_seal':file_hash(Path(j['output'])/'fit'/'DONE.json'),'repeat_seal':file_hash(p/'DONE.json')})
 for j in jobs:
  rows=query[j['domain_key']];scores=np.random.default_rng(j['support_seed']).normal(size=(len(rows),j['classes']))
  pred=[dict(sample_id=r['sample_id'],label_id=r['label_id'],mean_log_scores=sc.tolist(),probabilities=probabilities([sc])[0].tolist(),prompt_contract_hash='synthetic') for r,sc in zip(rows,scores)]
  y=[r['label_id'] for r in rows]
  r={k:j[k] for k in ('job_id','unit_id','model_key','model_id','domain_key','support_seed','arm')}
  r.update(source=source_signature(),query_ids_sha256=digest([x['sample_id'] for x in rows]),query_label_sha256=digest(y),raw=metrics(scores,y),temperature=metrics(scores,y,2.))
  if j['arm']=='frozen':r['bias']=metrics(scores,y,bias=np.zeros(j['classes']))
  p=Path(j['output'])/'evaluation';s=JobStore(p,{'job':j})
  atomic_json(p/'evaluation.json',r);(p/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pred));s.finish({'ok':True},['evaluation.json','predictions.jsonl'])
 result=export(c)
 assert len(result['cells'])==112 and len(result['rows'])==40
 assert json.loads((root/'paper_exports'/'coverage.json').read_text())['paper_ready']
 # Independently filled tiny manuscript: no publication-number fixtures.
 source=tmp_path/'main.tex';source.write_text('\n'.join('\\pending{'+k+'}' for k in sorted(result['cells'])))
 assert fill_main(source,tmp_path/'filled.tex',root/'paper_exports')==112
 # Tamper one prediction: do not leave the earlier valid tables in place.
 p=Path(jobs[-1]['output'])/'evaluation'/'predictions.jsonl';p.write_text('')
 import pytest
 with pytest.raises(ValueError):export(c)
 assert not (root/'paper_exports'/'main_table.tex').exists()
 assert not json.loads((root/'paper_exports'/'coverage.json').read_text())['paper_ready']
