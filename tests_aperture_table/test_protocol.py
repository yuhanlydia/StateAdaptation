import copy,json
from pathlib import Path
import pytest
from aperture_table.protocol import default_config, make_jobs, select_query, validate_triplet, write_locked, expected_cells

def rows(prefix,nc,n,group):
 return [dict(sample_id=f'{prefix}-{c}-{i}',label_id=c,label=chr(65+c),candidate_labels=[chr(65+k) for k in range(nc)],group_id=group,image=f'/tmp/{prefix}-{c}-{i}.png',metadata={'pixel_sha256':f'{prefix}-{c}-{i}'}) for c in range(nc) for i in range(n)]

def test_fixed_matrix():
 c=default_config();jobs=make_jobs(c)
 assert len(jobs)==64 and len({j['job_id'] for j in jobs})==64
 assert sum(j['arm']!='frozen' for j in jobs)==48
 assert {j['model_key'] for j in jobs}=={'q25','q34','q38','g34'}
 assert {j['support_seed'] for j in jobs}=={0,1}
 for j in jobs:
  assert j['query_count']==(200 if j['domain_key']=='h2' else 180)
  assert j['fit_count']==(12 if j['domain_key']=='h2' else 54)
  assert j['cal_count']==(4 if j['domain_key']=='h2' else 18)
 assert len(expected_cells())==112

def test_same_samples_across_methods_and_models():
 jobs=make_jobs(default_config())
 for d in ['h2','path']:
  for seed in [0,1]:
   js=[j for j in jobs if j['domain_key']==d and j['support_seed']==seed]
   for key in ['support','calibration','query','split_record']:
    assert len({j[key] for j in js})==1

def test_registry_cannot_silently_replace_model():
 c=default_config();c['models'][1]['id']='Qwen/Qwen3-4B'
 with pytest.raises(ValueError):make_jobs(c)
 c=default_config();c['support_seeds']=[0]
 with pytest.raises(ValueError):make_jobs(c)
 c=default_config();c['run_root']='runs/aperture_medical_v1'
 with pytest.raises(ValueError):make_jobs(c)

def test_query_is_predeclared_subset_not_prediction_selected():
 rr=rows('q',2,130,'qpatient');a=select_query(rr,2,100)
 assert len(a)==200
 assert a==select_query(list(reversed(rr)),2,100)
 assert set(r['sample_id'] for r in a)<=set(r['sample_id'] for r in rr)
 with pytest.raises(ValueError):select_query(rr,2,140)

def test_triplet_rejects_patient_and_pixel_overlap():
 f,c,q=rows('f',2,6,'fpatient'),rows('c',2,2,'cpatient'),rows('q',2,100,'qpatient')
 validate_triplet(f,c,q,2)
 c[0]['group_id']='fpatient'
 with pytest.raises(ValueError):validate_triplet(f,c,q,2)
 c[0]['group_id']='cpatient';c[0]['metadata']['pixel_sha256']=f[0]['metadata']['pixel_sha256']
 with pytest.raises(ValueError):validate_triplet(f,c,q,2)

def test_triplet_requires_candidate_and_content_identity():
 f,c,q=rows('f',2,6,'fpatient'),rows('c',2,2,'cpatient'),rows('q',2,100,'qpatient')
 q[0]['candidate_labels']=['B','A']
 with pytest.raises(ValueError):validate_triplet(f,c,q,2)
 q[0]['candidate_labels']=['A','B'];del q[0]['metadata']['pixel_sha256']
 with pytest.raises(ValueError):validate_triplet(f,c,q,2)

def test_lock_cannot_be_replaced(tmp_path):
 p=tmp_path/'lock.json';write_locked(p,{'v':1});write_locked(p,{'v':1})
 with pytest.raises(ValueError):write_locked(p,{'v':2})
 assert json.loads(p.read_text())=={'v':1}
