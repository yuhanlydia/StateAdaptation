import json
from pathlib import Path
import pytest
from vigor_handoff.protocol import JobStore,atomic_json
from aperture_table.protocol import default_config,make_jobs
from aperture_table.worker import verified_record,seal_unit,verify_selection,validate_rows

def test_tampering_invalidates_seal(tmp_path):
 s=JobStore(tmp_path,{'x':1});atomic_json(tmp_path/'payload.json',{'n':1});s.finish({'ok':True},['payload.json'])
 assert verified_record(tmp_path)
 atomic_json(tmp_path/'payload.json',{'n':2})
 with pytest.raises(ValueError):verified_record(tmp_path)

def test_selection_requires_every_fit_and_cannot_change(tmp_path):
 jobs=make_jobs(default_config())[:4]
 for j in jobs:j['output']=str(tmp_path/j['job_id'])
 dest=tmp_path/'selection.json'
 with pytest.raises((ValueError,FileNotFoundError)):seal_unit(jobs,dest)
 for j in jobs:
  p=Path(j['output'])/'fit';s=JobStore(p,{'job':j})
  atomic_json(p/'calibration.json',{'raw_nll':.5 if j['arm']=='lora1' else .6,
     'temperature':{'calibration_nll':.7 if j['arm']=='lora1' else .4,'temperature':2.}})
  s.finish({'ok':True},['calibration.json'])
 result=seal_unit(jobs,dest)
 assert result['selected_lora_raw']=='lora1' and result['selected_lora_temperature']=='lora4'
 assert result['query_metrics_used'] is False
 assert verify_selection(jobs[0],dest)
 p=Path(jobs[1]['output'])/'fit'/'calibration.json';p.write_text('{}')
 with pytest.raises(ValueError):verify_selection(jobs[0],dest)

def test_prediction_probability_and_order_checks():
 expected=[{'sample_id':'a','label_id':0,'candidate_labels':['A','B']}]
 good=[{'sample_id':'a','label_id':0,'mean_log_scores':[0.,0.],'probabilities':[.5,.5],'prompt_contract_hash':'x'}]
 validate_rows(good,expected)
 good[0]['probabilities']=[.9,.1]
 with pytest.raises(ValueError):validate_rows(good,expected)
 good[0]['probabilities']=[.5,.5];good[0]['sample_id']='other'
 with pytest.raises(ValueError):validate_rows(good,expected)

def test_repeat_gate_checks_the_original_and_repeat_seals(tmp_path):
 from aperture_table.worker import verify_repeat
 from aperture_table.protocol import source_signature
 job=make_jobs(default_config())[3];job['output']=str(tmp_path/'states'/job['job_id'])
 p=Path(job['output'])/'fit';s=JobStore(p,{'id':'original'});s.finish({'ok':True})
 q=tmp_path/'repeat'/job['model_key']/'state'/'fit';s=JobStore(q,{'id':'repeat'});s.finish({'ok':True})
 from vigor_handoff.protocol import file_hash
 gate=tmp_path/'repeat'/job['model_key']/'gate.json'
 atomic_json(gate,{'passed':True,'source':source_signature(),'first_output':job['output'],
   'repeat_output':str(q.parent),'first_seal':file_hash(p/'DONE.json'),'repeat_seal':file_hash(q/'DONE.json')})
 assert verify_repeat(job,tmp_path)
 atomic_json(q/'metrics.json',{'changed':True})
 with pytest.raises(ValueError):verify_repeat(job,tmp_path)
