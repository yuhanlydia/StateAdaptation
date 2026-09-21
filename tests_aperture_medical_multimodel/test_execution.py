from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from aperture_medical_multimodel.worker import seal_choices, check_repeatability, require_repeatability, verified_record
from vigor_handoff.protocol import JobStore, atomic_json
from aperture_medical_multimodel.protocol import source_signature
from tests_aperture_medical_multimodel.test_protocol import cfg
from aperture_medical_multimodel.protocol import make_jobs


def seal_fit(path,rows,raw=1.,temperature_nll=.9):
    st=JobStore(path,{'synthetic':True})
    atomic_json(Path(path)/'calibration.json',{'raw_calibration_nll':raw,'temperature':{'calibration_nll':temperature_nll,'temperature':1.2}})
    (Path(path)/'calibration_predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    st.finish({'fixture':True},['calibration.json','calibration_predictions.jsonl'])


def test_selection_accepts_random_arm(tmp_path):
    jobs=make_jobs(cfg());unit=jobs[0]['unit_id'];jobs=[j for j in jobs if j['unit_id']==unit]
    for j in jobs:
        j['output']=str(tmp_path/j['arm']);seal_fit(Path(j['output'])/'fit',[{'sample_id':'a','mean_log_scores':[-1.,-2.]}],raw=.5 if j['arm']=='lora4' else 1)
    result=seal_choices(jobs,tmp_path/'selections.json')
    assert result['units'][unit]['lora_raw']=='lora4'
    assert not result['query_metrics_used']


def test_selection_rejects_missing_control(tmp_path):
    jobs=make_jobs(cfg());unit=jobs[0]['unit_id'];jobs=[j for j in jobs if j['unit_id']==unit][:-1]
    with pytest.raises(ValueError,match='incomplete'):seal_choices(jobs,tmp_path/'s.json')


def test_repeatability_only_calibration(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';rows=[{'sample_id':'a','label_id':0,'mean_log_scores':[-1.,-2.]}]
    seal_fit(a,rows);seal_fit(b,rows)
    gate=tmp_path/'gate.json';r=check_repeatability(a,b,gate,.005)
    assert r['passed'] and r['query_forwards']==0
    require_repeatability({'repeat_gate':str(gate)})


def test_repeatability_fails_without_replacing(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';rows=[{'sample_id':'a','label_id':0,'mean_log_scores':[-1.,-2.]}]
    seal_fit(a,rows);changed=deepcopy(rows);changed[0]['mean_log_scores'][0]=-.5;seal_fit(b,changed)
    with pytest.raises(RuntimeError,match='repeatability'):
        check_repeatability(a,b,tmp_path/'gate.json',.005)
    assert not json.loads((tmp_path/'gate.json').read_text())['passed']


def test_gate_detects_tampered_fit(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';rows=[{'sample_id':'a','label_id':0,'mean_log_scores':[-1.,-2.]}]
    seal_fit(a,rows);seal_fit(b,rows);gate=tmp_path/'g.json';check_repeatability(a,b,gate,.005)
    (b/'calibration_predictions.jsonl').write_text('{}\n')
    with pytest.raises(ValueError):require_repeatability({'repeat_gate':str(gate)})
