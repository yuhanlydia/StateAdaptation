import json
from pathlib import Path
import pytest
from vigor_handoff.protocol import JobStore, atomic_json
from aperture_final.worker import seal_choices, verified_record, score_block
from aperture_final.protocol import make_jobs
from .test_protocol import cfg


def make_fits(jobs):
    for j in jobs:
        p=Path(j['output'])/'fit';s=JobStore(p,{'job':j,'test_fixture':True})
        cal={'raw_calibration_nll':{'lora1':.4,'lora4':.3}.get(j['arm'],.5),
             'temperature':{'temperature':2.,'calibration_nll':{'lora1':.2,'lora4':.25}.get(j['arm'],.3)}}
        atomic_json(p/'calibration.json',cal)
        s.finish({'fixture':True},['calibration.json'])


def test_selection_requires_all_fit_states_then_freezes_raw_and_scaled_choices(tmp_path):
    jobs=make_jobs(cfg(tmp_path))
    with pytest.raises((ValueError,FileNotFoundError)):seal_choices(jobs,tmp_path/'selection.json')
    make_fits(jobs)
    result=seal_choices(jobs,tmp_path/'selection.json')
    assert len(result['units'])==3
    for u in result['units'].values():
        assert u['lora_raw']=='lora4' and u['lora_temperature']=='lora1'
    assert result['query_metrics_used'] is False
    p=Path(jobs[0]['output'])/'fit'/'calibration.json';p.write_text('{}')
    with pytest.raises(ValueError,match='hash'):verified_record(p.parent)


def test_score_block_checks_order_labels_and_finite_metrics():
    rows=[{'sample_id':'a','label_id':0,'mean_log_scores':[2.,0.]},
          {'sample_id':'b','label_id':1,'mean_log_scores':[0.,2.]}]
    report=score_block(rows,temperature=2.)
    assert report['raw']['accuracy']==1 and report['temperature']['accuracy']==1
    with pytest.raises(ValueError):score_block(rows+[rows[0]],temperature=2.)
