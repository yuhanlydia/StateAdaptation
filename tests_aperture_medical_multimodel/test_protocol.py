import copy
import json
from pathlib import Path
import pytest
from aperture_medical_multimodel.protocol import make_jobs, source_signature


def cfg():
    return {'protocol':'aperture_medical_multimodel_v1','data_root':'data/camelyon17_v1.0',
      'prepared_root':'data/prepared/aperture_medical_multimodel_v1','run_root':'runs/aperture_medical_multimodel_v1',
      'hospitals':[0,1,2,3,4],'support_seeds':[10,11,12],'query_count':500,'split_seed':271021,
      'primary_support':32,'curve_supports':[16,64],
      'models':[{'name':'qwen2','family':'qwen2','path':'models/qwen','id':'Qwen/Qwen2.5-VL-7B-Instruct','image_size':448},
                {'name':'internvl3','family':'internvl3','path':'models/intern','id':'OpenGVLab/InternVL3-8B-Instruct','image_size':224}]}


def test_counts():
    jobs=make_jobs(cfg()); counts={p:sum(j['phase']==p for j in jobs) for p in ('main','replication','curve')}
    assert counts=={'main':75,'replication':60,'curve':120}
    assert len({j['job_id'] for j in jobs})==255


def test_same_manifest_within_comparison():
    jobs=make_jobs(cfg());u=jobs[0]['unit_id'];a=[j for j in jobs if j['unit_id']==u]
    assert len(a)==5
    for k in ('support','calibration','query','total_support'):
        assert len({j[k] for j in a})==1


def test_cross_backbone_same_data():
    a=[j for j in make_jobs(cfg()) if j['hospital']==0 and j['support_seed']==10 and j['total_support']==32]
    assert len({j['query'] for j in a})==1
    assert {j['family'] for j in a}=={'qwen2','internvl3'}


def test_calibration_fraction():
    for j in make_jobs(cfg()):
        assert j['support_per_class']*2+j['calibration_per_class']*2==j['total_support']
        assert j['calibration_per_class']*2==j['total_support']//4


@pytest.mark.parametrize('change',[{'hospitals':[0,0]},{'primary_support':17},{'query_count':0},{'support_seeds':[-1]}])
def test_bad_config(change):
    c=cfg();c.update(change)
    with pytest.raises(ValueError):make_jobs(c)


def test_source_is_hashed():
    assert len(source_signature())==64
