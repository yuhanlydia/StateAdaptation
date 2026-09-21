import json
from pathlib import Path
from aperture_medical_multimodel.cli import main, gates_for
from aperture_medical_multimodel.protocol import make_jobs
from tests_aperture_medical_multimodel.test_protocol import cfg


def test_plan_no_assets(tmp_path):
    c=cfg();c['run_root']=str(tmp_path/'aperture_medical_multimodel_test');c['prepared_root']=str(tmp_path/'aperture_medical_multimodel_data')
    p=tmp_path/'config.json';p.write_text(json.dumps(c))
    assert main(['plan','--config',str(p)])==0
    plan=json.loads((Path(c['run_root'])/'plan.json').read_text())
    assert len(plan['jobs'])==255


def test_distinct_model_gates():
    jobs=make_jobs(cfg());g=gates_for(jobs,'runs/aperture_medical_multimodel_v1')
    assert set(g)=={'qwen2','internvl3'}
    for name,gate in g.items():
        assert gate['reference']['total_support']==32
        assert gate['reference']['hospital']==0
        assert gate['reference']['arm']=='aperture'


def test_no_unknown_hospital(tmp_path):
    c=cfg();c['run_root']=str(tmp_path/'aperture_medical_multimodel_test');c['prepared_root']=str(tmp_path/'aperture_medical_multimodel_data')
    p=tmp_path/'c.json';p.write_text(json.dumps(c))
    import pytest
    with pytest.raises(SystemExit):main(['plan','--config',str(p),'--hospitals','9'])
