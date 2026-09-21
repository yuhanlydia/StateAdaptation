import json
from pathlib import Path
import pytest
from aperture_medical_multimodel.reporting import export
from aperture_medical_multimodel.worker import score_block
from aperture_medical_multimodel.data import locked_jsonl
from vigor_handoff.protocol import JobStore, atomic_json
from aperture_medical_multimodel.protocol import make_jobs
from tests_aperture_medical_multimodel.test_protocol import cfg


def test_complete_synthetic_export(tmp_path):
    c=cfg();c['hospitals']=[0];c['support_seeds']=[10];c['models']=c['models'][:1];c['curve_supports']=[]
    c['run_root']=str(tmp_path/'aperture_medical_multimodel_test');c['prepared_root']=str(tmp_path/'aperture_medical_multimodel_data')
    jobs=make_jobs(c)
    manifest=[{'sample_id':str(i),'label_id':i%2,'metadata':{'patient':f'p{i//2}'}} for i in range(8)]
    locked_jsonl(jobs[0]['query'],manifest)
    for j in jobs:
        fit=Path(j['output'])/'fit';ev=Path(j['output'])/'evaluation'
        fs=JobStore(fit,{'synthetic':True,'arm':j['arm']})
        cal={'temperature':{'temperature':1.4,'calibration_count':8},'raw_calibration_nll':.5,'bias':{'values':[0.,0.]}}
        atomic_json(fit/'calibration.json',cal);fs.finish({'optimized_scalars':1024},['calibration.json'])
        preds=[{'sample_id':r['sample_id'],'label_id':r['label_id'],'mean_log_scores':[-1.,-2.] if r['label_id']==0 else [-2.,-1.]} for r in manifest]
        es=JobStore(ev,{'synthetic':True,'arm':j['arm']});locked_jsonl(ev/'clean.jsonl',preds)
        result={'job_id':j['job_id'],'unit_id':j['unit_id'],'domain':j['domain'],'support_seed':10,'arm':j['arm'],
                   'clean':score_block(preds,1.4,[0,0] if j['arm']=='frozen' else None),
                   'selected_lora_raw':'lora1','selected_lora_temperature':'lora4'}
        atomic_json(ev/'evaluation.json',result)
        es.finish({'completed':True,'count':len(manifest)},['clean.jsonl','evaluation.json'])
    cov=export(jobs,c['run_root'],['main']);assert cov['paper_ready']
    out=Path(c['run_root'])/'paper_exports'/'main';r=json.loads((out/'results.json').read_text())
    assert len(r['rows'])==13 # five arms raw+T; bias; two selected LoRA rows
    assert all(x['auroc']==1 for x in r['rows'])
    assert (out/'qwen2_n32_patient_macro_nll.tex').is_file()
    (Path(jobs[-1]['output'])/'evaluation'/'clean.jsonl').unlink()
    with pytest.raises(RuntimeError):export(jobs,c['run_root'],['main'])


def test_empty_results_do_not_get_filled(tmp_path):
    c=cfg();c['run_root']=str(tmp_path/'aperture_medical_multimodel_test');j=make_jobs(c)
    r=export(j,c['run_root'],['main'],True)
    assert not r['paper_ready'] and r['completed']==0
