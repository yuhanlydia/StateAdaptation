import copy, json
from pathlib import Path
import numpy as np
import pytest
from aperture_medical.protocol import partition, select_balanced, make_jobs, validate_triplet, check_job, prepare
from aperture_medical.numeric import metrics, replay_difference
from aperture_medical.reporting import aggregate, export


def rows():
    return [dict(sample_id=f'p{p}-c{c}-{k}',group_id=f'p{p}',domain_id='hospital_0',
                 image=f'/unused/{p}-{c}-{k}.png',label_id=c,label=['normal','tumor'][c],
                 candidate_labels=['normal','tumor'],dataset='camelyon17-wilds',
                 question='Does the central region contain tumor tissue? Answer exactly normal or tumor.',
                 metadata={'patient':str(p),'pixel_sha256':f'{p}-{c}-{k}'})
            for p in range(10) for c in range(2) for k in range(30)]


def config(tmp):
    return dict(protocol='aperture_medical_v1',run_root=str(tmp/'runs'),prepared_root=str(tmp/'prepared'),
                model_path='/unused/model',support_seeds=[0,1,2],budgets_per_class=[8,16],
                split_seed=310927,calibration_per_class=2,domain_specs=[
                    dict(name=f'hospital_{h}',dataset='camelyon17-wilds',classes=2,query_per_class=25,
                         isolation='patient',source_root='/data/cam') for h in range(5)])


def test_partition_order_invariant_and_disjoint():
    p=partition(rows(),4);q=partition(list(reversed(rows())),4)
    assert p==q
    groups=[{r['group_id'] for r in part} for part in p]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    assert [len(x) for x in groups]==[4,2,4]


def test_nested_support_and_different_seeds():
    fit,cal,q=partition(rows(),4)
    a=select_balanced(fit,6,0);b=select_balanced(fit,14,0)
    assert {r['sample_id'] for r in a} < {r['sample_id'] for r in b}
    assert a!=select_balanced(fit,6,1)


def test_triplet_detects_patient_leakage():
    f,c,q=partition(rows(),4);c[0]['group_id']=f[0]['group_id']
    with pytest.raises(ValueError,match='group'):validate_triplet(f,c,q)


def test_triplet_detects_pixel_duplicates():
    f,c,q=partition(rows(),4);q[0]['metadata']['pixel_sha256']=f[0]['metadata']['pixel_sha256']
    with pytest.raises(ValueError,match='content'):validate_triplet(f,c,q)


def test_fail_insufficient_groups():
    with pytest.raises(ValueError):partition([r for r in rows() if r['group_id'] in ['p1','p2']],4)


def test_fail_insufficient_samples():
    with pytest.raises(ValueError):select_balanced(rows(),1000,0)


def test_missing_patient_is_not_replaced_by_sample_id():
    r=rows();r[0]['group_id']=''
    with pytest.raises(ValueError):partition(r,4)


def test_plan_counts_and_shared_data(tmp_path):
    cfg=config(tmp_path);jobs=make_jobs(cfg)
    assert len(jobs)==150
    for u in {j['unit_id'] for j in jobs}:
        jj=[j for j in jobs if j['unit_id']==u]
        assert {j['arm'] for j in jj}=={'frozen','lora1','lora4','random','aperture'}
        assert len({j['support'] for j in jj})==1 and len({j['query'] for j in jj})==1
        assert all(j['lora_rank']==16 for j in jj)
    assert jobs[0]['support_per_class']==6
    assert len({j['query'] for j in jobs if j['domain']=='hospital_0'})==1


def test_external_plan_count(tmp_path):
    c=config(tmp_path);c['domain_specs'].append(dict(name='pathmnist_crc',dataset='pathmnist',classes=9,
        query_per_class=100,isolation='image_content',source_root='/unused/path.npz'))
    j=make_jobs(c);assert len(j)==180
    assert {x['total_label_budget'] for x in j if x['domain']=='pathmnist_crc'}=={72,144}


def test_reject_duplicate_seeds(tmp_path):
    c=config(tmp_path);c['support_seeds']=[0,0]
    with pytest.raises(ValueError):make_jobs(c)


def test_auc_and_temperature_invariance():
    s=np.array([[3,0],[0,2],[1,0],[0,3.]])
    a=metrics(s,[0,1,0,1]);b=metrics(s,[0,1,0,1],3)
    assert a['auroc']==1 and a['auprc']==1
    assert a['macro_f1']==b['macro_f1'] and a['nll']!=b['nll']


def test_multiclass_auc():
    a=metrics(np.eye(3)*2,[0,1,2]);assert a['auroc']==1


def test_replay_difference():
    assert replay_difference([[1,2]],[[1,2]])['passed']
    assert not replay_difference([[1,2]],[[2,1]])['passed']


def test_aggregate_patient_macro_and_seeds():
    rr=[{'domain':'hospital_0','budget':16,'seed':s,'method':'Aperture','nll':n,'macro_f1':.6}
        for s,n in [(0,.4),(1,.5),(2,.6)]]
    a=aggregate(rr);assert a[0]['nll_mean']==pytest.approx(.5)
    assert a[0]['nll_sd']==pytest.approx(.1)


def test_missing_results_reject_paper_export(tmp_path):
    c=config(tmp_path)
    with pytest.raises(RuntimeError):export(make_jobs(c),tmp_path)
    assert not list(tmp_path.glob('paper_exports/table*.tex'))


def test_prepare_real_toy_images_and_tamper(tmp_path,monkeypatch):
    from PIL import Image
    from aperture_medical import protocol as p
    c=config(tmp_path);c['domain_specs']=c['domain_specs'][:1]
    rr=rows()
    for k,r in enumerate(rr):
        image=tmp_path/f'{k}.png';a=np.zeros((4,4,3),dtype=np.uint8)
        a[0,0,:]=[k%256,k//256,91];Image.fromarray(a).save(image);r['image']=str(image)
    monkeypatch.setattr(p,'population',lambda spec,cache:(rr,{'test_fixture':True}))
    prepare(c);j=make_jobs(c)[0];check_job(j,assets=False)
    rec=json.loads(Path(j['split_record']).read_text())
    assert not set(rec['fit_groups'])&set(rec['calibration_groups'])
    original=Path(j['support']).read_text();changed=original.replace('tumor tissue','test tissue')
    Path(j['support']).write_text(changed)
    with pytest.raises(ValueError,match='content'):check_job(j,assets=False)


def test_camelyon_loader_preserves_patient_and_label(tmp_path):
    import pandas as pd
    from aperture_medical.data import camelyon_population
    pd.DataFrame([dict(patient='003',node=1,x_coord=4,y_coord=6,center=0,tumor=1,slide=2)]).to_csv(tmp_path/'metadata.csv')
    rr,meta=camelyon_population(dict(name='hospital_0',source_root=str(tmp_path)))
    assert rr[0]['group_id']=='patient_003'
    assert rr[0]['label']=='tumor' and 'patient_003_node_1' in rr[0]['image']
    assert 'central 32x32' in rr[0]['metadata']['label_definition']


def test_path_loader_rejects_wrong_archive(tmp_path):
    from aperture_medical.data import path_population
    p=tmp_path/'wrong.npz';p.write_bytes(b'not official')
    with pytest.raises(ValueError,match='MD5'):path_population(dict(source_root=str(p)),tmp_path)


def test_score_block_auc_and_invariance():
    from aperture_medical.worker import score_block
    rr=[dict(sample_id=f'i{i}',mean_log_scores=s,label_id=c) for i,(s,c) in enumerate([([2.,0],0),([0.,3],1)])]
    b=score_block(rr,temperature=2)
    assert b['raw']['auroc']==1
    assert b['raw']['macro_f1']==b['temperature']['macro_f1']


def test_random_extraction_does_not_read_labels():
    import torch
    from vigor_handoff.core import extract_basis,ResidualController
    layer=torch.nn.Linear(8,8);layer.requires_grad_(False)
    def forbidden(_):raise AssertionError('random control may not inspect labels for directions')
    bb,meta=extract_basis(layer,[(0,'K',layer)],[1],forbidden,rank=2,mode='random',seed=0)
    c=ResidualController([(0,'K',layer)],bb,mode='full')
    c.set_mask(torch.tensor([[True,False]]));x=torch.ones(1,2,8)
    y=layer(x)
    with torch.no_grad():
        for p in c.parameters():p.fill_(.1)
    z=layer(x)
    assert torch.equal(y[:,1],z[:,1]) and not torch.equal(y[:,0],z[:,0])
    c.close()


def test_fit_contains_no_query_forward():
    import inspect
    from aperture_medical.worker import execute_fit
    source=inspect.getsource(execute_fit)
    assert "_samples(job,'query')" not in source and "_samples(job, 'query')" in source # comment only
    assert "mode=job['basis_mode']" in source
    assert 'a.detach().cpu()' in source


def test_multiclass_missing_group_auc_not_fabricated():
    r=metrics([[3,0,0],[0,3,0]],[0,1]);assert r['auroc'] is None


def test_selection_rejects_missing_arm(tmp_path):
    from aperture_medical.worker import seal_choices
    jobs=make_jobs(config(tmp_path))[:4]
    with pytest.raises(ValueError,match='incomplete'):seal_choices(jobs,tmp_path/'selection.json')


def test_h2_anchor_status(tmp_path):
    jobs=make_jobs(config(tmp_path))
    assert all(j['target_split_status']=='H2_previously_examined_anchor' for j in jobs if j['domain']=='hospital_2')


def test_balanced_sampling_spreads_over_patients():
    selected=select_balanced(rows(),8,0,2)
    for c in (0,1):assert len({r['group_id'] for r in selected if r['label_id']==c})==8


def test_full_synthetic_export(tmp_path,monkeypatch):
    from aperture_medical import reporting as rep
    from vigor_handoff.protocol import JobStore,atomic_json
    from aperture_medical.protocol import source_signature
    from aperture_medical.worker import score_block
    c=config(tmp_path);c['domain_specs']=c['domain_specs'][:1];c['budgets_per_class']=[8]
    jobs=make_jobs(c)
    monkeypatch.setattr(rep,'check_job',lambda *a,**k:None)
    pred=[dict(sample_id=f'i{i}',group_id=f'p{i//2}',label_id=i%2,mean_log_scores=([2.,0] if i%2==0 else [0,2.]),probabilities=([.8807970779778823,.11920292202211755] if i%2==0 else [.11920292202211755,.8807970779778823])) for i in range(50)]
    for j in jobs:
        f=Path(j['output'])/'fit';e=Path(j['output'])/'evaluation'
        JobStore(f,{}).finish(dict(optimized_scalars=0,basis_scalars=0,serialized_state_bytes=0,optimizer_updates=0,fit_wall_seconds=0))
        es=JobStore(e,{'source':source_signature()})
        r={k:j[k] for k in ['job_id','unit_id','domain','arm','support_seed']}
        r.update(clean=score_block(pred,bias=[0,0]),selected_lora_raw='lora1',selected_lora_temperature='lora1')
        atomic_json(e/'evaluation.json',r);(e/'clean.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pred))
        es.finish({},['evaluation.json','clean.jsonl'])
    cov=export(jobs,tmp_path)
    assert cov['paper_ready'] and cov['completed']==15
    assert (tmp_path/'paper_exports/table_n16_auroc.tex').is_file()
    assert (tmp_path/'paper_exports/query_group_metrics.csv').is_file()
    # A downstream plot/export exception also revokes paper-ready status.
    with monkeypatch.context() as patch:
        def broken_plot(*args, **kwargs):raise RuntimeError('plot failed')
        patch.setattr(rep,'plot',broken_plot)
        with pytest.raises(RuntimeError,match='plot failed'):export(jobs,tmp_path)
        assert not json.loads((tmp_path/'paper_exports/coverage.json').read_text())['paper_ready']
        assert not list((tmp_path/'paper_exports').glob('table*.tex'))
    # Exact source mismatch must revoke paper exports rather than reuse a stale table.
    monkeypatch.setattr(rep,'source_signature',lambda:'CHANGED')
    with pytest.raises(RuntimeError):export(jobs,tmp_path)
    assert not list((tmp_path/'paper_exports').glob('table*.tex'))


def test_eval_requires_replay_for_each_dataset(tmp_path):
    from aperture_medical.cli import validate_replay_gates
    jobs=make_jobs(config(tmp_path))
    with pytest.raises((FileNotFoundError,ValueError)):validate_replay_gates(jobs,tmp_path)
