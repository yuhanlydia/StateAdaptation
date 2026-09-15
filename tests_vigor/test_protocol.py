import json
from pathlib import Path
import pytest
from vigor_handoff.protocol import (digest,read_jsonl,atomic_json,validate_split,
                                    validate_predictions,JobStore,expand_plan)


def rows(ids):
    return [{'sample_id':str(i),'group_id':f'g{i}','image':f'{i}.png',
             'label_id':i%2,'label':['normal','tumor'][i%2],
             'candidate_labels':['normal','tumor']} for i in ids]


def test_hash_deterministic_and_semantic_change():
    assert digest({'a':1,'b':2})==digest({'b':2,'a':1})
    assert digest({'a':1})!=digest({'a':2})


def test_split_overlap_rejected():
    with pytest.raises(ValueError,match='overlap'):validate_split(rows([1,2]),rows([2,3]))
    a,b=rows([1]),rows([3]);b[0]['group_id']='g1'
    with pytest.raises(ValueError,match='group'):validate_split(a,b)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError,match='duplicate'):validate_split(rows([1,1]),rows([3]))


def test_same_image_under_different_id_rejected():
    a,b=rows([1]),rows([3]);b[0]['image']=a[0]['image']
    with pytest.raises(ValueError,match='image'):validate_split(a,b)


def test_missing_group_rejected():
    a,b=rows([1]),rows([3]);del a[0]['group_id']
    with pytest.raises(ValueError,match='group'):validate_split(a,b)


def test_prediction_complete_probability_and_labels():
    query=rows([1,2]);pred=[dict(r,probabilities=[.4,.6]) for r in query]
    validate_predictions(pred,query)
    with pytest.raises(ValueError,match='coverage'):validate_predictions(pred[:1],query)
    pred[0]['probabilities']=[float('nan'),.6]
    with pytest.raises(ValueError,match='probab'):validate_predictions(pred,query)


def test_store_refuses_config_collision_and_skips_only_completed(tmp_path):
    s=JobStore(tmp_path,{'rank':16});assert not s.complete()
    s.finish({'macro_f1':.5}); assert s.complete()
    with pytest.raises(ValueError,match='fingerprint'):JobStore(tmp_path,{'rank':32})


def test_corrupt_metrics_does_not_count_complete(tmp_path):
    s=JobStore(tmp_path,{'rank':16});s.finish({'macro_f1':.5})
    (tmp_path/'metrics.json').write_text('{}')
    assert not s.complete()


def test_lock_prevents_parallel_writer(tmp_path):
    s=JobStore(tmp_path,{'rank':16})
    with s.lock():
        with pytest.raises(RuntimeError,match='lock'):
            with s.lock():pass
    with s.lock():pass


def test_jsonl_partial_last_line_repaired_only_on_request(tmp_path):
    p=tmp_path/'x.jsonl';p.write_text('{"a":1}\n{"a":')
    with pytest.raises(ValueError):read_jsonl(p)
    assert read_jsonl(p,repair_tail=True)==[{'a':1}]
    assert p.read_text()=='{"a":1}\n'


def test_plan_is_fixed_and_oracles_separated():
    cfg={'models':[{'family':'internvl3','path':'models/i','id':'i'}],
         'datasets':[{'name':'bright','domain':'hawaii','kind':'paired','support':'s.jsonl','query':'q.jsonl',
                      'seeds':[0], 'diagnostics':True, 'group_key':'tile_id','support_per_class':8,'query_count':300}],
         'run_root':'runs/new','ablation_families':['internvl3']}
    main=expand_plan(cfg,['main']);all_=expand_plan(cfg,['main','ablation','diagnostic'])
    assert len(main)==4
    assert all(j['evidence']=='primary_reproduction' for j in main)
    assert any(j['evidence']=='oracle_diagnostic' for j in all_)
    assert len({j['job_id'] for j in all_})==len(all_)
    assert all(j['phase']!='main' for j in all_ if j['basis_source']=='query')
    assert main==expand_plan(cfg,['main'])
