import copy
import json
from pathlib import Path
import pytest
from aperture_final.protocol import make_jobs, split_support, probe_ids, prepare_job, check_job, write_locked


def rows(n=8, classes=('intact','damaged','destroyed')):
    return [dict(sample_id=f'{c}-{i}',label_id=k,label=c,candidate_labels=classes,
                 tile_id=f'train-tile-{k}',pre_image=f'/pre/{k}-{i}.png',post_image=f'/post/{k}-{i}.png')
            for k,c in enumerate(classes) for i in range(n)]


def cfg(tmp_path):
    return {'protocol':'aperture_final_v1','run_root':str(tmp_path/'runs'),
            'prepared_root':str(tmp_path/'prepared'),'model_path':str(tmp_path/'model'),
            'support_seeds':[0,1,2], 'calibration_per_class':2,
            'domains':[{'name':'hawaii','kind':'paired','support':str(tmp_path/'s{seed}.jsonl'),
                        'query':str(tmp_path/'q.jsonl'),'support_per_class':8,
                        'query_count':6, 'classes':3,'group_key':'tile_id'}]}


def test_splits_are_stable_disjoint_complete_and_order_invariant():
    data=rows();a,b=split_support(data,2,17)
    assert len(a)==18 and len(b)==6
    assert {r['sample_id'] for r in a}.isdisjoint(r['sample_id'] for r in b)
    assert set(r['sample_id'] for r in a+b)==set(r['sample_id'] for r in data)
    x,y=split_support(list(reversed(data)),2,17)
    assert a==x and b==y
    assert {c:sum(r['label_id']==c for r in b) for c in range(3)}=={0:2,1:2,2:2}


def test_overlap_in_calibration_images_is_rejected():
    data=rows();data[2]['pre_image']=data[0]['pre_image']
    with pytest.raises(ValueError,match='duplicate image'):split_support(data,2,17)


def test_plan_shares_manifests_and_never_adapts_to_query(tmp_path):
    jobs=make_jobs(cfg(tmp_path));assert len(jobs)==12
    for seed in range(3):
        js=[j for j in jobs if j['support_seed']==seed]
        assert {j['arm'] for j in js}=={'frozen','lora1','lora4','aperture'}
        for field in ['support','calibration','query','source_support']:
            assert len({j[field] for j in js})==1
        assert all(j['basis_source']=='support' for j in js)
        assert all(j['support_per_class']==6 for j in js)
    assert jobs==make_jobs(cfg(tmp_path))


def test_probe_selection_does_not_use_query_labels():
    q=rows();r=copy.deepcopy(q)
    for x in r:x['label_id']=999;x['label']='not-used'
    assert probe_ids(q,10,123)==probe_ids(r,10,123)


def test_prepare_preserves_locked_query_and_tracks_label_budget(tmp_path):
    c=cfg(tmp_path);s=rows();q=rows(2)
    for r in q:
        r['sample_id']='query-'+r['sample_id'];r['tile_id']='query-'+r['tile_id']
        r['pre_image']='/q'+r['pre_image'];r['post_image']='/q'+r['post_image']
    (tmp_path/'s0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in s))
    (tmp_path/'q.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in q))
    j=make_jobs(c)[0];audit=prepare_job(j)
    assert audit['total_labels']==24 and audit['fit_labels']==18 and audit['calibration_labels']==6
    assert json.loads(Path(j['query']).read_text().splitlines()[0])['sample_id']==q[0]['sample_id']
    assert check_job(j,assets=False)['total_labels']==24
    prepare_job(j)
    s[0]['label']='INVALID'
    (tmp_path/'s0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in s))
    with pytest.raises(ValueError):prepare_job(j)


def test_immutable_file_will_not_overwrite(tmp_path):
    p=tmp_path/'a.json';write_locked(p,{'x':1});write_locked(p,{'x':1})
    with pytest.raises(ValueError):write_locked(p,{'x':2})


def test_config_rejects_duplicate_domains_and_unsafe_root(tmp_path):
    c=cfg(tmp_path);c['domains']*=2
    with pytest.raises(ValueError):make_jobs(c)
    c=cfg(tmp_path);c['run_root']='runs/visual_lens_p0_v2'
    with pytest.raises(ValueError):make_jobs(c)


def test_preflight_rejects_modified_prepared_content_even_if_ids_unchanged(tmp_path):
    c=cfg(tmp_path);s=rows();q=rows(2)
    for r in q:
        r['sample_id']='q-'+r['sample_id'];r['tile_id']='q-'+r['tile_id']
        r['pre_image']='/q'+r['pre_image'];r['post_image']='/q'+r['post_image']
    (tmp_path/'s0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in s))
    (tmp_path/'q.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in q))
    j=make_jobs(c)[0];prepare_job(j)
    p=Path(j['support']);f=[json.loads(line) for line in p.read_text().splitlines()]
    f[0]['post_image']='/another/image.png'
    p.write_text(''.join(json.dumps(r)+'\n' for r in f))
    with pytest.raises(ValueError,match='prepared content'):
        check_job(j,assets=False)


def test_preflight_rejects_source_change_after_prepare(tmp_path):
    c=cfg(tmp_path);s=rows();q=rows(2)
    for r in q:
        r['sample_id']='q-'+r['sample_id'];r['tile_id']='q-'+r['tile_id']
        r['pre_image']='/q'+r['pre_image'];r['post_image']='/q'+r['post_image']
    (tmp_path/'s0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in s))
    (tmp_path/'q.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in q))
    j=make_jobs(c)[0];prepare_job(j)
    (tmp_path/'s0.jsonl').write_text((tmp_path/'s0.jsonl').read_text()+'\n')
    with pytest.raises(ValueError,match='source manifest'):
        check_job(j,assets=False)


def test_different_crops_from_shared_tile_are_allowed():
    s=rows()
    for n,r in enumerate(s):
        r['pre_image']='/tile/pre.png';r['post_image']='/tile/post.png'
        r['bbox_xyxy']=[n,0,n+1,1]
    fit,cal=split_support(s,2,17)
    assert len(fit)==18 and len(cal)==6
