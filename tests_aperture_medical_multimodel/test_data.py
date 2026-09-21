import json
from pathlib import Path
import pytest
from aperture_medical_multimodel.data import load_metadata, partition_patients, select_balanced, select_query, build_unit, validate_unit, locked_jsonl


def cohort(center=0, patients=10, perclass=40):
    rows=[]
    for p in range(patients):
        for c in (0,1):
            for i in range(perclass):
                sid=f'{center}-{p}-{c}-{i}'
                rows.append(dict(sample_id=sid,domain_id=f'hospital_{center}',group_id=f'p{center}-{p}',
                    image=f'/not-real/{sid}.png',label=('normal','tumor')[c],label_id=c,
                    candidate_labels=['normal','tumor'],question='Tumor?',dataset='camelyon17-wilds',
                    metadata=dict(patient=f'p{center}-{p}',slide=center*100+p,hospital=center)))
    return rows


def test_partition_patient_disjoint():
    pools=partition_patients(cohort(),seed=721,pool_per_class=64)
    sets=[{r['group_id'] for r in pool} for pool in pools.values()]
    assert len(sets)==3
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    assert sum(map(len,pools.values()))==len(cohort())


def test_invariant_to_metadata_order():
    a=partition_patients(cohort(),721,64)
    b=partition_patients(list(reversed(cohort())),721,64)
    assert a==b


def test_insufficient_patients_fails():
    with pytest.raises(ValueError,match='patient'):
        partition_patients(cohort(patients=2),721,64)


def test_nested_support():
    r=cohort();a=select_balanced(r,6,11);b=select_balanced(r,24,11)
    assert {x['sample_id'] for x in a}<={x['sample_id'] for x in b}
    assert len(a)==12 and len(b)==48


def test_labels_not_used_to_balance_query():
    r=[x for x in cohort() if x['label_id']==0]
    q=select_query(r,101,17)
    assert len(q)==101 and all(x['label_id']==0 for x in q)


def test_unit_budgets_and_fixed_queries():
    pools=partition_patients(cohort(),721,64)
    a=build_unit(pools,16,10,300,721)
    b=build_unit(pools,32,11,300,721)
    assert [x['sample_id'] for x in a['query']]==[x['sample_id'] for x in b['query']]
    assert (len(a['fit']),len(a['calibration']))==(12,4)
    assert (len(b['fit']),len(b['calibration']))==(24,8)
    validate_unit(b,32,300)


def test_strict_patient_not_patient_slide():
    pools=partition_patients(cohort(),721,64);u=build_unit(pools,32,10,300,721)
    u['calibration'][0]['group_id']=u['fit'][0]['group_id']
    u['calibration'][0]['metadata']['patient']=u['fit'][0]['metadata']['patient']
    with pytest.raises(ValueError,match='patient'):
        validate_unit(u,32,300)


def test_duplicate_patch_fails():
    r=cohort();r.append(r[0])
    with pytest.raises(ValueError,match='duplicate'):
        partition_patients(r,721,64)


def test_official_metadata_path(tmp_path):
    (tmp_path/'metadata.csv').write_text(',patient,node,x_coord,y_coord,center,slide,tumor,split\n0,004,2,96,192,0,3,1,0\n')
    r=load_metadata(tmp_path)
    assert r[0]['image'].endswith('patches/patient_004_node_2/patch_patient_004_node_2_x_96_y_192.png')
    assert r[0]['group_id']=='patient_004'
    assert r[0]['metadata']['hospital']==0


def test_bad_hospital_or_label(tmp_path):
    (tmp_path/'metadata.csv').write_text('patient,node,x_coord,y_coord,center,slide,tumor\n004,2,96,192,6,3,1\n')
    with pytest.raises(ValueError):load_metadata(tmp_path)


def test_lock_rejects_content_change(tmp_path):
    p=tmp_path/'x.jsonl';rows=cohort()[:3]
    locked_jsonl(p,rows);locked_jsonl(p,rows)
    rows[0]['label']='tampered'
    with pytest.raises(ValueError,match='locked'):locked_jsonl(p,rows)
