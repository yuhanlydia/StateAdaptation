import numpy as np
import pytest
from aperture_medical_multimodel.reporting import medical_metrics, aggregate, tables


def test_auc_and_patient_macro():
    s=[[-1,-2],[-1,-3],[-2,-1],[-3,-1]];y=[0,0,1,1];g=['p1','p1','p1','p2']
    m=medical_metrics(s,y,g)
    assert m['auroc']==1 and m['macro_f1']==1
    assert m['patient_count']==2
    assert m['patient_macro_nll']!=m['nll']


def test_temperature_keeps_auc_and_f1():
    s=[[-1,-3],[-2,-1],[-2,-1],[-2,-1]];y=[0,0,1,1];g=['a','b','c','c']
    a=medical_metrics(s,y,g,1);b=medical_metrics(s,y,g,3)
    assert a['macro_f1']==b['macro_f1'] and a['auroc']==b['auroc']
    assert a['nll']!=b['nll']


def test_missing_class_auc_is_none():
    assert medical_metrics([[0,-1],[0,-2]],[0,0],['a','b'])['auroc'] is None


def test_nan_scores_fail():
    with pytest.raises(ValueError):medical_metrics([[np.nan,0]],[0],['a'])


def test_patient_label_permutation_bad():
    with pytest.raises(ValueError):medical_metrics([[0,1]],[0],[])


def test_aggregate_mean_and_sd():
    rows=[dict(family='qwen2',budget=32,hospital=0,seed=s,method='Aperture',nll=x,macro_f1=.6,auroc=.7,patient_macro_nll=x,brier=.4,ece=.1) for s,x in enumerate([1,2,3])]
    a=aggregate(rows);assert a[0]['nll_mean']==2 and a[0]['nll_sd']==1
    assert '\\begin{tabular}' in tables(a,'qwen2',32,'nll')


def test_duplicate_seed_fails():
    r=dict(family='qwen2',budget=32,hospital=0,seed=1,method='Aperture',nll=1,macro_f1=.6,auroc=.7,patient_macro_nll=1,brier=.4,ece=.1)
    with pytest.raises(ValueError,match='duplicate'):aggregate([r,r])
