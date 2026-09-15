import pytest
import numpy as np
from vigor_handoff.statistics import macro_f1,paired_cluster_bootstrap


def predictions(values):
    return [{'sample_id':str(i),'group_id':f'g{i//2}','label_id':i%2,
             'probabilities':([1.-v,v])} for i,v in enumerate(values)]


def test_macro_f1_matches_hand_result():
    assert macro_f1(np.array([0,1,0,1]),np.array([0,1,0,1]),2)==1.
    assert macro_f1(np.array([0,1,0,1]),np.array([0,0,0,0]),2)==pytest.approx(1/3)


def test_paired_bootstrap_zero_for_identical_predictions():
    a=predictions([.1,.9,.3,.6,.2,.7])
    result=paired_cluster_bootstrap(a,list(reversed(a)),2,draws=50)
    assert result['delta']==0
    assert result['ci95']==[0.,0.]


def test_paired_bootstrap_mismatched_ids_fail():
    a=predictions([.1,.9,.3,.6])
    with pytest.raises(ValueError,match='IDs'):paired_cluster_bootstrap(a,a[:-1],2,draws=10)


def test_paired_bootstrap_mismatched_labels_fail():
    a=predictions([.1,.9,.3,.6]);b=predictions([.1,.9,.3,.6]);b[0]['label_id']=1
    with pytest.raises(ValueError,match='labels'):paired_cluster_bootstrap(a,b,2,draws=10)
