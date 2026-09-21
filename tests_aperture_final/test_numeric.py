import numpy as np
import pytest
from PIL import Image
from aperture_final.numeric import fit_temperature, probabilities, metrics, choose_lora, degrade


def test_temperature_is_positive_argmax_preserving_and_improves_overconfident_loss():
    s=np.array([[16.,0.],[0.,16.],[16.,0.],[0.,16.]])
    y=np.array([0,1,1,0])
    fit=fit_temperature(s,y)
    assert fit['temperature'] > 1
    p=probabilities(s,fit['temperature'])
    assert np.array_equal(p.argmax(1),s.argmax(1))
    assert metrics(s,y,fit['temperature'])['nll'] < metrics(s,y)['nll']


def test_temperature_ties_choose_identity_and_scores_are_offset_invariant():
    s=np.zeros((4,2));y=[0,1,0,1]
    assert fit_temperature(s,y)['temperature']==1
    assert np.allclose(probabilities(s),probabilities(s+np.arange(4)[:,None]*100))


@pytest.mark.parametrize('s,y',[(np.array([[np.nan,0]]),[0]),(np.zeros((0,2)),[]),(np.zeros((2,2)),[0,3])])
def test_bad_scores_rejected(s,y):
    with pytest.raises(ValueError):fit_temperature(s,y)


def test_metrics_preserve_rare_confident_errors():
    m=metrics([[0.,-1000.],[0.,-1.]],[1,0])
    assert m['nll'] > 499
    assert 0<=m['ece']<=1 and m['count']==2
    assert m['nll_clipped_legacy'] < m['nll']


def test_lora_selection_uses_only_calibration_and_ties_prefer_one_pass():
    assert choose_lora({'lora1':{'calibration_nll':.3},'lora4':{'calibration_nll':.2}})=='lora4'
    assert choose_lora({'lora1':{'calibration_nll':.2},'lora4':{'calibration_nll':.2}})=='lora1'
    with pytest.raises(ValueError):choose_lora({'lora1':{'calibration_nll':float('nan')},'lora4':{'calibration_nll':.2}})


@pytest.mark.parametrize('condition',['gaussian4','gaussian8','jpeg70','jpeg40'])
def test_degradation_is_deterministic_size_preserving_and_post_only(condition):
    rng=np.random.default_rng(7)
    a=Image.fromarray(rng.integers(0,256,(31,27,3),dtype=np.uint8))
    b=Image.fromarray(rng.integers(0,256,(31,27,3),dtype=np.uint8))
    x=degrade((a,b),condition,'sample-x',12)
    y=degrade((a,b),condition,'sample-x',12)
    assert np.array_equal(x[0],a)
    assert np.array_equal(x[1],y[1])
    assert not np.array_equal(x[1],b)
    assert x[1].size==b.size and x[1].mode=='RGB'


def test_centered_scores_do_not_overflow_common_offset():
    from aperture_final.numeric import probabilities, metrics
    scores=np.array([[1e308,1e308],[1e308,1e308]])
    assert np.allclose(probabilities(scores,.2),.5)
    assert np.isclose(metrics(scores,[0,1],.2)['nll'],np.log(2))
