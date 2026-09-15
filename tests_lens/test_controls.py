import importlib
import numpy as np
import pytest
import torch
from PIL import Image


def api():
    assert importlib.util.find_spec('visual_lens.controls') is not None, 'controls implementation required'
    return importlib.import_module('visual_lens.controls')


def test_permutation_is_bijective_nonself_and_order_independent():
    m=api(); ids=['d','a','c','b']
    a=m.image_permutation(ids,11);b=m.image_permutation(list(reversed(ids)),11)
    assert a==b and set(a.values())==set(ids)
    assert all(k!=v for k,v in a.items())
    with pytest.raises(ValueError):m.image_permutation(['a'],0)
    with pytest.raises(ValueError):m.image_permutation(['a','a'],0)


def test_neutral_and_swap_preserve_exact_size_and_pre_reference():
    m=api();pre=Image.new('RGB',(28,56),'red');post=Image.new('RGB',(56,28),'blue')
    dp=Image.new('RGB',(10,10),'black');dq=Image.new('RGB',(14,7),'white')
    pair=m.intervene_images((pre,post),'neutral_post')
    assert pair[0].tobytes()==pre.tobytes() and pair[1].size==post.size
    assert np.asarray(pair[1]).mean()==128
    pair=m.intervene_images((pre,post),'shuffle_post',(dp,dq))
    assert pair[0].tobytes()==pre.tobytes() and pair[1].size==post.size
    assert np.asarray(pair[1]).mean()==255
    assert post.getpixel((0,0))==(0,0,255)


def test_label_span_uses_contextual_suffix_not_earlier_question():
    m=api()
    class Tok:
        def decode(self,ids,**kwargs):
            vocab={1:'Answer',2:':',3:' tumor',4:'?',5:'normal'}
            return ''.join(vocab[i] for i in ids)
    ids=torch.tensor([[3,4,1,2,3]])
    assert m.select_label_span(ids,(2,5),'tumor',Tok())==(4,5)
    assert m.select_label_span(ids,(4,5),'tumor',Tok())==(4,5)
    with pytest.raises(ValueError):m.select_label_span(ids,(2,5),'normal',Tok())


def test_bias_has_k_minus_one_parameters_and_zero_sum():
    m=api();scores=np.array([[0.,0.]]*12);y=np.array([0]*9+[1]*3)
    bias,info=m.fit_candidate_bias(scores,y,l2=.01,max_iter=100)
    assert abs(sum(bias))<1e-9 and info['trainable_scalars']==1
    assert bias[0]>bias[1]
    p=m.biased_probabilities(scores[0],bias)
    assert p[0]>0.5 and sum(p)==pytest.approx(1.)


def test_difference_in_differences_uses_paired_query_clusters():
    m=api()
    def rows(pred):
        return [dict(sample_id=str(i),group_id=str(i//2),label_id=i%2,
                    probabilities=[1. if q==0 else 0.,1. if q==1 else 0.]) for i,q in enumerate(pred)]
    f=rows([0,0,0,0]);r=rows([0,1,0,1]);c=rows([0,0,0,0])
    stat=m.paired_did(f,r,f,c,draws=20,seed=3)
    assert stat['macro_f1']['difference_in_differences']>0
    assert stat['clusters']==2
    with pytest.raises(ValueError):m.paired_did(f,r,f,c[:-1],draws=5)


def test_support_sampling_never_draws_a_query_tile():
    m=api()
    pool=[dict(sample_id=f'{c}-{i}',event_id='e',tile_id=f't{i}',label_id=c,label=str(c))
          for c in range(3) for i in range(5)]
    q=[dict(sample_id='q',event_id='e',tile_id='t0',label_id=0,label='0')]
    a=m.balanced_support(pool,q,2,7)
    assert len(a)==6 and all(x['tile_id']!='t0' for x in a)
    assert a==m.balanced_support(list(reversed(pool)),q,2,7)
    with pytest.raises(ValueError):m.balanced_support(pool,q,8,7)
