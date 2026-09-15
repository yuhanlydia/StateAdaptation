import torch
import pytest
from vigor_handoff.core import Moments, ResidualController, visual_mask, fit_coefficients, extract_basis


def modules():
    return [(0, 'K', torch.nn.Linear(4, 4, bias=False)),
            (0, 'V', torch.nn.Linear(4, 4, bias=False))]


def test_uncentered_second_moment_matches_stacked_svd():
    g = torch.randn(11, 4, generator=torch.Generator().manual_seed(2))
    m = Moments(4); m.add(g[:5]); m.add(g[5:])
    b, meta = m.basis(2)
    expected = torch.linalg.svd(g, full_matrices=False).Vh[:2].T
    assert torch.allclose(b @ b.T, expected @ expected.T, atol=2e-5)
    assert meta['rows'] == 11


def test_centered_matches_covariance():
    g = torch.randn(13, 4) + 3
    m = Moments(4); m.add(g)
    assert torch.allclose(m.matrix('centered'), (g-g.mean(0)).T@(g-g.mean(0)), atol=5e-5)


def test_zero_energy_fails_closed():
    m = Moments(4); m.add(torch.zeros(2, 4))
    with pytest.raises(ValueError, match='energy'): m.basis(2)


def test_random_basis_is_orthogonal_deterministic():
    m = Moments(4)
    b1, _ = m.basis(2, 'random', seed=19); b2, _ = m.basis(2, 'random', seed=19)
    assert torch.equal(b1,b2)
    assert torch.allclose(b1.T @ b1, torch.eye(2),atol=1e-6)


def test_invalid_rank_and_nonfinite():
    m=Moments(4)
    with pytest.raises(ValueError): m.basis(5,'random')
    with pytest.raises(ValueError): m.add(torch.full((2,4),float('nan')))


@pytest.mark.parametrize('mode',['full','diagonal'])
def test_zero_identity_mask_isolation_and_bound(mode):
    mods=modules(); bases={(l,k):torch.eye(4)[:,:2] for l,k,_ in mods}
    original=mods[0][2](torch.ones(1,5,4)).detach()
    c=ResidualController(mods,bases,alpha=0.5,mode=mode)
    c.set_mask(torch.tensor([[False,True,True,False,False]]))
    assert torch.equal(mods[0][2](torch.ones(1,5,4)),original)
    with torch.no_grad():
        for p in c.parameters():p.fill_(0.4)
    changed=mods[0][2](torch.ones(1,5,4))
    assert torch.equal(changed[:,[0,3,4]],original[:,[0,3,4]])
    assert not torch.equal(changed[:,1:3],original[:,1:3])
    assert all(float(torch.linalg.matrix_norm(a.detach(),ord=2)) <= .500001 for a in c.operators().values())
    c.close()
    assert torch.equal(mods[0][2](torch.ones(1,5,4)),original)


def test_controller_only_registers_coefficients_not_backbone():
    mods=modules();c=ResidualController(mods,{(l,k):torch.eye(4)[:,:2] for l,k,_ in mods},1,'full')
    assert c.num_scalars()==8
    assert len(list(c.parameters()))==2
    c.close()


def test_full_operator_correct_formula_and_nonmutation():
    layer=torch.nn.Identity()
    b=torch.eye(4)[:,:2];c=ResidualController([(0,'K',layer)],{(0,'K'):b},.5,'full')
    with torch.no_grad(): c.raw['0:K'].copy_(torch.tensor([[1.,2.],[3.,4.]]))
    x=torch.arange(12,dtype=torch.float32).reshape(1,3,4);old=x.clone()
    c.set_mask(torch.ones(1,3,dtype=torch.bool))
    r=c.raw['0:K']; a=.5*r/(1+torch.linalg.vector_norm(r))
    assert torch.allclose(layer(x),x+(x@b)@a@b.T)
    assert torch.equal(x,old);c.close()


def test_post_mask_never_guesses_merged_images():
    ids=torch.tensor([[4,9,9,5,9,9,6,7]])
    mask=visual_mask(ids,9,'post',2)
    assert mask.tolist()==[[False,False,False,False,True,True,False,False]]
    with pytest.raises(ValueError, match='groups'):visual_mask(torch.tensor([[9,9,9,9]]),9,'post',2)


def test_text_mask_excludes_answer_and_padding():
    ids=torch.tensor([[2,9,9,4,5,6,0]])
    mask=visual_mask(ids,9,'text',1,answer_start=5,attention_mask=ids.ne(0))
    assert mask.tolist()==[[True,False,False,True,True,False,False]]


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.k=torch.nn.Linear(4,4,bias=False)
        torch.nn.init.eye_(self.k.weight)
    def forward(self,x): return self.k(x)


def test_fullsupport_fit_freezes_model_and_decreases_loss():
    model=Toy()
    for p in model.parameters():p.requires_grad_(False)
    b=torch.eye(4)[:,:2]; c=ResidualController([(0,'K',model.k)],{(0,'K'):b},1,'full')
    x=torch.ones(1,3,4); mask=torch.ones(1,3,dtype=torch.bool)
    def loss_fn(sample):
        c.set_mask(mask)
        return (model(x)[...,:2]-2).pow(2).mean()
    history=fit_coefficients(model,c,[0,1],loss_fn,steps=8,lr=.1,l2=0,reduction='mean')
    assert history[-1]['data_loss_mean'] < history[0]['data_loss_mean']
    assert model.k.weight.grad is None
    assert torch.equal(model.k.weight,torch.eye(4));c.close()


def test_gradient_extraction_does_not_need_trainable_backbone():
    model=Toy()
    for p in model.parameters():p.requires_grad_(False)
    mask=torch.ones(1,3,dtype=torch.bool)
    def closure(sample):
        return model(torch.ones(1,3,4)).pow(2).mean(),mask
    bases,meta=extract_basis(model,[(0,'K',model.k)],[0,1],closure,rank=2,mode='covariance',seed=0)
    assert bases[(0,'K')].shape==(4,2)
    assert meta['0:K']['energy']>0
    assert model.k.weight.grad is None
    assert not model.k._forward_hooks


def test_loss_reduction_is_explicit_not_silently_changed():
    def run(reduction):
        model=Toy()
        for p in model.parameters():p.requires_grad_(False)
        c=ResidualController([(0,'K',model.k)],{(0,'K'):torch.eye(4)[:,:2]},1,'full')
        def fn(_):
            c.set_mask(torch.ones(1,1,dtype=torch.bool));return model(torch.ones(1,1,4)).sum()
        h=fit_coefficients(model,c,[0,1],fn,steps=1,lr=.01,l2=0,reduction=reduction)
        c.close();return h[0]
    mean,sum_=run('mean'),run('sum')
    assert sum_['objective']==pytest.approx(2*mean['objective'])


def test_hidden_site_preserves_tuple_output_and_backprop():
    from vigor_handoff.core import HiddenSite
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Linear(4,4,bias=False)
        def forward(self,x):return self.w(x), 'unchanged-cache'
    layer=Layer()
    for p in layer.parameters():p.requires_grad_(False)
    site=HiddenSite(layer,4)
    c=ResidualController([(0,'H',site)],{(0,'H'):torch.eye(4)[:,:2]},1,'full')
    c.set_mask(torch.ones(1,2,dtype=torch.bool))
    z,cache=layer(torch.ones(1,2,4))
    assert cache=='unchanged-cache'
    z.sum().backward()
    assert c.raw['0:H'].grad is not None
    assert layer.w.weight.grad is None
    c.close()


def test_hidden_control_intervenes_before_attention_not_after_last_block():
    from vigor_handoff.core import HiddenSite
    class Layer(torch.nn.Module):
        def forward(self, hidden_states, scale=2):
            return hidden_states * scale, 'cache'
    layer=Layer();site=HiddenSite(layer,4)
    handle=site.register_forward_hook(lambda module, inputs, z: z+1)
    x=torch.ones(1,2,4)
    assert torch.equal(layer(x)[0], 2*(x+1))
    assert torch.equal(layer(hidden_states=x)[0], 2*(x+1))
    handle.remove()
    assert torch.equal(layer(x)[0],2*x)
