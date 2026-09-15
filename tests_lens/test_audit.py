import importlib
import torch
import pytest
from vigor_handoff.core import ResidualController


def module():
    assert importlib.util.find_spec('visual_lens.audit') is not None, 'audit implementation required'
    return importlib.import_module('visual_lens.audit')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.k=torch.nn.Linear(4,4,bias=False)
        self.gradient_checkpointing=False;torch.nn.init.eye_(self.k.weight)
        for p in self.parameters():p.requires_grad_(False)
    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None):
        assert gradient_checkpointing_kwargs['use_reentrant'] is False
        self.gradient_checkpointing=True
    def gradient_checkpointing_disable(self):self.gradient_checkpointing=False
    def forward(self,x):
        from torch.utils.checkpoint import checkpoint
        def f(z):return self.k(z).sin().square()
        return checkpoint(f,x,use_reentrant=False) if self.training and self.gradient_checkpointing else f(x)


class Back:
    def __init__(self,model):self.model=model
    def loss_and_mask(self,sample,controller=None):
        mask=torch.tensor([[False,True,False]])
        if controller is not None:controller.set_mask(mask)
        x=torch.ones(1,3,4,requires_grad=True)*sample
        return self.model(x).sum(),mask


def test_checkpoint_replay_preserves_controller_gradient_and_mask():
    m=module();model=Model();mods=[(0,'K',model.k)]
    c=ResidualController(mods,{(0,'K'):torch.eye(4)[:,:2]})
    with torch.no_grad():c.raw['0:K'].fill_(.04)
    a=m.gradient_signature(model,Back(model),[1.,2.],mods,c,checkpoint=False)
    b=m.gradient_signature(model,Back(model),[1.,2.],mods,c,checkpoint=True)
    assert torch.allclose(a['controller_gradient'],b['controller_gradient'],atol=1e-6)
    assert b['forward_counts']['0:K']>a['forward_counts']['0:K']
    assert c._mask is None;c.close()


def test_sum_mean_gradient_scaling_tracks_actual_objective():
    m=module();model=Model();mods=[(0,'K',model.k)]
    c=ResidualController(mods,{(0,'K'):torch.eye(4)[:,:2]})
    a=m.gradient_signature(model,Back(model),[1.,2.],mods,c,False,'mean')
    b=m.gradient_signature(model,Back(model),[1.,2.],mods,c,False,'sum')
    assert b['loss']==pytest.approx(2*a['loss'])
    assert torch.allclose(b['controller_gradient'],2*a['controller_gradient'],atol=1e-6)
    c.close()
