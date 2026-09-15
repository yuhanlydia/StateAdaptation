"""Auditable masked visual-state adaptation, independent of VLM architecture.

The operator matches EventTune @ 3150cae. This implementation adds explicit
loss-reduction accounting and one-factor controls without rewriting old runs.
All geometry statistics are descriptive; none certify target improvement.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable
import math

import torch
from torch import nn


class Moments:
    """Token-weighted FP64 CPU sufficient statistics; no gradient centering by default."""
    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError('dim must be positive')
        self.dim, self.n = dim, 0
        self.second = torch.zeros(dim, dim, dtype=torch.float64)
        self.total = torch.zeros(dim, dtype=torch.float64)

    def add(self, rows: torch.Tensor) -> None:
        rows = rows.detach().reshape(-1, self.dim).to(device='cpu', dtype=torch.float64)
        if not torch.isfinite(rows).all():
            raise ValueError('non-finite moment input')
        self.second.add_(rows.T @ rows)
        self.total.add_(rows.sum(0))
        self.n += len(rows)

    def matrix(self, mode='covariance') -> torch.Tensor:
        if self.n == 0:
            raise ValueError('no rows accumulated')
        if mode == 'covariance':
            result = self.second
        elif mode in ('centered', 'activation_pca'):
            result = self.second - torch.outer(self.total, self.total) / self.n
        elif mode == 'mean':
            result = torch.outer(self.total, self.total) / self.n**2
        else:
            raise ValueError(f'unknown moment mode {mode}')
        return ((result + result.T) * .5).float()

    def basis(self, rank: int, mode='covariance', seed=0):
        if rank < 1 or rank > self.dim:
            raise ValueError(f'rank {rank} outside [1,{self.dim}]')
        if mode == 'random':
            g = torch.Generator(device='cpu').manual_seed(seed)
            b = torch.linalg.qr(torch.randn(self.dim, rank, generator=g), mode='reduced').Q
            return b, {'rows': self.n, 'energy': None, 'effective_rank': rank, 'eigenvalues': []}
        c = self.matrix(mode)
        if not torch.isfinite(c).all() or float(torch.trace(c)) <= 0:
            raise ValueError('zero or non-finite gradient energy; do not silently replace by random basis')
        values, vectors = torch.linalg.eigh(c)
        order = torch.argsort(values, descending=True)
        values = values[order]
        b = vectors[:, order[:rank]].contiguous()
        effective_rank = int((values > values[0].clamp_min(1e-30)*1e-6).sum())
        return b, {'rows': self.n, 'energy': float(torch.trace(c)),
                   'effective_rank': effective_rank,
                   'rank_deficient': effective_rank < rank,
                   'eigenvalues': values[:min(3*rank,len(values))].tolist(),
                   'captured_support_energy': float(values[:rank].sum()/values.clamp_min(0).sum())}


def visual_mask(input_ids, image_token_id: int, mode: str, expected_groups: int,
                answer_start=None, attention_mask=None):
    """Fail closed on unknown image boundaries; never split one run at its midpoint.

    For the text ablation, exclude answer/completion and padding. Text includes
    prompt-format tokens and is a control, not an automatically localized cue.
    """
    if input_ids.ndim != 2:
        raise ValueError('input_ids must be [batch,sequence]')
    if mode not in ('post','pre','all_visual','text'):
        raise ValueError(f'unknown mask {mode}')
    result = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, ids in enumerate(input_ids):
        pos = (ids == image_token_id).nonzero(as_tuple=False).flatten()
        if not len(pos):
            raise ValueError('no image token groups')
        cuts = (pos[1:] != pos[:-1]+1).nonzero(as_tuple=False).flatten()+1
        groups = list(torch.tensor_split(pos, cuts.tolist()))
        if len(groups) != expected_groups:
            raise ValueError(f'expected {expected_groups} image groups, observed {len(groups)}')
        if mode == 'all_visual':
            result[row,pos] = True
        elif mode in ('post','pre'):
            if expected_groups != 2:
                raise ValueError('pre/post masks require a genuine two-image prompt')
            result[row,groups[1 if mode=='post' else 0]] = True
        else:
            if answer_start is None:
                raise ValueError('text mask requires the exact answer_start')
            starts = answer_start if isinstance(answer_start,(tuple,list)) else [answer_start]*len(input_ids)
            result[row,:int(starts[row])] = True
            result[row,pos] = False
    if attention_mask is not None:
        result &= attention_mask.bool()
    return result


class ResidualController(nn.Module):
    """Same bounded Full/Diagonal operator, with removable projection-output hooks.

    Module objects live in an ordinary list, never an nn.ModuleList: the optimizer
    must not accidentally discover the backbone as a child of this controller.
    """
    def __init__(self, modules, bases, alpha=3.0, mode='full', hard_projection=False):
        super().__init__()
        if mode not in ('full','diagonal') or not math.isfinite(alpha) or alpha < 0:
            raise ValueError('invalid controller mode/alpha')
        self.targets=list(modules); self.alpha=float(alpha); self.mode=mode
        self.hard_projection=hard_projection
        self.bases={k:v.detach().float() for k,v in bases.items()}
        self.raw=nn.ParameterDict()
        self._mask=None; self._handles=[]
        for layer,kind,module in self.targets:
            b=self.bases[(layer,kind)]
            rank=b.shape[1]
            if not torch.allclose(b.T@b,torch.eye(rank,device=b.device),atol=2e-4):
                raise ValueError('basis is not orthonormal')
            device=next(module.parameters(),b).device
            self.raw[f'{layer}:{kind}']=nn.Parameter(torch.zeros(
                (rank,rank) if mode=='full' else (rank,),device=device),
                requires_grad=not hard_projection)
            self._handles.append(module.register_forward_hook(self._hook(layer,kind)))

    def operators(self):
        result={}
        for key,r in self.raw.items():
            if self.mode=='full':
                a=self.alpha*r/(1+torch.linalg.vector_norm(r))
            else:
                a=torch.diag(self.alpha*torch.tanh(r))
            result[key]=a
        return result

    def _hook(self,layer,kind):
        def apply(module,inputs,z):
            mask=self._mask
            if mask is None:
                return z
            if not torch.is_tensor(z) or z.ndim!=3 or tuple(z.shape[:2])!=tuple(mask.shape):
                raise ValueError('projection output/mask shape mismatch; generation cache is unsupported')
            b=self.bases[(layer,kind)].to(z)
            low=z@b
            r=self.raw[f'{layer}:{kind}']
            if self.hard_projection:
                changed=low@b.T
                delta=changed-z
            elif self.mode=='diagonal':
                delta=(low*(self.alpha*torch.tanh(r)).to(z))@b.T
            else:
                a=(self.alpha*r/(1+torch.linalg.vector_norm(r))).to(z)
                delta=(low@a)@b.T
            return z+mask.to(z).unsqueeze(-1)*delta
        return apply

    def set_mask(self,mask):self._mask=mask
    def clear_mask(self):self._mask=None
    def num_scalars(self):return sum(p.numel() for p in self.raw.values() if p.requires_grad)
    def close(self):
        self.clear_mask()
        for handle in self._handles:handle.remove()
        self._handles.clear()
    def payload(self):
        return {'bases':{f'{l}:{k}':v.cpu() for (l,k),v in self.bases.items()},
                'raw':{k:p.detach().cpu() for k,p in self.raw.items()},
                'alpha':self.alpha,'mode':self.mode,'hard_projection':self.hard_projection}
    def restore(self,payload):
        with torch.no_grad():
            for k,v in payload['raw'].items():self.raw[k].copy_(v)


def extract_basis(model,modules,samples,loss_and_mask: Callable,rank=16,
                  mode='covariance',seed=0):
    """Compute bases without training weights. `loss_and_mask` owns each forward.

    Projection outputs become grad-requiring leaves only when no upstream
    differentiable state exists; an existing graph is never detached. This avoids
    unnecessary embedding gradients but preserves derivatives to earlier sites.
    """
    if mode not in ('covariance','centered','mean','activation_pca','random'):
        raise ValueError(f'unsupported basis mode {mode}')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('freeze every backbone parameter before basis extraction')
    model.eval()
    moments={(l,k):Moments(m.out_features) for l,k,m in modules}
    if mode=='random':
        out={};meta={}
        for j,(key,m) in enumerate(moments.items()):
            out[key],meta[f'{key[0]}:{key[1]}']=m.basis(rank,'random',seed+j)
        return out,meta
    saved={}; handles=[]
    def capture(key):
        def hook(module,inp,z):
            if not z.requires_grad:z.requires_grad_(True)
            z.retain_grad();saved[key]=z
        return hook
    for l,k,m in modules:handles.append(m.register_forward_hook(capture((l,k))))
    try:
        for sample in samples:
            saved.clear()
            with torch.enable_grad():
                loss,mask=loss_and_mask(sample)
                if not torch.isfinite(loss):raise ValueError('non-finite extraction loss')
                if mode!='activation_pca':loss.backward()
            for key,z in saved.items():
                rows=z if mode=='activation_pca' else z.grad
                if rows is None:raise ValueError(f'no gradient at {key}')
                if tuple(rows.shape[:2])!=tuple(mask.shape):raise ValueError('mask shape mismatch')
                moments[key].add(rows[mask])
            model.zero_grad(set_to_none=True)
            saved.clear();del loss
    finally:
        for handle in handles:handle.remove()
        saved.clear()
    bases={};meta={}
    for key,m in moments.items():
        bases[key],meta[f'{key[0]}:{key[1]}']=m.basis(rank,mode,seed)
    return bases,meta


def fit_coefficients(model,controller,samples,loss_fn: Callable,*,steps=4,
                     lr=.05,l2=.001,reduction='mean',clip=1.0):
    """One update per complete support pass; mask remains active through backward.

    reduction='sum' explicitly reproduces the archived task_kv.py gradient
    scaling; reduction='mean' matches the paper equation and BRIGHT backend.
    Both the actual objective and mean data loss are recorded separately.
    """
    samples=list(samples)
    if not samples or steps<0 or lr<=0 or l2<0 or reduction not in ('mean','sum'):
        raise ValueError('invalid fitting configuration')
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('model parameters must remain frozen')
    model.eval()
    params=[p for p in controller.parameters() if p.requires_grad]
    if not params or steps==0:return []
    optimizer=torch.optim.Adam(params,lr=lr)
    history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True);total=0.
        for sample in samples:
            try:
                loss=loss_fn(sample)
                if not torch.isfinite(loss):raise ValueError('non-finite adaptation loss')
                (loss/(len(samples) if reduction=='mean' else 1)).backward()
                total+=float(loss.detach())
            finally:
                controller.clear_mask()
        penalty=l2*sum(p.square().sum() for p in params)
        penalty.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,clip,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in params):raise ValueError('non-finite controller')
        history.append({'step':step+1,'data_loss_mean':total/len(samples),
                        'penalty':float(penalty.detach()),'reduction':reduction,
                        'objective':total/(len(samples) if reduction=='mean' else 1)+float(penalty.detach()),
                        'gradient_norm_before_clip':float(norm)})
    return history


class HiddenSite:
    """Expose decoder-block INPUT states as a hookable control site.

    An edit to final-block visual outputs cannot reach the later answer token
    in the same forward, because no attention layer follows it. Using the
    block input keeps this baseline causally comparable to pre-attention KV.
    This control is not a reproduction of ReFT's learned parameterization.
    """
    def __init__(self,module,dim):self.module=module;self.out_features=dim
    def parameters(self):return self.module.parameters()
    def register_forward_hook(self,hook):
        def wrapped(module,inputs,kwargs):
            positional=bool(inputs)
            if positional:
                z=inputs[0]
            elif 'hidden_states' in kwargs:
                z=kwargs['hidden_states']
            else:
                raise ValueError('decoder block has no identifiable hidden-state input')
            result=hook(self,inputs,z)
            if result is None:return None
            if positional:return (result,*inputs[1:]),kwargs
            changed=dict(kwargs);changed['hidden_states']=result
            return inputs,changed
        return self.module.register_forward_pre_hook(wrapped,with_kwargs=True)
