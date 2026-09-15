"""Gradient-Covariance KV-TTT for Phi's fused language qkv projection.

Phi exposes one ``qkv_proj`` rather than separate K/V Linear modules.  The
three contiguous hidden-size slices are introspected and the controller edits
only the K and V slices, preserving the same visual-mask and bounded-residual
contract as the Qwen implementation.
"""
from __future__ import annotations
import re
from typing import Sequence
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .kv_ttt import freeze_model
from .task_phi import labeled_batch

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.qkv_proj$")

def phi_visual_mask(input_ids):
    if input_ids.ndim != 2: raise ValueError("input_ids must be [B,T]")
    mask = input_ids < 0
    if not bool(mask.any()): raise ValueError("Phi batch has no negative visual token IDs")
    return mask

def discover_phi_kv(model):
    found=[]
    for name,module in model.named_modules():
        m=_LAYER_RE.search(name)
        if m is None: continue
        hidden=module.in_features
        if module.out_features != 3*hidden:
            raise RuntimeError(f"unexpected Phi qkv shape at {name}")
        found.append((int(m.group(1)),module,hidden))
    if not found: raise RuntimeError("Phi language decoder has no fused qkv_proj modules")
    return sorted(found), max(x[0] for x in found)+1

class PhiKVController(nn.Module):
    def __init__(self, modules, bases, rank=16, alpha_max=3.0, coefficient_mode='full', device=None):
        super().__init__(); self.modules=list(modules); self.rank=int(rank); self.alpha_max=float(alpha_max); self.coefficient_mode=coefficient_mode
        self._device=device or next(iter(bases.values())).device; self.bases={k:v.float().to(self._device) for k,v in bases.items()}
        self.coefficients=nn.ParameterDict()
        for layer,_,_ in self.modules:
            for kind in ('K','V'):
                shape=(rank,rank) if coefficient_mode=='full' else (rank,)
                self.coefficients[f'{layer}:{kind}']=nn.Parameter(torch.zeros(shape,device=self._device))
        self._active_mask=None; self._hooks=[m.register_forward_hook(self._hook(layer,m)) for layer,m,_ in self.modules]
    def _hook(self,layer,module):
        def hook(mod,inp,out):
            if self._active_mask is None: return out
            updated=out.clone(); hidden=out.shape[-1]//3
            mask=self._active_mask.to(out.device).unsqueeze(-1)
            for kind,start in [('K',hidden),('V',2*hidden)]:
                seg=out[...,start:start+hidden]; b=self.bases[(layer,kind)].to(seg.device,seg.dtype); raw=self.coefficients[f'{layer}:{kind}']; low=seg@b
                if self.coefficient_mode=='diagonal': mixed=low*(self.alpha_max*torch.tanh(raw)).to(seg.dtype)
                else: mixed=low@((self.alpha_max*raw/(1+torch.linalg.vector_norm(raw))).to(seg.dtype))
                updated[...,start:start+hidden]=seg+mask*(mixed@b.transpose(-1,-2))
            return updated
        return hook
    def set_mask(self,m): self._active_mask=m.to(self._device)
    def clear_mask(self): self._active_mask=None
    def close(self):
        self.clear_mask()
        for h in self._hooks: h.remove()
        self._hooks.clear()
    def ttt_parameters(self): return list(self.coefficients.parameters())
    def num_scalars(self): return sum(p.numel() for p in self.ttt_parameters())

def extract_phi_subspace(model,processor,samples,modules,rank=16,basis_mode='covariance',seed=0,batch_builder=None,mask_builder=None):
    freeze_model(model); model.eval(); device=next(model.parameters()).device
    batch_builder = batch_builder or labeled_batch
    mask_builder = mask_builder or phi_visual_mask
    model.config.use_cache=False
    checkpoint=getattr(model,'gradient_checkpointing_enable',None)
    if callable(checkpoint): checkpoint()
    enable=getattr(model,'enable_input_require_grads',None)
    if callable(enable): enable()
    dims={(layer,k):dim for layer,_,dim in modules for k in ('K','V')}
    if basis_mode=='random':
        g=torch.Generator().manual_seed(seed); return {(l,k):torch.linalg.qr(torch.randn(d,rank,generator=g))[0][:,:rank] for (l,k),d in dims.items()},{}
    cov={k:torch.zeros(d,d,device=device) for k,d in dims.items()}; saved={}; handles=[]
    def cap(layer):
        def f(mod,inp,out): out.retain_grad(); saved[layer]=out
        return f
    for layer,mod,_ in modules: handles.append(mod.register_forward_hook(cap(layer)))
    try:
        for sample in tqdm(samples,desc='Phi KV gradients',dynamic_ncols=True):
            batch,_=batch_builder(processor,sample); batch={k:v.to(device) for k,v in batch.items()}; loss=model(**batch).loss; loss.backward(); mask=mask_builder(batch['input_ids'])
            for layer,_,dim in modules:
                grad=saved[layer].grad
                for kind,start in [('K',dim),('V',2*dim)]:
                    g=grad[...,start:start+dim][mask].float(); cov[(layer,kind)].add_(g.t()@g)
            model.zero_grad(set_to_none=True)
    finally:
        for h in handles: h.remove()
        disable=getattr(model,'disable_input_require_grads',None)
        if callable(disable): disable()
    bases={}; spectra={}
    for key,matrix in cov.items():
        vals,vecs=torch.linalg.eigh(matrix.detach().cpu()); vals,order=torch.sort(vals,descending=True); bases[key]=vecs[:,order[:rank]].contiguous(); spectra[str(key)]=[float(v) for v in vals[:rank*3]]
    return bases,spectra

def fit_phi_coefficients(model,processor,samples,controller,steps=4,learning_rate=.05,l2=1e-3,batch_builder=None,mask_builder=None):
    device=next(model.parameters()).device; batch_builder = batch_builder or labeled_batch; mask_builder = mask_builder or phi_visual_mask; opt=torch.optim.Adam(controller.ttt_parameters(),lr=learning_rate); losses=[]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); total=0.
        for sample in samples:
            batch,_=batch_builder(processor,sample); batch={k:v.to(device) for k,v in batch.items()}; controller.set_mask(mask_builder(batch['input_ids'])); loss=model(**batch).loss; controller.clear_mask(); loss.backward(); total+=float(loss.detach())
        penalty=l2*sum(p.pow(2).sum() for p in controller.ttt_parameters()); penalty.backward(); torch.nn.utils.clip_grad_norm_(controller.ttt_parameters(),1.); opt.step(); losses.append(total/len(samples)+float(penalty.detach()))
    return losses

def save_phi_state(path,controller,model_id,metadata=None):
    torch.save({'version':1,'model_id':model_id,'rank':controller.rank,'layers':[l for l,_,_ in controller.modules],'alpha_max':controller.alpha_max,'coefficient_mode':controller.coefficient_mode,'bases':{f'{l}:{k}':v.cpu() for (l,k),v in controller.bases.items()},'coefficients_raw':{k:v.detach().cpu() for k,v in controller.coefficients.items()},'metadata':metadata or {}},path)
