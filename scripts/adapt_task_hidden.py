#!/usr/bin/env python3
"""Fit the frozen hidden-residual single-image baseline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from eventttt.io import read_task_samples
from eventttt.qwen import DEFAULT_MODEL, load_model, preflight
from eventttt.task_kv import task_visual_mask
from eventttt.hidden_residual import (HiddenResidualController, discover_hidden_layers,
    extract_hidden_subspace, fit_hidden_coefficients, save_hidden_state)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--support-manifest', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--model-id', default=DEFAULT_MODEL); p.add_argument('--rank', type=int, default=16)
    p.add_argument('--alpha-max', type=float, default=3.0); p.add_argument('--basis-mode', choices=('covariance','random'), default='covariance')
    p.add_argument('--coefficient-mode', choices=('full','diagonal'), default='full'); p.add_argument('--steps', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=.05); p.add_argument('--l2', type=float, default=1e-3); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--layers', nargs='+', type=int, default=None)
    a=p.parse_args(); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    print(json.dumps(preflight(require_gpu=True), indent=2))
    samples=read_task_samples(a.support_manifest)
    model, processor=load_model(a.model_id, gradient_checkpointing=False, use_lora=False)
    all_layers, count=discover_hidden_layers(model)
    selected=a.layers if a.layers is not None else [count//2, count-1]
    by={layer: module for layer,module in all_layers}
    if any(layer not in by for layer in selected): raise ValueError(f'missing layers: {selected}')
    modules=[(layer,by[layer]) for layer in selected]; mask=task_visual_mask(model)
    bases,spectra=extract_hidden_subspace(model, processor, samples, modules, mask, a.rank, a.basis_mode, a.seed)
    state={'bases':bases,'rank':a.rank,'alpha_max':a.alpha_max,'coefficient_mode':a.coefficient_mode}
    controller=HiddenResidualController(modules,bases,rank=a.rank,alpha_max=a.alpha_max,coefficient_mode=a.coefficient_mode,device=next(model.parameters()).device)
    losses=fit_hidden_coefficients(model,processor,samples,controller,mask,a.steps,a.learning_rate,a.l2)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    save_hidden_state(out/'hidden_state.pt',controller,a.model_id,{'support_manifest':str(Path(a.support_manifest).resolve())})
    (out/'extraction.json').write_text(json.dumps({'method':'hidden_residual','basis_mode':a.basis_mode,'support_examples':len(samples),'rank':a.rank,'layers':selected,'spectra':spectra,'arguments':vars(a)},indent=2)+'\n')
    (out/'adaptation.json').write_text(json.dumps({'losses':losses,'steps':a.steps,'learning_rate':a.learning_rate,'l2':a.l2,'hidden_scalars':controller.num_scalars()},indent=2)+'\n')
    print(json.dumps({'output_dir':str(out),'losses':losses,'hidden_scalars':controller.num_scalars()},indent=2))
if __name__=='__main__': main()
