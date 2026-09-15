#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from eventttt.io import read_task_samples
from eventttt.task_phi import load_phi
from eventttt.phi_kv import discover_phi_kv,extract_phi_subspace,PhiKVController,fit_phi_coefficients,save_phi_state

def main():
 p=argparse.ArgumentParser(); p.add_argument('--support-manifest',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--model-id',required=True); p.add_argument('--rank',type=int,default=16); p.add_argument('--alpha-max',type=float,default=3); p.add_argument('--basis-mode',choices=('covariance','random'),default='covariance'); p.add_argument('--steps',type=int,default=4); p.add_argument('--learning-rate',type=float,default=.05); p.add_argument('--l2',type=float,default=1e-3); p.add_argument('--seed',type=int,default=0); a=p.parse_args()
 samples=read_task_samples(a.support_manifest); model,proc=load_phi(a.model_id); modules,count=discover_phi_kv(model); modules=[x for x in modules if x[0] in {count//2,count-1}]
 bases,spectra=extract_phi_subspace(model,proc,samples,modules,a.rank,a.basis_mode,a.seed); c=PhiKVController(modules,bases,a.rank,a.alpha_max,'full',next(model.parameters()).device); losses=fit_phi_coefficients(model,proc,samples,c,a.steps,a.learning_rate,a.l2)
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); save_phi_state(out/'kv_state.pt',c,a.model_id,{'support_manifest':str(Path(a.support_manifest).resolve())}); (out/'extraction.json').write_text(json.dumps({'method':'phi_gradient_covariance_kv','basis_mode':a.basis_mode,'layers':[x[0] for x in modules],'rank':a.rank,'spectra':spectra,'arguments':vars(a)},indent=2)+'\n'); (out/'adaptation.json').write_text(json.dumps({'losses':losses,'kv_scalars':c.num_scalars()},indent=2)+'\n'); print(json.dumps({'output_dir':str(out),'losses':losses},indent=2))
if __name__=='__main__': main()
