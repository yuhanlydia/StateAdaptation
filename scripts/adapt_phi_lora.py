#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from eventttt.io import read_task_samples
from eventttt.task_phi import load_phi,fit_phi_lora

def main():
 p=argparse.ArgumentParser(); p.add_argument('--support-manifest',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--model-id',required=True); p.add_argument('--passes',type=int,default=4); p.add_argument('--learning-rate',type=float,default=2e-4); p.add_argument('--seed',type=int,default=0); a=p.parse_args(); torch.manual_seed(a.seed)
 samples=read_task_samples(a.support_manifest); model,proc=load_phi(a.model_id,use_lora=True); losses=fit_phi_lora(model,proc,samples,a.passes,a.learning_rate,a.seed)
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out)
 try: proc.save_pretrained(out)
 except AttributeError: pass  # Phi's remote processor lacks chat_template
 (out/'adaptation.json').write_text(json.dumps({'method':'phi_lora_tta','losses':losses,'arguments':vars(a)},indent=2)+'\n'); print(json.dumps({'output_dir':str(out),'losses':losses},indent=2))
if __name__=='__main__': main()
