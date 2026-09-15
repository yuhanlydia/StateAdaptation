"""One fresh-model GPU job. Uses EventTune's verified loaders and prompt builders.

All files are written under a versioned new run root. No legacy report is edited.
Weights and data must already be present locally; authentication is never logged.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from contextlib import nullcontext
import importlib.metadata
import json
import os
import random
import re
import time

import numpy as np
import torch

from .core import extract_basis,fit_coefficients,ResidualController,visual_mask,HiddenSite
from .protocol import (JobStore,atomic_json,digest,file_hash,preflight_job,
                        read_jsonl,resolved_manifest,validate_predictions)


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def fingerprint_assets(job):
    """Content-hash model shards once per path+size+mtime; never identify by name only."""
    cache_root=Path(job['output']).parents[1]/'_asset_fingerprints'
    cache_root.mkdir(parents=True,exist_ok=True)
    root=Path(job['model_path']).resolve()
    model_files=sorted(p for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts
                       and p.suffix in ('.json','.safetensors','.bin','.model','.py','.txt'))
    if not model_files:raise FileNotFoundError('empty model snapshot')
    statkey=digest([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in model_files])
    cache=cache_root/f'{statkey}.json'
    if cache.exists():model_hash=json.loads(cache.read_text())['hash']
    else:
        model_hash=digest([(str(p.relative_to(root)),file_hash(p)) for p in model_files])
        atomic_json(cache,{'hash':model_hash,'files':len(model_files),'cache_key':'path-size-mtime'})
    source_files=sorted(list(Path('src/eventttt').glob('*.py'))+list(Path('src/vigor_handoff').glob('*.py'))+list(Path('src/visual_lens').glob('*.py'))+list(Path('scripts').glob('*.py')))
    source_hash=digest([(str(p),file_hash(p)) for p in source_files])
    support=resolved_manifest(job['support']);query=resolved_manifest(job['query'])
    image_files=sorted({r[k] for r in support+query for k in ('image','pre_image','post_image') if r.get(k)})
    image_key=digest([(p,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in image_files])
    image_cache=cache_root/f'images-{image_key}.json'
    if image_cache.exists():image_hash=json.loads(image_cache.read_text())['hash']
    else:
        image_hash=digest([(p,file_hash(p)) for p in image_files])
        atomic_json(image_cache,{'hash':image_hash})
    return {'job':job,'model_content':model_hash,'source_content':source_hash,
            'support_manifest':file_hash(job['support']),'query_manifest':file_hash(job['query']),
            'image_content':image_hash,
            'torch':torch.__version__,
            'transformers':importlib.metadata.version('transformers'),
            'peft':importlib.metadata.version('peft')}


class Backend:
    """One labelled forward and one candidate scorer, with consistent token spans."""
    def __init__(self,job,model,processor):
        self.job=job;self.model=model;self.processor=processor
        self.device=next(model.parameters()).device
        from eventttt.bright_kv import image_token_id
        self.image_token=image_token_id(model,processor)
        self.groups=2 if job['kind']=='paired' else 1
        self._tokenization_logged={}
        self.image_transform=None
        self.last_token_contract=None

    def batch(self,sample,label):
        from eventttt.schemas import DAMAGE_LABELS
        if self.job['kind']=='single':
            from eventttt.task_qwen import _batch_for_candidates,load_task_image
            image=load_task_image(sample)
            if self.image_transform is not None:image=self.image_transform(sample,(image,))[0]
            b,spans=_batch_for_candidates(self.processor,sample,image,(label,),self.job['family'])
            span=spans[0]
        else:
            from eventttt.qwen import load_sample_pair
            pre,post=load_sample_pair(sample,self.job['crop_size'])
            if self.image_transform is not None:pre,post=self.image_transform(sample,(pre,post))
            if self.job['family']=='qwen2':
                from eventttt.qwen import _candidate_batch
                b,spans=_candidate_batch(self.processor,sample,pre,post,candidate_labels=(label,));span=spans[0]
            else:
                from eventttt.bright_vlm import _inputs_for_candidate
                b,span=_inputs_for_candidate(self.processor,self.job['family'],sample,label,pre,post)
        b={k:v.to(self.device) for k,v in b.items()}
        self._tokenization_logged[label]=b['input_ids'][0,span[0]:span[1]].tolist()
        self.last_token_contract={'input_ids':b['input_ids'][0].tolist(),
             'span':list(span),'visual_tokens':int((b['input_ids']==self.image_token).sum()),
             'grid':b['image_grid_thw'].tolist() if 'image_grid_thw' in b else None}
        return b,span

    def mask(self,b,span):
        return visual_mask(b['input_ids'],self.image_token,self.job['mask'],self.groups,
                           answer_start=span[0],attention_mask=b.get('attention_mask'))

    def loss_and_mask(self,sample,controller=None):
        b,span=self.batch(sample,sample.label)
        mask=self.mask(b,span)
        labels=torch.full_like(b['input_ids'],-100)
        loss_span=span
        if self.job.get('loss_span','full')=='label':
            from visual_lens.controls import select_label_span
            loss_span=select_label_span(b['input_ids'],span,sample.label,self.processor.tokenizer)
        elif self.job.get('loss_span','full')!='full':raise ValueError('unknown loss span')
        labels[0,loss_span[0]:loss_span[1]]=b['input_ids'][0,loss_span[0]:loss_span[1]]
        b['labels']=labels
        if controller is not None:controller.set_mask(mask)
        out=self.model(**b)
        return out.loss,mask

    @torch.inference_mode()
    def score(self,sample,controller=None):
        from eventttt.schemas import DAMAGE_LABELS
        labels=tuple(sample.candidate_labels) if self.job['kind']=='single' else DAMAGE_LABELS
        self.model.eval();scores=[];contracts=[]
        for label in labels:
            b,span=self.batch(sample,label)
            contracts.append(self.last_token_contract)
            if controller is not None:controller.set_mask(self.mask(b,span))
            try:
                logits=self.model(**b).logits
                targets=b['input_ids'][0,span[0]:span[1]]
                logp=logits[0,span[0]-1:span[1]-1].float().log_softmax(-1)
                scores.append(float(logp.gather(-1,targets[:,None]).sum()))
            finally:
                if controller is not None:controller.clear_mask()
        s=np.asarray(scores,dtype=float);p=np.exp(s-s.max());p/=p.sum()
        return {'sample_id':sample.sample_id,'group_id':getattr(sample,'group_id',getattr(sample,'tile_id','')),
                'label':sample.label,'label_id':sample.label_id,'probabilities':p.tolist(),
                'prediction':labels[int(p.argmax())],'mean_log_scores':scores,
                'prompt_contract_hash':digest(contracts),
                'visual_tokens':contracts[0]['visual_tokens']}


def discover_sites(model,layers,sites):
    pattern=re.compile(r'\.layers\.(\d+)\.self_attn\.([qkvo])_proj$')
    matches=[]
    if 'H' in sites:
        hidden_pattern=re.compile(r'\.layers\.(\d+)$')
        for name,module in model.named_modules():
            if any(s in name.lower() for s in ('vision_tower','vision_model','vision_encoder','visual.')):continue
            hit=hidden_pattern.search(name)
            if hit and int(hit[1]) in layers:
                attention=getattr(module,'self_attn',None)
                output_projection=getattr(attention,'o_proj',None)
                if output_projection is None:raise ValueError('cannot determine hidden width')
                matches.append((int(hit[1]),'H',HiddenSite(module,output_projection.out_features)))
    for name,module in model.named_modules():
        if any(s in name.lower() for s in ('vision_tower','vision_model','vision_encoder','visual.')):continue
        m=pattern.search(name)
        if m and int(m[1]) in layers and m[2].upper() in sites:
            if not hasattr(module,'out_features'):raise ValueError(f'{name} has no projection width')
            matches.append((int(m[1]),m[2].upper(),module))
    expected={(l,s) for l in layers for s in sites}
    found={(l,s) for l,s,_ in matches}
    if found!=expected or len(matches)!=len(expected):
        raise ValueError(f'missing/duplicate decoder sites: expected {expected}, got {found}')
    return matches


def train_lora(model,backend,support,job):
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise ValueError('no trainable LoRA parameters')
    if any('lora_' not in name for name,p in model.named_parameters() if p.requires_grad):
        raise ValueError('non-LoRA weights unexpectedly trainable')
    opt=torch.optim.AdamW(params,lr=job['lora_lr']);model.train();history=[];updates=0
    g=torch.Generator().manual_seed(job['optimization_seed'])
    accum=job['lora_accumulation']
    for epoch in range(job['lora_passes']):
        order=torch.randperm(len(support),generator=g).tolist();total=0.;opt.zero_grad(set_to_none=True)
        for position,index in enumerate(order):
            loss,_=backend.loss_and_mask(support[index])
            chunk_start=(position//accum)*accum
            denom=min(accum,len(order)-chunk_start)
            (loss/denom).backward();total+=float(loss.detach())
            if (position+1)%accum==0 or position+1==len(order):
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                opt.step();opt.zero_grad(set_to_none=True);updates+=1
        history.append({'pass':epoch+1,'data_loss_mean':total/len(support)})
    return history,updates,sum(p.numel() for p in params)


def execute_job(job):
    if not torch.cuda.is_available():raise RuntimeError('GPU execution requested but CUDA is unavailable')
    preflight_job(job)
    identity=fingerprint_assets(job)
    store=JobStore(job['output'],identity)
    if store.complete():return {'status':'skipped','job_id':job['job_id']}
    if (store.path/'DONE.json').exists():raise RuntimeError('completed artifacts were modified; inspect rather than overwrite')
    from eventttt.task_vlm import load_task_model
    from eventttt.io import read_samples,read_task_samples
    from eventttt.metrics import classification_metrics_nclass
    seed_everything(job['optimization_seed'])
    read=read_samples if job['kind']=='paired' else read_task_samples
    support=read(job['support']);query=read(job['query'])
    query_rows=resolved_manifest(job['query'])
    is_lora=job['arm'].startswith('lora')
    if job['lora_rank']!=16 or job['lora_alpha']!=32:
        raise ValueError('current loader is pinned to LoRA rank16/alpha32; do not silently ignore requested values')
    with store.lock():
        controller=None;model=None;start=time.perf_counter()
        try:
            model,processor=load_task_model(job['model_path'],job['family'],
                                           use_lora=is_lora,gradient_checkpointing=is_lora)
            device_types={p.device.type for p in model.parameters()}
            if device_types!={'cuda'}:
                raise RuntimeError(f'CPU/meta offload detected {device_types}; do not silently change TTA execution')
            backend=Backend(job,model,processor)
            model.eval();model.config.use_cache=False
            adaptation_path=store.path/'ADAPTED.json'
            resume=adaptation_path.exists()
            if not resume and (store.path/'predictions.jsonl').exists():
                raise RuntimeError('predictions exist without a sealed adaptation state; refuse unsafe resume')
            if resume:
                saved=json.loads(adaptation_path.read_text())
                for p,h in saved.get('artifact_hashes',{}).items():
                    if file_hash(store.path/p)!=h:raise ValueError('adaptation artifact hash mismatch')
            else:saved={}
            if is_lora:
                if resume:
                    # Reattach saved learned weights to an unadapted fresh base.
                    from peft import set_peft_model_state_dict
                    from safetensors.torch import load_file
                    set_peft_model_state_dict(model,load_file(str(store.path/'adapter/adapter_model.safetensors')))
                    adaptation=saved
                else:
                    losses,updates,count=train_lora(model,backend,support,job)
                    model.save_pretrained(store.path/'adapter',safe_serialization=True)
                    adaptation={'losses':losses,'optimizer_updates':updates,'trainable_scalars':count}
            elif job['arm']=='frozen':
                adaptation={'trainable_scalars':0,'optimizer_updates':0}
            else:
                for p in model.parameters():p.requires_grad_(False)
                disable=getattr(model,'gradient_checkpointing_disable',None)
                if callable(disable):disable()
                model.eval()
                mods=discover_sites(model,job['layers'],job['sites'])
                if resume:
                    payload=torch.load(store.path/'controller.pt',map_location='cpu',weights_only=True)
                    bases={(int(k.split(':')[0]),k.split(':')[1]):v for k,v in payload['bases'].items()}
                    spectra=saved.get('spectra',{})
                else:
                    if job['basis_source']=='query' and job['evidence']!='oracle_diagnostic':
                        raise ValueError('query-basis forbidden outside oracle diagnostics')
                    basis_samples=query if job['basis_source']=='query' else support
                    if job.get('shuffle_labels'):
                        labels=[s.label for s in basis_samples]
                        g=torch.Generator().manual_seed(job['optimization_seed']+971)
                        order=torch.randperm(len(labels),generator=g).tolist()
                        candidates=tuple(query_rows[0].get('candidate_labels',('intact','damaged','destroyed')))
                        basis_samples=[replace(s,label=labels[j],label_id=candidates.index(labels[j]))
                                       for s,j in zip(basis_samples,order)]
                    bases,spectra=extract_basis(model,mods,basis_samples,backend.loss_and_mask,
                                              job['rank'],job['basis_mode'],job['basis_seed'])
                controller=ResidualController(mods,bases,job['alpha'],job['coefficient_mode'],job.get('hard_projection',False))
                if resume:
                    controller.restore(payload);adaptation=saved
                else:
                    # Strong integration gate: zero state must reproduce frozen candidate scores.
                    if not job.get('hard_projection'):
                        a=backend.score(support[0]);b=backend.score(support[0],controller)
                        diff=float(np.max(np.abs(np.asarray(a['mean_log_scores'])-b['mean_log_scores'])))
                        if diff>1e-3:raise RuntimeError(f'zero-controller identity gate failed: {diff}')
                    losses=fit_coefficients(model,controller,support,
                        lambda s:backend.loss_and_mask(s,controller)[0],steps=job['steps'],lr=job['lr'],
                        l2=job['l2'],reduction=job['loss_reduction'])
                    torch.save(controller.payload(),store.path/'controller.pt')
                    adaptation={'trainable_scalars':controller.num_scalars(),'optimizer_updates':len(losses),
                                'losses':losses,'spectra':spectra,
                                'operator_norms':{k:float(torch.linalg.matrix_norm(a.detach(),ord=2)) for k,a in controller.operators().items()}}
            if not resume:
                names=['controller.pt'] if controller else (['adapter/adapter_config.json','adapter/adapter_model.safetensors'] if is_lora else [])
                adaptation.update(adaptation_seconds=time.perf_counter()-start,
                                  artifact_hashes={p:file_hash(store.path/p) for p in names},
                                  answer_token_ids=backend._tokenization_logged,
                                  loss_reduction=job['loss_reduction'],scoring='sequential_candidates_v1')
                atomic_json(adaptation_path,adaptation)
            pred_path=store.path/'predictions.jsonl'
            rows=read_jsonl(pred_path,repair_tail=True) if pred_path.exists() else []
            validate_predictions(rows,query_rows,complete=False)
            seen={r['sample_id'] for r in rows}
            pending_count=len(query)-len(rows)
            model.eval();torch.cuda.reset_peak_memory_stats();qstart=time.perf_counter()
            with pred_path.open('a',encoding='utf-8') as f:
                for sample in query:
                    if sample.sample_id in seen:continue
                    row=backend.score(sample,controller)
                    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                    rows.append(row);seen.add(sample.sample_id)
            torch.cuda.synchronize()
            validate_predictions(rows,query_rows)
            labels=query_rows[0].get('candidate_labels',('intact','damaged','destroyed'))
            metrics=classification_metrics_nclass([r['label_id'] for r in rows],np.asarray([r['probabilities'] for r in rows]),labels)
            metrics.update(n=len(rows),evidence=job['evidence'],job_id=job['job_id'])
            atomic_json(store.path/'runtime.json',{'query_seconds_this_invocation':time.perf_counter()-qstart,
                        'query_count_this_invocation':pending_count,
                        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,
                        'note':'resume timings are partial; benchmark separately for latency claims'})
            sealed=json.loads(adaptation_path.read_text())
            store.finish(metrics,['predictions.jsonl','ADAPTED.json','runtime.json']+list(sealed.get('artifact_hashes',{})))
            return {'status':'done','job_id':job['job_id'],'metrics':metrics}
        finally:
            if controller is not None:controller.close()
            if model is not None:del model
            torch.cuda.empty_cache()
