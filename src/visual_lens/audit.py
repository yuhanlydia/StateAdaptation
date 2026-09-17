"""Support-only real-model P0-C audits. A checkpoint flag alone is not a pass."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import numpy as np
import torch
from vigor_handoff.core import ResidualController,extract_basis
from vigor_handoff.worker import Backend,discover_sites,fingerprint_assets,seed_everything
from vigor_handoff.protocol import JobStore,atomic_json


def checkpoint_mode(model, enabled):
    targets=[model]
    if getattr(model,'language_model',None) is not None:targets.append(model.language_model)
    ok=False;errors=[]
    for target in targets:
        disable=getattr(target,'gradient_checkpointing_disable',None)
        if callable(disable):disable()
        if enabled:
            enable=getattr(target,'gradient_checkpointing_enable',None)
            if callable(enable):
                try:enable(gradient_checkpointing_kwargs={'use_reentrant':False});ok=True
                except (TypeError,ValueError) as exc:errors.append(str(exc))
        config=getattr(target,'config',None)
        if config is not None:config.use_cache=False
    if enabled and not ok:raise RuntimeError('non-reentrant checkpoint API unavailable: '+repr(errors))
    # Most Transformers decoders checkpoint only in train mode. Dropout must be
    # disabled on BOTH sides of the comparison; vision tower stays deterministic.
    model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.dropout._DropoutNd):m.eval()
    for attr in ('vision_model','vision_tower','visual'):
        tower=getattr(model,attr,None)
        if tower is not None:tower.eval()


def gradient_signature(model,backend,samples,modules,controller=None,checkpoint=False,reduction='mean'):
    if not samples or reduction not in ('mean','sum'):raise ValueError('invalid audit objective')
    checkpoint_mode(model,checkpoint)
    model.zero_grad(set_to_none=True)
    if controller is not None:controller.zero_grad(set_to_none=True)
    counts=Counter();rows={f'{l}:{k}':[] for l,k,_ in modules};mask_ref=[None];handles=[]
    def hook(key):
        def capture(module,inputs,z):
            counts[key]+=1
            if not z.requires_grad:z.requires_grad_(True)
            # Capture the actual derivative; checkpoint recomputation may call
            # the projection again. Do not replace the saved forward state.
            def grad(g):
                mask=mask_ref[0]
                if mask is None:raise RuntimeError('mask cleared before checkpoint backward completed')
                rows[key].append(g[mask].detach().float().cpu())
            z.register_hook(grad)
        return capture
    for l,k,m in modules:handles.append(m.register_forward_hook(hook(f'{l}:{k}')))
    total=0.
    try:
        for sample in samples:
            try:
                loss,mask=backend.loss_and_mask(sample,controller)
                mask_ref[0]=mask
                divisor=len(samples) if reduction=='mean' else 1
                (loss/divisor).backward();total+=float(loss.detach())/divisor
            finally:
                if controller is not None:controller.clear_mask()
                mask_ref[0]=None
        if controller is not None:
            grads=[]
            for name,p in controller.named_parameters():
                if p.grad is None:raise RuntimeError(f'no controller gradient: {name}')
                grads.append(p.grad.detach().float().cpu().reshape(-1))
            flat=torch.cat(grads)
        else:flat=torch.empty(0)
        return {'loss':total,'controller_gradient':flat,'forward_counts':dict(counts),
                'activation_gradients':{k:torch.cat(v) if v else torch.empty(0) for k,v in rows.items()}}
    finally:
        for h in handles:h.remove()
        model.zero_grad(set_to_none=True)


def compare_vectors(a,b):
    a=a.float().reshape(-1);b=b.float().reshape(-1)
    if a.shape!=b.shape:return {'shape_match':False,'relative_l2':None,'cosine':None}
    denom=float(a.norm()*b.norm())
    return {'shape_match':True,'relative_l2':float((a-b).norm()/a.norm().clamp_min(1e-12)),
            'max_abs':float((a-b).abs().max()) if len(a) else 0.,
            'cosine':float(torch.dot(a,b)/denom) if denom>1e-20 else None}


def gradient_agrees(comparison, relative_tol):
    """Apply the configured BF16 gradient tolerance to a vector comparison."""
    return bool(comparison.get('shape_match') and
                comparison.get('relative_l2', float('inf')) <= relative_tol)


def signature_summary(result):
    return {'loss':result['loss'],'controller_gradient_norm':float(result['controller_gradient'].norm()),
            'forward_counts':result['forward_counts'],
            'activation_gradients':{k:{'rows':len(v),'norm':float(v.norm()),
                 'finite':bool(torch.isfinite(v).all())} for k,v in result['activation_gradients'].items()}}


def execute_audit(job):
    from eventttt.task_vlm import load_task_model
    from eventttt.io import read_samples,read_task_samples
    if not torch.cuda.is_available():raise RuntimeError('real-model audit requires CUDA')
    identity=fingerprint_assets(job);store=JobStore(job['output'],identity)
    if store.complete():return {'status':'skipped','job_id':job['job_id']}
    if (store.path/'DONE.json').exists():raise RuntimeError('sealed outputs changed; refuse to overwrite')
    reader=read_samples if job['kind']=='paired' else read_task_samples
    all_support=reader(job['support'])
    # Include each represented class, not only the first items of class-sorted files.
    selected=[]
    for c in sorted({s.label_id for s in all_support}):
        selected.extend([s for s in all_support if s.label_id==c][:job.get('audit_per_class',1)])
    model=controller=None
    with store.lock():
        try:
            seed_everything(job['optimization_seed'])
            model,processor=load_task_model(job['model_path'],job['family'],gradient_checkpointing=False)
            if {p.device.type for p in model.parameters()}!={'cuda'}:raise RuntimeError('audit refuses CPU/meta offload')
            for p in model.parameters():p.requires_grad_(False)
            model.eval();backend=Backend(job,model,processor)
            mods=discover_sites(model,job['layers'],job['sites'])
            bases={};spectra={};spans={}
            for mode in ('full','label'):
                backend.job={**job,'loss_span':mode}
                model.eval()
                bases[mode],spectra[mode]=extract_basis(model,mods,selected,backend.loss_and_mask,
                    rank=job['rank'],mode='covariance',seed=job['basis_seed'])
                sample=selected[0];batch,span=backend.batch(sample,sample.label)
                from .controls import select_label_span
                target=span if mode=='full' else select_label_span(batch['input_ids'],span,sample.label,processor.tokenizer)
                spans[mode]={'token_ids':batch['input_ids'][0,target[0]:target[1]].tolist(),'span':list(target)}
            overlap={f'{l}:{k}':float((bases['full'][(l,k)].T@bases['label'][(l,k)]).square().sum()/job['rank'])
                     for l,k,_ in mods}
            backend.job={**job,'loss_span':'full'}
            reference=backend.score(selected[0])
            controller=ResidualController(mods,bases['full'],job['alpha'],'full')
            zero=backend.score(selected[0],controller)
            identity_diff=float(np.max(np.abs(np.asarray(reference['mean_log_scores'])-zero['mean_log_scores'])))
            with torch.no_grad():
                for p in controller.parameters():p.fill_(.01)
            # Compare direct before/after-hook states in the SAME forward.
            originals={};direct={};hooks=[]
            for l,k,m in mods:
                key=f'{l}:{k}'
                def before(mod,inp,z,key=key):originals[key]=z.detach().clone()
                def after(mod,inp,z,key=key):
                    mask=controller._mask
                    if mask is None:return
                    delta=(z.detach()-originals.pop(key)).float()
                    direct[key]={'outside_max_abs':float(delta[~mask].abs().max()) if (~mask).any() else 0.,
                                 'inside_norm':float(delta[mask].norm())}
                hooks.append(m.register_forward_hook(before,prepend=True));hooks.append(m.register_forward_hook(after))
            changed=backend.score(selected[0],controller)
            for h in hooks:h.remove()
            torch.save(controller.payload(),store.path/'roundtrip_controller.pt')
            controller.close();controller=ResidualController(mods,bases['full'],job['alpha'],'full')
            controller.restore(torch.load(store.path/'roundtrip_controller.pt',map_location='cpu',weights_only=True))
            restored=backend.score(selected[0],controller)
            roundtrip_diff=float(np.max(np.abs(np.asarray(changed['mean_log_scores'])-restored['mean_log_scores'])))
            with torch.no_grad():
                for p in controller.parameters():p.zero_()
            reset=backend.score(selected[0],controller)
            reset_diff=float(np.max(np.abs(np.asarray(reference['mean_log_scores'])-reset['mean_log_scores'])))
            with torch.no_grad():
                for p in controller.parameters():p.fill_(.01)
            enable=getattr(model,'enable_input_require_grads',None)
            if callable(enable):enable()
            off=gradient_signature(model,backend,selected,mods,controller,False,'mean')
            summed=gradient_signature(model,backend,selected,mods,controller,False,'sum')
            label_backend=Backend({**job,'loss_span':'label'},model,processor)
            label_sig=gradient_signature(model,label_backend,selected,mods,controller,False,'mean')
            on=gradient_signature(model,backend,selected,mods,controller,True,'mean')
            checkpoint_cmp=compare_vectors(off['controller_gradient'],on['controller_gradient'])
            arithmetic=compare_vectors(summed['controller_gradient'],len(selected)*off['controller_gradient'])
            replay=any(on['forward_counts'].get(k,0)>off['forward_counts'].get(k,0) for k in off['forward_counts'])
            activation_cmp={k:compare_vectors(v,on['activation_gradients'][k]) for k,v in off['activation_gradients'].items()}
            finite_nonzero=all(len(v)>0 and bool(torch.isfinite(v).all()) and float(v.norm())>0 for v in off['activation_gradients'].values())
            gradient_tol=job.get('audit_relative_tol',.05)
            checks={'zero_identity':identity_diff<=job.get('audit_abs_tol',.005),
                    'reset_identity':reset_diff<=job.get('audit_abs_tol',.005),
                    'save_load':roundtrip_diff<=job.get('audit_abs_tol',.005),
                    'direct_mask_isolation':all(x['outside_max_abs']==0 for x in direct.values()) and len(direct)==len(mods),
                    'finite_nonzero_visual_gradients':finite_nonzero,
                    'checkpoint_replay_observed':replay,
                    'checkpoint_controller_gradient':gradient_agrees(checkpoint_cmp,gradient_tol),
                    'checkpoint_activation_gradients':all(gradient_agrees(x,gradient_tol) for x in activation_cmp.values()),
                    'checkpoint_loss':abs(off['loss']-on['loss'])<=job.get('audit_abs_tol',.005),
                    'mean_sum_data_gradient':gradient_agrees(arithmetic,gradient_tol)}
            # Actual finite differences at nonzero R; BF16 quantization can make
            # these scale-sensitive, so report a curve rather than invent a pass.
            checkpoint_mode(model,False);model.eval()
            params=list(controller.parameters());generator=torch.Generator().manual_seed(23)
            direction=torch.randn(off['controller_gradient'].shape,generator=generator)
            direction/=direction.norm();original=[p.detach().clone() for p in params]
            def move(eps):
                offset=0
                with torch.no_grad():
                    for p,r in zip(params,original):
                        d=direction[offset:offset+p.numel()].reshape(p.shape).to(p);p.copy_(r+eps*d);offset+=p.numel()
            def value():
                total=0.
                for sample in selected:
                    with torch.no_grad():loss,_=backend.loss_and_mask(sample,controller)
                    total+=float(loss)/len(selected);controller.clear_mask()
                return total
            finite_diffs=[]
            for eps in (.005,.02,.1):
                move(eps);plus=value();move(-eps);minus=value()
                finite_diffs.append({'epsilon':eps,'central_difference':(plus-minus)/(2*eps)})
            move(0)
            report={'checks':checks,'passed':all(checks.values()),'support_ids':[s.sample_id for s in selected],
                    'identity_max_abs':identity_diff,'reset_max_abs':reset_diff,'roundtrip_max_abs':roundtrip_diff,
                    'direct_intervention':direct,'answer_spans':spans,'basis_overlap_full_vs_label':overlap,
                    'extraction_spectra':spectra,'checkpoint_off':signature_summary(off),'checkpoint_on':signature_summary(on),
                    'checkpoint_controller_comparison':checkpoint_cmp,'checkpoint_activation_comparison':activation_cmp,
                    'sum_objective':signature_summary(summed),'mean_sum_arithmetic':arithmetic,
                    'label_only_objective':signature_summary(label_sig),
                    'full_vs_label_controller':compare_vectors(off['controller_gradient'],label_sig['controller_gradient']),
                    'finite_difference':finite_diffs,'autograd_directional_derivative':float(torch.dot(off['controller_gradient'],direction)),
                    'note':'Support-only diagnostics. No new query performance. Dropout disabled in both checkpoint train-mode probes; actual recomputation must be observed.'}
            atomic_json(store.path/'audit.json',report)
            if not report['passed']:raise RuntimeError('P0-C audit gate failed: '+repr([k for k,v in checks.items() if not v]))
            store.finish({'passed':True,'job_id':job['job_id']},['audit.json','roundtrip_controller.pt'])
            return {'status':'done','job_id':job['job_id']}
        finally:
            if controller is not None:controller.close()
            if model is not None:del model
            torch.cuda.empty_cache()
