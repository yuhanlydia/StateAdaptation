"""Support-only fitting, sealed choices, and read-only query evaluation.

All stages are restartable in versioned JobStores. Runtime audits use real VLMs;
CPU tests do not certify the pretrained-model or medical-data paths.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import importlib.metadata
import json
import os
import random
import time
import gc
import numpy as np
import torch
from aperture_final.numeric import probabilities, metrics, fit_temperature, choose_lora
from vigor_handoff.protocol import JobStore, digest, file_hash, atomic_json, read_jsonl, resolved_manifest, validate_predictions
from vigor_handoff.core import ResidualController, extract_basis, fit_coefficients
from vigor_handoff.worker import train_lora
from visual_lens.controls import fit_candidate_bias
from eventttt.io import read_task_samples
from .protocol import check_job, assets_identity, source_signature, ARMS
from .backend import load_native, NativeBackend, projection_sites, enable_checkpointing


def verified_record(folder):
    folder=Path(folder)
    identity=json.loads((folder/'identity.json').read_text());done=json.loads((folder/'DONE.json').read_text())
    if identity['fingerprint']!=digest(identity['identity']) or done['fingerprint']!=identity['fingerprint']:
        raise ValueError('state identity is inconsistent')
    if not done.get('artifacts'):raise ValueError('empty completion record')
    for name,h in done['artifacts'].items():
        p=folder/name
        if not p.is_file() or file_hash(p)!=h:raise ValueError(f'sealed artifact changed: {p}')
    return done


def validate_score_rows(rows,manifest,complete=True):
    validate_predictions(rows,manifest,complete=complete)
    if [r['sample_id'] for r in rows]!=[r['sample_id'] for r in manifest[:len(rows)]]:
        raise ValueError('candidate records do not follow the fixed manifest order')
    for r in rows:
        p=probabilities([r['mean_log_scores']])[0]
        if not np.allclose(p,r['probabilities'],rtol=1e-7,atol=1e-8):raise ValueError('score/probability inconsistency')


def compare_repeats(a,b,tolerance=.005):
    if not a or [r['sample_id'] for r in a]!=[r['sample_id'] for r in b]:
        raise ValueError('repeated calibration IDs differ')
    x=np.asarray([r['mean_log_scores'] for r in a]);y=np.asarray([r['mean_log_scores'] for r in b])
    if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('invalid replay scores')
    delta=float(np.max(np.abs(x-y)))
    if delta>tolerance:raise ValueError(f'repeat/save-load gate failed: max score difference {delta:.8g} > {tolerance}')
    return {'passed':True,'samples':len(a),'max_abs_score_difference':delta,'tolerance':tolerance}


def deterministic(seed=0):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 BEFORE launching Python')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.set_num_threads(1)


def environment():
    deps={}
    for name in ('torch','transformers','peft','accelerate','numpy','pillow','scipy','scikit-learn'):
        try:deps[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:deps[name]=None
    return {'dependencies':deps,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'attention_implementation':'eager','dtype':'bfloat16','deterministic_algorithms':True}


@contextmanager
def loaded(job,restore=None):
    deterministic(job['optimization_seed'])
    model=backend=controller=None
    try:
        is_lora=job['arm'].startswith('lora')
        model,processor=load_native(job,is_lora,Path(restore)/'adapter' if restore is not None and is_lora else None)
        backend=NativeBackend(job,model,processor)
        if restore is not None and job['arm']=='aperture':
            payload=torch.load(Path(restore)/'controller.pt',map_location='cpu',weights_only=True)
            if payload['mode']!='full' or payload['alpha']!=job['alpha'] or payload['hard_projection']:
                raise ValueError('saved controller does not match the fixed operator')
            bases={(int(k.split(':')[0]),k.split(':')[1]):v for k,v in payload['bases'].items()}
            controller=ResidualController(projection_sites(model,job['layers']),bases,job['alpha'],'full')
            controller.restore(payload)
            for p in controller.parameters():p.requires_grad_(False)
        if restore is not None:
            for p in model.parameters():p.requires_grad_(False)
        yield model,backend,controller
    finally:
        if controller is not None:controller.close()
        del model,backend,controller
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def _write_rows(path,rows):
    Path(path).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')


def _gradient(model,backend,samples,controller,checkpoint):
    enable_checkpointing(model,checkpoint);model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.dropout._DropoutNd):m.eval()
    counts={};hooks=[]
    for l,k,m in controller.targets:
        name=f'{l}:{k}'
        def count(_m,_x,_z,n=name):counts[n]=counts.get(n,0)+1
        hooks.append(m.register_forward_hook(count))
    controller.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True);total=0.
    try:
        for s in samples:
            try:
                loss,_=backend.loss_and_mask(s,controller);(loss/len(samples)).backward();total+=float(loss.detach())/len(samples)
            finally:controller.clear_mask()
        grads=[p.grad.detach().float().cpu().flatten() for p in controller.parameters() if p.grad is not None]
        if len(grads)!=len(list(controller.parameters())):raise ValueError('missing controller gradient')
        return torch.cat(grads),total,counts
    finally:
        for h in hooks:h.remove()
        enable_checkpointing(model,False);model.eval();model.zero_grad(set_to_none=True)


def _base_audit(job,folder):
    samples=read_task_samples(job['support'])
    subset=[next(s for s in samples if s.label_id==c) for c in range(job['classes'])]
    with loaded({**job,'arm':'frozen'}) as (model,backend,_):
        baseline=backend.score(subset[0]);mods=projection_sites(model)
        bases,spectra=extract_basis(model,mods,subset,backend.loss_and_mask,rank=16)
        ctrl=ResidualController(mods,bases,job['alpha'])
        try:
            zero=backend.score(subset[0],ctrl);compare_repeats([baseline],[zero],.005)
            with torch.no_grad():
                for p in ctrl.parameters():p.fill_(.01)
            captured={};direct={};hooks=[]
            for l,k,module in mods:
                name=f'{l}:{k}'
                def before(_m,_x,z,n=name):captured[n]=z.detach().clone()
                def after(_m,_x,z,n=name):
                    mask=ctrl._mask;delta=z.detach()-captured.pop(n)
                    if mask is None:raise ValueError('audit residual mask missing')
                    direct[n]={'outside':float(delta[~mask].abs().max()) if (~mask).any() else 0.,
                               'inside':float(delta[mask].float().norm())}
                hooks.append(module.register_forward_hook(before,prepend=True));hooks.append(module.register_forward_hook(after))
            try:changed=backend.score(subset[0],ctrl)
            finally:
                for h in hooks:h.remove()
            if len(direct)!=4 or any(v['outside']!=0 for v in direct.values()):raise ValueError('visual write escaped mask')
            if not any(v['inside']>0 for v in direct.values()):raise ValueError('nonzero residual made no visual-state change')
            torch.save(ctrl.payload(),folder/'audit_controller.pt');ctrl.close()
            ctrl=ResidualController(mods,bases,job['alpha']);ctrl.restore(torch.load(folder/'audit_controller.pt',weights_only=True))
            compare_repeats([changed],[backend.score(subset[0],ctrl)],.005)
            off,loss0,c0=_gradient(model,backend,subset,ctrl,False)
            on,loss1,c1=_gradient(model,backend,subset,ctrl,True)
            rel=float((off-on).norm()/off.norm().clamp_min(1e-12))
            if not torch.isfinite(off).all() or not torch.isfinite(on).all() or float(off.norm())==0:raise ValueError('invalid coefficient gradients')
            if rel>.05 or abs(loss0-loss1)>.005:raise ValueError('checkpoint gradient/loss mismatch')
            if not any(c1.get(k,0)>c0.get(k,0) for k in c0):raise ValueError('checkpoint recomputation not observed')
            return dict(passed=True,support_ids=[s.sample_id for s in subset],spectra=spectra,
                checkpoint_relative_l2=rel,checkpoint_loss_delta=abs(loss0-loss1),counts_off=c0,counts_on=c1,
                direct_mask=direct,zero_identity=True,controller_save_load=True,visual_tokens=baseline['visual_tokens'],
                answer_token_ids=baseline['answer_token_ids'])
        finally:ctrl.close()


def _lora_audit_fit(job,folder):
    sample=read_task_samples(job['support'])[0]
    j={**job,'arm':'lora1'}
    with loaded(j) as (model,backend,_):
        trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
        if not trainable or any('lora_' not in n for n in trainable):raise ValueError('LoRA target isolation failed')
        before={n:p.detach().cpu().clone() for n,p in trainable.items()};counts=[0]
        h=projection_sites(model)[0][2].register_forward_hook(lambda *args:counts.__setitem__(0,counts[0]+1))
        try:
            model.train();opt=torch.optim.AdamW(list(trainable.values()),lr=job['lora_lr'])
            loss,_=backend.loss_and_mask(sample);forward_count=counts[0];loss.backward()
            if counts[0]<=forward_count:raise ValueError('LoRA checkpoint did not replay selected projection')
            torch.nn.utils.clip_grad_norm_(trainable.values(),1.,error_if_nonfinite=True);opt.step()
        finally:h.remove()
        if not any(not torch.equal(before[n],p.detach().cpu()) for n,p in trainable.items()):raise ValueError('LoRA parameters did not update')
        model.eval();row=backend.score(sample);model.save_pretrained(folder/'audit_adapter',safe_serialization=True)
        return row


def _lora_audit_restore(job,folder,expected):
    # load_native expects adapter subfolder under restore. Audit uses an explicit path.
    model,processor=load_native({**job,'arm':'lora1'},True,folder/'audit_adapter')
    try:
        backend=NativeBackend(job,model,processor);sample=read_task_samples(job['support'])[0]
        return compare_repeats([expected],[backend.score(sample)],.005)
    finally:
        del model,processor
        gc.collect();torch.cuda.empty_cache()


def execute_audit(job,path):
    identity={'assets':assets_identity(job),'environment':environment(),'kind':'support_only_native_audit'}
    store=JobStore(path,identity)
    if store.complete():return
    if (store.path/'DONE.json').exists():raise ValueError('changed completed audit')
    with store.lock():
        base=_base_audit(job,store.path);gc.collect();torch.cuda.empty_cache()
        row=_lora_audit_fit(job,store.path);gc.collect();torch.cuda.empty_cache()
        deterministic(job['optimization_seed'])
        lora=_lora_audit_restore(job,store.path,row);gc.collect();torch.cuda.empty_cache()
        atomic_json(store.path/'audit.json',{'passed':True,'visual':base,'lora_save_load':lora,'environment':environment()})
        store.finish({'passed':True},['audit.json'])


def execute_fit(job,audit_path):
    check_job(job);audit=verified_record(audit_path)
    ai=json.loads((Path(audit_path)/'identity.json').read_text())['identity']
    aj=ai['assets']['job']
    if any(aj[k]!=job[k] for k in ('model_key','model_id','family','domain','rank','layers','classes')) or ai['assets']['source_signature']!=source_signature() or ai['environment']!=environment():
        raise ValueError('audit belongs to another model, domain, source, or environment')
    if not json.loads((Path(audit_path)/'audit.json').read_text())['passed']:raise ValueError('native GPU audit failed')
    identity={'assets':assets_identity(job),'environment':environment(),'audit_seal':digest(audit),'kind':'fit_only'}
    store=JobStore(Path(job['output'])/'fit',identity)
    if store.complete():return
    if (store.path/'DONE.json').exists():raise ValueError('sealed fit changed')
    fit=read_task_samples(job['support']);cal=read_task_samples(job['calibration'])
    # No query sample is forwarded or fitted in this function.
    with store.lock(),loaded(job) as (model,backend,_):
        ctrl=None;artifacts=[];spectra={};extract_seconds=0.
        torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        stats={'job':job,'labels_used':len(fit)+len(cal),'query_forwards':0,'environment':environment()}
        try:
            if job['arm'].startswith('lora'):
                history,updates,n=train_lora(model,backend,fit,job)
                model.save_pretrained(store.path/'adapter',safe_serialization=True)
                artifacts.extend(str(p.relative_to(store.path)) for p in (store.path/'adapter').rglob('*') if p.is_file())
                stats.update(optimized_scalars=n,basis_scalars=0,optimizer_updates=updates,
                             fitting_visits=len(fit)*job['lora_passes'],basis_visits=0)
            elif job['arm']=='aperture':
                modules=projection_sites(model,job['layers']);torch.cuda.synchronize();t=time.perf_counter()
                bases,spectra=extract_basis(model,modules,fit,backend.loss_and_mask,rank=job['rank'])
                torch.cuda.synchronize();extract_seconds=time.perf_counter()-t
                ctrl=ResidualController(modules,bases,job['alpha'])
                compare_repeats([backend.score(fit[0])],[backend.score(fit[0],ctrl)],.005)
                history=fit_coefficients(model,ctrl,fit,lambda s:backend.loss_and_mask(s,ctrl)[0],
                         steps=job['steps'],lr=job['lr'],l2=job['l2'],reduction=job['loss_reduction'])
                torch.save(ctrl.payload(),store.path/'controller.pt');artifacts.append('controller.pt')
                stats.update(optimized_scalars=ctrl.num_scalars(),basis_scalars=sum(x.numel() for x in bases.values()),
                  optimizer_updates=len(history),fitting_visits=len(fit)*job['steps'],basis_visits=len(fit),
                  operator_norms={k:float(torch.linalg.matrix_norm(a.detach().cpu(),2)) for k,a in ctrl.operators().items()})
            else:
                history=[];stats.update(optimized_scalars=0,basis_scalars=0,optimizer_updates=0,fitting_visits=0,basis_visits=0)
            torch.cuda.synchronize();stats.update(fit_seconds=time.perf_counter()-start,extraction_seconds=extract_seconds,
                    peak_fit_bytes=torch.cuda.max_memory_allocated(),total_model_scalars=sum(p.numel() for p in model.parameters()))
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
            calrows=[backend.score(s,ctrl) for s in cal];validate_score_rows(calrows,resolved_manifest(job['calibration']))
            scores=[r['mean_log_scores'] for r in calrows];labels=[r['label_id'] for r in calrows]
            calibration={'temperature':fit_temperature(scores,labels,penalty=job['temperature_penalty']),
              'raw_calibration_nll':metrics(scores,labels)['nll'],'query_labels_used':False}
            if job['arm']=='frozen':
                allrows=[backend.score(s) for s in fit]+calrows
                bias,meta=fit_candidate_bias([r['mean_log_scores'] for r in allrows],[r['label_id'] for r in allrows],l2=job['bias_l2'])
                calibration['bias']={'values':bias.tolist(),'metadata':meta,'all_support_count':len(allrows)}
                _write_rows(store.path/'all_support_scores.jsonl',allrows);artifacts.append('all_support_scores.jsonl')
            stats['serialized_state_bytes']=sum((store.path/p).stat().st_size for p in artifacts if p=='controller.pt' or p.startswith('adapter/'))
            stats['visual_tokens']=calrows[0]['visual_tokens']
            _write_rows(store.path/'calibration_scores.jsonl',calrows)
            atomic_json(store.path/'calibration.json',calibration);atomic_json(store.path/'fitting.json',{'stats':stats,'history':history,'spectra':spectra})
            artifacts+=['calibration_scores.jsonl','calibration.json','fitting.json']
            store.finish(stats,artifacts)
        finally:
            if ctrl is not None:ctrl.close()


def seal_choices(jobs,path):
    units={}
    for j in jobs:units.setdefault(j['unit_id'],{})[j['arm']]=j
    result={'source_signature':source_signature(),'query_metrics_used':False,'units':{}}
    for unit,arms in units.items():
        if set(arms)!=set(ARMS):raise ValueError('selection requires all four prespecified arms')
        data={};seals={}
        for arm,j in arms.items():
            folder=Path(j['output'])/'fit';verified_record(folder);seals[arm]=file_hash(folder/'DONE.json')
            data[arm]=json.loads((folder/'calibration.json').read_text())
        result['units'][unit]={'fit_seals':seals,
          'lora_raw':choose_lora({a:{'calibration_nll':data[a]['raw_calibration_nll']} for a in ('lora1','lora4')}),
          'lora_temperature':choose_lora({a:data[a]['temperature'] for a in ('lora1','lora4')})}
    from aperture_final.protocol import write_locked
    write_locked(path,result);return result


def selection_identity(selections,unit):
    return digest({'source':selections['source_signature'],'unit_id':unit,'choice':selections['units'][unit]})


def execute_evaluate(job,selection_path):
    source=Path(job['output'])/'fit';done=verified_record(source)
    current=assets_identity(job);old=json.loads((source/'identity.json').read_text())['identity']
    if old['assets']!=current or old['environment']!=environment():raise ValueError('fit source/assets/runtime changed')
    selections=json.loads(Path(selection_path).read_text())
    if selections['source_signature']!=source_signature():raise ValueError('selection source changed')
    chosen=selections['units'][job['unit_id']]
    if chosen['fit_seals'][job['arm']]!=file_hash(source/'DONE.json'):raise ValueError('stale fitted-state selection')
    identity={'assets':current,'fit_seal':digest(done),'selection':selection_identity(selections,job['unit_id']),'environment':environment()}
    store=JobStore(Path(job['output'])/'evaluation',identity)
    if store.complete():return
    if (store.path/'DONE.json').exists():raise ValueError('sealed query evaluation changed')
    calibration=json.loads((source/'calibration.json').read_text())
    manifest=resolved_manifest(job['query']);samples=read_task_samples(job['query'])
    with store.lock(),loaded(job,source) as (model,backend,ctrl):
        replay=[backend.score(s,ctrl) for s in read_task_samples(job['calibration'])]
        compare_repeats(read_jsonl(source/'calibration_scores.jsonl'),replay,.005)
        p=store.path/'predictions.jsonl';rows=read_jsonl(p,repair_tail=True) if p.exists() else []
        validate_score_rows(rows,manifest,complete=False)
        torch.cuda.synchronize();start=time.perf_counter();pending=len(samples)-len(rows);torch.cuda.reset_peak_memory_stats()
        with p.open('a',encoding='utf-8') as f:
            for sample in samples[len(rows):]:
                row=backend.score(sample,ctrl);f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();rows.append(row)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        validate_score_rows(rows,manifest)
        scores=[r['mean_log_scores'] for r in rows];labels=[r['label_id'] for r in rows];t=calibration['temperature']['temperature']
        data={'job':job,'raw':metrics(scores,labels),'temperature':metrics(scores,labels,t),
          'selection':chosen,'query_ids_sha256':digest([r['sample_id'] for r in rows]),
          'runtime':{'new_queries_timed':pending,'elapsed_seconds':elapsed,
                     'seconds_per_query':elapsed/pending if pending else None,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}}
        if job['arm']=='frozen':data['bias']=metrics(scores,labels,bias=calibration['bias']['values'])
        if data['raw']['macro_f1']!=data['temperature']['macro_f1']:raise ValueError('temperature changed argmax')
        atomic_json(store.path/'evaluation.json',data);store.finish({'passed':True},['predictions.jsonl','evaluation.json'])
