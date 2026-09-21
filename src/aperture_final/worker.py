"""Fitting and read-only evaluation for the final protocol.

No query forward occurs in execute_fit(). Model selection and temperatures are
sealed by seal_choices() before execute_evaluate() is allowed to run.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np
from .numeric import fit_temperature, probabilities, metrics, choose_lora, degrade
from .protocol import check_job, source_signature, probe_ids, write_locked, ARMS
from vigor_handoff.protocol import JobStore, atomic_json, digest, file_hash, read_jsonl, resolved_manifest, validate_predictions


def verified_record(folder):
    folder=Path(folder)
    identity=json.loads((folder/'identity.json').read_text())
    done=json.loads((folder/'DONE.json').read_text())
    if identity['fingerprint']!=digest(identity['identity']) or done['fingerprint']!=identity['fingerprint']:
        raise ValueError(f'identity mismatch: {folder}')
    for name, h in done['artifacts'].items():
        p=folder/name
        if not p.is_file() or file_hash(p)!=h:raise ValueError(f'artifact hash mismatch: {p}')
    return done


def score_block(rows, temperature=1., bias=None):
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('empty or duplicate prediction rows')
    s=np.asarray([r['mean_log_scores'] for r in rows],dtype=float)
    y=np.asarray([r['label_id'] for r in rows])
    result={'raw':metrics(s,y), 'temperature':metrics(s,y,temperature),
            'ids_sha256':digest([r['sample_id'] for r in rows])}
    if bias is not None:result['candidate_bias']=metrics(s,y,bias=bias)
    if result['raw']['macro_f1']!=result['temperature']['macro_f1']:
        raise ValueError('positive-temperature argmax invariance failed')
    return result


def seal_choices(jobs, destination):
    units={}
    for j in jobs:units.setdefault(j['unit_id'],{})[j['arm']]=j
    result={'selection_rule':'held-out SUPPORT NLL only; ties prefer one pass',
            'query_metrics_used':False,'units':{},'source_signature':source_signature()}
    for key,arms in units.items():
        if set(arms)!=set(ARMS):raise ValueError(f'incomplete planned arms for {key}')
        data={};seals={}
        for arm,j in arms.items():
            folder=Path(j['output'])/'fit';verified_record(folder)
            seals[arm]=file_hash(folder/'DONE.json')
            data[arm]=json.loads((folder/'calibration.json').read_text())
        raw={k:{'calibration_nll':data[k]['raw_calibration_nll']} for k in ('lora1','lora4')}
        scaled={k:data[k]['temperature'] for k in ('lora1','lora4')}
        result['units'][key]={'lora_raw':choose_lora(raw),'lora_temperature':choose_lora(scaled),
                             'fit_seals':seals}
    write_locked(destination,result)
    return result


def _identity(job):
    from vigor_handoff.worker import fingerprint_assets
    base=fingerprint_assets(job)
    cal=resolved_manifest(job['calibration'])
    paths=sorted({r[k] for r in cal for k in ('image','pre_image','post_image') if r.get(k)})
    return {'base':base,'calibration_sha256':file_hash(job['calibration']),
            'calibration_image_content':digest([(p,file_hash(p)) for p in paths]),
            'split_record_sha256':file_hash(job['split_record']),
            'final_source':source_signature()}


def _samples(job, name):
    from eventttt.io import read_samples, read_task_samples
    reader=read_samples if job['kind']=='paired' else read_task_samples
    return reader(job[name])


def _gpu_ready():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU execution requires CUDA')
    # BF16 7B model + gradients: never silently switch quantization or resize.
    if torch.cuda.get_device_properties(0).total_memory < 22*1024**3:
        raise RuntimeError('This fixed BF16 7B protocol needs a 24GB-or-larger GPU. No implicit 16GB/quantized substitution.')


@contextmanager
def _loaded(job, restore=None):
    import torch
    from eventttt.task_vlm import load_task_model
    from vigor_handoff.worker import Backend, discover_sites, seed_everything
    from vigor_handoff.core import ResidualController
    model=controller=None
    try:
        seed_everything(job['optimization_seed'])
        lora=job['arm'].startswith('lora')
        model,processor=load_task_model(job['model_path'],'qwen2',use_lora=lora,
                                        gradient_checkpointing=lora and restore is None)
        if {p.device.type for p in model.parameters()}!={'cuda'}:
            raise RuntimeError('CPU/meta offload is not part of the fixed protocol')
        model.eval();model.config.use_cache=False
        if restore is not None:
            restore=Path(restore)
            if lora:
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                set_peft_model_state_dict(model,load_file(str(restore/'adapter/adapter_model.safetensors')))
            elif job['arm']=='aperture':
                payload=torch.load(restore/'controller.pt',map_location='cpu',weights_only=True)
                bases={(int(k.split(':')[0]),k.split(':')[1]):v for k,v in payload['bases'].items()}
                sites=discover_sites(model,job['layers'],job['sites'])
                controller=ResidualController(sites,bases,payload['alpha'],payload['mode'])
                controller.restore(payload)
            for p in model.parameters():p.requires_grad_(False)
        elif not lora:
            for p in model.parameters():p.requires_grad_(False)
        backend=Backend(job,model,processor)
        yield model,backend,controller
    finally:
        if controller is not None:controller.close()
        if model is not None:del model
        import gc
        gc.collect();torch.cuda.empty_cache()


def _write_predictions(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')


def execute_fit(job, audit_folder):
    import torch
    from vigor_handoff.worker import train_lora, discover_sites
    from vigor_handoff.core import extract_basis, ResidualController, fit_coefficients
    from visual_lens.controls import fit_candidate_bias
    _gpu_ready();check_job(job)
    audit_done=verified_record(audit_folder)
    audit=json.loads((Path(audit_folder)/'audit.json').read_text())
    if not audit.get('passed'):raise RuntimeError('support-only GPU audit did not pass')
    identity={**_identity(job),'audit_seal':digest(audit_done)}
    folder=Path(job['output'])/'fit';store=JobStore(folder,identity)
    if store.complete():return 'skipped'
    if (folder/'DONE.json').exists():raise ValueError('sealed fitting outputs changed')
    fit=_samples(job,'support');cal=_samples(job,'calibration')
    # Deliberately NO _samples(job, 'query') or backend query call here.
    with store.lock(),_loaded(job) as (model,backend,_):
        controller=None
        torch.cuda.synchronize();start=time.perf_counter();extraction_seconds=0.;spectra={}
        stats={'fitting_examples':len(fit),'calibration_examples':len(cal),
               'all_labeled_examples':len(fit)+len(cal),'query_examples_used_for_fitting':0,
               'recipe':job,'source_signature':source_signature(),
               'temperature_scalars':1,'output_bias_scalars':job['classes']-1 if job['arm']=='frozen' else 0}
        artifacts=[]
        try:
            if job['arm'].startswith('lora'):
                history,updates,scalars=train_lora(model,backend,fit,job)
                model.save_pretrained(folder/'adapter',safe_serialization=True)
                artifacts.extend(str(p.relative_to(folder)) for p in (folder/'adapter').rglob('*') if p.is_file())
                stats.update(optimizer_updates=updates,optimized_scalars=scalars,basis_scalars=0,
                             fitting_visits=len(fit)*job['lora_passes'],basis_visits=0)
            elif job['arm']=='aperture':
                sites=discover_sites(model,job['layers'],job['sites'])
                torch.cuda.synchronize();extract_start=time.perf_counter()
                bases,spectra=extract_basis(model,sites,fit,backend.loss_and_mask,rank=job['rank'],mode='covariance',seed=0)
                torch.cuda.synchronize();extraction_seconds=time.perf_counter()-extract_start
                controller=ResidualController(sites,bases,job['alpha'],'full')
                ref=backend.score(fit[0]);zero=backend.score(fit[0],controller)
                delta=float(np.max(np.abs(np.array(ref['mean_log_scores'])-zero['mean_log_scores'])))
                if delta>1e-3:raise RuntimeError(f'zero-controller identity failed: {delta}')
                history=fit_coefficients(model,controller,fit,lambda s:backend.loss_and_mask(s,controller)[0],
                       steps=job['steps'],lr=job['lr'],l2=job['l2'],reduction=job['loss_reduction'])
                torch.save(controller.payload(),folder/'controller.pt');artifacts.append('controller.pt')
                stats.update(optimizer_updates=len(history),optimized_scalars=controller.num_scalars(),
                             basis_scalars=sum(b.numel() for b in bases.values()),
                             fitting_visits=len(fit)*job['steps'],basis_visits=len(fit),
                             # These tiny descriptive matrices do not need a CUDA SVD.
                             # Keep cuSOLVER workspace available for the actual fit.
                             operator_norms={k:float(torch.linalg.matrix_norm(a.detach().cpu(),2))
                                             for k,a in controller.operators().items()})
            elif job['arm']=='frozen':
                history=[];stats.update(optimizer_updates=0,optimized_scalars=0,basis_scalars=0,fitting_visits=0,basis_visits=0)
            else:raise ValueError('unknown arm')
            torch.cuda.synchronize();stats['fit_wall_seconds']=time.perf_counter()-start
            stats['extraction_seconds']=extraction_seconds
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
            calibration_rows=[backend.score(s,controller) for s in cal]
            validate_predictions(calibration_rows,resolved_manifest(job['calibration']))
            scores=[r['mean_log_scores'] for r in calibration_rows];labels=[r['label_id'] for r in calibration_rows]
            temperature=fit_temperature(scores,labels,penalty=job['temperature_penalty'])
            calibration={'temperature':temperature,'raw_calibration_nll':metrics(scores,labels)['nll'],
                         'fit_ids_sha256':digest([s.sample_id for s in fit]),
                         'calibration_ids_sha256':digest([s.sample_id for s in cal]),
                         'query_labels_used':False,'query_forwards':0}
            if job['arm']=='frozen':
                # With no backbone fit, output bias may use ALL original labels;
                # do not weaken this control by discarding the fitting partition.
                fit_rows=[backend.score(s) for s in fit]
                bias_rows=fit_rows+calibration_rows
                bias,meta=fit_candidate_bias([r['mean_log_scores'] for r in bias_rows],
                          [r['label_id'] for r in bias_rows],l2=job['bias_l2'])
                calibration['bias']={'values':bias.tolist(),'metadata':meta,
                     'calibration_count':len(bias_rows),
                     'fit_source':'all_original_labeled_support; frozen predictor'}
                _write_predictions(folder/'bias_support_predictions.jsonl',bias_rows)
                artifacts.append('bias_support_predictions.jsonl')
            _write_predictions(folder/'calibration_predictions.jsonl',calibration_rows)
            atomic_json(folder/'calibration.json',calibration)
            atomic_json(folder/'fitting.json',{'stats':stats,'history':history,'spectra':spectra})
            artifacts+=['calibration_predictions.jsonl','calibration.json','fitting.json']
            stats['serialized_state_bytes']=sum((folder/p).stat().st_size for p in artifacts
                         if p=='controller.pt' or p.startswith('adapter/'))
            store.finish(stats,artifacts)
            return 'done'
        finally:
            if controller is not None:controller.close()


def _score_condition(backend, samples, controller, path, query_rows, real_index=None):
    # Row-wise resume is accepted only inside the surrounding fingerprinted store.
    path=Path(path)
    rows=read_jsonl(path,repair_tail=True) if path.exists() else []
    validate_predictions(rows,query_rows,complete=False)
    def check_row(row):
        p=probabilities([row['mean_log_scores']])[0]
        if not np.allclose(p,np.asarray(row['probabilities']),rtol=1e-7,atol=1e-8):
            raise ValueError('saved score/probability mismatch')
        if real_index is not None and row['prompt_contract_hash']!=real_index[row['sample_id']]['prompt_contract_hash']:
            raise ValueError('text, candidate span, image grid or visual-token budget changed')
    for row in rows:check_row(row)
    have={r['sample_id'] for r in rows}
    with path.open('a',encoding='utf-8') as f:
        for sample in samples:
            if sample.sample_id in have:continue
            row=backend.score(sample,controller)
            check_row(row)
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();rows.append(row)
    validate_predictions(rows,query_rows)
    index={r['sample_id']:r for r in rows}
    return [index[s.sample_id] for s in samples]


def execute_evaluate(job, selection_path):
    import torch
    _gpu_ready();check_job(job)
    source=Path(job['output'])/'fit';done=verified_record(source)
    # Recompute identities so changed models/images/source cannot silently reuse a state.
    fit_identity=json.loads((source/'identity.json').read_text())['identity']
    current=_identity(job)
    if any(fit_identity.get(k)!=v for k,v in current.items()):raise ValueError('fitted-state assets/source changed')
    selections=json.loads(Path(selection_path).read_text())
    if selections['source_signature']!=source_signature():raise ValueError('selection source changed')
    selected=selections['units'][job['unit_id']]
    if selected['fit_seals'][job['arm']]!=file_hash(source/'DONE.json'):raise ValueError('selection is stale')
    identity={'fit_done':digest(done),'selection_sha256':file_hash(selection_path),'job':job,'source':source_signature()}
    folder=Path(job['output'])/'evaluation';store=JobStore(folder,identity)
    if store.complete():return 'skipped'
    if (folder/'DONE.json').exists():raise ValueError('sealed evaluation outputs changed')
    calibration=json.loads((source/'calibration.json').read_text());t=calibration['temperature']['temperature']
    bias=calibration.get('bias',{}).get('values')
    query=_samples(job,'query');manifest=resolved_manifest(job['query'])
    ids=probe_ids(manifest,job['probe_count'],job['probe_seed']);idset=set(ids)
    samples=[s for s in query if s.sample_id in idset];probe_manifest=[r for r in manifest if r['sample_id'] in idset]
    with store.lock(),_loaded(job,source) as (model,backend,controller):
        real=_score_condition(backend,query,controller,folder/'clean.jsonl',manifest)
        index={r['sample_id']:r for r in real};probe_real=[index[s.sample_id] for s in samples]
        result={'job_id':job['job_id'],'unit_id':job['unit_id'],'domain':job['domain'],'support_seed':job['support_seed'],
                'arm':job['arm'],'clean':score_block(real,t,bias),
                'probe_clean':score_block(probe_real,t,bias),'probe_ids_sha256':digest(ids),
                'corruptions':{},'dose':{},'selected_lora_raw':selected['lora_raw'],
                'selected_lora_temperature':selected['lora_temperature'],'temperature':t}
        artifacts=['clean.jsonl']
        try:
            for condition in job['corruptions']:
                backend.image_transform=lambda sample,im,c=condition:degrade(im,c,sample.sample_id,job['probe_seed'])
                filename=f'{condition}.jsonl'
                rows=_score_condition(backend,samples,controller,folder/filename,probe_manifest,index)
                result['corruptions'][condition]=score_block(rows,t,bias);artifacts.append(filename)
        finally:backend.image_transform=None
        if controller is not None:
            original_alpha=controller.alpha
            try:
                for scale in job['dose_scales']:
                    if scale==1.:
                        result['dose']['1']=score_block(probe_real,t)
                        continue
                    controller.alpha=original_alpha*scale
                    name=f'dose_{scale:g}.jsonl'
                    rows=_score_condition(backend,samples,controller,folder/name,probe_manifest,index)
                    result['dose'][f'{scale:g}']=score_block(rows,t);artifacts.append(name)
            finally:controller.alpha=original_alpha
        timing=[]
        n=min(job['timing_samples'],len(samples))
        if n:
            # Repeat complete candidate scoring; no CPU offload, cold-load timing or
            # hidden partial-resume time is presented as steady-state latency.
            for s in samples[:min(2,n)]:backend.score(s,controller)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            for _ in range(job['timing_repeats']):
                torch.cuda.synchronize();start=time.perf_counter()
                for s in samples[:n]:backend.score(s,controller)
                torch.cuda.synchronize();timing.append((time.perf_counter()-start)/n)
        result['runtime']={'seconds_per_query_repeats':timing,'timed_queries_per_repeat':n,
             'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'timing_includes':'image preprocessing, full candidate scoring, CPU result transfer; clean fixed state'}
        atomic_json(folder/'evaluation.json',result);artifacts.append('evaluation.json')
        store.finish({'completed':True,'count':len(query),'query_labels_used_for_fitting':False},artifacts)
        return 'done'
