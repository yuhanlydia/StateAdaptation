"""Read-only evaluation of a sealed fitted state on all visual controls."""
from pathlib import Path
import copy,json,time
import numpy as np
import torch
from .controls import image_permutation,intervene_images,fit_candidate_bias,biased_probabilities
from vigor_handoff.protocol import JobStore,atomic_json,file_hash,read_jsonl,resolved_manifest,validate_predictions
from vigor_handoff.worker import Backend,discover_sites,fingerprint_assets,seed_everything
from vigor_handoff.core import ResidualController


def execute_evidence(job):
    base=job['base_job'];source=Path(base['output'])
    # All fitting belongs to the base job; query interventions never call fit.
    base_identity=fingerprint_assets(base)
    original=JobStore(source,base_identity)
    if not original.complete():raise RuntimeError(f'base job not sealed: {source}')
    identity={'job':job,'base_fingerprint':original.fingerprint,
              'base_done_sha256':file_hash(source/'DONE.json')}
    store=JobStore(job['output'],identity)
    if store.complete():return {'status':'skipped','job_id':job['job_id']}
    if (store.path/'DONE.json').exists():raise RuntimeError('sealed outputs changed; refuse to overwrite')
    from eventttt.task_vlm import load_task_model
    from eventttt.io import read_samples,read_task_samples
    from eventttt.metrics import classification_metrics_nclass
    reader=read_samples if base['kind']=='paired' else read_task_samples
    support=reader(base['support']);query=reader(base['query']);query_rows=resolved_manifest(base['query'])
    lookup={s.sample_id:s for s in query}
    mapping=image_permutation(list(lookup),job['shuffle_seed'])
    fitted=json.loads((source/'ADAPTED.json').read_text())
    controller=None;model=None
    with store.lock():
        try:
            seed_everything(base['optimization_seed'])
            lora=base['arm'].startswith('lora')
            model,processor=load_task_model(base['model_path'],base['family'],use_lora=lora,gradient_checkpointing=False)
            if {p.device.type for p in model.parameters()}!={'cuda'}:raise RuntimeError('evidence evaluation requires all-GPU model, no implicit offload')
            if lora:
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                set_peft_model_state_dict(model,load_file(str(source/'adapter/adapter_model.safetensors')))
            elif (source/'controller.pt').is_file():
                payload=torch.load(source/'controller.pt',map_location='cpu',weights_only=True)
                bases={(int(k.split(':')[0]),k.split(':')[1]):v for k,v in payload['bases'].items()}
                mods=discover_sites(model,base['layers'],base['sites'])
                controller=ResidualController(mods,bases,payload['alpha'],payload['mode'],payload['hard_projection'])
                controller.restore(payload)
            for p in model.parameters():p.requires_grad_(False)
            model.eval();backend=Backend(base,model,processor)
            # Use archived real predictions only after full identity verification.
            real=read_jsonl(source/'predictions.jsonl');validate_predictions(real,query_rows)
            real_index={r['sample_id']:r for r in real}
            def images(sample):
                if base['kind']=='paired':
                    from eventttt.qwen import load_sample_pair
                    return load_sample_pair(sample,base['crop_size'])
                from eventttt.task_qwen import load_task_image
                return (load_task_image(sample),)
            bias=None;bias_meta={}
            if base['arm']=='frozen':
                sb=[backend.score(s) for s in support]
                bias,bias_meta=fit_candidate_bias([r['mean_log_scores'] for r in sb],
                    [r['label_id'] for r in sb],l2=job['bias_l2'])
                atomic_json(store.path/'candidate_bias.json',{'bias':bias.tolist(),**bias_meta})
            labels=query_rows[0].get('candidate_labels',('intact','damaged','destroyed'))
            all_metrics={};artifacts=[]
            for condition in job['conditions']:
                target=store.path/condition;target.mkdir(parents=True,exist_ok=True)
                path=target/'predictions.jsonl'
                if condition=='real':
                    rows=real
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                else:
                    def transform(sample,im):
                        donor=images(lookup[mapping[sample.sample_id]]) if condition.startswith('shuffle') else None
                        return intervene_images(im,condition,donor)
                    backend.image_transform=transform
                    rows=read_jsonl(path,repair_tail=True) if path.exists() else []
                    validate_predictions(rows,query_rows,complete=False)
                    seen={r['sample_id'] for r in rows}
                    with path.open('a') as f:
                        for sample in query:
                            if sample.sample_id in seen:continue
                            row=backend.score(sample,controller)
                            reference=real_index[sample.sample_id]
                            if row['prompt_contract_hash']!=reference.get('prompt_contract_hash'):
                                raise ValueError(f'text/span/token-budget changed by {condition} for {sample.sample_id}')
                            row['visual_condition']=condition
                            row['donor_id']=mapping[sample.sample_id] if condition.startswith('shuffle') else None
                            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();rows.append(row)
                validate_predictions(rows,query_rows)
                metrics=classification_metrics_nclass([r['label_id'] for r in rows],np.asarray([r['probabilities'] for r in rows]),labels)
                all_metrics[condition]=metrics;atomic_json(target/'metrics.json',metrics)
                artifacts.extend([str(path.relative_to(store.path)),str((target/'metrics.json').relative_to(store.path))])
                if bias is not None:
                    adjusted=[]
                    for row in rows:
                        p=biased_probabilities(row['mean_log_scores'],bias)
                        adjusted.append({**row,'probabilities':p.tolist(),'prediction':labels[int(p.argmax())],
                                         'method':'support_candidate_bias'})
                    bp=target/'bias_predictions.jsonl';bp.write_text(''.join(json.dumps(r)+'\n' for r in adjusted))
                    bm=classification_metrics_nclass([r['label_id'] for r in adjusted],np.asarray([r['probabilities'] for r in adjusted]),labels)
                    all_metrics[condition+'_bias']=bm;artifacts.append(str(bp.relative_to(store.path)))
            atomic_json(store.path/'interventions.json',{'mapping':mapping,'conditions':job['conditions'],
               'base_job':base,'base_adaptation':fitted,'fit_on_controls':False,'bias_fit':bias_meta,
               'interpretation':'Input controls alter the distribution; difference-in-differences is descriptive, not a proof of attention causality.'})
            artifacts.append('interventions.json')
            if bias is not None:artifacts.append('candidate_bias.json')
            store.finish({'conditions':all_metrics,'evidence':'visual_dependence_control','job_id':job['job_id']},artifacts)
            return {'status':'done','job_id':job['job_id']}
        finally:
            if controller is not None:controller.close()
            if model is not None:del model
            torch.cuda.empty_cache()
