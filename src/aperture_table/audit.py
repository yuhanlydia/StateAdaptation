"""Real-model support-only gates for every backbone and candidate task."""
from pathlib import Path
import time
import numpy as np
import torch
from .backend import loaded, decoder_sites
from .protocol import read_roles
from .worker import deterministic, identity
from vigor_handoff.protocol import JobStore, atomic_json
from vigor_handoff.core import extract_basis, ResidualController
from visual_lens.audit import gradient_signature, compare_vectors, signature_summary


def signature_checks(off, on):
    def agrees(a,b):
        c=compare_vectors(a,b)
        return bool(c['shape_match'] and c['relative_l2'] is not None and c['relative_l2']<=.05)
    gradients=off['activation_gradients']
    return {
        'nonzero_controller_gradient':bool(torch.isfinite(off['controller_gradient']).all() and off['controller_gradient'].norm()>0),
        'nonzero_visual_gradients':bool(gradients) and all(bool(v.numel() and torch.isfinite(v).all() and v.norm()>0) for v in gradients.values()),
        'checkpoint_replayed':all(on['forward_counts'].get(k,0)>n for k,n in off['forward_counts'].items()),
        'controller_gradient_agrees':agrees(off['controller_gradient'],on['controller_gradient']),
        'activation_gradients_agree':set(gradients)==set(on['activation_gradients']) and all(agrees(v,on['activation_gradients'][k]) for k,v in gradients.items()),
        'checkpoint_loss_agrees':abs(off['loss']-on['loss'])<=.005,
    }


def execute_audit(job, root):
    deterministic(0)
    roles=read_roles(job,images=True)
    # One labeled fitting example per class, never query images.
    sample=[next(r for r in roles['support'] if r['label_id']==c) for c in range(job['classes'])]
    neutral={**job,'arm':'frozen'}
    folder=Path(root)/'audits'/job['model_key']/job['domain_key']
    store=JobStore(folder,identity(neutral,Path(root)/'assets'))
    if store.complete():return 'skipped'
    if (folder/'DONE.json').exists():raise ValueError('sealed model audit corrupted')
    with store.lock(),loaded(neutral) as (model,backend,_):
        sites=decoder_sites(model,job['layers']);controller=None
        try:
            base=backend.score(sample[0]);tic=time.perf_counter()
            bases,spectra=extract_basis(model,sites,sample,backend.loss_and_mask,job['rank'],'covariance',0)
            controller=ResidualController(sites,bases,job['alpha'],'full')
            zero=backend.score(sample[0],controller)
            zero_diff=float(np.max(np.abs(np.array(base['mean_log_scores'])-zero['mean_log_scores'])))
            with torch.no_grad():
                for p in controller.parameters():p.fill_(.01)
            before={};direct={};handles=[]
            for l,k,m in sites:
                key=f'{l}:{k}'
                def capture(module,inp,z,key=key):before[key]=z.detach().clone()
                def check(module,inp,z,key=key):
                    mask=controller._mask
                    delta=(z.detach()-before.pop(key)).float()
                    direct[key]={'outside_max_abs':float(delta[~mask].abs().max()) if (~mask).any() else 0.,
                                 'inside_norm':float(delta[mask].norm())}
                handles.extend([m.register_forward_hook(capture,prepend=True),m.register_forward_hook(check)])
            try:changed=backend.score(sample[0],controller)
            finally:
                for h in handles:h.remove()
            torch.save(controller.payload(),folder/'roundtrip.pt')
            controller.close();controller=ResidualController(sites,bases,job['alpha'],'full')
            controller.restore(torch.load(folder/'roundtrip.pt',weights_only=True,map_location='cpu'))
            restored=backend.score(sample[0],controller)
            roundtrip=float(np.max(np.abs(np.array(changed['mean_log_scores'])-restored['mean_log_scores'])))
            # All four module paths must carry finite gradients through replay.
            off=gradient_signature(model,backend,sample,sites,controller,False,'mean')
            on=gradient_signature(model,backend,sample,sites,controller,True,'mean')
            checks=signature_checks(off,on)
            checks.update(zero_identity=zero_diff<=.005,save_restore_identity=roundtrip<=.005,
                visual_mask_isolation=len(direct)==4 and all(d['outside_max_abs']==0. for d in direct.values()),
                residual_is_nonzero=any(d['inside_norm']>0 for d in direct.values()))
            model.gradient_checkpointing_disable();model.eval()
            durations=[]
            for row in sample:
                torch.cuda.synchronize();t=time.perf_counter();backend.score(row,controller)
                torch.cuda.synchronize();durations.append(time.perf_counter()-t)
            report={'passed':all(checks.values()),'checks':checks,'zero_max_abs':zero_diff,
                'roundtrip_max_abs':roundtrip,'direct_mask':direct,'spectra':spectra,
                'off':signature_summary(off),'on':signature_summary(on),
                'seconds_per_query_support_estimate':float(np.mean(durations)),
                'support_only':True,'query_forwards':0,'wall_seconds':time.perf_counter()-tic,
                'processor_contract':{'family':job['family'],'image_tokens':base['image_tokens']},
                'selected_sites':[{'layer':l,'site':k,'width':m.out_features} for l,k,m in sites]}
            atomic_json(folder/'audit.json',report)
            if not report['passed']:raise RuntimeError('support-only model audit failed: '+repr([k for k,v in checks.items() if not v]))
            store.finish({'passed':True},['audit.json','roundtrip.pt'])
            return 'done'
        finally:
            if controller is not None:controller.close()
