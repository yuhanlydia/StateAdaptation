"""Sealed support fitting and restored-state scoring. No query forwards in fitting."""
from pathlib import Path
from copy import deepcopy
import importlib.metadata
import json
import random
import time
import numpy as np
from vigor_handoff.protocol import JobStore, atomic_json, digest, file_hash, read_jsonl
from aperture_final.numeric import metrics, probabilities, fit_temperature
from .protocol import ARMS, MODELS, source_signature, read_roles, write_locked


def verified_record(folder):
    folder = Path(folder)
    ident = json.loads((folder / 'identity.json').read_text())
    done = json.loads((folder / 'DONE.json').read_text())
    if ident['fingerprint'] != digest(ident['identity']) or done['fingerprint'] != ident['fingerprint']:
        raise ValueError(f'identity mismatch: {folder}')
    if not done.get('artifacts'):
        raise ValueError('empty completion seal')
    for name, checksum in done['artifacts'].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or file_hash(p) != checksum:
            raise ValueError(f'artifact changed: {p}')
    return done


def validate_rows(rows, expected, complete=True):
    index = {r['sample_id']: r for r in expected}
    if len(index) != len(expected):
        raise ValueError('duplicate expected IDs')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate prediction IDs')
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in expected[:len(rows)]]:
        raise ValueError('prediction ID/order mismatch')
    if complete and len(rows) != len(expected):
        raise ValueError('incomplete predictions')
    for row in rows:
        ref = index[row['sample_id']]
        s = np.asarray(row['mean_log_scores'], dtype=float)
        p = np.asarray(row['probabilities'], dtype=float)
        if row['label_id'] != ref['label_id'] or len(s) != len(ref['candidate_labels']):
            raise ValueError('prediction labels/candidates mismatch')
        if not row.get('prompt_contract_hash') or p.shape != s.shape or not np.allclose(p, probabilities([s])[0], rtol=1e-7, atol=1e-8):
            raise ValueError('prediction score/probability contract mismatch')


def seal_unit(jobs, path):
    arms = {j['arm']: j for j in jobs}
    if set(arms) != set(ARMS) or len({j['unit_id'] for j in jobs}) != 1 or len(jobs) != 4:
        raise ValueError('selection requires one complete four-arm comparison unit')
    cal, seals, folders = {}, {}, {}
    for arm, job in arms.items():
        folder = Path(job['output']) / 'fit'
        verified_record(folder)
        cal[arm] = json.loads((folder / 'calibration.json').read_text())
        seals[arm] = file_hash(folder / 'DONE.json')
        folders[arm] = str(folder.resolve())
    def pick(scaled):
        vals = {a: float(cal[a]['temperature']['calibration_nll'] if scaled else cal[a]['raw_nll']) for a in ('lora1','lora4')}
        if not np.isfinite(list(vals.values())).all():
            raise ValueError('nonfinite model selection criterion')
        return min(vals, key=lambda a: (vals[a], a))
    result = dict(unit_id=jobs[0]['unit_id'], source=source_signature(), query_metrics_used=False,
                  selected_lora_raw=pick(False), selected_lora_temperature=pick(True),
                  fit_seals=seals, fit_folders=folders)
    write_locked(path, result)
    return result


def verify_selection(job, path):
    result = json.loads(Path(path).read_text())
    if result['unit_id'] != job['unit_id'] or result['source'] != source_signature() or result['query_metrics_used']:
        raise ValueError('stale or invalid model selection')
    for arm in ARMS:
        folder = Path(result['fit_folders'][arm])
        verified_record(folder)
        if file_hash(folder / 'DONE.json') != result['fit_seals'][arm]:
            raise ValueError('selection fit seal changed')
    if Path(result['fit_folders'][job['arm']]).resolve() != (Path(job['output'])/'fit').resolve():
        raise ValueError('selection belongs to another fitted state')
    return result


def model_files(job):
    root = Path(job['model_path'])
    cfg = json.loads((root / 'config.json').read_text())
    if cfg.get('model_type') != MODELS[job['model_key']][1] or cfg.get('quantization_config'):
        raise ValueError('wrong or quantized local checkpoint')
    text = cfg.get('text_config', cfg)
    if text.get('num_hidden_layers', 0) <= max(job['layers']):
        raise ValueError('checkpoint does not contain the prescribed decoder layers')
    weights = sorted(root.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError('BF16 safetensor weights are required')
    index = root / 'model.safetensors.index.json'
    if index.exists():
        required = set(json.loads(index.read_text())['weight_map'].values())
        if not required <= {p.name for p in weights}:
            raise FileNotFoundError('incomplete model weight shards')
    if not (root/'tokenizer_config.json').is_file() or not (root/'preprocessor_config.json').is_file():
        raise FileNotFoundError('native tokenizer and image processor files are required')
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.model','.jinja','.txt'))
    return files


def check_assets(job, asset_dir, refresh=False):
    """Hash weight bytes once, then fail on changed size/mtime before every job."""
    path = Path(asset_dir) / (job['model_key']+'.json')
    files = model_files(job)
    stat = {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['stat'] != stat:
            raise ValueError('model assets changed after preflight; use a new run root')
        if refresh and any(file_hash(p) != saved['sha256'][str(p.resolve())] for p in files):
            raise ValueError('model content fingerprint changed')
    else:
        saved = {'model_id':job['model_id'], 'stat':stat,
                 'sha256':{str(p.resolve()):file_hash(p) for p in files}}
        write_locked(path, saved)
    return digest(saved)


def environment():
    packages = {}
    for name in ('torch','transformers','peft','numpy','scipy','Pillow','safetensors'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return packages


def deterministic(seed=0):
    import os
    import torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('launch through the CLI to establish deterministic CUDA settings before imports')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def identity(job, asset_dir):
    return {'job':job, 'source':source_signature(), 'environment':environment(),
            'model_assets':check_assets(job, asset_dir),
            'split_sha256':file_hash(job['split_record']),
            'manifests':{k:file_hash(job[k]) for k in ('support','calibration','query')}}


def _write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))


def _scores(rows):
    return np.asarray([r['mean_log_scores'] for r in rows]), np.asarray([r['label_id'] for r in rows])


def score_all(backend, rows, controller, path=None):
    """Resume a prefix only within a surrounding fingerprinted state store."""
    output = read_jsonl(path, repair_tail=True) if path and Path(path).exists() else []
    validate_rows(output, rows, complete=False)
    if path:
        with Path(path).open('a') as f:
            for row in rows[len(output):]:
                result = backend.score(row, controller)
                validate_rows([result], [row])
                f.write(json.dumps(result, allow_nan=False)+'\n'); f.flush(); output.append(result)
    else:
        output = [backend.score(r, controller) for r in rows]
    validate_rows(output, rows)
    return output


def execute_fit(job, root):
    import torch
    from .backend import loaded, decoder_sites
    from vigor_handoff.core import extract_basis, ResidualController, fit_coefficients
    from vigor_handoff.worker import train_lora
    from visual_lens.controls import fit_candidate_bias
    deterministic(job['optimization_seed'])
    roles = read_roles(job, images=True)  # Integrity only; query is never forwarded here.
    audit = Path(root)/'audits'/job['model_key']/job['domain_key']
    verified_record(audit)
    if not json.loads((audit/'audit.json').read_text())['passed']:
        raise RuntimeError('required support-only model audit failed')
    current = identity(job,Path(root)/'assets')
    audit_identity = json.loads((audit/'identity.json').read_text())['identity']
    if any(audit_identity.get(k)!=current[k] for k in ('source','environment','model_assets')):
        raise ValueError('support-only audit source/environment/model changed')
    ident = {**current, 'audit_seal':file_hash(audit/'DONE.json')}
    folder = Path(job['output'])/'fit'; store = JobStore(folder, ident)
    if store.complete(): return 'skipped'
    if (folder/'DONE.json').exists(): raise ValueError('corrupt completed fitting state')
    # Fitting is short. Partial fitting checkpoints are never mixed with a refit.
    with store.lock(), loaded(job) as (model,backend,_):
        (folder/'calibration_predictions.jsonl').unlink(missing_ok=True)
        controller = None; artifacts=[]; extraction=0.; history=[]
        try:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
            if job['arm'].startswith('lora'):
                before = {n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
                history,updates,count = train_lora(model,backend,roles['support'],job)
                delta = sum(float((p.detach().float().cpu()-before[n]).square().sum()) for n,p in model.named_parameters() if n in before)
                if not np.isfinite(delta) or delta<=0: raise RuntimeError('LoRA did not make a finite nonzero update')
                model.save_pretrained(folder/'adapter',safe_serialization=True)
                artifacts += [str(p.relative_to(folder)) for p in (folder/'adapter').rglob('*') if p.is_file()]
                bases_count=0; visits=len(roles['support'])*job['lora_passes']; basis_visits=0
            elif job['arm']=='aperture':
                sites=decoder_sites(model,job['layers']); tic=time.perf_counter()
                bases,spectra=extract_basis(model,sites,roles['support'],backend.loss_and_mask,job['rank'],'covariance',0)
                torch.cuda.synchronize(); extraction=time.perf_counter()-tic
                controller=ResidualController(sites,bases,job['alpha'],'full')
                a=backend.score(roles['support'][0]); b=backend.score(roles['support'][0],controller)
                if np.max(np.abs(np.array(a['mean_log_scores'])-b['mean_log_scores']))>.005:
                    raise RuntimeError('zero-controller identity failed')
                history=fit_coefficients(model,controller,roles['support'],lambda r:backend.loss_and_mask(r,controller)[0],
                    steps=job['steps'],lr=job['lr'],l2=job['l2'],reduction=job['loss_reduction'])
                if not all(torch.isfinite(p).all() for p in controller.parameters()) or not any(float(p.detach().norm())>0 for p in controller.parameters()):
                    raise RuntimeError('controller did not make a finite nonzero update')
                torch.save(controller.payload(),folder/'controller.pt'); artifacts.append('controller.pt')
                atomic_json(folder/'basis_spectra.json',spectra); artifacts.append('basis_spectra.json')
                count=controller.num_scalars();updates=len(history); bases_count=sum(b.numel() for b in bases.values())
                visits=len(roles['support'])*updates;basis_visits=len(roles['support'])
            else:
                count=updates=bases_count=visits=basis_visits=0
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            model.eval();model.gradient_checkpointing_disable()
            for p in model.parameters(): p.requires_grad_(False)
            cal_rows=score_all(backend,roles['calibration'],controller)
            s,y=_scores(cal_rows); temperature=fit_temperature(s,y,penalty=job['temperature_penalty'])
            cal={'raw_nll':metrics(s,y)['nll'],'temperature':temperature,'query_forwards':0,
                 'fit_ids':digest([r['sample_id'] for r in roles['support']]),
                 'calibration_ids':digest([r['sample_id'] for r in roles['calibration']])}
            if job['arm']=='frozen':
                bias_rows=score_all(backend,roles['support'],None)+cal_rows
                bs,by=_scores(bias_rows); bias,meta=fit_candidate_bias(bs,by,l2=job['bias_l2'])
                cal['bias']={'values':bias.tolist(),'metadata':meta,'count':len(bias_rows),'source':'all_original_support'}
            _write_rows(folder/'calibration_predictions.jsonl',cal_rows)
            atomic_json(folder/'calibration.json',cal)
            stats={'optimizer_updates':updates,'optimized_scalars':count,'basis_scalars':bases_count,
              'fit_wall_seconds':elapsed,'basis_extraction_seconds':extraction,'fitting_visits':visits,
              'basis_visits':basis_visits,'fit_count':len(roles['support']),'cal_count':len(roles['calibration']),
              'total_labels':job['total_labels'],'query_forwards':0,'total_model_parameters':sum(p.numel() for p in model.parameters()),
              'serialized_state_bytes':sum((folder/p).stat().st_size for p in artifacts if p=='controller.pt' or p.startswith('adapter/')),
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'gpu':torch.cuda.get_device_name(0),'execution_gpu_id':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),'history':history}
            atomic_json(folder/'fitting.json',stats)
            store.finish({'job_id':job['job_id'],'query_forwards':0},artifacts+['calibration_predictions.jsonl','calibration.json','fitting.json'])
            return 'done'
        finally:
            if controller is not None: controller.close()


def compare_repeat(job, repeat_output, path):
    first=Path(job['output'])/'fit'; second=Path(repeat_output)/'fit'
    for p in (first,second): verified_record(p)
    ia=json.loads((first/'identity.json').read_text())['identity']
    ib=json.loads((second/'identity.json').read_text())['identity']
    ja={k:v for k,v in ia['job'].items() if k not in ('job_id','output')}
    jb={k:v for k,v in ib['job'].items() if k not in ('job_id','output')}
    if ja!=jb or any(ia.get(k)!=ib.get(k) for k in ('source','model_assets','environment','manifests')):
        raise ValueError('repeat-fit source/model/split differs from canonical')
    fa=json.loads((first/'fitting.json').read_text());fb=json.loads((second/'fitting.json').read_text())
    if fa['execution_gpu_id']!=fb['execution_gpu_id']:raise ValueError('repeat fits must use the same GPU')
    a=read_jsonl(first/'calibration_predictions.jsonl');b=read_jsonl(second/'calibration_predictions.jsonl')
    if [(r['sample_id'],r['label_id'],r['prompt_contract_hash']) for r in a] != [(r['sample_id'],r['label_id'],r['prompt_contract_hash']) for r in b]:
        raise ValueError('repeat-fit comparison inputs changed')
    diff=float(np.max(np.abs(_scores(a)[0]-_scores(b)[0])))
    result={'passed':diff<=.005,'max_abs_calibration_score_delta':diff,'tolerance':.005,
            'source':source_signature(),'query_metrics_used':False,
            'first_seal':file_hash(first/'DONE.json'),'repeat_seal':file_hash(second/'DONE.json'),
            'first_output':str(first.parent.resolve()),'repeat_output':str(second.parent.resolve())}
    atomic_json(path,result)
    if not result['passed']: raise RuntimeError('same-GPU fresh-process repeat fit failed; do not retune or replace canonical result')
    return result



def verify_repeat(job, root):
    gate=json.loads((Path(root)/'repeat'/job['model_key']/'gate.json').read_text())
    if not gate.get('passed') or gate.get('source')!=source_signature():
        raise ValueError('repeat-fit gate absent or stale')
    for name in ('first','repeat'):
        folder=Path(gate[name+'_output'])/'fit';verified_record(folder)
        if file_hash(folder/'DONE.json')!=gate[name+'_seal']:raise ValueError('repeat gate state changed')
    return gate

def execute_evaluate(job, root, selection_path):
    import torch
    from .backend import loaded
    deterministic(job['optimization_seed']); roles=read_roles(job,images=True)
    verify_repeat(job,root)
    selected=verify_selection(job,selection_path)
    fit=Path(job['output'])/'fit'; done=verified_record(fit)
    saved=json.loads((fit/'identity.json').read_text())['identity']
    now=identity(job,Path(root)/'assets')
    if any(saved.get(k)!=v for k,v in now.items()):raise ValueError('fit inputs/source/environment changed')
    folder=Path(job['output'])/'evaluation'
    store=JobStore(folder,{'fit_seal':digest(done),'selection':file_hash(selection_path),'identity':now})
    if store.complete():return 'skipped'
    if (folder/'DONE.json').exists():raise ValueError('corrupt completed evaluation')
    cal=json.loads((fit/'calibration.json').read_text()); t=cal['temperature']['temperature']
    with store.lock(),loaded(job,fit) as (model,backend,controller):
        # This gate uses only saved held-out support, before any query forward.
        old=read_jsonl(fit/'calibration_predictions.jsonl'); replay=score_all(backend,roles['calibration'],controller)
        if [r['prompt_contract_hash'] for r in old]!=[r['prompt_contract_hash'] for r in replay]:
            raise ValueError('saved-state processor contract changed')
        delta=float(np.max(np.abs(_scores(old)[0]-_scores(replay)[0])))
        if delta>.005:raise RuntimeError('saved-state replay failed before query scoring')
        torch.cuda.synchronize();start=time.perf_counter()
        rows=score_all(backend,roles['query'],controller,folder/'predictions.jsonl')
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        s,y=_scores(rows); raw=metrics(s,y);scaled=metrics(s,y,t)
        if raw['macro_f1']!=scaled['macro_f1']:raise RuntimeError('positive temperature changed class decisions')
        result={'job_id':job['job_id'],'unit_id':job['unit_id'],'model_key':job['model_key'],
           'model_id':job['model_id'],'domain_key':job['domain_key'],'support_seed':job['support_seed'],'arm':job['arm'],
           'source':source_signature(),'query_ids_sha256':digest([r['sample_id'] for r in rows]),
           'query_label_sha256':digest([r['label_id'] for r in rows]),'raw':raw,'temperature':scaled,
           'saved_state_replay_max_abs':delta,'elapsed_this_invocation_seconds':elapsed,
           'selection':{k:selected[k] for k in ('selected_lora_raw','selected_lora_temperature')}}
        if job['arm']=='frozen':result['bias']=metrics(s,y,bias=cal['bias']['values'])
        atomic_json(folder/'evaluation.json',result)
        store.finish({'job_id':job['job_id']},['predictions.jsonl','evaluation.json'])
        return 'done'
