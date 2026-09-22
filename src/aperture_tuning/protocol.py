from copy import deepcopy
import json
from pathlib import Path
from aperture_table.protocol import make_jobs, write_locked, source_signature
from vigor_handoff.protocol import digest, file_hash

PROTOCOL = 'aperture_tuning_v1'


def tuning_source():
    return digest({'table': source_signature(), 'tuning': [(p.name, file_hash(p))
        for p in sorted(Path(__file__).parent.glob('*.py'))]})


def candidates(depth):
    mid, last = depth // 2, depth - 1
    relative = [mid, last]
    position = relative if relative != [14, 27] else [depth // 4, 3 * depth // 4]
    aperture = [
        ('a0', [14, 27], 3., .05, 4),
        ('a1', [mid], 3., .05, 4),
        ('a2', position, 3., .05, 4),
        ('a3', relative, 1., .01, 4),
        ('a4', relative, 1., .01, 12),
        ('a5', relative, 3., .01, 12),
    ]
    return {'aperture': [(name, dict(layers=layers, alpha=alpha, lr=lr, steps=steps))
                for name, layers, alpha, lr, steps in aperture],
            'lora': [(f'l{i}', dict(lora_lr=lr, lora_passes=passes))
                for i, (lr, passes) in enumerate(( (lr,p) for lr in (5e-5,2e-4,8e-4) for p in (1,4)))]}


def build(base, root):
    root = Path(root).resolve()
    base_jobs = make_jobs(base)
    jobs, contexts = [], {}
    signature = tuning_source()
    for reference in base_jobs:
        if reference['arm'] != 'aperture': continue
        cfg = json.loads((Path(reference['model_path'])/'config.json').read_text())
        depth = cfg.get('text_config',cfg)['num_hidden_layers']
        for method, options in candidates(depth).items():
            for candidate, overrides in options:
                job = deepcopy(reference)
                job.update(overrides)
                job.update(arm='aperture' if method=='aperture' else 'lora4',
                           method=method,candidate=candidate,tuning_protocol=PROTOCOL,tuning_source=signature)
                job['job_id'] = f"{job['unit_id']}--{method}--{candidate}"
                job['output'] = str(root/'states'/job['job_id'])
                architecture = {k:job[k] for k in ('model_key','domain_key','layers','rank','alpha','family','image_size','attention','weight_dtype')}
                key = digest(architecture)[:16]
                job['context'] = str(root/'contexts'/key)
                contexts.setdefault(key, job)
                jobs.append(job)
    assert len(jobs)==192
    for unit in {j['unit_id'] for j in jobs}:
        for method in ('aperture','lora'):
            chosen=[j for j in jobs if j['unit_id']==unit and j['method']==method]
            assert len(chosen)==6
    return {'protocol':PROTOCOL,'tuning_source':signature,'base_config':base,
            'selection_rule':'max calibration macro_f1, then min raw NLL, then candidate ID',
            'query_status':'previously inspected; exploratory replication only',
            'query_used_for_selection':False,'candidate_fit_count':192,'selected_query_states':32,
            'jobs':jobs,'audits':list(contexts.values())}


def read_plan(root):
    plan=json.loads((Path(root)/'plan.json').read_text())
    if plan['tuning_source'] != tuning_source(): raise ValueError('tuning source changed; use a new root')
    return plan


def select_rows(records):
    # This interface deliberately accepts calibration-only records, never query metrics.
    if len(records)!=6 or len({r['candidate'] for r in records})!=6:
        raise ValueError('six distinct candidates required')
    ids={r['candidate'] for r in records}
    if ids not in ({f'a{i}' for i in range(6)}, {f'l{i}' for i in range(6)}):
        raise ValueError('complete single-method candidate registry required')
    if any(r.get('split')!='calibration' for r in records):
        raise ValueError('selection accepts held-out support calibration only')
    import math
    if any(not math.isfinite(r['macro_f1']) or not math.isfinite(r['nll']) for r in records):
        raise ValueError('nonfinite calibration metric')
    if any(not 0 <= r['macro_f1'] <= 1 or r['nll'] < 0 for r in records):
        raise ValueError('invalid metric range')
    return min(records,key=lambda r:(-r['macro_f1'],r['nll'],r['candidate']))['candidate']
