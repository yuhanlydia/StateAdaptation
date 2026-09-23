"""CPU tests of support-only CV selection and saved-fit comparison contracts."""
import json
from pathlib import Path

import pytest

from aperture_debug import worker
from aperture_debug.protocol import repeat_job
from aperture_final.numeric import probabilities
from vigor_handoff.protocol import atomic_json


def records():
    return [{'candidate': 'a', 'split': 'support_validation', 'f1': .5, 'nll': .9},
            {'candidate': 'b', 'split': 'support_validation', 'f1': .7, 'nll': 1.2}]


def test_choose_prioritizes_f1_then_nll_then_stable_candidate_id():
    rows = records()
    assert worker.choose(rows) == 'b'
    rows[0]['f1'] = .7
    assert worker.choose(rows) == 'a'
    rows[0]['nll'] = 1.2
    assert worker.choose(list(reversed(rows))) == 'a'


@pytest.mark.parametrize('split', ['query', 'test', 'fit', 'calibration', None])
def test_choose_rejects_non_support_validation_rows(split):
    rows = records()
    rows[0]['split'] = split
    with pytest.raises(ValueError, match='invalid support'):
        worker.choose(rows)


@pytest.mark.parametrize('field,value', [('f1', float('nan')), ('nll', float('inf')),
                                        ('f1', -1.), ('f1', 1.1), ('nll', -.1)])
def test_choose_rejects_invalid_metric_even_for_losing_candidate(field, value):
    rows = records()
    rows[0][field] = value
    with pytest.raises(ValueError, match='invalid support'):
        worker.choose(rows)


def test_choose_rejects_empty_or_duplicate_candidates():
    with pytest.raises(ValueError):
        worker.choose([])
    with pytest.raises(ValueError):
        worker.choose([records()[0], records()[0]])


def selection_fixture(tmp_path, monkeypatch):
    jobs, finals = [], []
    roles = {}
    for method in ('aperture', 'lora'):
        for candidate in ('c0', 'c1'):
            for fold in range(3):
                jid = f'u--{method}--{candidate}--fold{fold}'
                job = dict(unit_id='u', method=method, candidate=candidate, fold=fold,
                           job_id=jid, output=str(tmp_path/'search'/jid))
                rows = []
                for label in range(2):
                    scores = [0., -1.] if label == 0 else [-1., 0.]
                    rows.append(dict(sample_id=f'fold{fold}-sample{label}', label_id=label,
                                     mean_log_scores=scores, probabilities=probabilities([scores])[0].tolist(),
                                     prediction=label, prompt_contract_hash='test'))
                fit = Path(job['output'])/'fit'
                atomic_json(fit/'identity.json', {'identity': {'job': job}})
                atomic_json(fit/'DONE.json', {'test': jid})
                (fit/'calibration_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                roles[jid] = {'calibration': [dict(r, candidate_labels=['normal', 'tumor']) for r in rows]}
                jobs.append(job)
            finals.append(dict(unit_id='u', method=method, candidate=candidate,
                               job_id=f'u--{method}--{candidate}--final'))
    plan = {'debug_source': 'test', 'units': {'u': {'fold_count': 3}}, 'jobs': jobs, 'finals': finals}
    monkeypatch.setattr(worker, 'read_plan', lambda root: plan)
    monkeypatch.setattr(worker, 'verified_record', lambda folder: {})
    monkeypatch.setattr(worker, 'read_roles', lambda job: roles[job['job_id']])
    return plan, roles


def test_selection_pools_all_folds_and_returns_one_final_per_method(tmp_path, monkeypatch):
    selection_fixture(tmp_path, monkeypatch)
    winners = worker.select(tmp_path, 'u')
    assert [(j['method'], j['candidate']) for j in winners] == [('aperture', 'c0'), ('lora', 'c0')]
    saved = json.loads((tmp_path/'selections/u.json').read_text())
    assert saved['query_used'] is False
    assert len(saved['fit_seals']) == 12
    assert all(r['pooled_metrics']['count'] == 6 and len(r['fold_metrics']) == 3 for r in saved['records'])


def test_missing_fold_cannot_seal_selection(tmp_path, monkeypatch):
    plan, _ = selection_fixture(tmp_path, monkeypatch)
    plan['jobs'].pop(1)
    with pytest.raises(ValueError, match='missing folds'):
        worker.select(tmp_path, 'u')
    assert not (tmp_path/'selections/u.json').exists()


def test_overlapping_validation_folds_cannot_seal_selection(tmp_path, monkeypatch):
    plan, roles = selection_fixture(tmp_path, monkeypatch)
    job0, job1 = plan['jobs'][:2]
    rows = roles[job0['job_id']]['calibration']
    roles[job1['job_id']] = {'calibration': rows}
    target = Path(job1['output'])/'fit/calibration_predictions.jsonl'
    target.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='overlapping validation folds'):
        worker.select(tmp_path, 'u')
    assert not (tmp_path/'selections/u.json').exists()


def test_repeat_retains_all_fields_used_by_inherited_compare_repeat():
    job = {'job_id': 'u--final', 'output': '/tmp/u--final', 'role': 'final', 'layers': [18, 35]}
    repeat = repeat_job(job)
    excluded = {'job_id', 'output'}
    assert {k: v for k, v in job.items() if k not in excluded} == {
        k: v for k, v in repeat.items() if k not in excluded}
    assert repeat['job_id'] != job['job_id'] and repeat['output'] != job['output']


def test_candidates_cannot_be_compared_on_different_validation_populations(tmp_path, monkeypatch):
    plan, roles = selection_fixture(tmp_path, monkeypatch)
    job = plan['jobs'][3]
    changed = [dict(r, sample_id=r['sample_id']+'-other') for r in roles[job['job_id']]['calibration']]
    roles[job['job_id']] = {'calibration': changed}
    (Path(job['output'])/'fit/calibration_predictions.jsonl').write_text(
        ''.join(json.dumps(r)+'\n' for r in changed))
    with pytest.raises(ValueError, match='validation populations differ'):
        worker.select(tmp_path, 'u')
    assert not (tmp_path/'selections/u.json').exists()


def test_missing_final_winner_cannot_pass_global_barrier(tmp_path, monkeypatch):
    plan, _ = selection_fixture(tmp_path, monkeypatch)
    plan['finals'] = [j for j in plan['finals'] if j['method'] != 'lora']
    with pytest.raises(ValueError, match='final winner'):
        worker.seal_global(tmp_path)
    assert not (tmp_path/'SELECTION_COMPLETE.json').exists()
