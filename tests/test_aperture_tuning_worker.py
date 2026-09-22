"""CPU checks for query barriers, repeat placement, and cooperative GPU use."""
import json
from pathlib import Path

import pytest

from aperture_tuning import cli, worker
from vigor_handoff.protocol import atomic_json, file_hash


def test_gpu_available_only_when_memory_and_compute_process_checks_pass(monkeypatch):
    outputs = iter([
        '0, GPU-a, 24000\n1, GPU-b, 24000\n2, GPU-c, 22000\n3, GPU-d, 23500\n',
        'GPU-b, 1234\nGPU-b, 4567\n',
    ])
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **kw: next(outputs))
    assert cli.free_gpus() == ['0', '3']


def test_repeat_job_preserves_scientific_configuration_without_mutating_original():
    job = {'job_id': 'unit--aperture--a1', 'output': '/tmp/states/unit--aperture--a1',
           'layers': [18], 'support_seed': 0, 'optimization_seed': 0}
    repeat = worker.repeat_job(job)
    assert repeat['job_id'] == job['job_id'] + '--repeat'
    assert Path(repeat['output']).name == repeat['job_id']
    assert {k: v for k, v in repeat.items() if k not in ('job_id', 'output')} == {
        k: v for k, v in job.items() if k not in ('job_id', 'output')}
    repeat['layers'].append(35)
    assert job['layers'] == [18]


def test_unit_fits_then_selects_then_repeats_on_same_gpu(tmp_path, monkeypatch):
    import aperture_table.worker as table_worker
    jobs = [{'unit_id': 'u', 'job_id': f'u--{method}', 'output': str(tmp_path/method)}
            for method in ('aperture', 'lora')]
    calls = []
    monkeypatch.setattr(cli, 'child', lambda root, action, job, gpu:
                        calls.append((action, job['job_id'], gpu)))
    def selection(root, unit):
        assert calls == [('fit', j['job_id'], '2') for j in jobs]
        return jobs
    monkeypatch.setattr(worker, 'seal_selection', selection)
    monkeypatch.setattr(table_worker, 'compare_repeat', lambda job, output, gate:
                        calls.append(('repeat_gate', job['job_id'], '2')))
    cli.run_unit(tmp_path, {'jobs': jobs}, 'u', '2')
    assert json.loads((tmp_path/'devices/u.json').read_text()) == {'gpu': '2'}
    assert calls == [('fit', j['job_id'], '2') for j in jobs] + [
        ('fit', 'u--aperture--repeat', '2'), ('repeat_gate', 'u--aperture', '2'),
        ('fit', 'u--lora--repeat', '2'), ('repeat_gate', 'u--lora', '2')]


def test_resumed_unit_rejects_different_gpu_before_any_child(tmp_path, monkeypatch):
    atomic_json(tmp_path/'devices/u.json', {'gpu': '2'})
    monkeypatch.setattr(cli, 'child', lambda *a: pytest.fail('must not launch on a different GPU'))
    with pytest.raises(ValueError, match='locked content changed'):
        cli.run_unit(tmp_path, {'jobs': []}, 'u', '1')


def global_fixture(tmp_path, monkeypatch, failing=None):
    jobs = [{'unit_id': unit, 'job_id': unit + '--winner',
             'output': str(tmp_path/'states'/unit)} for unit in ('u0', 'u1')]
    monkeypatch.setattr(worker, 'read_plan', lambda root: {'jobs': jobs, 'tuning_source': 'test-source'})
    monkeypatch.setattr(worker, 'verified_record', lambda folder: {})
    def selection(root, unit):
        atomic_json(tmp_path/'selections'/f'{unit}.json', {'unit_id': unit})
        return [j for j in jobs if j['unit_id'] == unit]
    monkeypatch.setattr(worker, 'seal_selection', selection)
    for job in jobs:
        first = Path(job['output'])/'fit/DONE.json'
        second = Path(worker.repeat_job(job)['output'])/'fit/DONE.json'
        atomic_json(first, {'test': job['job_id']})
        atomic_json(second, {'test': job['job_id'] + '--repeat'})
        atomic_json(tmp_path/'repeat_gates'/f"{job['job_id']}.json", {
            'passed': job['unit_id'] != failing,
            'first_seal': file_hash(first), 'repeat_seal': file_hash(second)})
    return jobs


def test_global_barrier_not_written_if_any_unit_repeat_failed(tmp_path, monkeypatch):
    global_fixture(tmp_path, monkeypatch, failing='u1')
    with pytest.raises(ValueError, match='repeat failed'):
        worker.seal_global(tmp_path)
    assert not (tmp_path/'SELECTION_COMPLETE.json').exists()


def test_global_barrier_not_written_if_repeat_artifact_seal_changed(tmp_path, monkeypatch):
    jobs = global_fixture(tmp_path, monkeypatch)
    atomic_json(Path(worker.repeat_job(jobs[1])['output'])/'fit/DONE.json', {'changed': True})
    with pytest.raises(ValueError, match='repeat seal changed'):
        worker.seal_global(tmp_path)
    assert not (tmp_path/'SELECTION_COMPLETE.json').exists()


def test_global_barrier_covers_every_selected_unit_and_repeat_gate(tmp_path, monkeypatch):
    jobs = global_fixture(tmp_path, monkeypatch)
    assert worker.seal_global(tmp_path) == [j['job_id'] for j in jobs]
    barrier = json.loads((tmp_path/'SELECTION_COMPLETE.json').read_text())
    assert barrier['selection_complete_before_query'] is True
    assert barrier['query_used_for_selection'] is False
    assert len(barrier['files']) == 4
    for relative, expected in barrier['files'].items():
        assert file_hash(tmp_path/relative) == expected


def test_resume_recognizes_completed_unit_without_requiring_gpu(tmp_path, monkeypatch):
    jobs = global_fixture(tmp_path, monkeypatch)
    worker.seal_selection(tmp_path, 'u0')
    assert worker.unit_complete(tmp_path, 'u0') is True
    (tmp_path/'repeat_gates'/f"{jobs[0]['job_id']}.json").unlink()
    assert worker.unit_complete(tmp_path, 'u0') is False


def test_resume_does_not_silently_skip_failed_repeat_gate(tmp_path, monkeypatch):
    global_fixture(tmp_path, monkeypatch, failing='u0')
    worker.seal_selection(tmp_path, 'u0')
    with pytest.raises(ValueError, match='repeat failed'):
        worker.unit_complete(tmp_path, 'u0')


@pytest.mark.parametrize('source,selected', [('stale-source', ['winner']), ('test-source', ['other'])])
def test_evaluation_rejects_stale_or_unselected_job_before_reading_images(tmp_path, monkeypatch, source, selected):
    monkeypatch.setattr(worker, 'read_plan', lambda root: {'tuning_source': 'test-source'})
    monkeypatch.setattr(worker, 'read_roles', lambda *a, **kw: pytest.fail('must not read images'))
    atomic_json(tmp_path/'SELECTION_COMPLETE.json', {'tuning_source': source, 'selected_jobs': selected})
    with pytest.raises(ValueError, match='sealed selected winner'):
        worker.evaluate(tmp_path, {'job_id': 'winner'})


def test_evaluation_rejects_modified_global_selection_before_images(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'read_plan', lambda root: {'tuning_source': 'test-source'})
    monkeypatch.setattr(worker, 'read_roles', lambda *a, **kw: pytest.fail('must not read images'))
    atomic_json(tmp_path/'selections/u.json', {'winner': 'before'})
    atomic_json(tmp_path/'SELECTION_COMPLETE.json', {
        'tuning_source': 'test-source', 'selected_jobs': ['winner'],
        'files': {'selections/u.json': file_hash(tmp_path/'selections/u.json')}})
    atomic_json(tmp_path/'selections/u.json', {'winner': 'after'})
    with pytest.raises(ValueError, match='global selection changed'):
        worker.evaluate(tmp_path, {'job_id': 'winner'})
