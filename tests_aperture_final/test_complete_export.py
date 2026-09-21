"""Complete artifact fixtures for CPU integration only, not measured results."""
import json
from pathlib import Path
import numpy as np
import pytest
from aperture_final.protocol import make_jobs
from aperture_final.worker import score_block
from aperture_final.reporting import export
from vigor_handoff.protocol import JobStore, atomic_json
from .test_protocol import cfg


def fixture_states(jobs):
    for j in jobs:
        rows=[{'sample_id':f'q{i}','label_id':i%3,'mean_log_scores':[0.,.1,.2]} for i in range(j['query_count'])]
        block=score_block(rows,2.,bias=[0.,0.,0.])
        f=Path(j['output'])/'fit';store=JobStore(f,{'fixture':True,'job':j})
        atomic_json(f/'calibration.json',{'temperature':{'temperature':2.,'calibration_count':6,'calibration_nll':1.,'fit_source':'held_out_labeled_support'},'raw_calibration_nll':1.2})
        store.finish({'optimized_scalars':1,'basis_scalars':2,'optimizer_updates':4},['calibration.json'])
        e=Path(j['output'])/'evaluation';store=JobStore(e,{'fixture':True,'job':j})
        result={k:j[k] for k in ('job_id','unit_id','domain','support_seed','arm')}
        result.update(clean=block,probe_clean=block,probe_ids_sha256=block['ids_sha256'],
          corruptions={},dose={},selected_lora_raw='lora1',selected_lora_temperature='lora4',temperature=2.,
          runtime={'seconds_per_query_repeats':[], 'gpu':'SYNTHETIC_TEST_ONLY', 'peak_allocated_bytes':0})
        atomic_json(e/'evaluation.json',result)
        store.finish({'count':len(rows)},['evaluation.json'])


def test_complete_export_keeps_all_comparators_and_all_seeds(tmp_path):
    c=cfg(tmp_path);jobs=make_jobs(c);fixture_states(jobs)
    result=export(jobs,c['run_root'])
    assert result['paper_ready'] is True and result['completed_states']==12
    folder=Path(c['run_root'])/'paper_exports'
    full=json.loads((folder/'full_results.json').read_text())
    assert len(full['aggregate'])==11
    assert all(r['n_support_seeds']==3 for r in full['aggregate'])
    names={r['method'] for r in full['aggregate']}
    assert {'LoRA-1pass','LoRA-4passes','LoRA-selected + T','Frozen + all-support bias'}<=names
    assert (folder/'table_nll_part1.tex').is_file()
    assert (folder/'calibration.csv').is_file()


def test_missing_one_state_blocks_paper_table_exports(tmp_path):
    c=cfg(tmp_path);jobs=make_jobs(c);fixture_states(jobs[:-1])
    with pytest.raises(RuntimeError,match='missing/invalid'):export(jobs,c['run_root'])
    assert not (Path(c['run_root'])/'paper_exports/table_nll.tex').exists()
    partial=export(jobs,c['run_root'],allow_partial=True)
    assert partial['paper_ready'] is False


def test_all_arms_must_share_query_order_not_just_count(tmp_path):
    c=cfg(tmp_path);jobs=make_jobs(c);fixture_states(jobs)
    j=jobs[0];e=Path(j['output'])/'evaluation'
    r=json.loads((e/'evaluation.json').read_text());r['clean']['ids_sha256']='wrong-order'
    atomic_json(e/'evaluation.json',r)
    JobStore(e,{'fixture':True,'job':j}).finish({'count':6},['evaluation.json'])
    with pytest.raises(ValueError,match='query IDs/order'):export(jobs,c['run_root'])
    assert json.loads((Path(c['run_root'])/'paper_exports/coverage.json').read_text())['paper_ready'] is False


def test_invalidating_a_completed_state_revokes_old_paper_tables(tmp_path):
    c=cfg(tmp_path);jobs=make_jobs(c);fixture_states(jobs);export(jobs,c['run_root'])
    (Path(jobs[0]['output'])/'evaluation/evaluation.json').write_text('{}')
    with pytest.raises(RuntimeError):export(jobs,c['run_root'])
    folder=Path(c['run_root'])/'paper_exports'
    assert not (folder/'table_nll.tex').exists()
    assert not (folder/'full_results.json').exists()
