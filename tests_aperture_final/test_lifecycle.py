"""Synthetic CPU contract tests, never benchmark evidence."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from aperture_final import worker
from aperture_final.protocol import make_jobs,prepare_job
from vigor_handoff.protocol import JobStore,atomic_json,resolved_manifest
from .test_protocol import rows,cfg


def prepared(tmp_path):
    c=cfg(tmp_path);c['support_seeds']=[0]
    s=rows();q=rows(2)
    for r in q:
        r['sample_id']='q-'+r['sample_id'];r['tile_id']='q-'+r['tile_id']
        r['pre_image']='/q'+r['pre_image'];r['post_image']='/q'+r['post_image']
    (tmp_path/'s0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in s))
    (tmp_path/'q.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in q))
    jobs=make_jobs(c);prepare_job(jobs[0]);return jobs


def test_frozen_fit_never_queries_and_bias_can_use_whole_label_budget(tmp_path,monkeypatch):
    j=prepared(tmp_path)[0];calls=[];reads=[]
    audit=tmp_path/'audit';s=JobStore(audit,{'fixture':True})
    atomic_json(audit/'audit.json',{'passed':True});s.finish({'passed':True},['audit.json'])
    class Backend:
        def score(self,sample,controller=None):
            calls.append(sample.sample_id)
            # Fixed arbitrary output independent of truth: fixture only.
            scores=np.array([.4,.1,-.2]);p=np.exp(scores-scores.max());p/=p.sum()
            return {'sample_id':sample.sample_id,'label_id':sample.label_id,'label':sample.label,
                    'mean_log_scores':scores.tolist(),'probabilities':p.tolist()}
    @contextmanager
    def fake_loaded(job,restore=None):
        yield torch.nn.Linear(2,2),Backend(),None
    def reader(job,name):
        reads.append(name)
        return [SimpleNamespace(**r) for r in resolved_manifest(job[name])]
    monkeypatch.setattr(worker,'_loaded',fake_loaded)
    monkeypatch.setattr(worker,'_samples',reader)
    monkeypatch.setattr(worker,'_gpu_ready',lambda:None)
    monkeypatch.setattr(worker,'_identity',lambda j:{'fixture':True})
    monkeypatch.setattr(worker,'check_job',lambda j:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    assert worker.execute_fit(j,audit)=='done'
    assert 'query' not in reads and not any(s.startswith('q-') for s in calls)
    cal=json.loads((Path(j['output'])/'fit/calibration.json').read_text())
    assert cal['temperature']['calibration_count']==6
    assert cal['bias']['calibration_count']==24
    assert cal['bias']['fit_source']=='all_original_labeled_support; frozen predictor'
    assert worker.verified_record(Path(j['output'])/'fit')
    before=len(calls)
    assert worker.execute_fit(j,audit)=='skipped' and len(calls)==before


def test_resume_rejects_corrupt_cached_scores(tmp_path):
    manifest=rows(1)
    predictions=[]
    for r in manifest:
        predictions.append({'sample_id':r['sample_id'],'label_id':r['label_id'],
             'probabilities':[1/3]*3,'mean_log_scores':[100.,0.,0.], 'prompt_contract_hash':'x'})
    p=tmp_path/'cached.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    samples=[SimpleNamespace(**r) for r in manifest]
    with pytest.raises(ValueError,match='score/probability'):
        worker._score_condition(None,samples,None,p,manifest)


def test_resume_rechecks_saved_prompt_contract(tmp_path):
    manifest=rows(1);predictions=[]
    for r in manifest:
        predictions.append({'sample_id':r['sample_id'],'label_id':r['label_id'],
          'probabilities':[1/3]*3,'mean_log_scores':[0.,0.,0.], 'prompt_contract_hash':'changed'})
    p=tmp_path/'cached.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    samples=[SimpleNamespace(**r) for r in manifest]
    real={r['sample_id']:{'prompt_contract_hash':'original'} for r in manifest}
    with pytest.raises(ValueError,match='token budget'):
        worker._score_condition(None,samples,None,p,manifest,real)
