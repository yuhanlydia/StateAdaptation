import importlib
import json
from pathlib import Path


def test_p0_plan_has_matched_seeds_all_four_bright_events_and_strong_lora():
    assert importlib.util.find_spec('visual_lens.plan') is not None, 'P0 planner required'
    from visual_lens.plan import build_plan
    config=json.loads(Path('configs/visual_lens_p0.json').read_text())
    jobs=build_plan(config)
    primary=[j for j in jobs if j['stage']=='matched']
    bright=[j for j in primary if j['dataset']=='bright']
    assert len(bright)==60
    assert {j['support_seed'] for j in bright}=={0,1,2}
    for event in config['bright_events']:
        for seed in [0,1,2]:
            block=[j for j in bright if j['domain']==event and j['support_seed']==seed]
            assert {j['arm'] for j in block}=={'frozen','lora','lora_passes4','random_kv','ours'}
            assert len({(j['support'],j['query']) for j in block})==1
    assert any(j['stage']=='audit' for j in jobs)
    assert len([j for j in jobs if j['stage']=='objective'])==8
    assert all(j['engine']=='evidence' and j['base_job']['stage']=='matched' for j in jobs if j['stage']=='evidence')
    assert len({j['job_id'] for j in jobs})==len(jobs)
    assert jobs==build_plan(config)
