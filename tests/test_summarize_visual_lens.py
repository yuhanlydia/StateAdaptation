import json
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    'summarize_visual_lens',
    Path(__file__).resolve().parents[1] / 'scripts' / 'summarize_visual_lens.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
resolve_paths = module.resolve_paths


def test_summary_paths_follow_configured_run_root(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'run_root': 'runs/current'}))

    plan, output = resolve_paths(config)

    assert plan.as_posix() == 'runs/current/plan.json'
    assert output.as_posix() == 'runs/current/summary'


def test_explicit_summary_paths_override_config(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'run_root': 'runs/current'}))

    plan, output = resolve_paths(config, 'custom/plan.json', 'custom/summary')

    assert plan.as_posix() == 'custom/plan.json'
    assert output.as_posix() == 'custom/summary'
