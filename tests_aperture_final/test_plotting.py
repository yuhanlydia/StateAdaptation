import json
from pathlib import Path
import pytest
from aperture_final.plotting import make_plots
from aperture_final.reporting import export
from aperture_final.protocol import make_jobs
from .test_protocol import cfg
from .test_complete_export import fixture_states


def test_plotting_refuses_partial_results(tmp_path):
    p=tmp_path/'r.json';p.write_text(json.dumps({'coverage':{'paper_ready':False}}))
    with pytest.raises(ValueError,match='complete'):make_plots(p,tmp_path/'plots',formats=('png',))


def test_plotting_runs_from_complete_measurement_schema(tmp_path):
    c=cfg(tmp_path);jobs=make_jobs(c);fixture_states(jobs);export(jobs,c['run_root'])
    report=Path(c['run_root'])/'paper_exports/full_results.json'
    files=make_plots(report,tmp_path/'plots',formats=('png',))
    assert files and all(Path(p).is_file() for p in files)
    assert len(list((tmp_path/'plots').glob('*.png')))==2
