import json
from pathlib import Path
import sys
import pytest
from aperture_final.cli import main,run_partitioned,audits_for
from aperture_final.protocol import make_jobs
from .test_protocol import cfg


def test_plan_does_not_require_models_and_filtered_run_cannot_redefine_it(tmp_path):
    c=cfg(tmp_path);p=tmp_path/'config.json';p.write_text(json.dumps(c))
    assert main(['plan','--config',str(p),'--seeds','0'])==0
    record=json.loads((Path(c['run_root'])/'plan.json').read_text())
    assert len(record['jobs'])==12
    assert main(['plan','--config',str(p)])==0
    assert (Path(c['run_root'])/'plan.json').read_text()
    assert main(['check','--config',str(p)])==2
    assert main(['summarize','--config',str(p),'--allow-partial'])==0


def test_gpu_partition_stops_scheduling_after_error(tmp_path):
    script=tmp_path/'fail.py';script.write_text('import sys\nsys.exit(7)\n')
    tasks=[{'stage':'fit','job':{'job_id':f'j{i}'}} for i in range(3)]
    with pytest.raises(RuntimeError,match='stage failed'):
        run_partitioned(tasks,['0'],script,tmp_path/'logs')
    records=list((tmp_path/'logs').glob('*.exit.json'))
    assert len(records)==1 and json.loads(records[0].read_text())['returncode']==7


def test_gpu_partition_sets_one_visible_device_per_worker(tmp_path):
    script=tmp_path/'worker.py'
    script.write_text("import os,sys,json\nfrom pathlib import Path\np=Path(sys.argv[-1]);j=json.loads(p.read_text())\nassert os.environ['CUDA_VISIBLE_DEVICES'] in ('3','5')\nassert os.environ['HF_HUB_OFFLINE']=='1'\n")
    tasks=[{'stage':'fit','job':{'job_id':f'j{i}'}} for i in range(4)]
    records=run_partitioned(tasks,['3','5'],script,tmp_path/'logs')
    assert len(records)==4 and {r['gpu'] for r in records}=={'3','5'}
