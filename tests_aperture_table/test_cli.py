import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from aperture_table.protocol import default_config
from aperture_table.cli import parser,select_jobs,lock_plan


def test_plan_contains_exact_matrix_and_locks_changes(tmp_path):
 c=default_config();c['run_root']=str(tmp_path/'aperture_table_v1')
 p=lock_plan(c)
 assert len(p['jobs'])==64 and p['trainable_fits']==48
 c['models'][0]['path']='different_snapshot'
 with pytest.raises(ValueError):lock_plan(c)


def test_unknown_filters_raise_and_subset_not_new_protocol():
 from aperture_table.protocol import make_jobs
 jobs=make_jobs(default_config())
 assert len(select_jobs(jobs,['g34'],['h2'],[0]))==4
 with pytest.raises(ValueError):select_jobs(jobs,['gemma'],None,None)


def test_entry_help_no_gpu_or_model_import_required():
 cmd=[sys.executable,'scripts/run_aperture_table.py','--help']
 result=subprocess.run(cmd,capture_output=True,text=True)
 assert result.returncode==0 and 'fill-main' in result.stdout
