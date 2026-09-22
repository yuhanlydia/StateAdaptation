#!/usr/bin/env python3
"""Wait for available GPUs, run the locked search, then publish its actual results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root).resolve();repo=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env['PYTHONPATH']=str(repo/'src');env['OMP_NUM_THREADS']='1'
    phase='gpu_queue'
    def status(value):
        (root/'pipeline_status.json').write_text(json.dumps({'status':value,'phase':phase,'utc':time.time()},indent=2)+'\n')
    try:
        status('running')
        subprocess.run([sys.executable,'-u','-m','aperture_tuning.cli','--root',str(root),'run',
                        '--gpus','0,1,2,3','--wait-hours','24'],cwd=repo,env=env,check=True)
        phase='publish';status('running')
        subprocess.run([sys.executable,'-u','-m','aperture_tuning.publish','--root',str(root),
                        '--repository',str(repo),'--push'],cwd=repo,env=env,check=True)
        status('uploaded')
    except BaseException:
        status('failed');raise


if __name__=='__main__':main()
