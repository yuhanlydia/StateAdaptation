#!/usr/bin/env python3
"""Durable queue: prior campaign -> targeted debug -> verified publication."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--previous',required=True);a=p.parse_args()
    root=Path(a.root).resolve();repo=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env['PYTHONPATH']=str(repo/'src');env['OMP_NUM_THREADS']='1'
    def status(value):
        (root/'pipeline_status.json').write_text(json.dumps({'status':value,'utc':time.time()})+'\n')
    try:
        status('running')
        subprocess.run([sys.executable,'-u','-m','aperture_debug.cli','--root',str(root),'run','--previous',a.previous],cwd=repo,env=env,check=True)
        from aperture_debug.publish import publish
        publish(root,repo,a.previous);status('uploaded')
    except BaseException:status('failed');raise

if __name__=='__main__':main()
