#!/usr/bin/env python3
"""Explicit opt-in download, never an implicit side effect of run/check."""
import argparse,hashlib,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_medical.data import PATH_URL,PATH_MD5
p=argparse.ArgumentParser();p.add_argument('dataset',choices=['camelyon17','pathmnist']);p.add_argument('--root',default='data/medical_raw');a=p.parse_args()
root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
if a.dataset=='camelyon17':
    from wilds import get_dataset
    ds=get_dataset(dataset='camelyon17',root_dir=str(root),download=True)
    print('Set source_root to:',ds.data_dir)
else:
    out=root/'pathmnist_224.npz'
    if not out.exists():
        temp=out.with_suffix('.download')
        urllib.request.urlretrieve(PATH_URL,temp)
        h=hashlib.md5()
        with temp.open('rb') as f:
            for x in iter(lambda:f.read(8<<20),b''):h.update(x)
        if h.hexdigest()!=PATH_MD5:raise ValueError('official PathMNIST archive checksum mismatch')
        temp.rename(out)
    h=hashlib.md5()
    with out.open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    if h.hexdigest()!=PATH_MD5:raise ValueError('cached PathMNIST archive checksum mismatch')
    print('Set source_root to:',out.resolve())
