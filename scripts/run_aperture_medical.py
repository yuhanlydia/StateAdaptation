#!/usr/bin/env python3
"""Entry point: configure reproducible GPU math before any torch import."""
import os,sys
from pathlib import Path
for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}.items():os.environ[k]=v
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_medical.cli import main
if __name__=='__main__':raise SystemExit(main())
