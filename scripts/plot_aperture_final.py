#!/usr/bin/env python3
"""Generate final-round paper figures from a complete measurement export."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_final.plotting import make_plots

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',default='runs/aperture_final_v1/paper_exports/full_results.json')
    p.add_argument('--output',default='runs/aperture_final_v1/paper_exports/figures')
    a=p.parse_args()
    print(f'Created {len(make_plots(a.results,a.output))} plot files from completed measurements.')
