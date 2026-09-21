#!/usr/bin/env python3
"""Plot measured medical exports; no data/model call and no invented intervals."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports',default='runs/aperture_medical_multimodel_v1/paper_exports/main-replication')
    a=p.parse_args();root=Path(a.exports)
    if not json.loads((root/'coverage.json').read_text()).get('paper_ready'):raise RuntimeError('complete validated export required')
    rows=list(csv.DictReader((root/'mean_sd.csv').open()))
    out=root/'figures';out.mkdir(exist_ok=True)
    preferred=['Frozen','Frozen + all-support bias','LoRA-1pass','LoRA-4passes',
               'LoRA-selected + T','Random visual','Aperture','Aperture + T']
    for family,budget in sorted({(r['family'],r['budget']) for r in rows}):
        block=[r for r in rows if r['family']==family and r['budget']==budget]
        hs=sorted({int(r['hospital']) for r in block})
        for metric in ('nll','macro_f1','auroc','patient_macro_nll'):
            fig,ax=plt.subplots(figsize=(8,4.6))
            for i,method in enumerate(preferred):
                ms={int(r['hospital']):r for r in block if r['method']==method}
                if set(ms)!=set(hs):continue
                vals=[float(ms[h][metric+'_mean']) if ms[h][metric+'_mean'] else np.nan for h in hs]
                sd=[float(ms[h][metric+'_sd']) if ms[h][metric+'_sd'] else 0 for h in hs]
                ax.errorbar(np.arange(len(hs))+(i-(len(preferred)-1)/2)*.06,vals,yerr=sd,marker='o',capsize=2,linestyle='none',label=method)
            ax.set_xticks(range(len(hs)),[f'H{h}' for h in hs]);ax.set_xlabel('Hospital (official metadata ID)')
            ax.set_ylabel(metric);ax.set_title(f'{family}, {budget} total support labels; mean ± support-seed SD')
            ax.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False)
            fig.tight_layout()
            for ext in ('pdf','svg','png'):fig.savefig(out/f'{family}_n{budget}_{metric}.{ext}',dpi=220,bbox_inches='tight')
            plt.close(fig)
    print(out)


if __name__=='__main__':main()
