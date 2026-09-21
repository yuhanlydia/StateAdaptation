"""Plot actual completed exports; never replace absent measurements with guesses."""
from __future__ import annotations
import json
from pathlib import Path
import re
import numpy as np


def make_plots(report_path,outdir,formats=('pdf','svg','png')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(Path(report_path).read_text())
    if not data['coverage'].get('paper_ready'):
        raise ValueError('plots require a complete final-round export')
    if not formats or set(formats)-{'pdf','svg','png'}:raise ValueError('unsupported figure format')
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);paths=[]
    def slug(x):return re.sub('[^A-Za-z0-9_-]+','_',str(x))
    def save(fig,name):
        fig.tight_layout()
        for ext in formats:
            p=out/(name+'.'+ext);fig.savefig(p,dpi=240,bbox_inches='tight');paths.append(str(p))
        plt.close(fig)
    # One chart per figure, no combined axes, hidden winners or estimated errors.
    methods=['Frozen + T','Frozen + all-support bias','LoRA-selected',
             'LoRA-selected + T','Aperture','Aperture + T']
    for domain in dict.fromkeys(r['domain'] for r in data['aggregate']):
        lookup={r['method']:r for r in data['aggregate'] if r['domain']==domain}
        selected=[m for m in methods if m in lookup]
        for metric,axis in [('nll','Candidate NLL (lower is better)'),('macro_f1','Macro-F1 (higher is better)')]:
            fig,ax=plt.subplots(figsize=(7.8,4.0))
            x=np.arange(len(selected));values=[lookup[m][metric+'_mean'] for m in selected]
            ax.barh(x,values,height=.6)
            for i,m in enumerate(selected):
                sd=lookup[m][metric+'_sd']
                if sd is not None:ax.errorbar(values[i],i,xerr=sd,fmt='none',capsize=3)
            ax.set_yticks(x,selected);ax.invert_yaxis();ax.set_xlabel(axis)
            ax.set_title(domain.replace('_',' ')+' | same total labeled budget')
            save(fig,'calibration_'+slug(domain)+'_'+metric)
    # Corruptions: seed-0 diagnostic probes, fixed clean-fitted states/calibrators.
    cr=data.get('corruptions',[])
    for domain in dict.fromkeys(r['domain'] for r in cr):
        dr=[r for r in cr if r['domain']==domain and r['variant']=='raw']
        for prefix,conditions in [('gaussian',['clean','gaussian4','gaussian8']),('jpeg',['clean','jpeg70','jpeg40'])]:
            fig,ax=plt.subplots(figsize=(6.4,3.9))
            for arm,marker in zip(('frozen','lora1','lora4','aperture'),('o','s','^','D')):
                records={r['condition']:r for r in dr if r['arm']==arm}
                if set(conditions)-set(records):raise ValueError('corruption figure would omit missing conditions')
                ax.plot(range(3),[records[c]['nll'] for c in conditions],marker=marker,label=arm)
            ax.set_xticks(range(3),conditions);ax.set_ylabel('Candidate NLL')
            ax.set_title(domain.replace('_',' ')+' | fixed-state 60-query probe')
            ax.legend(frameon=False);save(fig,'degradation_'+slug(domain)+'_'+prefix)
    for domain in dict.fromkeys(r['domain'] for r in data.get('residual_dose',[])):
        dr=sorted([r for r in data['residual_dose'] if r['domain']==domain],key=lambda r:r['scale'])
        if len(dr)!=3 or [r['scale'] for r in dr]!=[0.,.5,1.]:raise ValueError('incomplete residual-dose diagnostic')
        fig,ax=plt.subplots(figsize=(5.5,3.6))
        ax.plot([r['scale'] for r in dr],[r['nll'] for r in dr],marker='o')
        ax.set_xticks([0,.5,1]);ax.set_xlabel('Residual multiplier (no refitting)');ax.set_ylabel('Raw candidate NLL')
        ax.set_title(domain.replace('_',' ')+' | diagnostic, not model selection')
        save(fig,'residual_dose_'+slug(domain))
    (out/'README.md').write_text(
      '# Measurement-based final-round figures\n\n'
      'Calibration bars: mean and sample SD across support seeds. No SD is fabricated for a single seed. '
      'Corruption and residual-dose curves use the fixed seed-0 probe, without error bars or query-selected parameters. '
      'Positive-temperature scaling preserves argmax, so raw and temperature-adjusted F1 coincide for a fixed arm. '
      'All LoRA schedules and all other comparators remain available in the CSV/LaTeX tables.\n')
    return paths
