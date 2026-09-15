"""Reproduce numerical figures from the bundled, source-attributed observations.

No model is trained here. No point, error bar, or uncertainty is synthesized.
Usage: python plotting/make_figures.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'plotting/figure_data.json').read_text())
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.035)
    fig.savefig(OUT / f'{name}.png', dpi=230, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)

# Each chart is a separate figure. Default color cycle; redundant shape encoding.
fig, ax = plt.subplots(figsize=(6.8, 1.85), layout='constrained')
rows = DATA['primary']['methods']
for row, marker in zip(rows, ['s','D','o']):
    ax.scatter(row['scalars'],100*row['macro_f1'],s=82,marker=marker,zorder=3)
    offsets = {'Diagonal lens': (12,-13), 'Full visual lens': (12,6), 'LoRA':(-11,9)}
    align = 'right' if row['name']=='LoRA' else 'left'
    ax.annotate(f"{row['name']}\n{100*row['macro_f1']:.2f}",
                (row['scalars'],100*row['macro_f1']), xytext=offsets[row['name']],
                textcoords='offset points',ha=align,fontsize=10.2)
ax.axhline(100*DATA['primary']['frozen_macro_f1'], linestyle='--',linewidth=1)
ax.text(43,22.34+0.5,'Frozen: 22.34',fontsize=10.2)
ax.set_xscale('log'); ax.set_xlim(35,3e7); ax.set_ylim(19,37)
ax.set_xticks([64,1024,10092544],['64','1,024','10,092,544'])
ax.set_yticks([20,25,30,35]);ax.set_ylabel('Mean macro-F1 (%)',fontsize=10)
ax.set_xlabel('Optimized adaptation scalars (log scale)',fontsize=10)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(labelsize=10.2)
save(fig,'fig1_observation')

fig, ax = plt.subplots(figsize=(6.8,1.8), layout='constrained')
rows=DATA['basis_control']['methods']
vals=[100*r['macro_f1'] for r in rows]
ax.bar(range(4),vals,width=.55)
ax.set_xticks(range(4),[r['name']+'\n'+f"rank {r['rank']}" for r in rows],fontsize=10.2)
ax.set_ylim(0,35); ax.set_ylabel('Mean macro-F1 (%)',fontsize=10)
for i,v in enumerate(vals): ax.text(i,v+.6,f'{v:.2f}',ha='center',fontsize=10.2)
ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=10.2)
save(fig,'fig3_basis')

fig,ax=plt.subplots(figsize=(6.7,2.0),layout='constrained')
d=DATA['prior_reweighting']
for seed,a,b,mark in zip(d['seed'],d['original_kappa'],d['reweighted_kappa'],['o','s','D']):
    ax.plot([0,1],[a,b],marker=mark,label=f'Support seed {seed}',linewidth=1.5,markersize=5)
ax.axhline(0,linestyle=':',linewidth=.8)
ax.set_xlim(-.17,1.17); ax.set_ylim(-1,1.12)
ax.set_xticks([0,1],['Original support mixture','Query-prior mixture (oracle)'],fontsize=10.2)
ax.set_ylabel(r'Directional agreement $\kappa$',fontsize=10)
ax.legend(loc='lower right',fontsize=10.2,frameon=False,ncol=1)
ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=10.2)
save(fig,'fig4_mixture')

fig,ax=plt.subplots(figsize=(6.7,2.25),layout='constrained')
d=DATA['module_alignment']; vals=np.asarray(d['kappa'])
im=ax.imshow(vals,vmin=-1,vmax=1,aspect='auto')
ax.set_xticks(range(4),d['modules']);ax.set_yticks(range(3),d['tasks'])
for i in range(3):
    for j in range(4):
        ax.text(j,i,f'{vals[i,j]:+.3f}',ha='center',va='center',fontsize=10,
                bbox={'facecolor':'white','edgecolor':'none','alpha':0.85,'pad':1.5})
fig.colorbar(im,ax=ax,label=r'Mean $\kappa$')
ax.set_xlabel('Projection module');ax.tick_params(labelsize=10.2)
save(fig,'figA_module_alignment')

assert 10092544//1024 == 9856
assert abs((.3208-.2986)-.0222)<1e-12
assert np.allclose(np.mean(d['kappa'],axis=1),np.mean(vals,axis=1))
print('Four numerical PDF/PNG figures regenerated from recorded data; no inferred error bars.')
