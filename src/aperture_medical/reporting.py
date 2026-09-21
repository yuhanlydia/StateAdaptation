"""Complete medical exports: never select hospitals or methods by query results."""
from collections import defaultdict
import csv,json
from pathlib import Path
import numpy as np
from vigor_handoff.protocol import atomic_json,digest
from .protocol import ARMS,source_signature,check_job
from .worker import verified_record

LABELS={'frozen':'Frozen','lora1':'LoRA-1pass','lora4':'LoRA-4passes','random':'Random visual','aperture':'Aperture'}
METRICS=('macro_f1','balanced_accuracy','nll','brier','ece','auroc','auprc')


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['domain'],r['budget'],r['method'])].append(r)
    result=[]
    for (d,b,m),rr in groups.items():
        if len({r['seed'] for r in rr})!=len(rr):raise ValueError('duplicate support seed')
        row=dict(domain=d,budget=b,method=m,seeds=len(rr))
        for key in METRICS:
            if key not in rr[0]:continue
            vals=[r[key] for r in rr]
            if any(v is None for v in vals):row[key+'_mean']=None;row[key+'_sd']=None;continue
            if not np.isfinite(vals).all():raise ValueError('nonfinite aggregate')
            row[key+'_mean']=float(np.mean(vals));row[key+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        result.append(row)
    return result


def table(rows, metric, budget, domains):
    selected=[r for r in rows if r['budget']==budget and r['domain'] in domains]
    methods=list(dict.fromkeys(r['method'] for r in selected));lookup={(r['domain'],r['method']):r for r in selected}
    lines=['% Prespecified medical protocol. Mean +/- sample SD over support sets.',
           r'\begin{tabular}{l'+'r'*len(domains)+'}',r'\toprule',
           'Method & '+' & '.join(d.replace('_',r'\_') for d in domains)+r' \\',r'\midrule']
    for m in methods:
        cells=[]
        for d in domains:
            r=lookup[(d,m)];v=r[metric+'_mean'];s=r[metric+'_sd']
            cells.append('--' if v is None else f'${v:.4f}'+(f'_{{\\pm {s:.4f}}}' if s is not None else '')+'$')
        lines.append(m+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n'


def _export_impl(jobs,root):
    out=Path(root)/'paper_exports';out.mkdir(parents=True,exist_ok=True)
    for pattern in ('table_*.tex','*.csv','fig_*.pdf','fig_*.png','full_results.json'):
        for p in out.glob(pattern):p.unlink()
    results=[];missing=[];stats=[]
    for j in jobs:
        try:
            check_job(j,assets=False)
            f=Path(j['output'])/'fit';e=Path(j['output'])/'evaluation'
            verified_record(f);verified_record(e)
            r=json.loads((e/'evaluation.json').read_text())
            if any(r[k]!=j[k] for k in ('job_id','unit_id','domain','arm','support_seed')):raise ValueError('result identity mismatch')
            if r['clean']['raw']['count']!=j['query_count']:raise ValueError('query count mismatch')
            ident=json.loads((e/'identity.json').read_text())['identity']
            if ident['source']!=source_signature():raise ValueError('result source signature differs')
            r['budget']=j['total_label_budget'];results.append(r)
            fs=json.loads((f/'metrics.json').read_text());stats.append(dict(domain=j['domain'],budget=j['total_label_budget'],seed=j['support_seed'],method=LABELS[j['arm']],
                **{k:fs[k] for k in ('optimized_scalars','basis_scalars','serialized_state_bytes','optimizer_updates','fit_wall_seconds')}))
        except (OSError,ValueError,KeyError) as exc:missing.append(dict(job=j['job_id'],error=str(exc)))
    cov=dict(planned=len(jobs),completed=len(results),missing=missing,paper_ready=not missing)
    atomic_json(out/'coverage.json',cov)
    if missing:raise RuntimeError('Incomplete medical matrix; only coverage written, no paper table')
    units=defaultdict(dict);qhash=defaultdict(set)
    for r in results:units[r['unit_id']][r['arm']]=r;qhash[r['domain']].add(r['clean']['ids_sha256'])
    if any(len(x)!=1 for x in qhash.values()):raise ValueError('query differs across budget/support seeds')
    rows=[];patient_rows=[]
    for unit,arms in units.items():
        if set(arms)!=set(ARMS):raise ValueError('incomplete comparison unit')
        for arm,r in arms.items():
            for v in ('raw','temperature'):
                rows.append(dict(domain=r['domain'],budget=r['budget'],seed=r['support_seed'],method=LABELS[arm]+(' + T' if v=='temperature' else ''),**{k:r['clean'][v][k] for k in METRICS}))
            if arm=='frozen':rows.append(dict(domain=r['domain'],budget=r['budget'],seed=r['support_seed'],method='Frozen + support bias',**{k:r['clean']['candidate_bias'][k] for k in METRICS}))
        a=arms['aperture']
        for v,key in [('raw','selected_lora_raw'),('temperature','selected_lora_temperature')]:
            r=arms[a[key]]
            rows.append(dict(domain=a['domain'],budget=a['budget'],seed=a['support_seed'],method='LoRA-selected'+(' + T' if v=='temperature' else ''),**{k:r['clean'][v][k] for k in METRICS}))
    ag=aggregate(rows);csv_write(out/'per_seed.csv',rows);csv_write(out/'mean_sd.csv',ag);csv_write(out/'resources.csv',stats)
    # CAMELYON per-patient means, not patch count presented as independent patients.
    from .numeric import metrics
    from vigor_handoff.protocol import read_jsonl
    for j in jobs:
        rr=read_jsonl(Path(j['output'])/'evaluation/clean.jsonl');groups=defaultdict(list)
        for r in rr:groups[r['group_id']].append(r)
        for g,gr in groups.items():
            mm=metrics([x['mean_log_scores'] for x in gr],[x['label_id'] for x in gr])
            patient_rows.append(dict(domain=j['domain'],budget=j['total_label_budget'],seed=j['support_seed'],method=LABELS[j['arm']],group=g,count=len(gr),nll=mm['nll'],macro_f1=mm['macro_f1']))
    csv_write(out/'query_group_metrics.csv',patient_rows)
    for budget in sorted({r['budget'] for r in rows}):
        ds=list(dict.fromkeys(r['domain'] for r in rows if r['budget']==budget))
        for m in METRICS:
            (out/f'table_n{budget}_{m}.tex').write_text(table(ag,m,budget,ds))
    # Paired mean and worst-hospital scores: never mix PathMNIST into center mean.
    center=[]
    for include,name in [(lambda d:d.startswith('hospital_'),'all_five_centers'),(lambda d:d in {'hospital_0','hospital_1','hospital_3','hospital_4'},'four_expansion_centers')]:
        gg=defaultdict(list)
        for r in rows:
            if include(r['domain']):gg[(r['budget'],r['seed'],r['method'])].append(r)
        for (b,s,m),rr in gg.items():
            center.append(dict(scope=name,budget=b,seed=s,method=m,centers=len(rr),
                mean_nll=float(np.mean([r['nll'] for r in rr])),worst_nll=max(r['nll'] for r in rr),
                mean_f1=float(np.mean([r['macro_f1'] for r in rr])),worst_f1=min(r['macro_f1'] for r in rr)))
    csv_write(out/'center_summary.csv',center)
    atomic_json(out/'full_results.json',dict(coverage=cov,rows=rows,aggregate=ag,resources=stats,centers=center))
    plot(ag,out)
    return cov


def plot(ag,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Frozen','Frozen + support bias','LoRA-1pass','LoRA-4passes','LoRA-selected + T','Random visual','Aperture']
    for b in sorted({r['budget'] for r in ag}):
        ds=list(dict.fromkeys(r['domain'] for r in ag if r['budget']==b))
        for metric in ('nll','auroc','macro_f1'):
            fig,ax=plt.subplots(figsize=(8,4.6));lookup={(r['domain'],r['method']):r for r in ag if r['budget']==b}
            for i,m in enumerate(names):
                yy=[lookup[d,m][metric+'_mean'] for d in ds]
                if any(v is None for v in yy):continue
                ee=[lookup[d,m][metric+'_sd'] or 0. for d in ds]
                ax.errorbar(np.arange(len(ds))+(i-3)*.07,yy,yerr=ee,marker='o',capsize=2,label=m)
            ax.set_xticks(np.arange(len(ds)),ds,rotation=20);ax.set_ylabel(metric)
            ax.set_title(f'{b} total support labels; three support sets')
            ax.legend(fontsize=8,ncol=2);fig.tight_layout()
            for ext in ('pdf','png'):fig.savefig(Path(out)/f'fig_n{b}_{metric}.{ext}',dpi=200)
            plt.close(fig)


def export(jobs,root):
    """Revoke every paper artifact if any collection, consistency or plotting step fails."""
    out=Path(root)/'paper_exports';out.mkdir(parents=True,exist_ok=True)
    try:
        return _export_impl(jobs,root)
    except Exception as exc:
        for pattern in ('table_*.tex','*.csv','fig_*.pdf','fig_*.png','full_results.json'):
            for p in out.glob(pattern):p.unlink()
        try:coverage=json.loads((out/'coverage.json').read_text())
        except (OSError,ValueError):coverage={'planned':len(jobs)}
        coverage.update(paper_ready=False,export_error=str(exc))
        atomic_json(out/'coverage.json',coverage)
        raise
