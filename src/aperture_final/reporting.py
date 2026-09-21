"""Complete-only paper exports. Missing/failed cells never become zeros."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from .worker import verified_record
from .protocol import write_locked
from vigor_handoff.protocol import atomic_json

METRICS=('macro_f1','balanced_accuracy','nll','brier','ece')
LABELS={'frozen':'Frozen','lora1':'LoRA-1pass','lora4':'LoRA-4passes','aperture':'Aperture'}


def aggregate_rows(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['domain'],r['method'])].append(r)
    out=[]
    for (domain,method), values in grouped.items():
        seeds=[r['support_seed'] for r in values]
        if len(seeds)!=len(set(seeds)):raise ValueError('duplicate support seed in aggregate')
        counts={r['count'] for r in values}
        if len(counts)!=1:raise ValueError('different query counts within a domain/method')
        row={'domain':domain,'method':method,'n_support_seeds':len(values),'query_count_per_state':counts.pop()}
        for k in METRICS:
            a=np.asarray([r[k] for r in values],float)
            if not np.isfinite(a).all():raise ValueError('nonfinite exported metric')
            row[k+'_mean']=float(a.mean());row[k+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None
        out.append(row)
    return out


def _escape(text):
    return str(text).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#')


def table_tex(rows, metric):
    domains=list(dict.fromkeys(r['domain'] for r in rows));methods=list(dict.fromkeys(r['method'] for r in rows))
    lookup={(r['domain'],r['method']):r for r in rows}
    text=[r'% New held-out-support protocol; do not replace historical whole-support results.',
          r'% Mean +/- sample SD over support seeds. Unclipped NLL uses saved log scores.',
          r'\begin{tabular}{l'+'r'*len(domains)+'}',r'\toprule',
          'Method & '+' & '.join(_escape(d) for d in domains)+r' \\',r'\midrule']
    for method in methods:
        cells=[]
        for domain in domains:
            r=lookup.get((domain,method))
            if r is None:raise ValueError('table would have missing cells')
            mean=r[metric+'_mean'];sd=r[metric+'_sd']
            cells.append(f'${mean:.4f}$' if sd is None else f'${mean:.4f}_{{\\pm {sd:.4f}}}$')
        text.append(_escape(method)+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(text+[r'\bottomrule',r'\end{tabular}',''])


def _csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('');return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def export(jobs, root, allow_partial=False):
    out=Path(root)/'paper_exports';out.mkdir(parents=True,exist_ok=True)
    # Revoke only this exporter's derived tables when rechecking completion.
    # Underlying predictions, fitted states and original reports are untouched.
    for pattern in ('table_*.tex','clean_*.csv','calibration.csv','paired_contrasts.csv',
                    'corruptions.csv','residual_dose.csv','resources.csv','full_results.json','RESULTS.md'):
        for path in out.glob(pattern):path.unlink()
    all_results=[];missing=[];fit_stats={};calibrations=[]
    for j in jobs:
        try:
            f=Path(j['output'])/'fit';e=Path(j['output'])/'evaluation'
            verified_record(f);verified_record(e)
            result=json.loads((e/'evaluation.json').read_text())
            if any(result.get(k)!=j[k] for k in ('job_id','unit_id','domain','support_seed','arm')):
                raise ValueError('result/config identity differs')
            if result['clean']['raw']['count']!=j['query_count']:raise ValueError('wrong query count')
            fit_stats[j['job_id']]=json.loads((f/'metrics.json').read_text())
            cal=json.loads((f/'calibration.json').read_text())
            calibrations.append({'domain':j['domain'],'support_seed':j['support_seed'],'arm':j['arm'],
              'temperature':cal['temperature']['temperature'],'calibration_count':cal['temperature']['calibration_count'],
              'raw_calibration_nll':cal['raw_calibration_nll'],'temperature_calibration_nll':cal['temperature']['calibration_nll'],
              'selected_lora_raw':result['selected_lora_raw'],'selected_lora_temperature':result['selected_lora_temperature']})
            all_results.append(result)
        except (OSError,ValueError,KeyError) as exc:
            missing.append({'job_id':j['job_id'],'error':str(exc)})
    coverage={'planned_states':len(jobs),'completed_states':len(all_results),'missing_or_invalid':missing,
              'paper_ready':False,'collection_complete':not missing,'new_gpu_metrics_only':True}
    atomic_json(out/'coverage.json',coverage)
    if missing and not allow_partial:
        raise RuntimeError(f'{len(missing)} states missing/invalid; only coverage.json written, not paper tables')
    if missing:
        atomic_json(out/'PARTIAL_NOT_FOR_PAPER.json',coverage)
        return coverage
    byunit=defaultdict(dict)
    for r in all_results:byunit[r['unit_id']][r['arm']]=r
    rows=[];corrupt=[];dose=[];resources=[];contrasts=[]
    for unit,arms in byunit.items():
        if len(arms)!=4:raise ValueError(f'incomplete unit: {unit}')
        if len({r['clean']['ids_sha256'] for r in arms.values()})!=1:
            raise ValueError(f'query IDs/order differ across arms: {unit}')
        if len({r['probe_ids_sha256'] for r in arms.values()})!=1:
            raise ValueError(f'probe IDs differ across arms: {unit}')
        for arm,r in arms.items():
            base={'domain':r['domain'],'support_seed':r['support_seed']}
            for variant,method in [('raw',LABELS[arm]),('temperature',LABELS[arm]+' + T')]:
                rows.append({**base,'method':method,**{k:r['clean'][variant][k] for k in ('count',)+METRICS}})
            if arm=='frozen':
                rows.append({**base,'method':'Frozen + all-support bias',**{k:r['clean']['candidate_bias'][k] for k in ('count',)+METRICS}})
            blocks={'clean':r['probe_clean'],**r['corruptions']}
            # Only include corruption rows for the predefined noise subset.
            if r['corruptions']:
                for condition,b in blocks.items():
                    for variant in ('raw','temperature'):
                        corrupt.append({**base,'arm':arm,'variant':variant,'condition':condition,
                           'temperature':r['temperature'],**{k:b[variant][k] for k in ('count',)+METRICS}})
            for scale,b in r['dose'].items():
                dose.append({**base,'scale':float(scale),**{k:b['raw'][k] for k in ('count',)+METRICS}})
            stats=fit_stats[r['job_id']];times=r['runtime']['seconds_per_query_repeats']
            resources.append({**base,'arm':arm,
                 **{k:stats.get(k) for k in ('optimized_scalars','basis_scalars','serialized_state_bytes',
                    'optimizer_updates','fitting_visits','basis_visits','fit_wall_seconds','extraction_seconds',
                    'temperature_scalars','output_bias_scalars')},
                 'gpu':r['runtime']['gpu'], 'latency_repeat_count':len(times),
                 'seconds_per_query_mean':float(np.mean(times)) if times else None,
                 'seconds_per_query_repeat_sd':float(np.std(times,ddof=1)) if len(times)>1 else None,
                 'peak_allocated_bytes':r['runtime']['peak_allocated_bytes']})
        a=arms['aperture'];base={'domain':a['domain'],'support_seed':a['support_seed']}
        for variant,key,name in [('raw','selected_lora_raw','LoRA-selected'),('temperature','selected_lora_temperature','LoRA-selected + T')]:
            chosen=a[key];b=arms[chosen]['clean'][variant]
            rows.append({**base,'method':name,**{k:b[k] for k in ('count',)+METRICS}})
            for ours_variant in ('raw','temperature'):
                ours=a['clean'][ours_variant]
                contrasts.append({**base,'comparison':f'Aperture-{ours_variant} minus {name}',
                     'selected_lora':chosen,**{k:ours[k]-b[k] for k in METRICS}})
    # Repeated supports must use a locked query set, not just identical counts.
    domains=defaultdict(set)
    for r in all_results:domains[r['domain']].add(r['clean']['ids_sha256'])
    if any(len(s)>1 for s in domains.values()):raise ValueError('query IDs/order changed across support seeds')
    aggr=aggregate_rows(rows)
    _csv(out/'clean_per_seed.csv',rows);_csv(out/'clean_mean_sd.csv',aggr)
    _csv(out/'paired_contrasts.csv',contrasts);_csv(out/'corruptions.csv',corrupt)
    _csv(out/'residual_dose.csv',dose);_csv(out/'resources.csv',resources)
    _csv(out/'calibration.csv',calibrations)
    domain_order=list(dict.fromkeys(r['domain'] for r in aggr))
    for metric in ('nll','macro_f1','ece'):
        (out/f'table_{metric}.tex').write_text(table_tex(aggr,metric))
        for start in range(0,len(domain_order),4):
            selected_domains=set(domain_order[start:start+4])
            part=[r for r in aggr if r['domain'] in selected_domains]
            (out/f'table_{metric}_part{start//4+1}.tex').write_text(table_tex(part,metric))
    coverage['paper_ready']=True
    atomic_json(out/'full_results.json',{'coverage':coverage,'clean':rows,'aggregate':aggr,
           'corruptions':corrupt,'residual_dose':dose,'resources':resources,'contrasts':contrasts,'calibration':calibrations})
    (out/'RESULTS.md').write_text(
        '# Aperture final round\n\nAll planned states complete. New protocol: original support labels are split '
        'between fitting and held-out calibration; no extra query labels are used.\n\n'
        'Primary analysis: stable candidate NLL, before and after positive temperature scaling, '
        'with LoRA schedule selected on calibration labels only. F1/BA/Brier/ECE and all arms are retained. '
        'NLL is computed from logsumexp without probability clipping; legacy-clipped NLL is also stored in raw JSON.\n\n'
        'Corruption results are a paired 60-query, seed-0 probe and are NOT an official corruption benchmark. '
        'They do not prove semantic nuisance removal. Residual-dose sweeps are read-only diagnostics, '
        'not query-selected replacement models. Latency SD describes repeats, not support-seed uncertainty.\n\n'
        'Use table_nll.tex, table_macro_f1.tex and table_ece.tex as editable LaTeX table fragments. '
        'Keep these separate from the historical full-support experiment tables.\n')
    atomic_json(out/'coverage.json',coverage)
    (out/'PARTIAL_NOT_FOR_PAPER.json').unlink(missing_ok=True)
    return coverage
