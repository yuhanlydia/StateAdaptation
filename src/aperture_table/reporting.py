"""Full-coverage export and exact, non-destructive filling of 112 manuscript cells."""
from collections import Counter
from pathlib import Path
import csv
import json
import re
import numpy as np
from .protocol import MODELS,DOMAINS,ARMS,expected_cells,make_jobs,source_signature,read_roles
from .worker import verified_record,verify_selection,validate_rows
from aperture_final.numeric import metrics
from vigor_handoff.protocol import atomic_json,file_hash,read_jsonl,digest


def summarize_records(records):
    jobs=make_jobs(__import__('aperture_table.protocol',fromlist=['default_config']).default_config())
    expected={j['job_id']:j for j in jobs}
    if len(records)!=64 or set(r['job_id'] for r in records)!=set(expected):
        raise ValueError('all 64 prespecified states are required; no partial mean')
    hashes={};groups={};sources=set()
    for r in records:
        j=expected[r['job_id']]
        for key in ('model_key','model_id','domain_key','support_seed','arm','unit_id'):
            if r[key]!=j[key]:raise ValueError('record identity/protocol mismatch')
        sources.add(r['source'])
        pair=(r['query_ids_sha256'],r['query_label_sha256'])
        if r['domain_key'] in hashes and hashes[r['domain_key']]!=pair:raise ValueError('query mismatch between methods/models/seeds')
        hashes[r['domain_key']]=pair
        if r['raw']['count']!=j['query_count'] or r['temperature']['count']!=j['query_count']:raise ValueError('wrong query count')
        if r['raw']['macro_f1']!=r['temperature']['macro_f1']:raise ValueError('positive temperature changed F1')
        groups.setdefault((r['model_key'],r['domain_key'],r['arm']),[]).append(r)
        if r['arm']=='frozen':
            if 'bias' not in r:raise ValueError('required output-bias comparator missing')
            groups.setdefault((r['model_key'],r['domain_key'],'bias'),[]).append({**r,'raw':r['bias']})
    if len(sources)!=1:raise ValueError('mixed implementation sources')
    cells={};rows=[]
    for (model,domain,arm),rr in groups.items():
        if sorted(r['support_seed'] for r in rr)!=[0,1]:raise ValueError('two fixed support seeds required')
        row=dict(model=model,domain=domain,method=arm,seeds=2)
        for field,slot,scale in [('f1',('raw','macro_f1'),100.),('nll',('raw','nll'),1.),('nllt',('temperature','nll'),1.)]:
            if arm=='bias' and field=='nllt':continue
            values=np.array([r[slot[0]][slot[1]] for r in rr],dtype=float)*scale
            if not np.isfinite(values).all():raise ValueError('nonfinite table measurement')
            value={'mean':float(values.mean()),'sd':float(values.std(ddof=1)),'n':2,
                   'support_values':{str(r['support_seed']):float(v) for r,v in zip(rr,values)}}
            cells[f'{model}-{domain}-{arm}-{field}']=value
            row[field+'_mean']=value['mean'];row[field+'_sd']=value['sd']
        rows.append(row)
    if set(cells)!=expected_cells():raise ValueError('manuscript cell coverage mismatch')
    return {'source':sources.pop(),'protocol':'aperture_table_v1','cells':cells,'rows':rows,
            'query_hashes':hashes,'completed_states':64,'description':'new 200/180-query protocol; sample SD over support seeds 0/1'}


def format_cell(key,cells):
    v=cells[key];dp=2 if key.endswith('-f1') else 4
    m,d,a,metric=key.split('-')
    peers=[x['mean'] for k,x in cells.items() if k.startswith(m+'-'+d+'-') and k.endswith('-'+metric)]
    best=max(peers) if metric=='f1' else min(peers)
    mean=f'{v["mean"]:.{dp}f}';sd=f'{v["sd"]:.{dp}f}'
    if v['mean']==best:mean=r'\mathbf{'+mean+'}'
    return '$'+mean+r'_{\pm '+sd+'}$'


def table_tex(summary):
    cells=summary['cells'];lines=[r'% Measured aperture_table_v1; two support seeds, same query lists.',
      r'\begin{tabular}{@{}llrrrrrr@{}}\toprule',
      r'& & \multicolumn{3}{c}{CAMELYON17--H2} & \multicolumn{3}{c}{PathMNIST--224} \\',
      r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
      r'Backbone & Method & F1 $\uparrow$ & NLL $\downarrow$ & NLL+$T$ $\downarrow$ & F1 $\uparrow$ & NLL $\downarrow$ & NLL+$T$ $\downarrow$ \\']
    labels={'frozen':'Frozen','bias':'Output bias','lora1':'LoRA (1 pass)','lora4':'LoRA (4 passes)','aperture':'Aperture'}
    names={'q25':'Qwen2.5-VL / 7B','q34':'Qwen3-VL / 4B','q38':'Qwen3-VL / 8B','g34':'Gemma 3 / 4B'}
    for m in MODELS:
        lines.append(r'\midrule')
        for i,a in enumerate(('frozen','bias','lora1','lora4','aperture')):
            values=[(r'\textit{n/a}' if a=='bias' and field=='nllt' else format_cell(f'{m}-{d}-{a}-{field}',cells)) for d in DOMAINS for field in ('f1','nll','nllt')]
            lines.append((names[m] if i==0 else '')+' & '+labels[a]+' & '+' & '.join(values)+r' \\')
    return '\n'.join(lines+[r'\bottomrule\end{tabular}'])+'\n'


def export(config):
    root=Path(config['run_root']);out=root/'paper_exports';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'coverage.json',{'paper_ready':False,'planned_states':64,'completed_states':0})
    for p in [out/'main_table.tex',out/'main_table_cells.json',out/'clean_mean_sd.csv']:
        p.unlink(missing_ok=True)
    records=[];resources=[]
    # Do not describe CPU fixture success as real model audit completion.
    for m in MODELS:
        for d in DOMAINS:
            folder=root/'audits'/m/d;verified_record(folder)
            if not json.loads((folder/'audit.json').read_text())['passed']:raise ValueError('failed model audit')
        repeat=json.loads((root/'repeat'/m/'gate.json').read_text())
        if not repeat['passed'] or repeat['source']!=source_signature():raise ValueError('repeat-fit gate missing/stale')
        rep_job=next(j for j in make_jobs(config) if j['model_key']==m and j['domain_key']=='h2' and j['support_seed']==0 and j['arm']=='aperture')
        for p,field in [(Path(rep_job['output'])/'fit','first_seal'),(root/'repeat'/m/'state'/'fit','repeat_seal')]:
            verified_record(p)
            if file_hash(p/'DONE.json')!=repeat[field]:raise ValueError('repeat-fit seal changed')
    for j in make_jobs(config):
        verified_record(Path(j['output'])/'fit');verified_record(Path(j['output'])/'evaluation')
        verify_selection(j,root/'selections'/(j['unit_id']+'.json'))
        roles=read_roles(j)
        folder=Path(j['output'])/'evaluation';r=json.loads((folder/'evaluation.json').read_text())
        if r['source']!=source_signature():raise ValueError('mixed/stale evaluation source')
        pred=read_jsonl(folder/'predictions.jsonl');validate_rows(pred,roles['query'])
        cal=json.loads((Path(j['output'])/'fit'/'calibration.json').read_text())
        scores=[p['mean_log_scores'] for p in pred];y=[p['label_id'] for p in pred]
        versions={'raw':metrics(scores,y),'temperature':metrics(scores,y,cal['temperature']['temperature'])}
        if j['arm']=='frozen':versions['bias']=metrics(scores,y,bias=cal['bias']['values'])
        for variant,values in versions.items():
            for k in ('macro_f1','balanced_accuracy','nll','brier','ece','count'):
                if not np.isclose(values[k],r[variant][k],rtol=1e-10,atol=1e-10):raise ValueError('published metrics do not match saved raw predictions')
        records.append(r)
        resources.append({'job_id':j['job_id'],**json.loads((Path(j['output'])/'fit'/'fitting.json').read_text())})
    summary=summarize_records(records)
    atomic_json(out/'main_table_cells.json',summary)
    atomic_json(out/'full_results.json',records);atomic_json(out/'resources.json',resources)
    (out/'main_table.tex').write_text(table_tex(summary))
    with (out/'clean_mean_sd.csv').open('w') as f:
        fields=sorted({k for r in summary['rows'] for k in r});w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary['rows'])
    atomic_json(out/'coverage.json',{'paper_ready':True,'planned_states':64,'completed_states':64,
        'cells_sha256':file_hash(out/'main_table_cells.json'),'source':source_signature(),
        'exports':{p.name:file_hash(p) for p in out.iterdir() if p.is_file() and p.name!='coverage.json'}})
    return summary


def fill_main(source,destination,exports):
    source=Path(source);destination=Path(destination);exports=Path(exports)
    if source.resolve()==destination.resolve() or destination.exists():raise ValueError('write to a NEW file, never overwrite the manuscript')
    coverage=json.loads((exports/'coverage.json').read_text())
    data_path=exports/'main_table_cells.json'
    if not coverage.get('paper_ready') or coverage.get('completed_states')!=64 or coverage.get('cells_sha256')!=file_hash(data_path):
        raise ValueError('table is incomplete, unverified, or changed')
    summary=json.loads(data_path.read_text());cells=summary['cells']
    if set(cells)!=expected_cells():raise ValueError('wrong table cells')
    # Ignore LaTeX comments, including the illustrative pending{unique-key} comment.
    pattern=re.compile(r'\\pending\{([^{}]+)\}')
    def split_comment(line):
        hit=re.search(r'(?<!\\)%',line)
        return (line[:hit.start()],line[hit.start():]) if hit else (line,'')
    text=source.read_text();pairs=[split_comment(line) for line in text.splitlines(keepends=True)]
    found=[key for active,_ in pairs for key in pattern.findall(active)]
    if Counter(found)!=Counter(expected_cells()):raise ValueError('manuscript must contain each of the 112 expected keys exactly once')
    result=''.join(pattern.sub(lambda m:format_cell(m[1],cells),active)+comment for active,comment in pairs)
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(result)
    atomic_json(destination.with_suffix('.fill_provenance.json'),{'source_sha256':file_hash(source),
        'cells_sha256':file_hash(data_path),'output_sha256':file_hash(destination),'filled_cells':112,
        'note':'Only numeric placeholders changed. Authors must update pending wording and recompile.'})
    return 112
