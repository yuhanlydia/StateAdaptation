"""Publish all targeted candidates and actual outcomes, including failures to improve."""
import json
from pathlib import Path
import shutil
import subprocess
from vigor_handoff.protocol import atomic_json,file_hash
from .protocol import read_plan,repeat_job
from .worker import summarize

BRANCH='aperture-targeted-debug-20260923'

def publish(root,repo,previous):
    root=Path(root);repo=Path(repo);summarize(root);plan=read_plan(root)
    subprocess.run(['git','diff','--quiet'],cwd=repo,check=True)
    subprocess.run(['git','diff','--cached','--quiet'],cwd=repo,check=True)
    if subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()!=BRANCH:raise ValueError('wrong publishing branch')
    target=repo/'results/aperture_debug_v2'
    if target.exists():raise ValueError('existing export; inspect and retry git push only if appropriate')
    target.mkdir(parents=True)
    def copy(p,rel):
        dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for name in ('plan.json','CHECKED.json','SELECTION_COMPLETE.json','summary.json','status.json','assignments.jsonl'):
        copy(root/name,name)
    for name in ('selections','repeat_gates','devices','folds'):
        for p in (root/name).rglob('*'):
            if p.is_file():copy(p,p.relative_to(root))
    for job in plan['jobs']:
        for name in ('fitting.json','calibration.json','calibration_predictions.jsonl','basis_spectra.json'):
            p=Path(job['output'])/'fit'/name
            if p.exists():copy(p,Path('search')/job['job_id']/name)
    summary=json.loads((root/'summary.json').read_text())
    for row in summary['records']:
        job=next(j for j in plan['finals'] if j['job_id']==row['job_id'])
        for record in (job,repeat_job(job)):
            for name in ('fitting.json','calibration.json','calibration_predictions.jsonl'):
                copy(Path(record['output'])/'fit'/name,Path('final_fits')/record['job_id']/name)
        for p in (Path(job['output'])/'evaluation').glob('*.json*'):copy(p,Path('evaluations')/job['job_id']/p.name)
    for job in plan['audits']:
        copy(Path(job['context'])/'audits'/job['model_key']/job['domain_key']/'audit.json',Path('audits')/(Path(job['context']).name+'.json'))
    old=json.loads((Path(previous)/'summary.json').read_text()) if previous else {'groups':[]}
    atomic_json(target/'v1_reference_groups.json',{'groups':old['groups'],'use':'historical comparison only, not v2 candidate selection'})
    lines=['# Targeted debug v2','',plan['query_status'],'','H2 uses three patient-separated support folds; Path uses the original 18-image calibration. Final training and temperature calibration reuse the original roles. Loss/rank unchanged. Every attempted candidate is reported. Details: ../../docs/APERTURE_DEBUG_V2.md.','', '| Model | Domain | Aperture F1 % | v2 LoRA F1 % | v1 LoRA F1 % |','|---|---|---:|---:|---:|']
    for t in plan['design']['targets']:
        g={r['method']:r for r in summary['groups'] if (r['model'],r['domain'])==(t['model'],t['domain'])}
        ref=next((r['f1_mean']*100 for r in old['groups'] if (r['model'],r['domain'],r['method'])==(t['model'],t['domain'],'lora')),None)
        reftext=f'{ref:.2f}' if ref is not None else 'unavailable'
        lines.append(f"| {t['model']} | {t['domain']} | {100*g['aperture']['f1_mean']:.2f} | {100*g['lora']['f1_mean']:.2f} | {reftext} |")
    lines+=['','A higher v2 score is exploratory evidence only. Report both LoRA references; do not claim success solely by replacing a stronger old baseline with a weaker selected baseline. Expanded cumulative search and tiny support sets limit conclusions.','']
    (target/'README.md').write_text('\n'.join(lines))
    atomic_json(target/'SHA256SUMS.json',{str(p.relative_to(target)):file_hash(p) for p in sorted(target.rglob('*')) if p.is_file()})
    subprocess.run(['git','add',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','user.name=CHENWENJIE0423','-c','user.email=161920995+CHENWENJIE0423@users.noreply.github.com','commit','-m','Publish targeted support-only debug and tuning outcomes'],cwd=repo,check=True)
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+BRANCH],cwd=repo,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    atomic_json(root/'UPLOAD_STATUS.json',{'status':'uploaded','commit':commit,'branch':BRANCH})
