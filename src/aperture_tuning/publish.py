"""Export a complete, verified tuning result without model/controller weights."""
import json
from pathlib import Path
import shutil
import subprocess
from vigor_handoff.protocol import atomic_json, file_hash
from .protocol import read_plan
from .worker import summarize, repeat_job


def export(root, destination):
    root=Path(root);destination=Path(destination)
    summarize(root)
    plan=read_plan(root);summary=json.loads((root/'summary.json').read_text())
    if destination.exists() and any(destination.iterdir()):raise ValueError('export destination must be new')
    destination.mkdir(parents=True,exist_ok=True)
    def copy(source,rel):
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for name in ('plan.json','CHECKED.json','SELECTION_COMPLETE.json','summary.json','assignments.jsonl','status.json'):
        copy(root/name,name)
    for directory in ('selections','repeat_gates','devices'):
        for source in (root/directory).glob('*.json'):copy(source,source.relative_to(root))
    for job in plan['jobs']:
        folder=Path(job['output'])/'fit'
        for name in ('calibration.json','calibration_predictions.jsonl','fitting.json'):
            copy(folder/name,Path('fits')/job['job_id']/name)
    for record in summary['records']:
        job=next(j for j in plan['jobs'] if j['job_id']==record['job_id'])
        repeat=Path(repeat_job(job)['output'])/'fit'
        for name in ('calibration_predictions.jsonl','fitting.json'):
            copy(repeat/name,Path('repeats')/job['job_id']/name)
        for name in ('predictions.jsonl','evaluation.json'):
            copy(Path(job['output'])/'evaluation'/name,Path('evaluations')/job['job_id']/name)
    for job in plan['audits']:
        source=Path(job['context'])/'audits'/job['model_key']/job['domain_key']/'audit.json'
        copy(source,Path('audits')/(Path(job['context']).name+'.json'))
    lines=['# Support-selected Aperture tuning v1','',
        'Exploratory follow-up to the fixed table at 41738d9. Six predetermined candidates per method/unit; calibration Macro-F1 selects, raw NLL breaks ties. All 192 candidate fits and 32 selected-fit repeats completed before the 32 selected query evaluations. The same previously inspected query sets are reused, so these results are not new blind-test evidence.','',
        '| Model | Dataset | Tuned Aperture F1 (%) | Tuned LoRA F1 (%) | Difference (pp) |',
        '|---|---|---:|---:|---:|']
    wins=0
    for model in ('q25','q34','q38','g34'):
        for domain in ('h2','path'):
            block={g['method']:g for g in summary['groups'] if g['model']==model and g['domain']==domain}
            a,l=(100*block[m]['f1_mean'] for m in ('aperture','lora'))
            wins+=a>l
            lines.append(f'| {model} | {domain} | {a:.2f} | {l:.2f} | {a-l:+.2f} |')
    lines += ['',f'Aperture exceeds tuned LoRA in {wins}/8 combinations by mean Macro-F1. Full per-seed outcomes, SD, candidate IDs and NLL are in summary.json. All candidate validation scores and training records are included, including unfavorable candidates.',
              '', 'H2 uses only four calibration examples per seed (Path uses 18). Selection uncertainty is substantial; candidate count is matched but compute and parameter count are not. No guarantee of reaching an optimal configuration or of beating LoRA is made.',
              '', 'Protocol and audit findings: ../../docs/APERTURE_TUNING_V1.md. Fixed images/manifests: ../../data/aperture_table_v1_release/. No model weights or credentials are included.','']
    (destination/'README.md').write_text('\n'.join(lines))
    atomic_json(destination/'SHA256SUMS.json',{str(p.relative_to(destination)):file_hash(p)
                for p in sorted(destination.rglob('*')) if p.is_file()})
    return destination


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--repository',required=True)
    p.add_argument('--push',action='store_true');a=p.parse_args()
    repo=Path(a.repository).resolve();root=Path(a.root).resolve()
    if a.push:
        subprocess.run(['git','diff','--quiet'],cwd=repo,check=True)
        subprocess.run(['git','diff','--cached','--quiet'],cwd=repo,check=True)
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
        if branch!='aperture-support-tuning-20260922':raise ValueError('unexpected publishing branch')
    target=export(root,repo/'results/aperture_tuning_v1')
    if a.push:
        subprocess.run(['git','add',str(target.relative_to(repo))],cwd=repo,check=True)
        subprocess.run(['git','-c','user.name=CHENWENJIE0423','-c','user.email=161920995+CHENWENJIE0423@users.noreply.github.com',
                        'commit','-m','Publish support-selected Aperture and LoRA tuning results','--quiet'],cwd=repo,check=True)
        subprocess.run(['git','push','origin','HEAD:refs/heads/aperture-support-tuning-20260922'],cwd=repo,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        atomic_json(root/'UPLOAD_STATUS.json',{'status':'uploaded','commit':commit,'branch':'aperture-support-tuning-20260922'})
    print(target)


if __name__=='__main__':main()
