#!/usr/bin/env python3
"""Materialize matched BRIGHT support seeds without changing any locked query."""
import argparse,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from visual_lens.controls import balanced_support
from vigor_handoff.protocol import resolved_manifest,validate_split,atomic_json,digest


def save_manifest(path,rows):
    text=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=text:
        raise ValueError(f'{path} exists with different content: use a new manifest root')
    path.write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',default='data/bright.jsonl')
    p.add_argument('--legacy-root',default='data/prepared/neurips')
    p.add_argument('--output',default='data/prepared/visual_lens/bright')
    p.add_argument('--events',nargs='+',default=['hawaii-wildfire','libya-flood','noto-earthquake','turkey-earthquake'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    a=p.parse_args();pool=resolved_manifest(a.pool);summary=[]
    for event in a.events:
        old=Path(a.legacy_root)/event
        q=resolved_manifest(old/'target_query.jsonl')
        original=resolved_manifest(old/'target_support.jsonl')
        ep=[r for r in pool if r['event_id']==event]
        if len(q)!=300:raise ValueError(f'{event}: expected locked query300')
        for seed in a.seeds:
            support=original if seed==0 else balanced_support(ep,q,8,seed)
            check=validate_split(support,q,'tile_id')
            if len(support)!=24 or set(check['support_counts'].values())!={8}:raise ValueError('support24 must be balanced')
            out=Path(a.output)/event/f'seed_{seed}'
            # Rebase paths relative to each new manifest so copied folds stay portable.
            def portable(rows):
                return [{**r,**{k:os.path.relpath(r[k],out.resolve()) for k in ('pre_image','post_image','mask_path') if r.get(k)}} for r in rows]
            save_manifest(out/'support.jsonl',portable(support));save_manifest(out/'query.jsonl',portable(q))
            record={'event':event,'seed':seed,**check,'query_ids_sha256':digest([r['sample_id'] for r in q]),
                    'support_ids_sha256':digest([r['sample_id'] for r in support]),
                    'test_status':'previously inspected query, replication not new blind holdout'}
            atomic_json(out/'manifest_audit.json',record);summary.append(record)
    atomic_json(Path(a.output)/'preparation.json',summary)
    print(f'Prepared {len(summary)} folds; historical manifests unchanged.')
if __name__=='__main__':main()
