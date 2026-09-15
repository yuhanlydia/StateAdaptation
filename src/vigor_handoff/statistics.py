"""Descriptive metrics and paired cluster bootstrap; no seed-as-sample inflation."""
from __future__ import annotations
import numpy as np


def macro_f1(y,pred,nclass):
    values=[]
    for c in range(nclass):
        tp=np.sum((y==c)&(pred==c));fp=np.sum((y!=c)&(pred==c));fn=np.sum((y==c)&(pred!=c))
        denom=2*tp+fp+fn;values.append(2*tp/denom if denom else 0.)
    return float(np.mean(values))


def paired_cluster_bootstrap(a,b,nclass,draws=2000,seed=20260914):
    """Delta B-A conditional on one fitted support/controller pair.

    Resamples tile/slide/episode clusters, not individual correlated crops.
    This does NOT estimate across-support-seed uncertainty. Do not pool the
    same query images across seeds as independent observations.
    """
    aa={r['sample_id']:r for r in a};bb={r['sample_id']:r for r in b}
    if len(aa)!=len(a) or len(bb)!=len(b) or set(aa)!=set(bb):raise ValueError('unmatched/duplicate query IDs')
    ids=sorted(aa)
    if any(aa[i]['label_id']!=bb[i]['label_id'] for i in ids):raise ValueError('unmatched labels')
    if any(not aa[i].get('group_id') or aa[i].get('group_id')!=bb[i].get('group_id') for i in ids):
        raise ValueError('missing/unmatched group IDs')
    y=np.array([aa[i]['label_id'] for i in ids])
    pa=np.array([np.argmax(aa[i]['probabilities']) for i in ids]);pb=np.array([np.argmax(bb[i]['probabilities']) for i in ids])
    groups={}
    for j,i in enumerate(ids):groups.setdefault(aa[i]['group_id'],[]).append(j)
    blocks=[np.asarray(v,dtype=int) for v in groups.values()]
    if len(blocks)<2:raise ValueError('at least two independent clusters required')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        idx=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
        values.append(macro_f1(y[idx],pb[idx],nclass)-macro_f1(y[idx],pa[idx],nclass))
    return {'delta':macro_f1(y,pb,nclass)-macro_f1(y,pa,nclass),
            'ci95':np.quantile(values,[.025,.975]).tolist(),'draws':draws,
            'clusters':len(blocks),'query_n':len(ids),'estimand':'paired query-cluster uncertainty conditional on fixed support'}
