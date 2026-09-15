"""Pure controls for P0: no query-label fitting, no image-size changes."""
from __future__ import annotations
from collections import Counter
import hashlib
import numpy as np
from PIL import Image


def image_permutation(sample_ids, seed=0):
    ids=list(sample_ids)
    if len(ids)<2 or len(ids)!=len(set(ids)):
        raise ValueError('shuffle requires at least two unique sample IDs')
    ordered=sorted(ids,key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).digest())
    return dict(zip(ordered,ordered[1:]+ordered[:1]))


def intervene_images(images, condition='real', donor_images=None):
    allowed={'real','shuffle','neutral','shuffle_post','neutral_post'}
    if condition not in allowed:raise ValueError(f'unknown visual condition: {condition}')
    images=tuple(images)
    if condition.endswith('_post') and len(images)!=2:
        raise ValueError('post-only intervention requires two separate images')
    if condition.startswith('shuffle') and (donor_images is None or len(donor_images)!=len(images)):
        raise ValueError('a same-arity donor is required for shuffle')
    result=[]
    for n,image in enumerate(images):
        if condition=='real' or (condition.endswith('_post') and n==0):
            out=image.copy()
        elif condition.startswith('neutral'):
            out=Image.new('RGB',image.size,(128,128,128))
        else:
            out=donor_images[n].convert('RGB').resize(image.size,Image.Resampling.BICUBIC)
        result.append(out)
    return tuple(result)


def select_label_span(input_ids, span, label, tokenizer):
    """Find the exact decoded label suffix inside an already known answer span.

    Works with slow tokenizers and contextual leading-space tokens. Never searches
    the question or constructs a standalone-label tokenization assumption.
    """
    start,end=map(int,span)
    if input_ids.ndim!=2 or input_ids.shape[0]!=1 or not 0<start<end<=input_ids.shape[1]:
        raise ValueError('invalid single-example answer span')
    ids=input_ids[0].tolist()
    for p in range(start,end):
        decoded=tokenizer.decode(ids[p:end],skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
        if decoded.strip()==label.strip():return p,end
    raise ValueError(f'cannot isolate contextual label {label!r} inside answer span {span}')


def balanced_support(pool, query, per_class=8, seed=0):
    """Label-stratified sampling from a pool excluding ALL locked query tiles."""
    if per_class<1:raise ValueError('per_class must be positive')
    if any(not r.get('tile_id') or not r.get('event_id') for r in pool+query):
        raise ValueError('event and tile identity required')
    if len({r['sample_id'] for r in pool})!=len(pool):raise ValueError('duplicate pool sample IDs')
    excluded={(r['event_id'],r['tile_id']) for r in query};qids={r['sample_id'] for r in query}
    result=[]
    for c in sorted({r['label_id'] for r in pool}):
        rows=[r for r in pool if r['label_id']==c and r['sample_id'] not in qids
              and (r['event_id'],r['tile_id']) not in excluded]
        rows.sort(key=lambda r:hashlib.sha256(f"{seed}:{r['event_id']}:{r['sample_id']}".encode()).digest())
        if len(rows)<per_class:raise ValueError(f'class {c}: only {len(rows)} eligible examples, need {per_class}')
        result.extend(rows[:per_class])
    return result


def fit_candidate_bias(scores, labels, l2=.001, max_iter=100):
    """Fit exactly K-1 support-only zero-sum logit-bias parameters on CPU."""
    import torch
    x=torch.as_tensor(np.asarray(scores),dtype=torch.float64)
    y=torch.as_tensor(np.asarray(labels),dtype=torch.long)
    if x.ndim!=2 or len(x)!=len(y) or not len(y) or x.shape[1]<2 or not torch.isfinite(x).all():
        raise ValueError('finite N by K support scores and N labels required')
    if l2<0 or max_iter<1 or int(y.min())<0 or int(y.max())>=x.shape[1]:raise ValueError('invalid bias fit arguments')
    v=torch.zeros(x.shape[1]-1,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([v],lr=1.,max_iter=max_iter,line_search_fn='strong_wolfe')
    calls=0
    def objective():
        nonlocal calls
        opt.zero_grad();b=torch.cat((v,-v.sum().view(1)))
        loss=torch.nn.functional.cross_entropy(x+b,y)+l2*b.square().sum()
        loss.backward();calls+=1;return loss
    opt.step(objective)
    bias=torch.cat((v,-v.sum().view(1))).detach().numpy()
    return bias,{'trainable_scalars':x.shape[1]-1,'support_examples':len(y),
                 'closure_evaluations':calls,'l2':l2,'optimizer':'LBFGS',
                 'supervision':'support labels only','sum_bias':float(bias.sum())}


def biased_probabilities(scores,bias):
    s=np.asarray(scores,dtype=float)+np.asarray(bias,dtype=float)
    p=np.exp(s-s.max());return p/p.sum()


def metric_vector(rows, indices=None):
    y=np.asarray([r['label_id'] for r in rows]);p=np.asarray([r['probabilities'] for r in rows],float)
    if indices is not None:y,p=y[indices],p[indices]
    nclass=p.shape[1];pred=p.argmax(1)
    cm=np.bincount(nclass*y+pred,minlength=nclass*nclass).reshape(nclass,nclass)
    tp=cm.diagonal();den=cm.sum(0)+cm.sum(1);counts=cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(nclass,float),where=den!=0).mean()
    ba=np.divide(tp,counts,out=np.zeros(nclass,float),where=counts!=0).mean()
    nll=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()
    return np.asarray([f1,ba,nll])


def paired_did(frozen_real, method_real, frozen_control, method_control, draws=2000,seed=0):
    """Paired cluster bootstrap of (M-F)_real - (M-F)_control.

    Same sampled query clusters are used in every condition and method. NLL uses
    the same signed formula as F1; lower NLL is better, so its sign is reversed
    in interpretation (not silently flipped in the output).
    """
    blocks=[frozen_real,method_real,frozen_control,method_control]
    if not frozen_real or draws<0:raise ValueError('nonempty predictions and nonnegative draws required')
    ids=sorted(r['sample_id'] for r in frozen_real)
    if len(ids)!=len(set(ids)):raise ValueError('duplicate query IDs')
    aligned=[]
    for block in blocks:
        index={r['sample_id']:r for r in block}
        if len(index)!=len(block) or set(index)!=set(ids):raise ValueError('query ID sets differ')
        aligned.append([index[s] for s in ids])
    for i in range(len(ids)):
        if len({b[i]['label_id'] for b in aligned})!=1:raise ValueError('query labels differ')
        if len({b[i].get('group_id') for b in aligned})!=1 or not aligned[0][i].get('group_id'):
            raise ValueError('query cluster identity missing/different')
    groups=sorted({r['group_id'] for r in aligned[0]})
    indices=[np.array([i for i,r in enumerate(aligned[0]) if r['group_id']==g]) for g in groups]
    def effect(ix=None):
        vals=[metric_vector(b,ix) for b in aligned]
        return vals[1]-vals[0], vals[3]-vals[2]
    gr,gc=effect();base=gr-gc;rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        ix=np.concatenate([indices[j] for j in rng.integers(len(groups),size=len(groups))])
        a,b=effect(ix);samples.append(a-b)
    result={'n':len(ids),'clusters':len(groups),'draws':draws,
            'interval_type':'paired query-cluster percentile, conditional on fitted support state'}
    for k,name in enumerate(('macro_f1','balanced_accuracy','nll')):
        result[name]={'gain_real':float(gr[k]),'gain_control':float(gc[k]),
                      'difference_in_differences':float(base[k]),
                      'ci95':np.quantile(np.asarray(samples)[:,k],[.025,.975]).tolist() if samples and len(groups)>1 else None}
    return result
