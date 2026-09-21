"""Medical endpoints in addition to the existing stable log-score metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from aperture_final.numeric import metrics as base_metrics,probabilities,fit_temperature,choose_lora,degrade


def metrics(scores, labels, temperature=1., *, bias=None):
    result=base_metrics(scores,labels,temperature,bias=bias)
    s=np.asarray(scores,dtype=float);y=np.asarray(labels,dtype=int)
    if bias is not None:s=s+np.asarray(bias)
    p=probabilities(s,temperature);classes=p.shape[1]
    if set(y)!=set(range(classes)):
        result.update(auroc=None,auprc=None)
    elif classes==2:
        result.update(auroc=float(roc_auc_score(y,p[:,1])),auprc=float(average_precision_score(y,p[:,1])))
    else:
        result.update(auroc=float(roc_auc_score(y,p,multi_class='ovr',average='macro')),
            auprc=float(average_precision_score(np.eye(classes)[y],p,average='macro')))
    return result


def replay_difference(a,b,tolerance=1e-3):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid replay scores')
    error=float(np.max(np.abs(a-b)));flips=int((a.argmax(1)!=b.argmax(1)).sum())
    return {'max_abs_score_delta':error,'prediction_disagreements':flips,
            'passed':error<=tolerance and flips==0,'tolerance':tolerance}
