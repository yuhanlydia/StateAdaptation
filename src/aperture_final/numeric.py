"""CPU-only scoring, held-out calibration, and deterministic image controls."""
from __future__ import annotations
import hashlib
import io
from typing import Mapping
import numpy as np
from PIL import Image
from scipy.special import logsumexp
from sklearn.metrics import f1_score, confusion_matrix


def _arrays(scores, labels=None):
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or not len(s) or s.shape[1] < 2 or not np.isfinite(s).all():
        raise ValueError('scores must be a nonempty finite N x C array, C>=2')
    if labels is None:
        return s, None
    original = np.asarray(labels)
    if original.shape != (len(s),) or not np.isfinite(original).all():
        raise ValueError('one finite integer label per score row is required')
    y = original.astype(np.int64)
    if not np.array_equal(original, y) or (y < 0).any() or (y >= s.shape[1]).any():
        raise ValueError('label index outside candidate range')
    return s, y


def _log_probabilities(scores, temperature=1.0):
    s, _ = _arrays(scores)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and strictly positive')
    # Subtract a common offset BEFORE scaling: preserves the distribution and
    # prevents a large irrelevant score offset overflowing at small T.
    with np.errstate(over='ignore', invalid='ignore'):
        x = (s - s.max(axis=1,keepdims=True)) / temperature
    if not np.isfinite(x).all():
        raise ValueError('score range exceeds float64; refuse nonfinite probabilities')
    return x - logsumexp(x, axis=1,keepdims=True)


def probabilities(scores, temperature=1.0):
    return np.exp(_log_probabilities(scores, temperature))


def fit_temperature(scores, labels, *, penalty=0.01, grid=None):
    """Only pass held-out SUPPORT scores/labels. Never query data.

    Small-support shrinkage toward T=1 is fixed in advance. Grid and penalty
    are not selected by query performance. A positive T preserves argmax.
    """
    s, y = _arrays(scores, labels)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('temperature penalty must be finite and nonnegative')
    ts = np.unique(np.r_[np.geomspace(.2, 20., 81) if grid is None else grid, 1.])
    if not np.isfinite(ts).all() or (ts <= 0).any():
        raise ValueError('temperature candidates must be positive and finite')
    values = []
    for t in ts:
        lp = _log_probabilities(s,t)
        loss = float(-lp[np.arange(len(y)),y].mean())
        objective = loss + penalty * float(np.log(t)**2)
        values.append((objective, abs(float(np.log(t))), float(t), loss))
    _, _, t, loss = min(values)
    return {'temperature': t, 'calibration_nll': loss,
            'selection_objective': min(values)[0], 'penalty': float(penalty),
            'calibration_count': len(y), 'grid': ts.tolist(),
            'fit_source': 'held_out_labeled_support', 'query_labels_used': False}


def metrics(scores, labels, temperature=1.0, *, bias=None):
    s, y = _arrays(scores, labels)
    if bias is not None:
        b = np.asarray(bias, dtype=np.float64)
        if b.shape != (s.shape[1],) or not np.isfinite(b).all():
            raise ValueError('invalid candidate-bias vector')
        s = s + b
    p = probabilities(s, temperature)
    logp = _log_probabilities(s,temperature)
    pred = s.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(s.shape[1]))
    totals = cm.sum(1)
    recalls = np.divide(np.diag(cm), totals, out=np.zeros(s.shape[1]), where=totals>0)
    confidence = p.max(1); correct = pred == y
    ece = 0.; bins = []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        idx = (confidence > lo) & (confidence <= hi)
        n = int(idx.sum())
        acc = float(correct[idx].mean()) if n else None
        conf = float(confidence[idx].mean()) if n else None
        if n: ece += n / len(y) * abs(acc-conf)
        bins.append({'lower':float(lo), 'upper':float(hi),'count':n,'accuracy':acc,'confidence':conf})
    clipped = np.clip(p, 1e-12, 1.); clipped /= clipped.sum(1,keepdims=True)
    return {'count':len(y), 'macro_f1':float(f1_score(y,pred,average='macro',labels=np.arange(s.shape[1]),zero_division=0)),
            'accuracy':float(correct.mean()), 'balanced_accuracy':float(recalls[totals>0].mean()),
            'nll':float(-logp[np.arange(len(y)),y].mean()),
            'nll_clipped_legacy':float(-np.log(clipped[np.arange(len(y)),y]).mean()),
            'brier':float(np.square(p-np.eye(s.shape[1])[y]).sum(1).mean()),
            'ece':float(ece), 'ece_bins':bins,'confusion_matrix':cm.tolist(),
            'temperature':float(temperature),
            'score_contract':'sum_answer_token_log_probabilities; candidate softmax; stable unclipped NLL'}


def choose_lora(calibrations: Mapping[str, dict]) -> str:
    """Select schedule solely from already sealed calibration NLL; tie -> 1 pass."""
    keys = ('lora1', 'lora4')
    if set(calibrations) != set(keys):
        raise ValueError('both LoRA schedules must have calibration results')
    values = [float(calibrations[k]['calibration_nll']) for k in keys]
    if not np.isfinite(values).all():
        raise ValueError('nonfinite calibration criterion')
    return min(keys, key=lambda k:(values[keys.index(k)], keys.index(k)))


def degrade(images, condition: str, sample_id: str, seed: int=270921):
    """Pixel-space degradation, not a claim of label-preserving nuisance removal.

    For a paired input modify POST only; for one image modify the single image.
    No labels, model outputs or method names enter the random generator.
    """
    if len(images) not in (1, 2):
        raise ValueError('expected a single image or a PRE/POST pair')
    allowed = {'gaussian4':4., 'gaussian8':8., 'jpeg70':70, 'jpeg40':40}
    if condition not in allowed:raise ValueError(f'unknown condition: {condition}')
    out = list(images); image = out[-1].convert('RGB')
    if condition.startswith('gaussian'):
        token = f'{seed}|{sample_id}|{condition}'.encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], 'little'))
        a = np.asarray(image,dtype=np.float32)
        noise = rng.normal(0,allowed[condition],a.shape)
        out[-1] = Image.fromarray(np.clip(np.rint(a+noise),0,255).astype(np.uint8))
    else:
        buf = io.BytesIO(); image.save(buf,format='JPEG',quality=int(allowed[condition]),subsampling=2)
        buf.seek(0)
        with Image.open(buf) as im:out[-1]=im.convert('RGB').copy()
    return tuple(out)
