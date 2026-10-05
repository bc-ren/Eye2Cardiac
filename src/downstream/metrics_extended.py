"""Additional binary metrics used in the paper's frozen-prediction completion.

Core AUROC, average precision and confusion metrics are the unchanged metrics.py
implementation. Thresholds must be frozen before test/external evaluation.
"""
import numpy as np
import metrics as M

NAMES = M.METRICS + ['BalancedAccuracy', 'MCC', 'LogLoss']


def values(y, p, w, threshold, order):
    """Return all 16 metrics in NAMES order, preserving undefined denominators."""
    v = M.binary(y, p, order, w, threshold)
    tp, fp, tn, fn = v[-4:]
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    pp = np.clip(p, 1e-15, 1-1e-15)
    ll = -np.sum(w*(y*np.log(pp)+(1-y)*np.log1p(-pp)))/w.sum()
    return np.r_[v, (v[3]+v[4])/2,
                 (tp*tn-fp*fn)/den if den > 0 else np.nan, ll]
