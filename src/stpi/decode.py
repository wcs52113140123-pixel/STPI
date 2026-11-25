"""Recoverability of positional information, in bits, under a given fold assignment.

The same estimator as the buffered decoding in analysis/fig5_buffered_all.py and the
held-out predictions in analysis/figS5_support.py; the number
of PCA components and the ridge alpha are keyword arguments with the published values
as defaults.

Estimator. For each fold, a PCA (at most n_components) is fitted on the training units'
expression and a ridge regression from the components to the coordinate; the held-out
units are predicted. With the out-of-fold predictions yhat,

    I = 0.5 * log2( Var(pos) / mean((pos - yhat)^2) )

which is the Gaussian-channel information between the coordinate and its prediction. It is
not clipped at zero: a negative I means the prediction is worse than the mean.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


@dataclass
class Recoverability:
    I: float                 # bits
    rho: float               # Spearman correlation between coordinate and prediction
    frac_dropped: float      # fraction of training units removed by the buffer
    prediction: np.ndarray   # out-of-fold prediction for every unit

    def __iter__(self):      # allows  I, rho, frac = recoverability(...)
        return iter((self.I, self.rho, self.frac_dropped))


def information_bits(pos, yhat):
    """0.5 * log2(Var(pos) / MSE); 0 if either is zero."""
    pos = np.asarray(pos, dtype=float)
    resid = pos - yhat
    mse = float(np.mean(resid ** 2))
    var = float(np.var(pos))
    return 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0


def recoverability(X, pos, folds, xy=None, buffer=0.0, n_components=30, alpha=1.0,
                   min_test=5, min_train=20):
    """Out-of-fold decoding of `pos` from expression `X`.

    X        units x features (log-normalised expression, see stpi.features.log_hvg)
    pos      expression-independent coordinate, one value per unit
    folds    fold label per unit (stpi.cv.checkerboard_folds or random_folds)
    xy       unit positions; needed only when buffer > 0
    buffer   training units closer than `buffer` (same units as xy) to ANY unit of the
             test fold are removed from that fold's training set. 0 = no buffer.

    A fold with fewer than `min_test` test units or `min_train` training units is
    predicted by the training mean.
    """
    X = np.asarray(X)
    pos = np.asarray(pos, dtype=float)
    folds = np.asarray(folds)
    if buffer > 0:
        if xy is None:
            raise ValueError("xy is required when buffer > 0")
        xy = np.asarray(xy, dtype=float)
    yhat = np.zeros_like(pos); dropped = 0; n_tr = 0
    for b in np.unique(folds):
        te = folds == b; tr = ~te
        if buffer > 0 and te.sum():
            d, _ = cKDTree(xy[te]).query(xy, k=1)
            tr = tr & (d > buffer)
        dropped += int((~te).sum() - tr.sum()); n_tr += int((~te).sum())
        if te.sum() < min_test or tr.sum() < min_train:
            yhat[te] = pos[tr].mean() if tr.sum() else pos.mean(); continue
        npc = min(n_components, X.shape[1], tr.sum() - 1)
        # random_state pinned: the randomized SVD is otherwise nondeterministic between
        # runs (up to about 0.015 bit on the published sections).
        pc = PCA(n_components=npc, svd_solver="randomized", random_state=0).fit(X[tr])
        yhat[te] = Ridge(alpha=alpha).fit(pc.transform(X[tr]), pos[tr]).predict(pc.transform(X[te]))
    I = information_bits(pos, yhat)
    return Recoverability(I=I, rho=float(spearmanr(pos, yhat)[0]),
                          frac_dropped=dropped / max(n_tr, 1), prediction=yhat)
