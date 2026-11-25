"""Recoverability across sampling-unit size and sequencing depth.

This scan reproduces the analysis behind Fig 4 and Fig S4 (analysis/fig4_tendon_scan.py).
Units are aggregated into square pseudospots of side `scale`; each pseudospot's counts
are thinned by multinomial sampling to exactly `depth`; the coordinate is decoded under
a 4 x 4 quantile checkerboard, and the same decoder is run with the checkerboard labels
permuted across pseudospots (`n_perm` times), which keeps the number and sizes of the
folds and changes only whether a fold is contiguous.

The random stream is consumed in a fixed order (scale, then depth; thinning, then the
permutations), so a whole section must be scanned in one call to reproduce stored values.

This decoder differs in two details from stpi.decode.recoverability: PCA uses the
default solver, and a fold with fewer than 20 training or 8 test pseudospots is left
unpredicted and excluded from I (rather than predicted by the training mean).
"""
import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def thin_to(Sm, d, rng):
    """Multinomial thinning of every row of a sparse count matrix to total d."""
    Sm = Sm.tocsr(); data = []; ind = []; iptr = [0]
    for i in range(Sm.shape[0]):
        s, e = Sm.indptr[i], Sm.indptr[i + 1]
        dd_ = Sm.data[s:e]; ix = Sm.indices[s:e]; tot = dd_.sum()
        if tot <= 0:
            iptr.append(iptr[-1]); continue
        nc = rng.multinomial(int(d), dd_ / tot); nz = nc > 0
        data.append(nc[nz].astype(np.float32)); ind.append(ix[nz])
        iptr.append(iptr[-1] + int(nz.sum()))
    return sp.csr_matrix((np.concatenate(data), np.concatenate(ind), np.array(iptr)),
                         shape=Sm.shape)


def lognorm(sub):
    """Library-size normalisation to the median library and log1p (all features)."""
    lib = np.asarray(sub.sum(1)).ravel(); lib[lib == 0] = 1
    L = sub.multiply(1.0 / lib[:, None]).multiply(np.median(lib)).tocsr()
    L.data = np.log1p(L.data)
    return np.asarray(L.todense())


def oof(X, y, blk, n_components=30, alpha=1.0, min_train=20, min_test=8):
    """One out-of-fold prediction pass; folds too small to fit are left as NaN."""
    yh = np.full(len(y), np.nan)
    for b in np.unique(blk):
        te = blk == b; tr = ~te
        if tr.sum() < min_train or te.sum() < min_test: continue
        pca = PCA(n_components=min(n_components, X.shape[1], tr.sum() - 1), random_state=0).fit(X[tr])
        m = Ridge(alpha).fit(pca.transform(X[tr]), y[tr])
        yh[te] = m.predict(pca.transform(X[te]))
    return yh


def I_of(y, yh):
    """0.5 * log2(Var / MSE) over the predicted units; NaN if fewer than 20 or degenerate."""
    ok = ~np.isnan(yh)
    if ok.sum() < 20: return np.nan
    mse = np.mean((y[ok] - yh[ok]) ** 2); var = np.var(y[ok])
    return 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else np.nan


def pseudospots(M, xy, coord, scale):
    """Aggregate units into square bins of side `scale` (same units as xy).

    M is units x genes (sparse). Returns (counts pseudospots x genes, cx, cy, coord mean).
    """
    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    cbx = ((xy[:, 0] - x0) / scale).astype(np.int64)
    cby = ((xy[:, 1] - y0) / scale).astype(np.int64)
    uc, inv = np.unique(cbx * 100000 + cby, return_inverse=True)
    P = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                      shape=(len(uc), M.shape[0]))
    agg = (P @ M).tocsr()
    cnt = np.bincount(inv)
    cx = np.bincount(inv, weights=xy[:, 0]) / cnt
    cy = np.bincount(inv, weights=xy[:, 1]) / cnt
    Ragg = np.bincount(inv, weights=coord) / cnt
    return agg, cx, cy, Ragg


def size_depth_scan(M, xy, coord, scales, depths, seed, n_min=45, n_perm=20, name=""):
    """The Fig 4 scan for one section.

    M       units x genes counts (sparse)
    coord   expression-independent coordinate per unit (e.g. rank01 of backbone_offset)
    seed    integer seed of the section's random stream (stpi.seed_of(name))

    Returns a DataFrame with one row per (scale, depth) cell that has at least n_min
    pseudospots at that depth: I_checker, and the median and 2.5-97.5 % range of I under
    permuted checkerboard labels.
    """
    rng = np.random.default_rng(seed)
    Mg = sp.csr_matrix(M)
    xy = np.asarray(xy, dtype=float); coord = np.asarray(coord, dtype=float)
    rows = []
    for sc in scales:
        agg, cx, cy, Ragg = pseudospots(Mg, xy, coord, sc)
        umi = np.asarray(agg.sum(1)).ravel()
        for d in depths:
            elig = umi >= d
            if elig.sum() < n_min: continue
            idx = np.where(elig)[0]
            X = lognorm(thin_to(agg[idx], d, rng)); y = Ragg[idx]
            ab = pd.qcut(cx[idx], 4, labels=False, duplicates="drop")
            pb = pd.qcut(cy[idx], 4, labels=False, duplicates="drop")
            blk = np.asarray((ab + pb) % 2)
            I_chk = I_of(y, oof(X, y, blk))
            perm = [I_of(y, oof(X, y, rng.permutation(blk))) for _ in range(n_perm)]
            perm = np.array([v for v in perm if np.isfinite(v)])
            rows.append(dict(section=name, scale_um=sc, depth=d, n=int(elig.sum()),
                             I_checker=I_chk,
                             I_randlab_med=float(np.median(perm)) if len(perm) else np.nan,
                             I_randlab_lo=float(np.percentile(perm, 2.5)) if len(perm) else np.nan,
                             I_randlab_hi=float(np.percentile(perm, 97.5)) if len(perm) else np.nan,
                             n_perm=len(perm)))
    return pd.DataFrame(rows)
