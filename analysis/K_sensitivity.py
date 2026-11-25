"""Sensitivity of the recoverability estimate to decoder dimensionality K.

Panel e of Fig2 plots I against the dimensionless ratio C/K, where K = 30 PCs is
an analysis choice. This script re-runs the same estimand (geometry-only
radial coordinate, 50 um bins, multinomial thinning to depth d, checkerboard CV,
I = 0.5*log2(Var/MSE)) with K in {10, 20, 30, 50} to show whether
the non-collapse conclusion depends on K.

Scope: 50 um only (the scale panel e uses), all five stages, all six depths.
Output: results/K_sensitivity.csv
  stage, scale_um, depth, K, n, I, R2, rho, feasible
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 fixes the PCA and ridge fits so results are reproducible across runs
import numpy as np, pandas as pd, math
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

base = DATA_
rng = np.random.default_rng(0)

# reuse the stage loaders and radial coordinate defined in the main scan
src = open(f"{HERE_}/multiGW_scan.py").read().split("def decode_I")[0]
exec(src)

UMPP = 0.5
SCALE_UM = 50
DEPTHS = [16, 32, 64, 128, 256, 512]
N_MIN = 100
KS = [10, 20, 30, 50]
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]


def decode_I_K(X, pos, cb, K):
    """Same as decode_I in the main scan, but takes K as a parameter instead of fixing it at 30."""
    yhat = np.zeros_like(pos)
    for b in np.unique(cb):
        te = cb == b
        tr = ~te
        if te.sum() < 5 or tr.sum() < 20:
            yhat[te] = pos[tr].mean()
            continue
        npc = min(K, X.shape[1], tr.sum() - 1)
        pc = PCA(n_components=npc, random_state=0).fit(X[tr])
        m = Ridge(alpha=1.0).fit(pc.transform(X[tr]), pos[tr])
        yhat[te] = m.predict(pc.transform(X[te]))
    resid = pos - yhat
    mse = float(np.mean(resid ** 2))
    var = float(np.var(pos))
    I = 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0
    return I, 1 - mse / var, spearmanr(pos, yhat)[0]


rows = []
for gw in STAGES:
    counts, coords, fib = load_stage(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], [3, 90])
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys = coords[sel, 0], coords[sel, 1]
    Cs = counts[sel].astype(np.float64)
    rs = r[sel]

    step = SCALE_UM / UMPP
    gx = np.floor((xs - xs.min()) / step).astype(int)
    gy = np.floor((ys - ys.min()) / step).astype(int)
    key = gx * 100000 + gy
    uk, inv = np.unique(key, return_inverse=True)
    nps = len(uk)
    G = Cs.shape[1]
    agg = np.zeros((nps, G))
    np.add.at(agg, inv, Cs)
    cnt = np.bincount(inv, minlength=nps).astype(float)
    cnt[cnt == 0] = 1
    Rp = np.bincount(inv, weights=rs, minlength=nps) / cnt
    cx = np.bincount(inv, weights=xs, minlength=nps) / cnt
    cy = np.bincount(inv, weights=ys, minlength=nps) / cnt
    umi = agg.sum(1)

    for d in DEPTHS:
        ok = umi >= d
        if ok.sum() < N_MIN:
            for K in KS:
                rows.append(dict(stage=gw, scale_um=SCALE_UM, depth=d, K=K,
                                 n=int(ok.sum()), I=np.nan, R2=np.nan, rho=np.nan,
                                 feasible=False))
            continue
        A = agg[ok]; Rk = Rp[ok]; cxk = cx[ok]; cyk = cy[ok]
        At = np.zeros_like(A)
        for i in range(A.shape[0]):
            tot = A[i].sum()
            At[i] = A[i] if tot <= d else rng.multinomial(d, A[i] / tot)
        lib = At.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(At / lib * np.median(lib))
        pos = (Rk - Rk.min()) / (np.ptp(Rk) + 1e-9)
        nb = 4
        bx = pd.cut(cxk, nb, labels=False).astype(int)
        by = pd.cut(cyk, nb, labels=False).astype(int)
        cb = (bx + by) % 2
        for K in KS:
            I, r2, rho = decode_I_K(X, pos, cb, K)
            rows.append(dict(stage=gw, scale_um=SCALE_UM, depth=d, K=K,
                             n=int(ok.sum()), I=I, R2=r2, rho=rho, feasible=True))
        print(f"{gw} d={d} n={int(ok.sum())} done", flush=True)

pd.DataFrame(rows).to_csv(f"{RES_}/K_sensitivity.csv", index=False)
print("written results/K_sensitivity.csv")
