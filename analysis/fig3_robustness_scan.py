"""Fig 3 robustness re-runs: does the five-stage result depend on (a) how the coordinate
is constructed, or (b) which cross-validation scheme is used?

Both axes reuse the five-stage pipeline of analysis/multiGW_scan.py verbatim -
same loaders, same QC crop (R percentile 3-90 on fibroblasts), same pseudospot binning,
same multinomial depth thinning, same log-normalisation, same ridge-on-30-PCs decoder,
same I = 0.5*log2(Var/MSE). Exactly one thing changes per variant.

(a) COORDINATE. Mainline R is a centreline-normal offset: fit a smooth centreline by
    binned medians in polar angle, then project onto the local normal. The alternative
    is a graph construction with no centreline at all: build a kNN graph over the
    fibroblast cloud, take the normalised-Laplacian eigenvectors, and keep the one that
    runs ACROSS the band. Selection is geometric, never by agreement with the mainline
    coordinate: for each candidate eigenvector |corr| is measured against the projection
    onto the LOCAL minor axis (local PCA in a k-neighbourhood) and take the largest.
    Zero expression input either way.

(b) CROSS-VALIDATION. Mainline is a parity checkerboard over a 4x4 spatial grid with the
    two parities as the two folds, trained on the complement. The alternative keeps the
    same folds but deletes from the TRAINING set every pseudospot within BUFFER_MULT
    pseudospot widths of any held-out pseudospot, so neighbouring units cannot leak
    across the fold boundary. A buffered fold that loses too much training data is
    skipped rather than silently falling back.

Outputs
  results/fig3_altcoord_5stage.csv   stage, scale_um, depth, n, I, R2, rho, coord, feasible
  results/fig3_bufferedcv_5stage.csv stage, scale_um, depth, n, I, R2, rho, cv, buffer_um,
                                     frac_train_dropped, feasible
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 pinned so PCA and ridge fits are reproducible across runs
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, pandas as pd, math, importlib.util, sys
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

spec = importlib.util.spec_from_file_location("ms", f"{HERE_}/multiGW_scan.py")
# multiGW_scan.py runs its scan at import time; load only its function definitions
src = open(f"{HERE_}/multiGW_scan.py").read()
src = src[:src.index("UMPP=0.5")]
ms = {}
exec(compile(src, "multiGW_scan_defs", "exec"), ms)
radial_coord_full, load_stage = ms["radial_coord_full"], ms["load_stage"]

UMPP = 0.5
SCALES_UM = [50, 100, 200, 400, 800]
DEPTHS = [16, 32, 64, 128, 256, 512]
N_MIN = 100
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]
KNN = 12
BUFFER_MULT = 1.0          # buffer = 1 pseudospot width
rng = np.random.default_rng(0)


def laplacian_across_band(x, y, knn=KNN, ncomp=6):
    """Across-band coordinate from a kNN graph Laplacian. No centreline, no expression."""
    P = np.c_[x, y].astype(float)
    n = len(P)
    tree = cKDTree(P)
    d, idx = tree.query(P, k=min(knn + 1, n))
    sig = np.median(d[:, 1:]) + 1e-9
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    cols = idx[:, 1:].ravel()
    w = np.exp(-(d[:, 1:].ravel() / sig) ** 2)
    W = csr_matrix((w, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    deg = np.asarray(W.sum(1)).ravel() + 1e-12
    Dm = diags(deg ** -0.5)
    L = diags(np.ones(n)) - Dm @ W @ Dm
    k = min(ncomp, n - 2)
    vals, vecs = eigsh(L, k=k, sigma=-1e-5, which="LM")
    vecs = vecs[:, np.argsort(vals)][:, 1:]              # drop the trivial constant

    # local minor axis: local PCA in each point's neighbourhood, projection onto it
    _, nb = tree.query(P, k=min(knn + 1, n))
    minor = np.zeros(n)
    for i in range(n):
        Q = P[nb[i]] - P[nb[i]].mean(0)
        u, s, vt = np.linalg.svd(Q, full_matrices=False)
        minor[i] = (P[i] - P[nb[i]].mean(0)) @ vt[1]     # offset along the local minor axis
    score = [abs(np.corrcoef(vecs[:, j], minor)[0, 1]) for j in range(vecs.shape[1])]
    j = int(np.nanargmax(score))
    return vecs[:, j], float(score[j])


def decode(X, pos, folds, pts=None, buffer_um=None):
    """Ridge on 30 PCs, out-of-fold prediction. With buffer_um, training units within
    that distance of any held-out unit are dropped."""
    yhat = np.zeros_like(pos)
    dropped = []
    for b in np.unique(folds):
        te = folds == b
        tr = ~te
        if buffer_um is not None and te.sum() > 0:
            dmin, _ = cKDTree(pts[te]).query(pts)
            keep = tr & (dmin > buffer_um / UMPP)
            dropped.append(1.0 - keep.sum() / max(tr.sum(), 1))
            if keep.sum() < 20:
                return np.nan, np.nan, np.nan, np.nan   # buffered fold unusable
            tr = keep
        if te.sum() < 5 or tr.sum() < 20:
            yhat[te] = pos[tr].mean() if tr.sum() else pos.mean()
            continue
        npc = min(30, X.shape[1], tr.sum() - 1)
        pc = PCA(n_components=npc, random_state=0).fit(X[tr])
        m = Ridge(alpha=1.0).fit(pc.transform(X[tr]), pos[tr])
        yhat[te] = m.predict(pc.transform(X[te]))
    mse = np.mean((pos - yhat) ** 2); var = np.var(pos)
    I = 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0
    return I, 1 - mse / var, spearmanr(pos, yhat)[0], (float(np.mean(dropped)) if dropped else 0.0)


rows_c, rows_b = [], []
for gw in STAGES:
    counts, coords, fib = load_stage(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], [3, 90])
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys = coords[sel, 0], coords[sel, 1]
    Cs = counts[sel]; rs = r[sel]
    r_alt, sc_sel = laplacian_across_band(xs, ys)
    agree = abs(spearmanr(rs, r_alt)[0])
    print(f"{gw}: n={sel.sum()} altcoord|corr with local minor axis|={sc_sel:.2f} "
          f"|rho(alt, mainline)|={agree:.3f}", flush=True)

    for s_um in SCALES_UM:
        step = s_um / UMPP
        gx = np.floor((xs - xs.min()) / step).astype(int)
        gy = np.floor((ys - ys.min()) / step).astype(int)
        uk, inv = np.unique(gx * 100000 + gy, return_inverse=True)
        nps = len(uk)
        agg = np.zeros((nps, Cs.shape[1])); Rp = np.zeros(nps); Ap = np.zeros(nps)
        cx = np.zeros(nps); cy = np.zeros(nps)
        for i in range(nps):
            mm = inv == i
            agg[i] = Cs[mm].sum(0); Rp[i] = rs[mm].mean(); Ap[i] = r_alt[mm].mean()
            cx[i] = xs[mm].mean(); cy[i] = ys[mm].mean()
        umi = agg.sum(1)
        for d in DEPTHS:
            ok = umi >= d
            base = dict(stage=gw, scale_um=s_um, depth=d, n=int(ok.sum()))
            if ok.sum() < N_MIN:
                rows_c.append({**base, "I": np.nan, "R2": np.nan, "rho": np.nan,
                               "coord": "laplacian", "feasible": False})
                rows_b.append({**base, "I": np.nan, "R2": np.nan, "rho": np.nan,
                               "cv": "buffered", "buffer_um": BUFFER_MULT * s_um,
                               "frac_train_dropped": np.nan, "feasible": False})
                continue
            A = agg[ok]; cxk = cx[ok]; cyk = cy[ok]
            At = np.zeros_like(A)
            for i in range(A.shape[0]):
                tot = A[i].sum()
                At[i] = A[i] if tot <= d else rng.multinomial(d, A[i] / tot)
            lib = At.sum(1, keepdims=True); lib[lib == 0] = 1
            X = np.log1p(At / lib * np.median(lib))
            nb = 4
            bx = pd.cut(cxk, nb, labels=False).astype(int)
            by = pd.cut(cyk, nb, labels=False).astype(int)
            cb = (bx + by) % 2
            pts = np.c_[cxk, cyk]

            posA = (Ap[ok] - Ap[ok].min()) / (np.ptp(Ap[ok]) + 1e-9)
            I, r2, rho, _ = decode(X, posA, cb)
            rows_c.append({**base, "I": round(I, 4), "R2": round(r2, 4),
                           "rho": round(rho, 4), "coord": "laplacian", "feasible": True})

            posM = (Rp[ok] - Rp[ok].min()) / (np.ptp(Rp[ok]) + 1e-9)
            I, r2, rho, fd = decode(X, posM, cb, pts=pts, buffer_um=BUFFER_MULT * s_um)
            rows_b.append({**base, "I": None if np.isnan(I) else round(I, 4),
                           "R2": None if np.isnan(r2) else round(r2, 4),
                           "rho": None if np.isnan(rho) else round(rho, 4),
                           "cv": "buffered", "buffer_um": BUFFER_MULT * s_um,
                           "frac_train_dropped": round(fd, 3),
                           "feasible": bool(not np.isnan(I))})
    print(f"  {gw} done", flush=True)

pd.DataFrame(rows_c).to_csv(f"{RES_}/fig3_altcoord_5stage.csv", index=False)
pd.DataFrame(rows_b).to_csv(f"{RES_}/fig3_bufferedcv_5stage.csv", index=False)
print("WROTE fig3_altcoord_5stage.csv / fig3_bufferedcv_5stage.csv")
