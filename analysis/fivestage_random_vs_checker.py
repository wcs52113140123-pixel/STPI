"""Writes results/fivestage_random_vs_checker_C.csv.

Computes positional information from spatial transcriptomics counts across five
developmental stages, comparing random train/test folds to spatial checkerboard
folds to estimate the contribution of spatial autocorrelation to the decoding
score.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 fixes the seed for PCA and the multinomial downsampling draws so results are reproducible
import numpy as np
import pandas as pd
import scipy.io as sio
import math
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

rng = np.random.default_rng(0)

order = ["Fibroblast_Pia", "Fibroblast_Arachnoid", "Fibroblast_Dura"]
UMPP = 0.5

def radial_coord_full(x, y, nbin=60, ksm=5):
    cx, cy = np.median(x), np.percentile(y, 15)
    theta = np.arctan2(y - cy, x - cx)
    tb = pd.cut(theta, nbin, labels=False)
    cl = []
    for b in range(nbin):
        m = tb == b
        if m.sum() < 15:
            continue
        cl.append((np.median(theta[m]), np.median(x[m]), np.median(y[m])))
    cl = np.array(cl)
    cl = cl[np.argsort(cl[:, 0])]
    sm = lambda a: np.convolve(a, np.ones(ksm) / ksm, mode="same")
    clx, cly = sm(cl[:, 1]), sm(cl[:, 2])
    tang = np.gradient(np.column_stack([clx, cly]), axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    ro = np.column_stack([clx - cx, cly - cy])
    ro /= np.linalg.norm(ro, axis=1, keepdims=True) + 1e-9
    fl = np.sign((nrm * ro).sum(1))
    fl[fl == 0] = 1
    nrm *= fl[:, None]
    tree = cKDTree(np.column_stack([clx, cly]))
    _, idx = tree.query(np.column_stack([x, y]))
    r = ((np.column_stack([x - clx[idx], y - cly[idx]])) * nrm[idx]).sum(1)
    return r

def load_stage(gw):
    if gw == "GW16":
        counts = np.load(f"{DATA_}/gaston_gw16/counts_mat.npy")
        coords = np.load(f"{DATA_}/gaston_gw16/coords_mat.npy")
        clusters = np.load(f"{DATA_}/gaston_gw16/clusters_kept.npy", allow_pickle=True)
        keep = counts.sum(1) > 0
        counts, coords, clusters = counts[keep], coords[keep], clusters[keep]
        fib = np.isin(clusters, order)
    elif gw == "GW6":
        m = pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv")
        c = sio.mmread(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_counts.mtx")
        counts = np.asarray(c.todense()).T
        coords = m[["x", "y"]].values.astype(float)
        ct = m["celltype"].values
        fib = np.isin(ct, ["Fibro-pia-pro", "Fibro-others"])
    elif gw == "GW8":
        counts = np.asarray(sio.mmread(f"{DATA_}/gaston_multi/GW8/counts.mtx").T.todense(), dtype=np.float64)
        coords = np.loadtxt(f"{DATA_}/gaston_multi/GW8/coords.tsv")
        clusters = np.array([l.strip() for l in open(f"{DATA_}/gaston_multi/GW8/clusters.txt")])
        fib = np.isin(clusters, order)
    elif gw == "GW10":
        counts = np.asarray(sio.mmread(f"{DATA_}/gaston_multi/GW10/counts.mtx").T.todense(), dtype=np.float64)
        coords = np.loadtxt(f"{DATA_}/gaston_multi/GW10/coords.tsv")
        clusters = np.array([l.strip() for l in open(f"{DATA_}/gaston_multi/GW10/clusters.txt")])
        fib = np.isin(clusters, order)
    elif gw == "GW12":
        counts = np.asarray(sio.mmread(f"{DATA_}/gaston_multi/GW12/counts.mtx").T.todense(), dtype=np.float64)
        coords = np.loadtxt(f"{DATA_}/gaston_multi/GW12/coords.tsv")
        clusters = np.array([l.strip() for l in open(f"{DATA_}/gaston_multi/GW12/clusters.txt")])
        fib = np.isin(clusters, order)
    return counts, coords, fib

def decode_folds(X, pos, folds):
    yhat = np.zeros_like(pos)
    for b in np.unique(folds):
        te = folds == b
        tr = ~te
        if te.sum() < 5 or tr.sum() < 20:
            yhat[te] = pos[tr].mean()
            continue
        npc = min(30, X.shape[1], tr.sum() - 1)
        pc = PCA(n_components=npc, random_state=0).fit(X[tr])
        Xt = pc.transform(X[tr])
        Xv = pc.transform(X[te])
        yhat[te] = Ridge(alpha=1.0).fit(Xt, pos[tr]).predict(Xv)
    resid = pos - yhat
    mse = np.mean(resid ** 2)
    var = np.var(pos)
    I = 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0
    return I

CFG = {"GW6": (50, 128), "GW8": (50, 256), "GW10": (100, 512), "GW12": (200, 256), "GW16": (200, 256)}
QC = (3, 90)

res = []
for gw, (s_um, d) in CFG.items():
    counts, coords, fib = load_stage(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], list(QC))
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys, rs, Cs = coords[sel, 0], coords[sel, 1], r[sel], counts[sel]
    step = s_um / UMPP
    gx = np.floor((xs - xs.min()) / step).astype(int)
    gy = np.floor((ys - ys.min()) / step).astype(int)
    key = gx * 100000 + gy
    uk, inv = np.unique(key, return_inverse=True)
    nps = len(uk)
    agg = np.zeros((nps, Cs.shape[1]))
    Rp = np.zeros(nps)
    cx = np.zeros(nps)
    cy = np.zeros(nps)
    for i in range(nps):
        m = inv == i
        agg[i] = Cs[m].sum(0)
        Rp[i] = rs[m].mean()
        cx[i] = xs[m].mean()
        cy[i] = ys[m].mean()
    umi = agg.sum(1)
    ok = umi >= d
    A = agg[ok]
    Rk = Rp[ok]
    cxk = cx[ok]
    cyk = cy[ok]
    At = np.zeros_like(A)
    for i in range(A.shape[0]):
        tot = A[i].sum()
        At[i] = A[i] if tot <= d else rng.multinomial(d, A[i] / tot)
    lib = At.sum(1, keepdims=True)
    lib[lib == 0] = 1
    X = np.log1p(At / lib * np.median(lib))
    pos = (Rk - Rk.min()) / (np.ptp(Rk) + 1e-9)
    bx = pd.cut(cxk, 4, labels=False).astype(int)
    by = pd.cut(cyk, 4, labels=False).astype(int)
    cb = (bx + by) % 2
    rf = rng.integers(0, 2, size=len(pos))
    Ir = decode_folds(X, pos, rf)
    Ic = decode_folds(X, pos, cb)
    frac = (Ir - Ic) / Ir if Ir > 0 else np.nan
    res.append(dict(stage=gw, scale_um=s_um, depth=d, n=int(ok.sum()),
                    I_random=round(Ir, 4), I_checker=round(Ic, 4),
                    frac_artifact=round(frac, 3)))

dfc = pd.DataFrame(res)
dfc.to_csv(f'{RES_}/fivestage_random_vs_checker_C.csv', index=False)
