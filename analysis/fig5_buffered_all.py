#!/usr/bin/env python3
"""Fig 5d: buffered spatial cross-validation sweep across all 18 samples in the field-wide panel.

Reproduces the decoding settings used in analysis/fig5_audit.py:
  coordinate   geometric tissue interiority, geodesic distance from the boundary
               (geom_interiority; boundary = lowest-degree 15% of spots). No expression.
  features     library-size normalised, log1p, top 2000 HVG by variance
  decoder      PCA <= 30 components, Ridge alpha = 1
  folds        4x4 checkerboard parity, the same two spatial folds as in fig5_audit.py
  metric       I = 0.5*log2(Var/MSE), unclipped; rho = Spearman(true, predicted)

Adds a buffer: a training spot is dropped when it lies within k * (median nearest-neighbour
spacing) of ANY test spot. k = 0 reproduces the plain checkerboard arm, so the 0 row serves
as a consistency check against fieldwide_audit.csv.

The buffer is expressed in spot pitches rather than as a fraction of the section span. Under
the checkerboard scheme, a band of 15% of the section span around every test spot removes
94.6-100% of the training spots across all 18 samples (50.6% already at 5%), leaving too little
training data for a meaningful buffered estimate at that resolution. Spot pitches keep the
deletion interpretable and comparable across tissues, since all datasets are Visium. The
span-fraction run is kept separately as results/fig5_buffered_cv_span.csv.

Seeding is deterministic: SHA-256 of the sample name for the per-sample RNG, and PCA's
randomized SVD is pinned with random_state = 0 so that repeated runs give identical results.
fig5_audit.py leaves the PCA unpinned, so its I values vary by about 0.015 bit between
runs, comparable to the gap between the buffer = 0 column here and I_checker in
fieldwide_audit.csv.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os, gzip, glob, hashlib, math, json
import numpy as np, pandas as pd
import scipy.io as sio
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

BASE = DATA_
DIRS = {"kidney": f"{DATA_}/kidney_GSE273978",
        "intestine": f"{DATA_}/intestine_GSE245274",
        "cerebellum": f"{DATA_}/cerebellum_GSE296375",
        "retina": f"{DATA_}/retina_GSE248517",
        "finch_retina": f"{DATA_}/retina_GSE309408_finch"}
BUFFERS_PITCH = [0, 1, 2, 4]      # buffer width in median nearest-neighbour spacings


def _seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ───────────────────────────── loaders, one per packaging ─────────────────────────────
def _positions(path):
    """tissue_positions_list.csv.gz - spaceranger v1, headerless."""
    p = pd.read_csv(path, header=None,
                    names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"])
    p = p[p.in_tissue == 1]
    return p.set_index("barcode")[["pxl_col", "pxl_row"]].astype(float)


def load_mtx(d, prefix):
    M = sio.mmread(glob.glob(f"{d}/{prefix}*matrix.mtx.gz")[0]).tocsr()      # genes x spots
    bc = [l.decode().split("\t")[0].strip()
          for l in gzip.open(glob.glob(f"{d}/{prefix}*barcodes.tsv.gz")[0])]
    P = _positions(glob.glob(f"{d}/{prefix}*tissue_positions_list.csv.gz")[0])
    keep = [i for i, b in enumerate(bc) if b in P.index]
    xy = P.loc[[bc[i] for i in keep]].values
    return M[:, keep].T.tocsr(), xy


def load_h5(d, prefix):
    import h5py
    f = glob.glob(f"{d}/{prefix}*matrix.h5")[0]
    with h5py.File(f, "r") as h:
        g = h["matrix"]
        M = csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                       shape=g["shape"][:][::-1])                      # spots x genes
        bc = [b.decode() for b in g["barcodes"][:]]
    P = _positions(glob.glob(f"{d}/{prefix}*tissue_positions_list.csv.gz")[0])
    keep = [i for i, b in enumerate(bc) if b in P.index]
    return M[keep], P.loc[[bc[i] for i in keep]].values


def load_finch(d, sub):
    base = f"{d}/{sub}/filtered_feature_bc_matrix"
    M = sio.mmread(f"{base}/matrix.mtx.gz").tocsr()
    bc = [l.decode().strip() for l in gzip.open(f"{base}/barcodes.tsv.gz")]
    P = pd.read_csv(f"{d}/{sub}/spatial/tissue_positions.csv")
    P.columns = [c.strip() for c in P.columns]
    P = P.set_index(P.columns[0])
    xc = [c for c in P.columns if "pxl_col" in c][0]
    yc = [c for c in P.columns if "pxl_row" in c][0]
    keep = [i for i, b in enumerate(bc) if b in P.index]
    return M[:, keep].T.tocsr(), P.loc[[bc[i] for i in keep], [xc, yc]].values.astype(float)


# ───────────────────── coordinate / features / decode, fixed settings ─────────────────────
def geom_interiority(xy):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    n = len(xy); tree = cKDTree(xy)
    r0 = 1.8 * np.median(tree.query(xy, k=2)[0][:, 1])
    deg = np.array([len(p) - 1 for p in tree.query_ball_point(xy, r0)])
    # Guards against a degenerate section: on a lattice whose degree distribution is nearly
    # constant, the 15th percentile coincides with the modal degree, so `deg <= thr` flags
    # every spot as boundary and the coordinate collapses to a constant. This occurs on two
    # cerebellum sections (GSM8969380, GSM8969382: 100% flagged, one unique value), which
    # would otherwise read exactly zero under both CV schemes even though the coordinate
    # carries signal. The fallback below leaves the other 16 sections unchanged (15.0-21.4%
    # flagged there); see results/fig5_boundary_diagnostic.csv for diagnostics.
    boundary = deg <= np.percentile(deg, 15)
    if boundary.mean() > 0.40:
        boundary = deg < np.median(deg)
    d, idx = tree.query(xy, k=9)
    rows = np.repeat(np.arange(n), 8)
    G = coo_matrix((d[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n))
    G = G.maximum(G.T)
    gd = dijkstra(G, directed=False, indices=np.where(boundary)[0], min_only=True)
    gd[~np.isfinite(gd)] = np.nanmax(gd[np.isfinite(gd)])
    return (gd - gd.min()) / (np.ptp(gd) + 1e-9)


def features(M, hvg=2000):
    Xc = np.asarray(M.todense(), dtype=np.float64)
    lib = Xc.sum(1, keepdims=True); lib[lib == 0] = 1
    logn = np.log1p(Xc / lib * np.median(lib))
    return logn[:, np.argsort(logn.var(0))[-hvg:]]


def decode_buffered(X, pos, folds, xy, rad):
    """Checkerboard CV with a buffer of radius `rad` (data units): training spots within
    `rad` of any test spot are dropped. rad = 0 reproduces the plain checkerboard scheme used
    in analysis/fig5_audit.py."""
    yhat = np.zeros_like(pos); dropped = 0; n_tr = 0
    for b in np.unique(folds):
        te = folds == b; tr = ~te
        if rad > 0 and te.sum():
            d, _ = cKDTree(xy[te]).query(xy, k=1)
            tr = tr & (d > rad)
        dropped += int((~te).sum() - tr.sum()); n_tr += int((~te).sum())
        if te.sum() < 5 or tr.sum() < 20:
            yhat[te] = pos[tr].mean() if tr.sum() else pos.mean(); continue
        npc = min(30, X.shape[1], tr.sum() - 1)
        # random_state is pinned because sklearn's randomized SVD is otherwise
        # nondeterministic: without pinning, repeated runs of this script can differ by
        # up to 0.013 bit in the resulting I value.
        pc = PCA(n_components=npc, svd_solver="randomized", random_state=0).fit(X[tr])
        yhat[te] = Ridge(alpha=1.0).fit(pc.transform(X[tr]), pos[tr]).predict(pc.transform(X[te]))
    resid = pos - yhat; mse = float(np.mean(resid ** 2)); var = float(np.var(pos))
    I = 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0
    return I, float(spearmanr(pos, yhat)[0]), dropped / max(n_tr, 1)


# ───────────────────────────────── manifest and run ─────────────────────────────────
def manifest():
    A = pd.read_csv(f"{RES_}/fieldwide_audit.csv")
    out = []
    for _, r in A.iterrows():
        t, s = r.tissue, r["sample"]
        if t == "finch_retina":
            sub = "25" if s.endswith("rep1") else "26"
            out.append((t, s, "finch", sub))
        elif t == "cerebellum":
            out.append((t, s, "h5", s))
        else:
            out.append((t, s, "mtx", s))
    return out


def main():
    rows = []
    for tissue, sample, kind, key in manifest():
        d = DIRS[tissue]
        try:
            M, xy = (load_finch(d, key) if kind == "finch" else
                     load_h5(d, key) if kind == "h5" else load_mtx(d, key))
        except Exception as e:
            print(f"SKIP {sample}: {type(e).__name__} {e}", flush=True); continue
        rng = np.random.default_rng(_seed(sample))
        R = geom_interiority(xy); X = features(M)
        bx = pd.cut(xy[:, 0], 4, labels=False).astype(int)
        by = pd.cut(xy[:, 1], 4, labels=False).astype(int)
        folds = (bx + by) % 2
        pitch = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
        span = float(np.mean([np.ptp(xy[:, 0]), np.ptp(xy[:, 1])]))
        for k in BUFFERS_PITCH:
            I, rho, frac_drop = decode_buffered(X, R, folds, xy, k * pitch)
            rows.append(dict(tissue=tissue, sample=sample, n=len(R), buffer_pitch=k,
                             buffer_units=round(k * pitch, 1), pitch=round(pitch, 1),
                             buffer_frac_of_span=round(k * pitch / span, 4),
                             I_bit=round(I, 4), rho=round(rho, 4),
                             frac_train_dropped=round(frac_drop, 4)))
            print(f"{sample:38s} buf={k} pitch  I={I:+.4f}  rho={rho:+.4f}  "
                  f"dropped={frac_drop:.1%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES_}/fig5_buffered_cv_all.csv", index=False)
    print("WROTE fig5_buffered_cv_all.csv", df.shape)


if __name__ == "__main__":
    main()
