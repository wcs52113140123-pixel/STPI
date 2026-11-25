"""Fig 3a: three across-band coordinates, judged against an external criterion.

The three constructions are deliberately from different families:

  polar      mainline (multiGW_scan.radial_coord_full). Centreline from binned medians
             in polar angle about a hand-placed origin, then local-normal offset.
  curve      MST principal curve. K-means knots, minimum spanning tree, longest path as
             backbone, arclength smoothing, signed perpendicular offset. Same family as
             polar (centreline-normal) but without its three arbitrary choices: no polar
             origin, no angular binning, no fixed smoothing width.
  harmonic   Dirichlet solve, no centreline at all. Boundary units are found from kNN
             one-sidedness; the boundary loop is cut at its two graph-diameter endpoints
             (the short ends of the band) into two arcs; u = 0 on one arc, u = 1 on the
             other, harmonic in between on the kNN graph Laplacian.

None uses expression. None uses the layer labels.

ARBITRATION. The layer labels (Fibroblast_Pia / _Arachnoid / _Dura) are an external
criterion: they never enter any coordinate's construction, and they are independent of
the decoding outcome I, so ranking coordinates by |rho(coordinate, layer order)| is not
selection on the result. This matters because the curve coordinate raises I for the
curved stages relative to the polar coordinate (GW10: 0.048 -> 0.549); whether the curve
coordinate tracks the known layer order better, or merely predicts something easier,
decides how to read that difference.

Also reported per stage and coordinate: the I(d, s) grid, so the response surface can be
compared, not just the coordinate itself.

Output
  results/fig3_coord_arbitration.csv   per stage x coordinate: rho_layer, n_layer, and
                                       grid summaries (I_max, argmax scale/depth)
  results/fig3_coord_grids.csv         full per-cell grid for all three coordinates
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 pinned throughout for reproducible results
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, pandas as pd, math
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, connected_components
from scipy.sparse.linalg import spsolve
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import scipy.io as sio

BASE = DATA_
src = open(f"{HERE_}/multiGW_scan.py").read()
ms = {}
exec(compile(src[:src.index("UMPP=0.5")], "defs", "exec"), ms)
radial_coord_full, load_stage = ms["radial_coord_full"], ms["load_stage"]

UMPP = 0.5
SCALES_UM = [50, 100, 200, 400, 800]
DEPTHS = [16, 32, 64, 128, 256, 512]
N_MIN = 100
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]
ORDER = ["Fibroblast_Pia", "Fibroblast_Arachnoid", "Fibroblast_Dura"]
KNN = 12
rng = np.random.default_rng(0)


def labels_for(gw):
    if gw == "GW16":
        d = f"{DATA_}/gaston_gw16"
        counts = np.load(f"{d}/counts_mat.npy")
        cl = np.load(f"{d}/clusters_kept.npy", allow_pickle=True)
        return cl[counts.sum(1) > 0]
    if gw == "GW6":
        return pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv")["celltype"].values
    return np.array([l.strip() for l in open(f"{DATA_}/gaston_multi/{gw}/clusters.txt")])


def coord_curve(x, y, n_knots=60, smooth=5):
    P = np.c_[x, y].astype(float)
    k = min(n_knots, max(8, len(P) // 50))
    C = KMeans(n_clusters=k, n_init=4, random_state=0).fit(P).cluster_centers_
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    T = minimum_spanning_tree(csr_matrix(D)); Tsym = T + T.T
    d0 = dijkstra(Tsym, indices=0)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(Tsym, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    bb = C[np.array(path[::-1])]
    if len(bb) >= smooth:
        w = np.ones(smooth) / smooth
        bb = np.c_[np.convolve(bb[:, 0], w, "valid"), np.convolve(bb[:, 1], w, "valid")]
    seg = np.diff(bb, axis=0); L = np.linalg.norm(seg, axis=1) + 1e-12
    tang = seg / L[:, None]; nrm = np.c_[-tang[:, 1], tang[:, 0]]
    _, si = cKDTree((bb[:-1] + bb[1:]) / 2).query(P)
    return ((P - bb[:-1][si]) * nrm[si]).sum(1)


def knn_graph(P, knn=KNN):
    n = len(P)
    tree = cKDTree(P)
    d, idx = tree.query(P, k=min(knn + 1, n))
    sig = np.median(d[:, 1:]) + 1e-9
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    cols = idx[:, 1:].ravel()
    w = np.exp(-(d[:, 1:].ravel() / sig) ** 2)
    W = csr_matrix((w, (rows, cols)), shape=(n, n))
    return W.maximum(W.T), idx


def coord_harmonic(x, y, knn=KNN, bfrac=0.18):
    """u = 0 on one side of the band, 1 on the other, harmonic in between. No centreline."""
    P = np.c_[x, y].astype(float)
    n = len(P)
    W, idx = knn_graph(P, knn)
    # one-sidedness: boundary units have neighbours concentrated in one direction
    V = P[idx[:, 1:]] - P[:, None, :]
    V = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-12)
    onesided = np.linalg.norm(V.mean(1), axis=1)
    bmask = onesided >= np.quantile(onesided, 1 - bfrac)
    bidx = np.flatnonzero(bmask)
    # cut the boundary loop at its two graph-diameter endpoints -> two arcs
    Wb = W[bidx][:, bidx]
    d0 = dijkstra(Wb, indices=0, unweighted=True)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da = dijkstra(Wb, indices=a, unweighted=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    db = dijkstra(Wb, indices=b, unweighted=True)
    cut = (da <= 2) | (db <= 2)                       # drop the two short end caps
    keep = np.flatnonzero(~cut)
    ncomp, lab = connected_components(Wb[keep][:, keep], directed=False)
    if ncomp < 2:
        return None, dict(ok=False, reason=f"boundary did not split (ncomp={ncomp})")
    sizes = np.bincount(lab)
    two = np.argsort(sizes)[-2:]
    s1 = bidx[keep[lab == two[0]]]; s2 = bidx[keep[lab == two[1]]]
    deg = np.asarray(W.sum(1)).ravel() + 1e-12
    L = diags(deg) - W
    fixed = np.concatenate([s1, s2])
    vals = np.concatenate([np.zeros(len(s1)), np.ones(len(s2))])
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n); u[fixed] = vals
    A = L[free][:, free].tocsc()
    rhs = -L[free][:, fixed] @ vals
    u[free] = spsolve(A, rhs)
    return u, dict(ok=True, n_boundary=len(bidx), side1=len(s1), side2=len(s2),
                   frac_boundary=round(len(bidx) / n, 3))


def grid_for(coord_vals, xs, ys, Cs):
    out = []
    for s_um in SCALES_UM:
        step = s_um / UMPP
        gx = np.floor((xs - xs.min()) / step).astype(int)
        gy = np.floor((ys - ys.min()) / step).astype(int)
        uk, inv = np.unique(gx * 100000 + gy, return_inverse=True)
        nps = len(uk)
        agg = np.zeros((nps, Cs.shape[1])); Ap = np.zeros(nps)
        cx = np.zeros(nps); cy = np.zeros(nps)
        for i in range(nps):
            mm = inv == i
            agg[i] = Cs[mm].sum(0); Ap[i] = coord_vals[mm].mean()
            cx[i] = xs[mm].mean(); cy[i] = ys[mm].mean()
        umi = agg.sum(1)
        for d in DEPTHS:
            ok = umi >= d
            if ok.sum() < N_MIN:
                out.append(dict(scale_um=s_um, depth=d, n=int(ok.sum()),
                                I=np.nan, feasible=False)); continue
            A = agg[ok]
            At = np.zeros_like(A)
            for i in range(A.shape[0]):
                tot = A[i].sum()
                At[i] = A[i] if tot <= d else rng.multinomial(d, A[i] / tot)
            lib = At.sum(1, keepdims=True); lib[lib == 0] = 1
            X = np.log1p(At / lib * np.median(lib))
            pos = (Ap[ok] - Ap[ok].min()) / (np.ptp(Ap[ok]) + 1e-9)
            bx = pd.cut(cx[ok], 4, labels=False).astype(int)
            by = pd.cut(cy[ok], 4, labels=False).astype(int)
            cb = (bx + by) % 2
            yhat = np.zeros_like(pos)
            for bl in np.unique(cb):
                te = cb == bl; tr = ~te
                if te.sum() < 5 or tr.sum() < 20:
                    yhat[te] = pos[tr].mean(); continue
                pc = PCA(n_components=min(30, X.shape[1], tr.sum() - 1), random_state=0).fit(X[tr])
                yhat[te] = Ridge(alpha=1.0).fit(pc.transform(X[tr]), pos[tr]).predict(
                    pc.transform(X[te]))
            mse = np.mean((pos - yhat) ** 2); var = np.var(pos)
            out.append(dict(scale_um=s_um, depth=d, n=int(ok.sum()),
                            I=round(0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0, 4),
                            feasible=True))
    return pd.DataFrame(out)


summary, grids = [], []
for gw in STAGES:
    counts, coords, fib = load_stage(gw)
    lab = labels_for(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], [3, 90])
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys = coords[sel, 0], coords[sel, 1]
    Cs = counts[sel]
    rank = {c: i for i, c in enumerate(ORDER)}
    yl = np.array([rank.get(c, np.nan) for c in lab[sel]])
    okl = ~np.isnan(yl)
    has_layers = okl.sum() >= 50 and len(np.unique(yl[okl])) >= 3

    cands = {"polar": r[sel], "curve": coord_curve(xs, ys)}
    uh, info = coord_harmonic(xs, ys)
    print(f"{gw}: harmonic {info}", flush=True)
    if uh is not None:
        cands["harmonic"] = uh

    for name, v in cands.items():
        rl = abs(spearmanr(v[okl], yl[okl])[0]) if has_layers else np.nan
        gdf = grid_for(v, xs, ys, Cs)
        gdf.insert(0, "coord", name); gdf.insert(0, "stage", gw)
        grids.append(gdf)
        f = gdf[gdf.feasible]
        best = f.loc[f["I"].idxmax()] if len(f) else None
        summary.append(dict(stage=gw, coord=name,
                            rho_layer=None if np.isnan(rl) else round(float(rl), 4),
                            n_layer=int(okl.sum()) if has_layers else 0,
                            I_max=None if best is None else float(best["I"]),
                            best_scale=None if best is None else int(best.scale_um),
                            best_depth=None if best is None else int(best.depth),
                            n_feasible=int(len(f))))
        print(f"   {gw} {name:9s} rho_layer={rl if has_layers else float('nan'):.3f} "
              f"I_max={summary[-1]['I_max']}", flush=True)

pd.DataFrame(summary).to_csv(f"{RES_}/fig3_coord_arbitration.csv", index=False)
pd.concat(grids, ignore_index=True).to_csv(f"{RES_}/fig3_coord_grids.csv", index=False)
print("WROTE fig3_coord_arbitration.csv / fig3_coord_grids.csv")
