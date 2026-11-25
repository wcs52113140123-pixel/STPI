"""Fig 3a: tests whether the five-stage result depends on how the across-band coordinate is constructed.

Mainline (multiGW_scan.radial_coord_full) fits the centreline by binning points in polar
angle about a fixed origin (median x, 15th percentile y), takes binned medians,
smooths over 5 bins, and projects each point onto the local normal. Three parameter
choices enter this construction: the polar origin, the 60 angular bins, and the smoothing width.

This script rebuilds the centreline without any of them. K-means places knots on the point
cloud, a minimum spanning tree links them, the tree's longest path is taken as the
backbone (this is what makes it work on the C-shaped stages), the backbone is smoothed
by arclength, and each point gets the signed perpendicular offset to its nearest backbone
segment. No origin, no angular binning, no expression input.

Scope note: this tests the construction algorithm within the centreline-normal family.
A cross-family comparison (harmonic/Laplacian coordinate) is available only for GW16,
where the response surface agreed with the mainline at Spearman 0.952 over 36 grid cells.

The low Laplacian eigenvectors of a kNN graph are not used as the alternative
coordinate: on a band this elongated the low modes oscillate along the band, and the
selected eigenvector correlates only 0.08-0.19 with the local minor axis, i.e. it is
not an across-band coordinate.

Everything downstream of the coordinate follows the same processing steps as the
mainline pipeline.

Output: results/fig3_altcoord_5stage.csv
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 pins the K-means/PCA initialization so the centreline and downstream fit are reproducible across runs
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, pandas as pd, math
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order, dijkstra
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

src = open(f"{HERE_}/multiGW_scan.py").read()
ms = {}
exec(compile(src[:src.index("UMPP=0.5")], "multiGW_scan_defs", "exec"), ms)
radial_coord_full, load_stage = ms["radial_coord_full"], ms["load_stage"]

UMPP = 0.5
SCALES_UM = [50, 100, 200, 400, 800]
DEPTHS = [16, 32, 64, 128, 256, 512]
N_MIN = 100
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]
rng = np.random.default_rng(0)


def principal_curve_coord(x, y, n_knots=60, smooth=5):
    """Signed offset from an MST-backbone centreline. No polar origin, no angle bins."""
    P = np.c_[x, y].astype(float)
    k = min(n_knots, max(8, len(P) // 50))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(P)
    C = km.cluster_centers_
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    T = minimum_spanning_tree(csr_matrix(D))
    Tsym = T + T.T
    # tree diameter: farthest node from an arbitrary node, then farthest from that one
    d0 = dijkstra(Tsym, indices=0)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(Tsym, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    path = np.array(path[::-1])
    bb = C[path]
    if len(bb) >= smooth:                       # smooth along arclength
        w = np.ones(smooth) / smooth
        bb = np.c_[np.convolve(bb[:, 0], w, "valid"), np.convolve(bb[:, 1], w, "valid")]
    seg = np.diff(bb, axis=0)
    L = np.linalg.norm(seg, axis=1) + 1e-12
    tang = seg / L[:, None]
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    mid = (bb[:-1] + bb[1:]) / 2
    _, si = cKDTree(mid).query(P)               # nearest backbone segment
    off = ((P - bb[:-1][si]) * nrm[si]).sum(1)  # signed perpendicular offset
    return off, len(bb)


rows = []
for gw in STAGES:
    counts, coords, fib = load_stage(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], [3, 90])
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys = coords[sel, 0], coords[sel, 1]
    Cs = counts[sel]; rs = r[sel]
    r_alt, nknot = principal_curve_coord(xs, ys)
    rho_cc = spearmanr(rs, r_alt)[0]
    print(f"{gw}: n={sel.sum()} backbone={nknot} knots  rho(alt, mainline)={rho_cc:+.3f}",
          flush=True)

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
            agg[i] = Cs[mm].sum(0); Ap[i] = r_alt[mm].mean()
            cx[i] = xs[mm].mean(); cy[i] = ys[mm].mean()
        umi = agg.sum(1)
        for d in DEPTHS:
            ok = umi >= d
            base = dict(stage=gw, scale_um=s_um, depth=d, n=int(ok.sum()),
                        coord="principal_curve", rho_coord=round(float(rho_cc), 4))
            if ok.sum() < N_MIN:
                rows.append({**base, "I": np.nan, "R2": np.nan, "rho": np.nan,
                             "feasible": False}); continue
            A = agg[ok]
            At = np.zeros_like(A)
            for i in range(A.shape[0]):
                tot = A[i].sum()
                At[i] = A[i] if tot <= d else rng.multinomial(d, A[i] / tot)
            lib = At.sum(1, keepdims=True); lib[lib == 0] = 1
            X = np.log1p(At / lib * np.median(lib))
            pos = (Ap[ok] - Ap[ok].min()) / (np.ptp(Ap[ok]) + 1e-9)
            nb = 4
            bx = pd.cut(cx[ok], nb, labels=False).astype(int)
            by = pd.cut(cy[ok], nb, labels=False).astype(int)
            cb = (bx + by) % 2
            yhat = np.zeros_like(pos)
            for bl in np.unique(cb):
                te = cb == bl; tr = ~te
                if te.sum() < 5 or tr.sum() < 20:
                    yhat[te] = pos[tr].mean(); continue
                pc = PCA(n_components=min(30, X.shape[1], tr.sum() - 1), random_state=0).fit(X[tr])
                m = Ridge(alpha=1.0).fit(pc.transform(X[tr]), pos[tr])
                yhat[te] = m.predict(pc.transform(X[te]))
            mse = np.mean((pos - yhat) ** 2); var = np.var(pos)
            I = 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0
            rows.append({**base, "I": round(I, 4), "R2": round(1 - mse / var, 4),
                         "rho": round(spearmanr(pos, yhat)[0], 4), "feasible": True})
    print(f"  {gw} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{RES_}/fig3_altcoord_5stage.csv", index=False)
print("WROTE fig3_altcoord_5stage.csv", df.shape, "feasible", int(df.feasible.sum()))
