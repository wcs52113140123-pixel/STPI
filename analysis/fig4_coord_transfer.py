#!/usr/bin/env python3
"""Fig 4: apply the MENINGES coordinate construction to the tendon.

The tendon analysis uses a boundary-field coordinate (geodesic distance from the tissue
surface, skin_markerfree_coord.py). The meninges coordinate belongs to a different family:
a centreline built from angular-binned medians about a hand-placed origin, followed by a
signed local-normal offset (multiGW_scan.radial_coord_full). A tissue-appropriate geometric
coordinate working in a second tissue is not the same thing as the meninges construction
itself transferring to that tissue, and this figure is built to show the latter.

The hand-placed polar origin cannot be carried to a new tissue without manual input, so the
transfer uses the origin-free form of the same family: the MST principal-curve construction
used in Fig 3a (fig3_altcoord_scan.principal_curve_coord) - KMeans knots, minimum spanning
tree, longest path as backbone, arclength smoothing, signed perpendicular offset. This is the
same family as the meninges coordinate (centreline + normal offset), without its arbitrary
choices (hand-placed origin, angular bins).

Of the three constructions compared in Fig 3a, the tendon's existing coordinate belongs to
the BOUNDARY family (Fig 3a's 'boundary harmonic' solves a Dirichlet problem; here it is a
boundary geodesic field - same family, different solver). This run adds the CENTRELINE
family, so the tendon is read with both, on the same cells, through the same pipeline.

Everything downstream is the same: 100-800 um pseudospots, multinomial depth thinning,
spatially structured checkerboard CV, PCA(30) + Ridge(1.0), I = 0.5 log2(var/MSE), N_MIN = 45.
Coordinates are rank-normalised to 0-1 before aggregation so the two are on one scale.

Outputs
  results/fig4_coord_transfer.csv   section, coord, scale_um, depth, n, I_checker
  results/fig4_coord_agreement.csv  section, n_spots, rho_curve_surface, frac_*
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os
# KMeans parallel reductions are not bit-reproducible across thread counts and the
# backbone moves with them; threads are pinned so every run gives the same grid.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import hashlib
import numpy as np, pandas as pd, scipy.sparse as sp, math
from multiprocessing import Pool
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
_al = open(f"{HERE_}/fig4_along_coord.py").read()
exec(compile(_al[_al.index("def along_backbone_coord"):], "along", "exec"), globals())


BASE = DATA_
DD, RES, SCR = f"{DER_}", f"{RES_}", f"{HERE_}"
SECTIONS = ["CTRL1", "CTRL2", "INMPP1", "INMPP1-2"]
SCALES = [100, 200, 400, 600, 800]
DEPTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
N_MIN = 45

# the centreline construction is imported directly from the Fig 3a script rather than reimplemented
_alt = open(f"{SCR}/fig3_altcoord_scan.py").read()
exec(compile(_alt[_alt.index("def principal_curve_coord"):_alt.index("rows = []")],
             "curve", "exec"), globals())




def rank01(v):
    o = np.argsort(np.argsort(v))
    return o / max(len(v) - 1, 1)


def thin_to(Sm, d, rng):
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
    lib = np.asarray(sub.sum(1)).ravel(); lib[lib == 0] = 1
    L = sub.multiply(1.0 / lib[:, None]).multiply(np.median(lib)).tocsr()
    L.data = np.log1p(L.data)
    return np.asarray(L.todense())


def checker_oof(X, y, cx, cy):
    ab = pd.qcut(cx, 4, labels=False, duplicates="drop")
    pb = pd.qcut(cy, 4, labels=False, duplicates="drop")
    blk = np.asarray((ab + pb) % 2); yh = np.full(len(y), np.nan)
    for b in np.unique(blk):
        te = blk == b; tr = ~te
        if tr.sum() < 20 or te.sum() < 8: continue
        pca = PCA(n_components=min(30, X.shape[1], tr.sum() - 1), random_state=0).fit(X[tr])
        m = Ridge(1.0).fit(pca.transform(X[tr]), y[tr])
        yh[te] = m.predict(pca.transform(X[te]))
    return yh


def I_of(y, yh):
    ok = ~np.isnan(yh)
    if ok.sum() < 20: return np.nan
    mse = np.mean((y[ok] - yh[ok]) ** 2); var = np.var(y[ok])
    return 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else np.nan


def _seed(name):
    """Deterministic per-section seed, computed from the SHA-256 digest of the section name
    so the thinning draws are identical across runs and across processes. Python's built-in
    string hash is randomised per process, so hashing the name directly would draw a
    different thinning stream each run and shift the grid by up to ~0.03 bit."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


def one_section(section):
    rng = np.random.default_rng(_seed(section))
    Bz = np.load(f"{DD}/skin_{section}_markerfree.npz", allow_pickle=True)
    M = sp.csr_matrix((Bz["counts_data"], Bz["counts_indices"], Bz["counts_indptr"]),
                      shape=tuple(Bz["counts_shape"]))
    xy = Bz["xy_um"]
    coords = {"surface": rank01(np.asarray(Bz["R"], float)),
              "centreline": rank01(principal_curve_coord(xy[:, 0], xy[:, 1])[0]),
              "along": rank01(along_backbone_coord(xy[:, 0], xy[:, 1]))}
    # GASTON isodepth (published method, trained by fig4_gaston_tendon.py on this same spot
    # set) is included as a third coordinate when its output file is present, so all three
    # coordinates are produced by the same pipeline
    _gp = f"{RES}/fig4_gaston_isodepth.npz"
    if os.path.exists(_gp):
        _g = np.load(_gp)
        iso = np.asarray(_g[f"{section}_isodepth_nl3"], float)
        assert len(iso) == len(xy), f"{section}: isodepth length {len(iso)} != {len(xy)}"
        assert np.allclose(_g[f"{section}_xy"], xy), f"{section}: isodepth spot set differs"
        coords["gaston"] = rank01(iso)
    rho = abs(spearmanr(coords["surface"], coords["centreline"])[0])
    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    rows = []
    for sc in SCALES:
        cbx = ((xy[:, 0] - x0) / sc).astype(np.int64)
        cby = ((xy[:, 1] - y0) / sc).astype(np.int64)
        uc, inv = np.unique(cbx * 100000 + cby, return_inverse=True)
        P = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                          shape=(len(uc), M.shape[0]))
        agg = (P @ M).tocsr()
        cnt = np.bincount(inv)
        cx = np.bincount(inv, weights=xy[:, 0]) / cnt
        cy = np.bincount(inv, weights=xy[:, 1]) / cnt
        agg_coord = {k: np.bincount(inv, weights=v) / cnt for k, v in coords.items()}
        umi = np.asarray(agg.sum(1)).ravel()
        for d in DEPTHS:
            elig = umi >= d
            if elig.sum() < N_MIN: continue
            idx = np.where(elig)[0]
            X = lognorm(thin_to(agg[idx], d, rng))
            for k in coords:
                y = agg_coord[k][idx]
                rows.append(dict(section=section, coord=k, scale_um=sc, depth=d,
                                 n=int(elig.sum()),
                                 I_checker=I_of(y, checker_oof(X, y, cx[idx], cy[idx]))))
        print(f"[{section} s={sc}] done", flush=True)
    return pd.DataFrame(rows), dict(section=section, n_spots=len(xy),
                                    rho_curve_surface=round(float(rho), 4))


if __name__ == "__main__":
    with Pool(len(SECTIONS)) as p:
        parts = p.map(one_section, SECTIONS)
    df = pd.concat([a for a, _ in parts], ignore_index=True)
    ag = pd.DataFrame([b for _, b in parts])
    df.to_csv(f"{RES}/fig4_coord_transfer.csv", index=False)
    ag.to_csv(f"{RES}/fig4_coord_agreement.csv", index=False)
    print("WROTE fig4_coord_transfer.csv", df.shape)
    print(ag.to_string(index=False))
    for sec in SECTIONS:
        for k in ("surface", "centreline"):
            q = df[(df.section == sec) & (df.coord == k)].dropna(subset=["I_checker"])
            b = q.loc[q.I_checker.idxmax()]
            lo = q[q.depth <= 32].I_checker
            e48 = [q[(q.scale_um == 800) & (q.depth == d)].I_checker.values[0]
                   - q[(q.scale_um == 400) & (q.depth == d)].I_checker.values[0]
                   for d in DEPTHS
                   if len(q[(q.scale_um == 800) & (q.depth == d)])
                   and len(q[(q.scale_um == 400) & (q.depth == d)])]
            print(f"  {sec:9s} {k:11s} best {int(b.scale_um):>3}um/d{int(b.depth):<5} "
                  f"I={b.I_checker:+.3f} | low-depth median {lo.median():+.3f} | "
                  f"E_scale(400->800) median {np.median(e48):+.3f} neg {sum(v<0 for v in e48)}/{len(e48)}")
