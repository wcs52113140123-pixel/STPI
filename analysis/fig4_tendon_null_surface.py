#!/usr/bin/env python3
"""Geodesic surface-distance null for the tendon autocorrelation-preserving null test.

Adds a coordinate to the tendon null analysis: geodesic depth from the tissue boundary
(R in skin_<sec>_markerfree.npz), rank-normalised. The device, cell list, draw count,
acceptance rule and P definition are shared with the across-band and along-band tests and
are exec'd here from fig4_tendon_null.py; only the target coordinate changes. The
across-band and along-band tests themselves are read from results/fig4_tendon_null.npz
and are not recomputed by this script.

Writes results/fig4_tendon_null_surface.{csv,npz}.

Fig 4 applies the two method guards used in Fig 1: (2) spatially structured
cross-validation, which stops neighbour leakage, and (3) an autocorrelation-preserving
null, which stops a reading that any smooth spatial field would produce. This script
supplies guard (3) for the surface-distance coordinate on the external tissue.

The meninges null used for comparison is stored on disk as 200 draws per stage
(results/tangential_null_draws.npz); the device used here is specified directly from the
method's definition, matching the meninges run in two respects: 200 draws per section,
and P = (#{null >= observed} + 1) / (n_draws + 1).

Two surrogate coordinates, both preserving the spatial smoothness of the real one:

  spin     The real coordinate FIELD is rigidly rotated about the section centroid by a
           random angle, then read back at the true spot positions by nearest neighbour.
           A rigid rotation leaves the field's own autocorrelation structure untouched and
           only misaligns it with the tissue's band geometry. 200 draws. Rotated positions
           that fall outside the tissue take the nearest in-tissue value - a limitation of
           the device, noted here.
  along    Arclength ALONG the backbone instead of signed offset ACROSS it: a single
           deterministic surrogate, orthogonal to the real axis by construction, with a
           comparable spatial range. One value, not a distribution.

The tendon pipeline, imported from fig4_coord_transfer.py, performs pseudospot binning,
multinomial depth thinning, spatially structured checkerboard cross-validation,
PCA(30) + Ridge(1.0), and computes I = 0.5 log2(var/MSE).

Cell: 200 um pseudospots at depth 2048 - the cell feasible in all four sections with the
highest four-section median I (0.321). Fixed for every section, so the four columns are
read under the same settings.

Both targets, across-band (the transferred coordinate) and along-band (arclength on the
same backbone), get their own 200-draw spin null computed within a single process:
KMeans parallel reductions are not bit-reproducible across thread counts, so the backbone,
and the along-band surrogate derived from it, can vary with the thread count unless this
is controlled (see the threading note below).

Output: results/fig4_tendon_null_surface.npz   <section>_<scale>um_d<depth>_spin_surface
        (null draws), <section>_<scale>um_d<depth>_surface (observed I)
        results/fig4_tendon_null_surface.csv   section, scale_um, depth, n_pseudospots,
        I_surface, null_median_surface, null_p95_surface, P_surface
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os, hashlib
# KMeans parallel reductions are not bit-reproducible across thread counts, so the
# backbone, and therefore the along-band surrogate, can differ depending on the thread
# count used to compute it. Threads are pinned here, and both targets are computed in
# this process so the across-band and along-band readings always come from one run.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
import math

DD, RES = f"{DER_}", f"{RES_}"
SECTIONS = ["CTRL1", "CTRL2", "INMPP1", "INMPP1-2"]
_al = open(f"{HERE_}/fig4_along_coord.py").read()
exec(compile(_al[_al.index("def along_backbone_coord"):], "along", "exec"), globals())

# Three cells, fixed in advance and ALL reported - not the cell that maximises I, which
# would make the P value a selected one. 200/2048 is the cell feasible in all four
# sections with the highest four-section median I; 100/2048 and 100/1024 are the native
# pitch at its two deepest feasible depths.
CELLS = [(200, 2048), (100, 2048), (100, 1024)]
DECORR_MAX = 0.30   # a surrogate must be decorrelated from the real axis to be a null
N_DRAWS = 200
SCALE, DEPTH = CELLS[0]

_tr = open(f"{HERE_}/fig4_coord_transfer.py").read()
exec(compile(_tr[_tr.index("def rank01"):_tr.index("def one_section")], "pipe", "exec"),
     globals())
_alt = open(f"{HERE_}/fig3_altcoord_scan.py").read()
exec(compile(_alt[_alt.index("def principal_curve_coord"):_alt.index("rows = []")],
             "curve", "exec"), globals())


def _seed(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


def spin_field(xy, vals, theta):
    """Rigid rotation of the coordinate field, read back at the true positions."""
    c = xy.mean(0)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    rot = (xy - c) @ R.T + c
    _, idx = cKDTree(xy).query(rot, k=1)     # nearest true spot to each rotated position
    return vals[idx]


def decode_I(M, xy, coord, rng):
    """One reading of the pipeline at the fixed cell."""
    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    cbx = ((xy[:, 0] - x0) / SCALE).astype(np.int64)
    cby = ((xy[:, 1] - y0) / SCALE).astype(np.int64)
    uc, inv = np.unique(cbx * 100000 + cby, return_inverse=True)
    P = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                      shape=(len(uc), M.shape[0]))
    agg = (P @ M).tocsr()
    cnt = np.bincount(inv)
    cx = np.bincount(inv, weights=xy[:, 0]) / cnt
    cy = np.bincount(inv, weights=xy[:, 1]) / cnt
    ac = np.bincount(inv, weights=coord) / cnt
    umi = np.asarray(agg.sum(1)).ravel()
    idx = np.where(umi >= DEPTH)[0]
    X = lognorm(thin_to(agg[idx], DEPTH, rng))
    return I_of(ac[idx], checker_oof(X, ac[idx], cx[idx], cy[idx])), len(idx)



out, rows = {}, []
for sec in SECTIONS:
    z = np.load(f"{DD}/skin_{sec}_markerfree.npz", allow_pickle=True)
    xy = np.asarray(z["xy_um"], float)
    M = sp.csr_matrix((z["counts_data"], z["counts_indices"], z["counts_indptr"]),
                      shape=tuple(z["counts_shape"]))
    surf = rank01(np.asarray(z["R"], float))

    def spin_null(target, tag):
        rng = np.random.default_rng(_seed(sec + tag))
        kept, tried = [], 0
        while len(kept) < N_DRAWS and tried < 60 * N_DRAWS:
            th = rng.uniform(0, 2 * np.pi); tried += 1
            surr = spin_field(xy, target, th)
            if abs(spearmanr(surr, target)[0]) < DECORR_MAX:
                kept.append(surr)
        if len(kept) < N_DRAWS:
            # the device has no valid instantiation here: a boundary-distance field is
            # near-radially symmetric, so on a compact section NO rotation decorrelates
            # it (CTRL2: 0 of 1000 angles below |rho| = 0.30, median |rho| 0.48).
            # Recorded as not evaluable rather than relaxing the acceptance rule, which
            # would seed the null with re-labelled copies of the axis under test.
            print(f"{sec}: NOT EVALUABLE - {len(kept)} decorrelated draws in {tried}", flush=True)
            return None
        draws = np.array([decode_I(M, xy, sr, np.random.default_rng(_seed(sec)))[0]
                          for sr in kept])
        obs, n = decode_I(M, xy, target, np.random.default_rng(_seed(sec)))
        return obs, draws, (np.sum(draws >= obs) + 1) / (len(draws) + 1), n

    for _cell in CELLS:
        globals()["SCALE"], globals()["DEPTH"] = _cell
        _r = spin_null(surf, f"-surface-spin-{_cell}")
        if _r is None:
            rows.append(dict(section=sec, scale_um=_cell[0], depth=_cell[1],
                             n_pseudospots=-1, I_surface=np.nan,
                             null_median_surface=np.nan, null_p95_surface=np.nan,
                             P_surface=np.nan))
            continue
        i_s, d_s, p_s, n_ps = _r
        tag = f"{sec}_{_cell[0]}um_d{_cell[1]}"
        out[f"{tag}_spin_surface"] = d_s
        out[f"{tag}_surface"] = np.array([i_s])
        rows.append(dict(section=sec, scale_um=_cell[0], depth=_cell[1],
                         n_pseudospots=int(n_ps),
                         I_surface=round(float(i_s), 4),
                         null_median_surface=round(float(np.median(d_s)), 4),
                         null_p95_surface=round(float(np.percentile(d_s, 95)), 4),
                         P_surface=round(float(p_s), 4)))
        print(f"{sec} {_cell}: surface {i_s:+.3f} vs null {np.median(d_s):+.3f} P={p_s:.3f}",
              flush=True)

np.savez_compressed(f"{RES}/fig4_tendon_null_surface.npz", **out)
pd.DataFrame(rows).to_csv(f"{RES}/fig4_tendon_null_surface.csv", index=False)
