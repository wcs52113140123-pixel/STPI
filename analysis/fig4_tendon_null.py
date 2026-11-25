#!/usr/bin/env python3
"""Fig 4: autocorrelation-preserving null for the tendon coordinate.

Fig 1 uses two method safeguards: (2) spatially structured cross-validation, which prevents
neighbour leakage, and (3) an autocorrelation-preserving null, which rules out a reading that
any smooth spatial field would produce. Fig 4's checkerboard cross-validation already
implements safeguard (2); this script implements safeguard (3) for the tendon tissue.

Note: the meninges null is stored on disk as draws only (results/tangential_null_draws.npz,
200 draws per stage) without the code that generated them, so this null is reimplemented from
the method's definition rather than ported line for line. It matches two properties of the
meninges run: 200 draws per section, and P = (#{null >= observed} + 1) / (n_draws + 1).

Two surrogate coordinates, both preserving the spatial smoothness of the real one:

  spin     The real coordinate field is rigidly rotated about the section centroid by a
           random angle, then read back at the true spot positions by nearest neighbour.
           A rigid rotation leaves the field's own autocorrelation structure untouched and
           only misaligns it with the tissue's band geometry. 200 draws. Rotated positions
           that fall outside the tissue take the nearest in-tissue value.
  along    Arclength along the backbone instead of signed offset across it: a single
           deterministic surrogate, orthogonal to the real axis by construction, with a
           comparable spatial range. One value, not a distribution.

The pipeline downstream of the surrogate coordinates is the tendon pipeline from
analysis/fig4_coord_transfer.py: pseudospot binning, multinomial depth thinning, spatially
structured checkerboard cross-validation, PCA(30) + Ridge(1.0), I = 0.5 log2(var/MSE).

Cell: 200 um pseudospots at depth 2048, the cell feasible in all four sections with the
highest four-section median I (0.321). Fixed for every section, so the four columns are
read under the same settings.

Both targets (across-band, the transferred coordinate, and along-band, arclength on the same
backbone) get their own 200-draw spin null, computed in the same process. This is because
KMeans reductions used for the backbone fit are not bit-reproducible across thread counts
(see below); computing both targets together ensures the across-band and along-band readings
come from the same backbone fit.

Output: results/fig4_tendon_null.npz   <section>_spin, <section>_spin_along, _iobs, _along
        results/fig4_tendon_null.csv   I_across / P_across / I_along / P_along per section
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os, hashlib
# KMeans parallel reductions are not bit-reproducible across thread counts, which can shift
# the fitted backbone and therefore the along-band surrogate. Threads are pinned to one here,
# and both targets are computed in this process so the across-band and along-band readings
# always come from the same backbone fit.
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

DD, RES = DER_, RES_
SECTIONS = ["CTRL1", "CTRL2", "INMPP1", "INMPP1-2"]
_al = open(f"{HERE_}/fig4_along_coord.py").read()
exec(compile(_al[_al.index("def along_backbone_coord"):], "along", "exec"), globals())

# Three cells, fixed in advance and all reported, not the cell that maximises I, which
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
    off, _ = principal_curve_coord(xy[:, 0], xy[:, 1])
    real = rank01(off)

    # the along-band axis: continuous arclength on the same backbone (analysis/fig4_along_coord.py)
    along = rank01(along_backbone_coord(xy[:, 0], xy[:, 1]))

    def spin_null(target, tag):
        """Rejection-sampled rotation null. An unconstrained rotation is NOT a null on an
        elongated section: rotating by ~0 or ~180 degrees returns nearly the same field, so
        the draws include re-labelled copies of the real axis (median |rho| with the real
        coordinate 0.51-0.66, and 16-31% of draws above 0.8), which pushes P up mechanically.
        Angles are therefore accepted only when the surrogate is decorrelated from the real
        axis, |rho| < DECORR_MAX - which is what the meninges' own tangential null does by
        construction, taking a direction orthogonal to the coordinate."""
        rng = np.random.default_rng(_seed(sec + tag))
        kept, tried = [], 0
        while len(kept) < N_DRAWS and tried < 60 * N_DRAWS:
            th = rng.uniform(0, 2 * np.pi); tried += 1
            surr = spin_field(xy, target, th)
            if abs(spearmanr(surr, target)[0]) < DECORR_MAX:
                kept.append(surr)
        assert len(kept) == N_DRAWS, f"{sec}: only {len(kept)} decorrelated draws in {tried}"
        draws = np.array([decode_I(M, xy, sr, np.random.default_rng(_seed(sec)))[0]
                          for sr in kept])
        obs, n = decode_I(M, xy, target, np.random.default_rng(_seed(sec)))
        return obs, draws, (np.sum(draws >= obs) + 1) / (len(draws) + 1), n

    for _cell in CELLS:
        globals()["SCALE"], globals()["DEPTH"] = _cell
        i_acr, d_acr, p_acr, n_ps = spin_null(real, f"-across-spin-{_cell}")
        i_alo, d_alo, p_alo, _ = spin_null(along, f"-along-spin-{_cell}")
        tag = f"{_cell[0]}um_d{_cell[1]}"
        out[f"{sec}_{tag}_spin"] = d_acr
        out[f"{sec}_{tag}_spin_along"] = d_alo
        out[f"{sec}_{tag}_iobs"] = np.array([i_acr])
        out[f"{sec}_{tag}_along"] = np.array([i_alo])
        rows.append(dict(section=sec, scale_um=_cell[0], depth=_cell[1],
                         n_pseudospots=int(n_ps),
                         I_across=round(float(i_acr), 4),
                         null_median_across=round(float(np.median(d_acr)), 4),
                         P_across=round(float(p_acr), 4),
                         I_along=round(float(i_alo), 4),
                         null_median_along=round(float(np.median(d_alo)), 4),
                         null_p95_along=round(float(np.percentile(d_alo, 95)), 4),
                         P_along=round(float(p_alo), 4)))
        print(f"{sec} {tag}: n={n_ps} | across {i_acr:+.3f} P={p_acr:.3f} "
              f"| along {i_alo:+.3f} vs null {np.median(d_alo):+.3f} P={p_alo:.3f}",
              flush=True)

np.savez_compressed(f"{RES}/fig4_tendon_null.npz", **out)
pd.DataFrame(rows).to_csv(f"{RES}/fig4_tendon_null.csv", index=False)
print("WROTE fig4_tendon_null.npz / fig4_tendon_null.csv")
