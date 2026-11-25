#!/usr/bin/env python3
"""Fig 4a: export the geometry of the transferred coordinate so it can be drawn.

principal_curve_coord (in fig3_altcoord_scan.py) returns only the signed offset and the backbone
length; the backbone itself and the local normals, which make the construction legible, are not
recoverable from its output. This script repeats that function's body (KMeans knots -> minimum
spanning tree -> tree diameter as the backbone -> arclength smoothing -> signed perpendicular
offset) and additionally writes out:

  <section>_backbone   the smoothed backbone polyline, um
  <section>_arrow_p0   foot points on the backbone for a sample of spots
  <section>_arrow_p1   the same spots, i.e. the head of each normal offset arrow
  <section>_xy         spot positions, um
  <section>_off        the signed offset itself (same quantity the scan uses)

The coordinate itself is not redefined here: the offsets written out are checked against
principal_curve_coord's own return value, and the script raises an error if they differ.

A second, separate question is addressed here but kept out of the figure: whether the geometric
coordinate lines up with a known biological gradient, the tendon counterpart of the Pia->Dura
layer-order criterion used for the meninges. Marker scores (Scx, Tnmd, Thbs4, Col1a1, Col1a2,
Col3a1) and epidermal scores (Krt10, Krt14) are scored per spot and correlated with the
coordinate. The correlation is weak and inconsistent across sections (|rho| 0.02-0.27 for tendon
markers, 0.04-0.40 for epidermal), so it supports no claim and is kept as source data only,
written to fig4_coord_validation.csv. A signed offset from a centreline puts the two sides of the
band at the two ends of the scale, so a marker gradient that is symmetric about the backbone
gives |rho| near zero even when the coordinate is exactly right; this is one reason this
criterion transfers poorly from the meninges, where the coordinate runs monotonically across
three named layers.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.spatial import cKDTree

DD, RES = f"{DER_}", f"{RES_}"
SECTIONS = ["CTRL1", "CTRL2", "INMPP1", "INMPP1-2"]
MK = ["Scx", "Tnmd", "Thbs4", "Col1a1", "Col1a2", "Col3a1"]
EPI = ["Krt10", "Krt14"]
N_ARROW, N_PERM = 26, 500

_alt = open(f"{HERE_}/fig3_altcoord_scan.py").read()
exec(compile(_alt[_alt.index("def principal_curve_coord"):_alt.index("rows = []")],
             "curve", "exec"), globals())


def geometry(x, y, n_knots=60, smooth=5):
    """principal_curve_coord's body, returning the backbone and normals as well."""
    P = np.c_[x, y].astype(float)
    k = min(n_knots, max(8, len(P) // 50))
    C = KMeans(n_clusters=k, n_init=4, random_state=0).fit(P).cluster_centers_
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    Tsym = minimum_spanning_tree(csr_matrix(D)); Tsym = Tsym + Tsym.T
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
    seg = np.diff(bb, axis=0)
    L = np.linalg.norm(seg, axis=1) + 1e-12
    tang = seg / L[:, None]
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    mid = (bb[:-1] + bb[1:]) / 2
    _, si = cKDTree(mid).query(P)
    off = ((P - bb[:-1][si]) * nrm[si]).sum(1)
    foot = P - off[:, None] * nrm[si]
    return off, bb, foot


def marker_score(M, genes, names):
    idx = [int(np.where(genes == n)[0][0]) for n in names if (genes == n).any()]
    tot = np.asarray(M.sum(1)).ravel(); tot[tot == 0] = 1
    X = np.log1p(np.asarray(M[:, idx].todense()) / tot[:, None] * 1e4)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X.mean(1), [str(genes[i]) for i in idx]


out, rows = {}, []
for s in SECTIONS:
    z = np.load(f"{DD}/skin_{s}_markerfree.npz", allow_pickle=True)
    xy = np.asarray(z["xy_um"], float)
    M = sp.csr_matrix((z["counts_data"], z["counts_indices"], z["counts_indptr"]),
                      shape=tuple(z["counts_shape"]))
    genes = np.array([str(g) for g in z["genes"]])
    off, bb, foot = geometry(xy[:, 0], xy[:, 1])
    ref, _ = principal_curve_coord(xy[:, 0], xy[:, 1])
    assert np.allclose(off, ref), f"{s}: geometry() diverged from principal_curve_coord"
    rg = np.random.default_rng(3)
    pick = rg.choice(len(xy), size=min(N_ARROW, len(xy)), replace=False)
    out[f"{s}_xy"] = xy
    out[f"{s}_off"] = off
    out[f"{s}_backbone"] = bb
    out[f"{s}_arrow_p0"] = foot[pick]
    out[f"{s}_arrow_p1"] = xy[pick]
    ten, ten_used = marker_score(M, genes, MK)
    epi, epi_used = marker_score(M, genes, EPI)
    null = np.array([abs(spearmanr(off, rg.permutation(ten))[0]) for _ in range(N_PERM)])
    rows.append(dict(section=s, n_spots=len(xy), n_knots_backbone=len(bb),
                     rho_tendon_marker=round(float(spearmanr(off, ten)[0]), 4),
                     rho_epidermal=round(float(spearmanr(off, epi)[0]), 4),
                     null95_abs=round(float(np.quantile(null, 0.95)), 4),
                     markers_used="|".join(ten_used), epidermal_used="|".join(epi_used)))
    print(f"{s}: spots={len(xy)} backbone={len(bb)} "
          f"rho_marker={rows[-1]['rho_tendon_marker']:+.3f} "
          f"rho_epi={rows[-1]['rho_epidermal']:+.3f}", flush=True)

np.savez_compressed(f"{RES}/fig4_coord_geometry.npz", **out)
pd.DataFrame(rows).to_csv(f"{RES}/fig4_coord_validation.csv", index=False)
print("WROTE fig4_coord_geometry.npz / fig4_coord_validation.csv")
