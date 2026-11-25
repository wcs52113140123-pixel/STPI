"""The tendon coordinate: continuous arclength along the transferred backbone.

Same backbone as fig3_altcoord_scan.principal_curve_coord: identical knots, MST, tree
diameter, arclength smoothing and nearest-segment assignment. That function returns the
signed offset across the band; this one returns the position along it, which is the axis
that clears the autocorrelation-preserving null on this tissue (fig4_tendon_null.csv).

Each spot is projected onto its nearest backbone segment and the within-segment distance is
added, so the field is continuous along the backbone. Using the graph distance of a spot's
KMeans knot from one backbone end instead would be piecewise constant (every spot sharing a
knot would get one value), producing visible patches in the coordinate maps rather than a
true arclength.

Imported by exec() in analysis/fig4_coord_transfer.py, analysis/fig4_tendon_scan.py and
analysis/fig4_tendon_null.py so each uses the same definition.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.spatial import cKDTree


def along_backbone_coord(x, y, n_knots=60, smooth=5):
    """Continuous arclength along the MST backbone. Returns one value per spot, in um."""
    P = np.c_[x, y].astype(float)
    k = min(n_knots, max(8, len(P) // 50))
    C = KMeans(n_clusters=k, n_init=4, random_state=0).fit(P).cluster_centers_
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    Tsym = minimum_spanning_tree(csr_matrix(D))
    Tsym = Tsym + Tsym.T
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
    cum = np.concatenate([[0.0], np.cumsum(L)])          # arclength at each backbone vertex
    mid = (bb[:-1] + bb[1:]) / 2
    _, si = cKDTree(mid).query(P)                         # nearest backbone segment
    t = ((P - bb[:-1][si]) * tang[si]).sum(1)             # distance along that segment
    return cum[si] + np.clip(t, 0, L[si])
