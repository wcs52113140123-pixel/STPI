"""Expression-independent coordinates built from tissue geometry.

Nothing here reads expression. Each function takes unit positions and returns one value
per unit.

  interiority      geodesic distance from the tissue boundary, scaled to 0-1.
                   Same as geom_interiority in analysis/fig5_buffered_all.py (Fig 5).
  across_band      signed distance along the normal of a smoothed centreline traced in
                   polar angle around the band's centre; for a thin curved band such as
                   the meninges. Same as radial_coord_full in analysis/multiGW_scan.py.
  backbone_offset  signed distance from a backbone traced through k-means knots by a
                   minimum spanning tree (a principal-curve construction, Hastie and
                   Stuetzle 1989), as in analysis/fig4_coord_construction.py;
                   the same construction appears as principal_curve_coord in the tendon
                   analysis scripts and as coord_curve in analysis/fig3_coord_arbitration.py.
  along_backbone   arclength along the same backbone (the axis orthogonal to
                   backbone_offset), as in analysis/fig4_along_coord.py.
  harmonic         harmonic field between the two long sides of a band (Laplace equation
                   on a kNN graph, Jones et al. 2000). Same as coord_harmonic in
                   analysis/fig3_coord_arbitration.py.
  rank01           ranks scaled to 0-1.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree, connected_components
from scipy.sparse.linalg import spsolve


def rank01(v):
    v = np.asarray(v, dtype=float)
    r = np.argsort(np.argsort(v)).astype(float)
    return r / max(len(r) - 1, 1)


def interiority(xy, boundary_pct=15, k=9, r_factor=1.8):
    """Geodesic distance from the tissue boundary on a kNN graph, scaled to 0-1.

    Boundary units are those with the fewest neighbours within r_factor x the median
    nearest-neighbour distance (degree at or below the boundary_pct percentile). On a
    section whose degree distribution is nearly constant that rule flags almost every
    unit; then units below the median degree are used instead.
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy); tree = cKDTree(xy)
    r0 = r_factor * np.median(tree.query(xy, k=2)[0][:, 1])
    deg = np.array([len(p) - 1 for p in tree.query_ball_point(xy, r0)])
    boundary = deg <= np.percentile(deg, boundary_pct)
    if boundary.mean() > 0.40:
        boundary = deg < np.median(deg)
    d, idx = tree.query(xy, k=k)
    rows = np.repeat(np.arange(n), k - 1)
    G = coo_matrix((d[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n))
    G = G.maximum(G.T)
    gd = dijkstra(G, directed=False, indices=np.where(boundary)[0], min_only=True)
    gd[~np.isfinite(gd)] = np.nanmax(gd[np.isfinite(gd)])
    return (gd - gd.min()) / (np.ptp(gd) + 1e-9)


def across_band(x, y, nbin=60, ksm=5, centre_y_pct=15, min_per_bin=15):
    """Signed distance across a curved band (unscaled, data units).

    The centre is (median x, centre_y_pct-th percentile of y): the meninges sections
    arch over the brain, and this point lies inside the arch. Units are binned by polar
    angle around it, the median position of each bin forms a centreline that is smoothed
    over ksm bins, and each unit is projected on the outward normal of its nearest
    centreline point. Scale to 0-1 (as the figures do) with (r - r.min()) / ptp(r).

    The smoothing is a zero-padded moving average (numpy.convolve, mode "same"), so the
    first and last ksm // 2 centreline points are pulled towards the origin of the data
    coordinates; units near the two ends of the band are projected less accurately than
    units along the rest of it.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    cx, cy = np.median(x), np.percentile(y, centre_y_pct)
    theta = np.arctan2(y - cy, x - cx); tb = pd.cut(theta, nbin, labels=False)
    cl = []
    for b in range(nbin):
        m = tb == b
        if m.sum() < min_per_bin: continue
        cl.append((np.median(theta[m]), np.median(x[m]), np.median(y[m])))
    cl = np.array(cl); cl = cl[np.argsort(cl[:, 0])]
    sm = lambda a: np.convolve(a, np.ones(ksm) / ksm, mode="same")
    clx, cly = sm(cl[:, 1]), sm(cl[:, 2])
    tang = np.gradient(np.column_stack([clx, cly]), axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    ro = np.column_stack([clx - cx, cly - cy]); ro /= np.linalg.norm(ro, axis=1, keepdims=True) + 1e-9
    fl = np.sign((nrm * ro).sum(1)); fl[fl == 0] = 1; nrm *= fl[:, None]
    tree = cKDTree(np.column_stack([clx, cly])); _, idx = tree.query(np.column_stack([x, y]))
    r = ((np.column_stack([x - clx[idx], y - cly[idx]])) * nrm[idx]).sum(1)
    return r


def backbone_offset(x, y, n_knots=60, smooth=5, return_geometry=False):
    """Signed distance from a backbone through the band (unscaled, data units).

    Knots are k-means centres (k = min(n_knots, max(8, n // 50))); the backbone is the
    longest path of their minimum spanning tree, smoothed over `smooth` knots; each unit
    is projected on the normal of its nearest backbone segment.
    """
    from sklearn.cluster import KMeans
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
    if return_geometry:
        return off, bb, P - off[:, None] * nrm[si]
    return off


def along_backbone(x, y, n_knots=60, smooth=5):
    """Arclength along the backbone of backbone_offset, one value per unit (data units)."""
    from sklearn.cluster import KMeans
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
    cum = np.concatenate([[0.0], np.cumsum(L)])
    mid = (bb[:-1] + bb[1:]) / 2
    _, si = cKDTree(mid).query(P)
    t = ((P - bb[:-1][si]) * tang[si]).sum(1)
    return cum[si] + np.clip(t, 0, L[si])


def _knn_graph(P, knn=12):
    n = len(P)
    tree = cKDTree(P)
    d, idx = tree.query(P, k=min(knn + 1, n))
    sig = np.median(d[:, 1:]) + 1e-9
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    cols = idx[:, 1:].ravel()
    w = np.exp(-(d[:, 1:].ravel() / sig) ** 2)
    W = csr_matrix((w, (rows, cols)), shape=(n, n))
    return W.maximum(W.T), idx


def harmonic(x, y, knn=12, bfrac=0.18):
    """u = 0 on one long side of the band, 1 on the other, harmonic in between.

    Boundary units are the bfrac most one-sided units (neighbours concentrated in one
    direction); the boundary loop is cut at its two graph-diameter endpoints into two
    arcs. Returns (u, info); u is None when the boundary does not split into two arcs.
    """
    P = np.c_[x, y].astype(float)
    n = len(P)
    W, idx = _knn_graph(P, knn)
    V = P[idx[:, 1:]] - P[:, None, :]
    V = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-12)
    onesided = np.linalg.norm(V.mean(1), axis=1)
    bmask = onesided >= np.quantile(onesided, 1 - bfrac)
    bidx = np.flatnonzero(bmask)
    Wb = W[bidx][:, bidx]
    d0 = dijkstra(Wb, indices=0, unweighted=True)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da = dijkstra(Wb, indices=a, unweighted=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    db = dijkstra(Wb, indices=b, unweighted=True)
    cut = (da <= 2) | (db <= 2)
    keep = np.flatnonzero(~cut)
    ncomp, lab = connected_components(Wb[keep][:, keep], directed=False)
    if ncomp < 2:
        return None, dict(ok=False, reason=f"boundary did not split (ncomp={ncomp})")
    sizes = np.bincount(lab)
    two = np.argsort(sizes)[-2:]
    s1 = bidx[keep[lab == two[0]]]; s2 = bidx[keep[lab == two[1]]]
    deg = np.asarray(W.sum(1)).ravel() + 1e-12
    Lap = diags(deg) - W
    fixed = np.concatenate([s1, s2])
    vals = np.concatenate([np.zeros(len(s1)), np.ones(len(s2))])
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n); u[fixed] = vals
    A = Lap[free][:, free].tocsc()
    rhs = -Lap[free][:, fixed] @ vals
    u[free] = spsolve(A, rhs)
    return u, dict(ok=True, n_boundary=len(bidx), side1=len(s1), side2=len(s2),
                   frac_boundary=round(len(bidx) / n, 3))
