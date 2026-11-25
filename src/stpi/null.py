"""Autocorrelation-preserving nulls for the coordinate.

A surrogate coordinate keeps the spatial smoothness of the real one but carries no
relation to the tissue's actual layout; decoding it gives the estimate that smoothness
alone produces. A surrogate is accepted only if it is decorrelated from the real
coordinate (|Spearman rho| < decorr_max), so the null is not seeded with copies of the
axis under test.

  msr_surrogates   Moran spectral randomisation: the coordinate is expanded in the
                   eigenvectors of the doubly centred kNN weights, coefficients are
                   permuted and sign-flipped within bands of `band` neighbouring
                   eigenvectors, and the result is rank-matched to the coordinate.
                   Used by analysis/fig5_null.py (Fig 5, Fig S5 A).
  spin_surrogates  rigid rotation of the coordinate field about the section centre, read
                   back at the true positions. Same rotation as in
                   analysis/fig4_tendon_null_surface.py (Fig S4 D).
  null_p           (number of null estimates >= observed + 1) / (draws + 1).
"""
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


def knn_weights(xy, k=9):
    """Symmetric binary kNN weights (dense)."""
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    _, idx = cKDTree(xy).query(xy, k=k)
    rows = np.repeat(np.arange(n), k - 1)
    W = coo_matrix((np.ones(rows.size), (rows, idx[:, 1:].ravel())), shape=(n, n)).toarray()
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def morans_I(x, W):
    z = x - x.mean()
    return (len(x) / W.sum()) * float(z @ W @ z) / float(z @ z)


def moran_basis(W):
    """Moran eigenvector maps: eigenvectors of the doubly centred weights."""
    r = W.mean(axis=1, keepdims=True)
    c = W.mean(axis=0, keepdims=True)
    g = float(W.mean())
    M = W - r - c + g
    M = (M + M.T) / 2.0
    # The binary kNN weights of a regular spot grid have many repeated eigenvalues, and
    # within a repeated eigenvalue LAPACK may return a different basis depending on the
    # number of BLAS threads; the surrogates (coefficients permuted across eigenvectors)
    # then change with the machine's thread setting. Limiting to one thread keeps the
    # basis fixed.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        vals, vecs = np.linalg.eigh(M)
    return vecs


def rank_match(surr, target):
    """Give the surrogate the real coordinate's exact marginal distribution."""
    out = np.empty_like(target)
    out[np.argsort(surr)] = np.sort(target)
    return out


def msr_surrogates(coord, xy, n_draws=200, seed=0, k=9, band=25, decorr_max=0.30,
                   max_tries_factor=40):
    """Moran spectral surrogates of `coord`. Returns (surrogates, info).

    surrogates   array n_draws x units
    info         dict: rho (|Spearman| with the coordinate per draw), moran (Moran's I per
                 draw), moran_obs, tried
    Raises RuntimeError when fewer than n_draws decorrelated surrogates are found in
    max_tries_factor x n_draws attempts.
    """
    coord = np.asarray(coord, dtype=float)
    W = knn_weights(xy, k)
    V = moran_basis(W)
    a = V.T @ (coord - coord.mean())
    rng = np.random.default_rng(seed)
    kept, rhos, mors, tried = [], [], [], 0
    while len(kept) < n_draws and tried < max_tries_factor * n_draws:
        tried += 1
        b = a.copy()
        for st in range(0, b.size, band):
            sl = slice(st, min(st + band, b.size))
            w = sl.stop - sl.start
            b[sl] = b[sl][rng.permutation(w)] * rng.choice([-1.0, 1.0], size=w)
        surr = rank_match(V @ b, coord)
        if np.std(surr) == 0:
            continue
        r = abs(spearmanr(surr, coord)[0])
        if r < decorr_max:
            kept.append(surr)
            rhos.append(r)
            mors.append(morans_I(surr, W))
    if len(kept) < n_draws:
        raise RuntimeError(f"only {len(kept)} decorrelated surrogates in {tried} attempts")
    return np.array(kept), dict(rho=np.array(rhos), moran=np.array(mors),
                                moran_obs=morans_I(coord, W), tried=tried)


def spin_field(xy, vals, theta):
    """Rigid rotation of the coordinate field, read back at the true positions."""
    xy = np.asarray(xy, dtype=float)
    c = xy.mean(0)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    rot = (xy - c) @ R.T + c
    _, idx = cKDTree(xy).query(rot, k=1)
    return np.asarray(vals)[idx]


def spin_surrogates(coord, xy, n_draws=200, seed=0, decorr_max=0.30, max_tries_factor=60):
    """Rotation surrogates of `coord`. Returns (surrogates, tried), or (None, tried) when
    fewer than n_draws rotations decorrelate the field - a near-radially symmetric field
    on a compact section has no valid rotation, and the null is then not evaluable."""
    coord = np.asarray(coord, dtype=float)
    rng = np.random.default_rng(seed)
    kept, tried = [], 0
    while len(kept) < n_draws and tried < max_tries_factor * n_draws:
        th = rng.uniform(0, 2 * np.pi); tried += 1
        surr = spin_field(xy, coord, th)
        if abs(spearmanr(surr, coord)[0]) < decorr_max:
            kept.append(surr)
    if len(kept) < n_draws:
        return None, tried
    return np.array(kept), tried


def null_p(observed, null):
    null = np.asarray(null, dtype=float)
    return (float((null >= observed).sum()) + 1.0) / (len(null) + 1.0)
