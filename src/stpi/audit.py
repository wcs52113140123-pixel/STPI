"""Audit of one section: the published Fig 5 procedure as one call.

Steps (the computations of analysis/fig5_audit.py, analysis/fig5_buffered_all.py and
analysis/fig5_null.py, and the decision rules used for Fig 5):

  coordinate   interiority (geodesic distance from the boundary), unless one is given
  features     log_hvg (2,000 most variable features)
  random CV    5 folds drawn independently per unit                -> I_random
  spatial CV   4 x 4 checkerboard, two folds                       -> I_checker
  buffer       checkerboard with training units within k spot pitches of the test fold
               removed, k in `buffers`                              -> I_buffer_k
  null         Moran spectral surrogates of the coordinate, decoded under the checkerboard
               -> P

frac_artifact = (I_random - I_checker) / I_random, the fraction of the random-CV estimate
that does not survive spatial blocking (undefined when I_random <= 0).
"""
import numpy as np

from .coords import interiority
from .cv import checkerboard_folds, random_folds, spot_pitch
from .decode import recoverability
from .features import log_hvg
from .null import msr_surrogates, null_p
from ._seed import seed_of


def audit_section(M, xy, name, coord=None, buffers=(0, 1, 2, 3, 4), null_draws=200,
                  random_seed=0, hvg=2000, n_components=30, alpha=1.0, blocks=4):
    """M: units x genes counts; xy: unit positions; name: section name (seeds the null).

    Returns a dict of estimates (bits), P and verdict. Set null_draws = 0 to skip the null.
    """
    xy = np.asarray(xy, dtype=float)
    R = interiority(xy) if coord is None else np.asarray(coord, dtype=float)
    X = log_hvg(M, hvg)
    kw = dict(n_components=n_components, alpha=alpha)
    folds = checkerboard_folds(xy, blocks)
    rnd = recoverability(X, R, random_folds(len(R), 5, random_seed), **kw)
    chk = recoverability(X, R, folds, **kw)
    out = dict(sample=name, n=len(R), I_random=rnd.I, I_checker=chk.I,
               rho_random=rnd.rho, rho_checker=chk.rho,
               frac_artifact=(rnd.I - chk.I) / rnd.I if rnd.I > 1e-6 else float("nan"))
    pitch = spot_pitch(xy)
    out["pitch"] = pitch
    for k in buffers:
        r = recoverability(X, R, folds, xy, k * pitch, **kw)
        out[f"I_buffer_{k}"] = r.I
        out[f"frac_dropped_{k}"] = r.frac_dropped
    if null_draws:
        S, info = msr_surrogates(R, xy, n_draws=null_draws, seed=seed_of(name + "msr"))
        obs = recoverability(X, R, folds, xy, 0.0, **kw).I
        null = np.array([recoverability(X, s, folds, xy, 0.0, **kw).I for s in S])
        out.update(P=null_p(obs, null), null_median=float(np.median(null)),
                   null_p95=float(np.quantile(null, 0.95)))
    out["verdict"] = verdict(out, max(buffers) if buffers else None)
    return out


def verdict(r, widest=4):
    """Outcome of the two guards applied to the spatial coordinate recovery results.

    coordinate-limited   I_random <= 0: not even random CV recovers the coordinate
    collapse             I_checker <= 0: the random-CV estimate does not survive spatial
                         blocking (guard 1)
    falls under buffer   I_checker > 0 but I at the widest buffer <= 0 (guard 2)
    holds                positive under spatial blocking and under the widest buffer
    """
    if r["I_random"] <= 1e-6:
        return "coordinate-limited"
    if r["I_checker"] <= 0:
        return "collapse"
    key = f"I_buffer_{widest}"
    if widest is None or key not in r:
        return "survives spatial blocking"
    return "holds under the buffer" if r[key] > 0 else "falls under the buffer"
