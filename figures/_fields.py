"""Helpers for drawing values on tissue sections (Fig S1 B, Fig 5 B, Fig S4 B, Fig S5 B)."""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def rank01(v):
    """Ranks scaled to [0, 1]."""
    o = np.argsort(np.argsort(np.asarray(v, float)))
    return o / max(len(o) - 1, 1)


def spot_field(Q, vals, pitch, up=8.0, reach=0.62, soft_frac=0.70):
    """Render spots as a filled field with a smooth outline.

    Every grid node takes the value of its nearest spot (no smoothing of the data, no
    value is invented). The tissue boundary is carried by an alpha channel: the occupancy
    indicator (grid nodes within reach * pitch of a spot) is blurred with a kernel of
    soft_frac * pitch and thresholded with a narrow ramp, which gives a smooth outline.
    Interior gaps narrower than about one pitch close up and isolated islands of one or
    two spots can disappear; the coloured values are unaffected.

    Q      (n, 2) spot coordinates
    vals   values to draw, one per spot
    pitch  spot spacing in the units of Q (measured per section)
    up     grid refinement (grid step = pitch / up)
    """
    step = pitch / up
    pad = 2.0 * pitch
    xs = np.arange(Q[:, 0].min() - pad, Q[:, 0].max() + pad + step, step)
    ys = np.arange(Q[:, 1].min() - pad, Q[:, 1].max() + pad + step, step)
    Xg, Yg = np.meshgrid(xs, ys)
    d, idx = cKDTree(Q).query(np.c_[Xg.ravel(), Yg.ravel()])
    Z = np.asarray(vals, float)[idx].reshape(Xg.shape)
    occ = (d.reshape(Xg.shape) <= reach * pitch).astype(float)
    occ = gaussian_filter(occ, soft_frac * pitch / step)
    A = np.clip((occ - 0.50) / 0.10, 0.0, 1.0)
    return xs, ys, Z, A
