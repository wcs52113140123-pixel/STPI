
"""Fold assignments for cross-validation.

The same fold rules as analysis/fig5_buffered_all.py and analysis/figS5_support.py
(folds_of); used by analysis/fig5_audit.py.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def checkerboard_folds(xy, k=4):
    """Spatial block folds: a k x k grid over the section's bounding box, two folds by
    the parity of the grid cell (a checkerboard).

    The grid is equal-width in x and in y (pandas.cut), so the blocks are rectangles of
    the section's aspect ratio. k = 4 is the setting used in the audit.
    """
    xy = np.asarray(xy, dtype=float)
    bx = pd.cut(xy[:, 0], k, labels=False).astype(int)
    by = pd.cut(xy[:, 1], k, labels=False).astype(int)
    return (bx + by) % 2


def random_folds(n, k=5, seed=0):
    """Random CV folds: each unit draws one of k fold labels independently."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, n)


def spot_pitch(xy):
    """Median distance from each unit to its nearest neighbour (the sampling pitch)."""
    xy = np.asarray(xy, dtype=float)
    return float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
