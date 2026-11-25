"""Writes results/fig4_coord_maps.npz.

Computes, for each tissue section, the surface coordinate and centreline coordinate
used in the coordinate map panels.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
from scipy.spatial import cKDTree

SEC = ["CTRL1", "CTRL2", "INMPP1", "INMPP1-2"]

_alt = open(f"{HERE_}/fig3_altcoord_scan.py").read()
exec(compile(_alt[_alt.index("def principal_curve_coord"):_alt.index("rows = []")],
             "curve", "exec"), globals())

def _rank01(v):
    o = np.argsort(np.argsort(v))
    return o / max(len(v) - 1, 1)

section_files = {
    "CTRL1": f"{DER_}/skin_CTRL1_markerfree.npz",
    "CTRL2": f"{DER_}/skin_CTRL2_markerfree.npz",
    "INMPP1": f"{DER_}/skin_INMPP1_markerfree.npz",
    "INMPP1-2": f"{DER_}/skin_INMPP1-2_markerfree.npz",
}

out = {}
for s in SEC:
    z = np.load(section_files[s], allow_pickle=True)
    xy = z["xy_um"]
    out[f"{s}_xy"] = xy
    out[f"{s}_surface"] = _rank01(np.asarray(z["R"], float))
    out[f"{s}_centreline"] = _rank01(principal_curve_coord(xy[:, 0], xy[:, 1])[0])

np.savez_compressed(f"{RES_}/fig4_coord_maps.npz", **out)
