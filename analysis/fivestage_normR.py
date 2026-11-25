"""Writes results/fivestage_normR.npz."""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_, DD_   # data, derived inputs, results, this folder
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial import cKDTree

base = DATA_
order = ["Fibroblast_Pia", "Fibroblast_Arachnoid", "Fibroblast_Dura"]

def radial_coord_full(x, y, nbin=60, ksm=5):
    # estimate a centre point near the ventral side of the tissue
    cx, cy = np.median(x), np.percentile(y, 15)
    theta = np.arctan2(y - cy, x - cx)
    tb = pd.cut(theta, nbin, labels=False)
    cl = []
    # build a centreline by taking the median position in each angular bin
    for b in range(nbin):
        m = tb == b
        if m.sum() < 15:
            continue
        cl.append((np.median(theta[m]), np.median(x[m]), np.median(y[m])))
    cl = np.array(cl)
    cl = cl[np.argsort(cl[:, 0])]
    sm = lambda a: np.convolve(a, np.ones(ksm) / ksm, mode="same")
    clx, cly = sm(cl[:, 1]), sm(cl[:, 2])
    # tangent and outward normal directions along the smoothed centreline
    tang = np.gradient(np.column_stack([clx, cly]), axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    ro = np.column_stack([clx - cx, cly - cy])
    ro /= np.linalg.norm(ro, axis=1, keepdims=True) + 1e-9
    fl = np.sign((nrm * ro).sum(1))
    fl[fl == 0] = 1
    nrm *= fl[:, None]
    # assign each point to its nearest centreline vertex and project onto the local normal
    tree = cKDTree(np.column_stack([clx, cly]))
    _, idx = tree.query(np.column_stack([x, y]))
    r = ((np.column_stack([x - clx[idx], y - cly[idx]])) * nrm[idx]).sum(1)
    return r

def load_coords(gw):
    # each developmental stage stores coordinates and cluster labels in a different layout
    if gw == "GW16":
        d = f"{DATA_}/gaston_gw16"
        coords = np.load(f"{d}/coords_mat.npy")
        counts = np.load(f"{d}/counts_mat.npy")
        clusters = np.load(f"{d}/clusters_kept.npy", allow_pickle=True)
        keep = counts.sum(1) > 0
        coords, clusters = coords[keep], clusters[keep]
        fib = np.isin(clusters, order)
    elif gw == "GW6":
        m = pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv")
        coords = m[["x", "y"]].values.astype(float)
        ct = m["celltype"].values
        fib = np.isin(ct, ["Fibro-pia-pro", "Fibro-others"])
    else:
        d = f"{DATA_}/gaston_multi/{gw}"
        coords = np.loadtxt(f"{d}/coords.tsv")
        clusters = np.array([l.strip() for l in open(f"{d}/clusters.txt")])
        fib = np.isin(clusters, order)
    return coords, fib

FS = {}
for gw in ["GW6", "GW8", "GW10", "GW12", "GW16"]:
    coords, fib = load_coords(gw)
    rr = radial_coord_full(coords[:, 0], coords[:, 1])
    # restrict to the fibroblast layer and trim the outer percentiles of the radial coordinate
    lo, hi = np.percentile(rr[fib], [3, 90])
    sel = fib & (rr >= lo) & (rr <= hi)
    xs, ys, rs = coords[sel, 0], coords[sel, 1], rr[sel]
    # rescale the radial coordinate to [0, 1] within each stage for cross-stage comparison
    R = (rs - rs.min()) / (np.ptp(rs) + 1e-9)
    FS[gw] = dict(x=xs, y=ys, R=R, n=len(xs))

np.savez(
    f"{DD_}/fivestage_normR.npz",
    **{f"{gw}_{k}": FS[gw][k] for gw in FS for k in ["x", "y", "R"]},
    stages=np.array(list(FS.keys()))
)
print("five-stage normalized coords saved | n:", {g: FS[g]["n"] for g in FS})
