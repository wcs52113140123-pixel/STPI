"""Writes results/stage_feasible_counts.csv.

For each developmental stage, computes the number of 50 um spatial bins
covering the pial/arachnoid/dural fibroblast layer and how many of those bins
reach minimum UMI count thresholds (16, 64, 128, 256, 512), together with the
median per-bin UMI count. This quantifies how feasible a given bin size is
for downstream counting analyses at each stage.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np
import pandas as pd
import scipy.io as sio
import math
from scipy.spatial import cKDTree

order = ["Fibroblast_Pia", "Fibroblast_Arachnoid", "Fibroblast_Dura"]
UMPP = 0.5
SCALE_UM = 50

def radial_coord_full(x, y, nbin=60, ksm=5):
    cx, cy = np.median(x), np.percentile(y, 15)
    theta = np.arctan2(y - cy, x - cx)
    tb = pd.cut(theta, nbin, labels=False)
    cl = []
    for b in range(nbin):
        m = tb == b
        if m.sum() < 15:
            continue
        cl.append((np.median(theta[m]), np.median(x[m]), np.median(y[m])))
    cl = np.array(cl)
    cl = cl[np.argsort(cl[:, 0])]
    sm = lambda a: np.convolve(a, np.ones(ksm) / ksm, mode="same")
    clx, cly = sm(cl[:, 1]), sm(cl[:, 2])
    tang = np.gradient(np.column_stack([clx, cly]), axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    ro = np.column_stack([clx - cx, cly - cy])
    ro /= np.linalg.norm(ro, axis=1, keepdims=True) + 1e-9
    fl = np.sign((nrm * ro).sum(1))
    fl[fl == 0] = 1
    nrm *= fl[:, None]
    tree = cKDTree(np.column_stack([clx, cly]))
    _, idx = tree.query(np.column_stack([x, y]))
    return ((np.column_stack([x - clx[idx], y - cly[idx]])) * nrm[idx]).sum(1)

def load_stage(gw):
    if gw == "GW16":
        counts = np.load(f'{DATA_}/gaston_gw16/counts_mat.npy')
        coords = np.load(f'{DATA_}/gaston_gw16/coords_mat.npy')
        clusters = np.load(f'{DATA_}/gaston_gw16/clusters_kept.npy', allow_pickle=True)
        keep = counts.sum(1) > 0
        counts, coords, clusters = counts[keep], coords[keep], clusters[keep]
        fib = np.isin(clusters, order)
    elif gw == "GW6":
        m = pd.read_csv(f'{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv')
        c = sio.mmread(f'{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_counts.mtx')
        counts = np.asarray(c.todense()).T
        coords = m[["x", "y"]].values.astype(float)
        ct = m["celltype"].values
        fib = np.isin(ct, ["Fibro-pia-pro", "Fibro-others"])
    elif gw == "GW8":
        counts = np.asarray(sio.mmread(f'{DATA_}/gaston_multi/GW8/counts.mtx').T.todense(), dtype=np.float64)
        coords = np.loadtxt(f'{DATA_}/gaston_multi/GW8/coords.tsv')
        clusters = np.array([l.strip() for l in open(f'{DATA_}/gaston_multi/GW8/clusters.txt')])
        fib = np.isin(clusters, order)
    elif gw == "GW10":
        counts = np.asarray(sio.mmread(f'{DATA_}/gaston_multi/GW10/counts.mtx').T.todense(), dtype=np.float64)
        coords = np.loadtxt(f'{DATA_}/gaston_multi/GW10/coords.tsv')
        clusters = np.array([l.strip() for l in open(f'{DATA_}/gaston_multi/GW10/clusters.txt')])
        fib = np.isin(clusters, order)
    elif gw == "GW12":
        counts = np.asarray(sio.mmread(f'{DATA_}/gaston_multi/GW12/counts.mtx').T.todense(), dtype=np.float64)
        coords = np.loadtxt(f'{DATA_}/gaston_multi/GW12/coords.tsv')
        clusters = np.array([l.strip() for l in open(f'{DATA_}/gaston_multi/GW12/clusters.txt')])
        fib = np.isin(clusters, order)
    return counts, coords, fib

rows = []
for gw in ["GW6", "GW8", "GW10", "GW12", "GW16"]:
    counts, coords, fib = load_stage(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], [3, 90])
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys = coords[sel, 0], coords[sel, 1]
    Cs = counts[sel]
    step = SCALE_UM / UMPP
    gx = np.floor((xs - xs.min()) / step).astype(int)
    gy = np.floor((ys - ys.min()) / step).astype(int)
    key = gx * 100000 + gy
    uk, inv = np.unique(key, return_inverse=True)
    nps = len(uk)
    umi = np.zeros(nps)
    for i in range(nps):
        umi[i] = Cs[inv == i].sum()
    rows.append(dict(stage=gw, total_bins=nps,
        floor_d16=int((umi >= 16).sum()), floor_d64=int((umi >= 64).sum()),
        floor_d128=int((umi >= 128).sum()), floor_d256=int((umi >= 256).sum()),
        floor_d512=int((umi >= 512).sum()), median_umi=int(np.median(umi))))

df = pd.DataFrame(rows)
df.to_csv(f"{RES_}/stage_feasible_counts.csv", index=False)
print(df.to_string(index=False))
