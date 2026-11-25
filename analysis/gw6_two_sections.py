"""Writes results/gw6_two_sections.npz."""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_, DD_   # data, derived inputs, results, this folder
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

base = DATA_

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
    r = ((np.column_stack([x - clx[idx], y - cly[idx]])) * nrm[idx]).sum(1)
    return r

fibset = ["Fibro-pia-pro", "Fibro-others"]
GW6 = {}
for R in ["R1", "R2"]:
    m = pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_{R}_meta.csv")
    fib = m["celltype"].isin(fibset).values
    x, y = m["x"].values.astype(float), m["y"].values.astype(float)
    rr = radial_coord_full(x, y)
    lo, hi = np.percentile(rr[fib], [3, 90])
    sel = fib & (rr >= lo) & (rr <= hi)
    xs, ys, rs = x[sel], y[sel], rr[sel]
    Rn = (rs - rs.min()) / (np.ptp(rs) + 1e-9)
    GW6[R] = dict(xs=xs, ys=ys, Rn=Rn)

np.savez(f"{DD_}/gw6_two_sections.npz",
         S1_xs=GW6["R1"]["xs"], S1_ys=GW6["R1"]["ys"], S1_Rn=GW6["R1"]["Rn"],
         S2_xs=GW6["R2"]["xs"], S2_ys=GW6["R2"]["ys"], S2_Rn=GW6["R2"]["Rn"])
print("saved GW6 two-section coords | S1 n=%d S2 n=%d" % (len(GW6["R1"]["xs"]), len(GW6["R2"]["xs"])))
