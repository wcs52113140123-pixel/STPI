"""Fig2 f: depth-matched spatial scale scan across all five developmental stages (GW6..GW16).

Applies the same estimand and pipeline used for the GW16-only depth-matched scan, extended here to
all five stages so that the spatial scale optimum can be checked for consistency across stages
rather than relying on a single stage.

Design: at each spatial scale s, downsample every eligible pseudospot to the same absolute per-bin
count C (matched depth). This removes the confound that coarser bins naturally have higher depth.
If the scale optimum s* persists under matched depth, it reflects a real geometric component; if it
flattens, the apparent optimum was a depth-ceiling artifact.

Output: results/depth_matched_sscan_5stage.csv
  stage, scale_um, C, n_elig, retention, mean_frac_discarded, n_pseudospots,
  med_umi, I, rho, feasible
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
# random_state=0 fixes the PCA/ridge fit and multinomial resampling draws for reproducibility
import numpy as np, pandas as pd, math, scipy.io as sio
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

rng = np.random.default_rng(0)
base = DATA_
order = ["Fibroblast_Pia", "Fibroblast_Arachnoid", "Fibroblast_Dura"]
UMPP = 0.5
SCALES_UM = [50, 100, 200, 400, 800]
MATCHED_C = [32, 64, 128]
NREP = 20
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]


def load_stage(gw):
    """Loads the counts matrix, spatial coordinates and fibroblast mask for the given stage."""
    if gw == "GW16":
        d = f"{DATA_}/gaston_gw16"
        counts = np.load(f"{d}/counts_mat.npy")
        coords = np.load(f"{d}/coords_mat.npy")
        clusters = np.load(f"{d}/clusters_kept.npy", allow_pickle=True)
        keep = counts.sum(1) > 0
        counts, coords, clusters = counts[keep], coords[keep], clusters[keep]
        fib = np.isin(clusters, order)
    elif gw == "GW6":
        m = pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv")
        c = sio.mmread(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_counts.mtx")
        counts = np.asarray(c.todense()).T
        coords = m[["x", "y"]].values.astype(float)
        fib = np.isin(m["celltype"].values, ["Fibro-pia-pro", "Fibro-others"])
    else:
        d = f"{DATA_}/gaston_multi/{gw}"
        counts = np.asarray(sio.mmread(f"{d}/counts.mtx").T.todense(), dtype=np.float64)
        coords = np.loadtxt(f"{d}/coords.tsv")
        clusters = np.array([l.strip() for l in open(f"{d}/clusters.txt")])
        fib = np.isin(clusters, order)
    return counts, coords, fib


def radial_coord_full(x, y, nbin=60, ksm=5):
    """Computes a geometry-only signed radial coordinate (uses only spatial coordinates, no expression data)."""
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


def decode_I(X, pos, blocks):
    """Checkerboard-CV ridge decoder; returns I (bit), R2, and Spearman rho."""
    yhat = np.zeros_like(pos)
    for b in np.unique(blocks):
        te = blocks == b
        tr = ~te
        if te.sum() < 5 or tr.sum() < 20:
            yhat[te] = pos[tr].mean()
            continue
        npc = min(30, X.shape[1], tr.sum() - 1)
        pc = PCA(n_components=npc, random_state=0).fit(X[tr])
        m = Ridge(alpha=1.0).fit(pc.transform(X[tr]), pos[tr])
        yhat[te] = m.predict(pc.transform(X[te]))
    mse = np.mean((pos - yhat) ** 2)
    var = np.var(pos)
    I = 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0
    return I, (1 - mse / var), spearmanr(pos, yhat)[0]


rows = []
for gw in STAGES:
    counts, coords, fib = load_stage(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], [3, 90])
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys = coords[sel, 0], coords[sel, 1]
    Cs = counts[sel]
    rs = r[sel]
    for s_um in SCALES_UM:
        step = s_um / UMPP
        gx = np.floor((xs - xs.min()) / step).astype(int)
        gy = np.floor((ys - ys.min()) / step).astype(int)
        key = gx * 100000 + gy
        uk, inv = np.unique(key, return_inverse=True)
        nps = len(uk)
        agg = np.zeros((nps, Cs.shape[1]))
        Rp = np.zeros(nps); cx = np.zeros(nps); cy = np.zeros(nps)
        for i in range(nps):
            m = inv == i
            agg[i] = Cs[m].sum(0); Rp[i] = rs[m].mean()
            cx[i] = xs[m].mean(); cy[i] = ys[m].mean()
        umi = agg.sum(1)
        med_umi = float(np.median(umi))
        for C in MATCHED_C:
            elig = umi >= C
            n_elig = int(elig.sum())
            retention = n_elig / nps
            disc = 1 - (C / umi[elig])
            disc = disc[disc > 0]
            mean_disc = float(np.mean(disc)) if len(disc) else 0.0
            base_row = dict(stage=gw, scale_um=s_um, C=C, n_elig=n_elig,
                            retention=round(retention, 3),
                            mean_frac_discarded=round(mean_disc, 3),
                            n_pseudospots=nps, med_umi=round(med_umi, 1))
            if n_elig < 100:
                rows.append({**base_row, "I": np.nan, "rho": np.nan, "feasible": False})
                continue
            A = agg[elig]; Rk = Rp[elig]; cxk = cx[elig]; cyk = cy[elig]
            pos = (Rk - Rk.min()) / (np.ptp(Rk) + 1e-9)
            nb = 4
            bx = pd.cut(cxk, nb, labels=False).astype(int)
            by = pd.cut(cyk, nb, labels=False).astype(int)
            cb = (bx + by) % 2
            Is = []; rhos = []
            for _ in range(NREP):
                At = np.zeros_like(A)
                for i in range(A.shape[0]):
                    tot = A[i].sum()
                    At[i] = rng.multinomial(C, A[i] / tot)
                lib = At.sum(1, keepdims=True)
                lib[lib == 0] = 1
                X = np.log1p(At / lib * np.median(lib))
                try:
                    I, _, rho = decode_I(X, pos, cb)
                    Is.append(I); rhos.append(rho)
                except Exception:
                    pass
            rows.append({**base_row,
                         "I": round(float(np.median(Is)), 4) if Is else np.nan,
                         "rho": round(float(np.median(rhos)), 4) if rhos else np.nan,
                         "feasible": bool(Is)})
        print(f"{gw} scale {s_um}um done: nps={nps} med_umi={med_umi:.0f}", flush=True)

df = pd.DataFrame(rows)
out = f"{RES_}/depth_matched_sscan_5stage.csv"
df.to_csv(out, index=False)
print("\n=== s* under matched depth, per stage ===")
for gw in STAGES:
    for C in MATCHED_C:
        sub = df[(df.stage == gw) & (df.C == C) & (df.feasible)]
        if len(sub):
            best = sub.loc[sub.rho.idxmax()]
            print(f"{gw} C={C}: s*={int(best.scale_um)}um rho={best.rho} "
                  f"(scales={list(sub.scale_um)})")
print("SAVED", out)
print("DONE")
