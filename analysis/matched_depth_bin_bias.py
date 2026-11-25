"""Does the C-threshold in the depth-matched control retain a spatially biased subset?

At 50 um with C = 64 only 64% (GW12) / 58% (GW16) of bins clear the threshold, so the
matched comparison runs on the denser bins. If bin density covaries with the radial
coordinate R, that subset is biased and the "fair race" in Fig 2d is compromised.

Test: for every stage x scale, compare the R distribution of retained (umi >= C) vs
discarded bins - mean R, coverage of the full R range, and a two-sample KS test.

Output: results/matched_depth_bin_bias.csv
  stage, scale_um, C, n_keep, n_drop, R_mean_keep, R_mean_drop, R_p05_keep, R_p95_keep,
  R_p05_all, R_p95_all, cover_frac, ks_stat, ks_p
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

base = DATA_
src = open(f"{HERE_}/multiGW_scan.py").read().split("def decode_I")[0]
exec(src)

UMPP = 0.5
SCALES = [50, 100, 200, 400, 800]
CS = [32, 64, 128]
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]

rows = []
for gw in STAGES:
    counts, coords, fib = load_stage(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], [3, 90])
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys = coords[sel, 0], coords[sel, 1]
    Cs = counts[sel].astype(np.float64)
    rs = r[sel]
    for s_um in SCALES:
        step = s_um / UMPP
        gx = np.floor((xs - xs.min()) / step).astype(int)
        gy = np.floor((ys - ys.min()) / step).astype(int)
        key = gx * 100000 + gy
        uk, inv = np.unique(key, return_inverse=True)
        nps = len(uk)
        cnt = np.bincount(inv, minlength=nps).astype(float); cnt[cnt == 0] = 1
        Rp = np.bincount(inv, weights=rs, minlength=nps) / cnt
        umi = np.zeros(nps)
        np.add.at(umi, inv, Cs.sum(1))
        # normalise R to [0,1] on this stage/scale, as the decoder does
        pos = (Rp - Rp.min()) / (np.ptp(Rp) + 1e-9)
        for C in CS:
            keep = umi >= C
            if keep.sum() < 10 or (~keep).sum() < 5:
                rows.append(dict(stage=gw, scale_um=s_um, C=C, n_keep=int(keep.sum()),
                                 n_drop=int((~keep).sum()), R_mean_keep=np.nan,
                                 R_mean_drop=np.nan, R_p05_keep=np.nan, R_p95_keep=np.nan,
                                 R_p05_all=np.nan, R_p95_all=np.nan, cover_frac=np.nan,
                                 ks_stat=np.nan, ks_p=np.nan))
                continue
            a, b = pos[keep], pos[~keep]
            ks = ks_2samp(a, b)
            p05k, p95k = np.percentile(a, [5, 95])
            p05a, p95a = np.percentile(pos, [5, 95])
            rows.append(dict(stage=gw, scale_um=s_um, C=C,
                             n_keep=int(keep.sum()), n_drop=int((~keep).sum()),
                             R_mean_keep=round(float(a.mean()), 4),
                             R_mean_drop=round(float(b.mean()), 4),
                             R_p05_keep=round(float(p05k), 4), R_p95_keep=round(float(p95k), 4),
                             R_p05_all=round(float(p05a), 4), R_p95_all=round(float(p95a), 4),
                             cover_frac=round(float((p95k - p05k) / (p95a - p05a + 1e-9)), 4),
                             ks_stat=round(float(ks.statistic), 4),
                             ks_p=float(ks.pvalue)))
    print(gw, "done", flush=True)

pd.DataFrame(rows).to_csv(f"{RES_}/matched_depth_bin_bias.csv", index=False)
print("written results/matched_depth_bin_bias.csv")
