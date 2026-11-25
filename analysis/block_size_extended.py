"""Supplementary analysis S2b: does the depth-gain interval depend on the spatial-block
size used for resampling?

Fig 2b reports E_depth with a 95% interval from a block bootstrap whose block edge is
800 um; that edge is an analysis choice. Here the same out-of-fold predictions are
resampled with blocks of 50 / 100 / 200 / 400 / 800 / 1600 um, so the only thing that
changes between runs is the resampling unit. Blocks of 50/100/200 um are smaller than
most pseudospots, so resampling at these sizes approaches resampling pseudospots
individually, which ignores spatial correlation between them. If the interval width is
sensitive to the resampling unit, it should narrow as the block shrinks, and some
intervals may come to exclude zero.

build_oof() is run once per stage and shared by all block sizes, so the comparison
isolates the block-size effect from the stochastic thinning inside build_oof.

Output: results/block_size_extended.csv   (per-draw values)
        results/block_size_extended_summary.csv       (E_depth uses depth>=64,
                                              E_scale uses scale>=200)
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np, pandas as pd, math, os

base = DATA_
os.chdir(HERE_)   # switch into this folder so the helper scripts below can be loaded by relative path

# reuse the loaders, build_oof() and I_from() defined in multiGW_bootstrap.py
src = open("multiGW_bootstrap.py").read().split("rows=[]")[0]
exec(src)

BLOCKS = [50, 100, 200, 400, 800, 1600]
NBOOT = 500
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]

rows = []
for gw in STAGES:
    oof, T_um = build_oof(gw)
    cells = list(oof.keys())
    if not cells:
        print(gw, "no cells", flush=True)
        continue
    for BLK in BLOCKS:
        brng = np.random.default_rng(1000 + BLK)      # own stream per block size
        # pre-compute block labels for every cell at this block size
        lab = {}
        for (s, d) in cells:
            o = oof[(s, d)]
            lab[(s, d)] = (np.floor((o["cx"] - o["cx"].min()) / (BLK / UMPP)).astype(int) * 1000
                           + np.floor((o["cy"] - o["cy"].min()) / (BLK / UMPP)).astype(int))
        nblk = len(np.unique(lab[cells[0]]))
        for bi in range(NBOOT):
            for (s, d) in cells:
                blk = lab[(s, d)]
                ub = np.unique(blk)
                samp = brng.choice(ub, len(ub), replace=True)
                idx = np.concatenate([np.where(blk == b)[0] for b in samp])
                I0 = I_from(oof[(s, d)]["pos"], oof[(s, d)]["yhat"], idx)
                for kind, key in (("E_depth", (s, 2 * d)), ("E_scale", (2 * s, d))):
                    if key not in oof:
                        continue
                    o2 = oof[key]
                    blk2 = lab[key]
                    # at small block sizes many resampled blocks are absent from the
                    # paired (deeper / coarser) cell, so the match list can be empty
                    parts = [np.where(blk2 == b)[0] for b in samp if (blk2 == b).any()]
                    if not parts:
                        continue
                    idx2 = np.concatenate(parts)
                    if len(idx2) <= 10:
                        continue
                    rows.append(dict(stage=gw, block_um=BLK, kind=kind, scale_um=s, depth=d,
                                     boot=bi, val=I_from(o2["pos"], o2["yhat"], idx2) - I0))
        print(f"{gw} block={BLK} done (n_blocks={nblk})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{RES_}/block_size_extended.csv", index=False)

summ = []
for gw in STAGES:
    for BLK in BLOCKS:
        for kind in ["E_depth", "E_scale"]:
            m = (df.stage == gw) & (df.block_um == BLK) & (df.kind == kind)
            m &= (df.depth >= 64) if kind == "E_depth" else (df.scale_um >= 200)
            v = df.loc[m, "val"]
            if len(v) == 0:
                continue
            summ.append(dict(stage=gw, block_um=BLK, kind=kind,
                             median=round(float(v.median()), 4),
                             lo=round(float(np.percentile(v, 2.5)), 4),
                             hi=round(float(np.percentile(v, 97.5)), 4),
                             pct_pos=round(float((v > 0).mean()), 3), n=int(len(v))))
sdf = pd.DataFrame(summ)
sdf.to_csv(f"{RES_}/block_size_extended_summary.csv", index=False)
print("BLOCK_DONE")
print(sdf[sdf.kind == "E_depth"].to_string(index=False))
