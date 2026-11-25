#!/usr/bin/env python3
"""Fig 5 and Fig S5 A: Moran spectral null for the 18 published sections.

Writes results/fig5_fieldwide_null_msr.csv (and the null draws, .npz).

200 surrogates of the interiority coordinate per section, each decorrelated from it
(|Spearman rho| < 0.30) and rank-matched to it; every surrogate is decoded under the
same 4 x 4 checkerboard as the observed coordinate. P = (#null >= observed + 1) / 201.
Seed: SHA-256 of the section name + "msr". About 4 minutes on 9 cores.
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))   # works under python -P too
from _paths import RESULTS, data, ROOT
import stpi
from stpi import io

N_DRAWS = 200
READ = {"geo_mtx": io.read_geo_mtx, "geo_h5": io.read_geo_h5, "spaceranger": io.read_spaceranger}


def one(row):
    M, xy = READ[row.reader](data(row.folder), row.key)
    R = stpi.interiority(xy)
    X = stpi.log_hvg(M)
    folds = stpi.checkerboard_folds(xy)
    S, info = stpi.msr_surrogates(R, xy, n_draws=N_DRAWS, seed=stpi.seed_of(row["sample"] + "msr"))
    obs = stpi.recoverability(X, R, folds).I
    null = np.array([stpi.recoverability(X, s, folds).I for s in S])
    rec = dict(tissue=row.tissue, sample=row["sample"], n=len(xy), I_obs=round(float(obs), 4),
               null_median=round(float(np.median(null)), 4),
               null_p95=round(float(np.quantile(null, 0.95)), 4),
               P=round(stpi.null_p(obs, null), 4),
               moran_obs=round(info["moran_obs"], 4),
               moran_surr_med=round(float(np.median(info["moran"])), 4),
               rho_med=round(float(np.median(info["rho"])), 4),
               accept_rate=round(N_DRAWS / info["tried"], 3), clamped=0.0, n_draws=N_DRAWS)
    print(f"{row["sample"]:38s} I={obs:+.4f} P={rec['P']:.4f}", flush=True)
    return rec, null


if __name__ == "__main__":
    man = pd.read_csv(ROOT / "analysis" / "sections_fig5.csv")
    man = man.sort_values("audit_order")                       # process sections in a fixed order for reproducibility
    with Pool(min(9, len(man))) as p:
        out = p.map(one, [r for _, r in man.iterrows()])
    pd.DataFrame([r for r, _ in out]).to_csv(RESULTS / "fig5_fieldwide_null_msr.csv", index=False)
    np.savez_compressed(RESULTS / "fig5_fieldwide_null_msr.npz",
                        **{f"{r['sample']}_msr": n for r, n in out})
    print("wrote fig5_fieldwide_null_msr.csv")
