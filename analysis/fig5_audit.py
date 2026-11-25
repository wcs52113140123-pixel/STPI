#!/usr/bin/env python3
"""Fig 5 and Fig S5: the audit of 18 published Visium sections.

Writes
  results/fieldwide_audit.csv   random-CV and spatial-block estimates
  results/fig5_buffered_cv_all.csv           spatial block CV with buffers of 0-4 pitches

Per section: interiority coordinate (no expression), 2,000 most variable features,
5 random folds (seed 0) and the 4 x 4 checkerboard, PCA <= 30 + ridge (alpha 1).
Sections and their readers are listed in sections_fig5.csv. About 3 minutes on 9 cores.
"""
import numpy as np
import pandas as pd
from multiprocessing import Pool

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))   # works under python -P too
from _paths import RESULTS, data, ROOT
import stpi
from stpi import io

BUFFERS_PITCH = [0, 1, 2, 3, 4]
READ = {"geo_mtx": io.read_geo_mtx, "geo_h5": io.read_geo_h5, "spaceranger": io.read_spaceranger}


def category(I_random, I_checker):
    if I_random <= 0:
        return "no-signal (I_random<=0; CV drop undefined)"
    if I_checker < 0:
        return "collapse (checkerboard CV drives I<0; strong spatial confounding)"
    frac = (I_random - I_checker) / I_random
    if frac >= 0.5: return f"high inflation ({frac:.2f})"
    if frac >= 0.2: return f"moderate inflation ({frac:.2f})"
    return f"low inflation ({frac:.2f})"


def one(row):
    M, xy = READ[row.reader](data(row.folder), row.key)
    R = stpi.interiority(xy)
    X = stpi.log_hvg(M)
    folds = stpi.checkerboard_folds(xy)
    rnd = stpi.recoverability(X, R, stpi.random_folds(len(R), 5, 0))
    chk = stpi.recoverability(X, R, folds)
    var = float(np.var(R))
    r2 = 1 - float(np.mean((R - chk.prediction) ** 2)) / var
    aud = dict(sample=row["sample"], n=len(R), I_random=round(rnd.I, 4), I_checker=round(chk.I, 4),
               inflation=round(rnd.I - chk.I, 4), rho_random=round(rnd.rho, 4),
               rho_checker=round(chk.rho, 4), R2_checker=round(r2, 4),
               frac_artifact=round((rnd.I - chk.I) / rnd.I, 4) if rnd.I > 1e-6 else np.nan,
               tissue=row.tissue, I_random_pos=rnd.I > 0,
               frac_valid=bool(rnd.I > 0 and chk.I >= 0),
               category=category(rnd.I, chk.I),
               frac_valid_val=(rnd.I - chk.I) / rnd.I if (rnd.I > 0 and chk.I >= 0) else np.nan)
    pitch = stpi.spot_pitch(xy)
    span = float(np.mean([np.ptp(xy[:, 0]), np.ptp(xy[:, 1])]))
    buf = []
    for k in BUFFERS_PITCH:
        r = stpi.recoverability(X, R, folds, xy, k * pitch)
        buf.append(dict(tissue=row.tissue, sample=row["sample"], n=len(R), buffer_pitch=k,
                        buffer_units=round(k * pitch, 1), pitch=round(pitch, 1),
                        buffer_frac_of_span=round(k * pitch / span, 4),
                        I_bit=round(r.I, 4), rho=round(r.rho, 4),
                        frac_train_dropped=round(r.frac_dropped, 4)))
    print(f"{row["sample"]:38s} I_random={rnd.I:+.4f} I_checker={chk.I:+.4f}", flush=True)
    return aud, buf


if __name__ == "__main__":
    man = pd.read_csv(ROOT / "analysis" / "sections_fig5.csv")
    with Pool(min(9, len(man))) as p:
        out = p.map(one, [r for _, r in man.iterrows()])
    aud = pd.DataFrame([a for a, _ in out])
    aud = aud.iloc[np.argsort(man.audit_order.values)]      # sorts rows into the order used in the published table
    aud.to_csv(RESULTS / "fieldwide_audit.csv", index=False)
    pd.DataFrame([b for _, bb in out for b in bb]).to_csv(RESULTS / "fig5_buffered_cv_all.csv", index=False)
    print("wrote fieldwide_audit.csv, fig5_buffered_cv_all.csv")
