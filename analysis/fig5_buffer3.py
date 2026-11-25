#!/usr/bin/env python3
"""Compute the buffer_pitch = 3 rung for the buffered sweep (Fig 5e) and its matched
random-deletion control (Fig S5e).

BUFFERS_PITCH in fig5_buffered_all.py is [0, 1, 2, 4], a doubling ladder, so on the
linear x axis of both panels the position for 3 is empty; this script fills that gap.

decode_buffered takes no rng, so computing pitch = 3 on its own returns the same value
it would have as part of a full sweep over [0, 1, 2, 3, 4]; the other rungs are not
recomputed. The deletion control draws from a separate random stream (salt "del3"), so
inserting rung 3 does not change the draws used for rungs 0, 1, 2 and 4.

Outputs
  results/fig5_buffer3.csv              the new rung only, both arms
  results/fig5_buffered_cv_all.csv      buffered-sweep table with rung 3 merged in
  results/fig5_deletion_control.csv     deletion-control table with rung 3 merged in
"""
import os, sys, importlib.util
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import RES_
R = RES_

spec = importlib.util.spec_from_file_location("fig5_buffered_all", f"{HERE}/fig5_buffered_all.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)                      # main() is __main__-guarded, so nothing runs

# the support script's helpers (decode_core takes drop_frac + rng); exec only the prefix
# that defines them, stopping before its own analysis loop
src = open(f"{HERE}/figS5_support.py").read()
ns = {"__name__": "figS5_support_prefix"}
exec(src[:src.index('BUF = pd.read_csv')], ns)
decode_core, folds_of, seed_of, load, N_DRAWS = (ns["decode_core"], ns["folds_of"],
                                                 ns["seed_of"], ns["load"], ns["N_DRAWS"])
RUNG = 3

rows = []
for tissue, sample, kind, key in m.manifest():
    Mx, xy = load(tissue, kind, key)
    coord = m.geom_interiority(xy)
    X = m.features(Mx)
    folds4 = folds_of(xy, 4)
    pitch = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    span = float(np.mean([np.ptp(xy[:, 0]), np.ptp(xy[:, 1])]))

    I, rho, frac_drop = m.decode_buffered(X, coord, folds4, xy, RUNG * pitch)
    rng = np.random.default_rng(seed_of(sample, "del3"))
    vals = [decode_core(X, coord, folds4, xy, 0.0, rng=rng, drop_frac=frac_drop)[0]
            for _ in range(N_DRAWS)]
    rows.append(dict(tissue=tissue, sample=sample, n=len(coord), buffer_pitch=RUNG,
                     buffer_units=round(RUNG * pitch, 1), pitch=round(pitch, 1),
                     buffer_frac_of_span=round(RUNG * pitch / span, 4),
                     I_bit=round(float(I), 4), rho=round(float(rho), 4),
                     frac_train_dropped=round(float(frac_drop), 4),
                     I_random_deletion=round(float(np.median(vals)), 4),
                     I_rd_lo=round(float(np.min(vals)), 4),
                     I_rd_hi=round(float(np.max(vals)), 4), n_draws=N_DRAWS))
    print(f"{sample:38s} buf=3 pitch  I={I:+.4f}  rho={rho:+.4f}  dropped={frac_drop:.1%}  "
          f"random-deletion median={np.median(vals):+.4f}", flush=True)

new = pd.DataFrame(rows)
new.to_csv(f"{R}/fig5_buffer3.csv", index=False)

BUFP, DELP = f"{R}/fig5_buffered_cv_all.csv", f"{R}/fig5_deletion_control.csv"

BUF = pd.read_csv(BUFP)
BUF = BUF[BUF.buffer_pitch != RUNG]
BUF = pd.concat([BUF, new[BUF.columns]], ignore_index=True)
BUF = BUF.sort_values(["tissue", "sample", "buffer_pitch"]).reset_index(drop=True)
BUF.to_csv(BUFP, index=False)

DEL = pd.read_csv(DELP)
DEL = DEL[DEL.buffer_pitch != RUNG]
cols = DEL.columns
add = new.rename(columns={"frac_train_dropped": "frac_dropped", "I_bit": "I_buffered"})
DEL = pd.concat([DEL, add[cols]], ignore_index=True)
DEL = DEL.sort_values(["tissue", "sample", "buffer_pitch"]).reset_index(drop=True)
DEL.to_csv(DELP, index=False)

print("\nWROTE fig5_buffer3.csv", new.shape, "| buffered table", BUF.shape, "| deletion table", DEL.shape)
print("buffered rungs:", sorted(BUF.buffer_pitch.unique()), "| deletion rungs:", sorted(DEL.buffer_pitch.unique()))
print(new.groupby("tissue")[["I_bit", "I_random_deletion", "frac_train_dropped"]].median().round(3).to_string())
