#!/usr/bin/env python3
"""Supplementary data for Figure S5, which extends Figure 5 with one additional diagnostic
panel per main panel.

S5 is one-to-one with Figure 5: each panel examines a specific question about the
corresponding main panel. Three of the five panels require additional data, produced here.
The loaders, geometric coordinate, feature matrix, PCA+Ridge decoder and 4x4 checkerboard
folds are imported from fig5_buffered_all.py and used unchanged. Only the quantity under
test varies in each block below.

  (1) all_maps        S5b extends Fig 5b, which displays 5 of the 18 sections. This block
                      decodes all 18 sections (fig5_pred_maps.npz holds only five), using the
                      same decode as Fig 5 (checkerboard folds, buffer 0), and keeps the
                      held-out predictions for every section.

  (2) block size      S5c extends Fig 5c, whose quantity is the share of the random-CV
                      reading that spatial block CV removes. This share is computed on a 4x4
                      checkerboard; here it is recomputed on 3x3 and 5x5 checkerboards to show
                      it is not a property of one particular grid size.

  (3) deletion        S5e extends Fig 5e, the widening buffer. The buffer does two things at
                      once: it separates training from test spots AND it deletes training
                      spots (5 / 35 / 65 % at the three rungs). This block tests whether the
                      decline comes from the deletion alone by removing the same share of
                      training spots AT RANDOM, with no spatial gap, and comparing the result
                      to the buffered reading.
                      Matched per section and per rung to that section's own measured
                      frac_train_dropped, not to the median, and averaged over N_DRAWS draws.

  (4) covariates      S5d extends Fig 5d, which tests the removed share against the apparent
                      reading and against spot count. Three more section-level covariates are
                      measured here: spot pitch, section extent, and sequencing depth (median
                      counts per spot).

Seeding: SHA-256 of the sample name, not Python's hash(), since the latter is randomised per
process and would make runs irreproducible (see fig4_coord_transfer.py).

Writes results/fig5_pred_maps_all.npz, fig5_blocksize_sensitivity.csv,
results/fig5_deletion_control.csv, results/fig5_section_covariates.csv
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import hashlib
import importlib.util
import os
import sys

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

R = f"{RES_}"
os.chdir(HERE_)   # change into the project folder so relative paths to helper scripts resolve

_spec = importlib.util.spec_from_file_location("f5b", f"{HERE_}/fig5_buffered_all.py")
m = importlib.util.module_from_spec(_spec)
sys.modules["f5b"] = m
_spec.loader.exec_module(m)

TIS = ["kidney", "intestine", "cerebellum", "retina", "finch_retina"]
N_DRAWS = 5                      # random-deletion draws per section and rung
BLOCKS = (3, 4, 5)               # checkerboard grids compared for sensitivity; 4x4 is the grid used in Fig 5


def seed_of(name, salt=""):
    return int(hashlib.sha256((name + salt).encode()).hexdigest()[:8], 16)


def folds_of(xy, k):
    """k x k checkerboard parity folds, generalising the block-CV grid to arbitrary k."""
    bx = pd.cut(xy[:, 0], k, labels=False).astype(int)
    by = pd.cut(xy[:, 1], k, labels=False).astype(int)
    return (bx + by) % 2


def load(tissue, kind, key):
    if kind == "finch":
        return m.load_finch(m.DIRS[tissue], key)
    if kind == "h5":
        return m.load_h5(m.DIRS[tissue], key)
    return m.load_mtx(m.DIRS[tissue], key)


def decode_core(X, pos, folds, xy, rad=0.0, rng=None, drop_frac=0.0):
    """Reimplements fig5_buffered_all.decode_buffered with two additions: the held-out
    prediction vector is returned (decode_buffered discards it, and S5b needs it), and
    training spots can optionally be deleted at random, fold by fold, at the same point
    where the buffer deletes training spots by distance. This isolates the effect of the
    deletion from the effect of the spatial gap, since the two procedures then differ only
    in which training spots are removed.

    With rad = 0 and drop_frac = 0 the returned I matches decode_buffered's output to
    within 1e-9 (checked for every section below).
    """
    yhat = np.zeros_like(pos)
    dropped = 0
    n_tr = 0
    for b in np.unique(folds):
        te = folds == b
        tr = ~te
        if rad > 0 and te.sum():
            d, _ = cKDTree(xy[te]).query(xy, k=1)
            tr = tr & (d > rad)
        if drop_frac > 0 and tr.sum():
            idx = np.where(tr)[0]
            kill = rng.choice(idx, size=int(round(drop_frac * len(idx))), replace=False)
            tr = tr.copy()
            tr[kill] = False
        dropped += int((~te).sum() - tr.sum())
        n_tr += int((~te).sum())
        if te.sum() < 5 or tr.sum() < 20:
            yhat[te] = pos[tr].mean() if tr.sum() else pos.mean()
            continue
        npc = min(30, X.shape[1], tr.sum() - 1)
        pc = PCA(n_components=npc, svd_solver="randomized", random_state=0).fit(X[tr])
        yhat[te] = Ridge(alpha=1.0).fit(pc.transform(X[tr]), pos[tr]).predict(pc.transform(X[te]))
    resid = pos - yhat
    mse = float(np.mean(resid ** 2))
    var = float(np.var(pos))
    I = 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0
    return I, float(spearmanr(pos, yhat)[0]), yhat, dropped / max(n_tr, 1)


BUF = pd.read_csv(f"{R}/fig5_buffered_cv_all.csv")
maps, rows_bs, rows_del, rows_cov = {}, [], [], []
for tissue, sample, kind, key in m.manifest():
    Mx, xy = load(tissue, kind, key)
    coord = m.geom_interiority(xy)
    X = m.features(Mx)
    pitch = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))

    # (1) decode the field with the same procedure as Fig 5, keeping the held-out prediction
    folds4 = folds_of(xy, 4)
    I4, rho4, yhat, _ = decode_core(X, coord, folds4, xy, 0.0)
    I_ref, rho_ref, _ = m.decode_buffered(X, coord, folds4, xy, 0.0)
    assert abs(I4 - I_ref) < 1e-9, f"{sample}: replay {I4} != reference {I_ref}"
    maps[f"{sample}_xy"] = xy
    maps[f"{sample}_true"] = coord
    maps[f"{sample}_pred"] = yhat
    maps[f"{sample}_meta"] = np.array([I4, rho4, len(xy), pitch], float)

    # (2) the same quantity recomputed under 3x3 and 5x5 checkerboards
    for k in BLOCKS:
        Ik, rhok, _, _ = (I4, rho4, None, None) if k == 4 else decode_core(
            X, coord, folds_of(xy, k), xy, 0.0)
        rows_bs.append(dict(tissue=tissue, sample=sample, n=len(xy), blocks=k,
                            I_checker=round(float(Ik), 4), rho=round(float(rhok), 4)))

    # (3) random deletion matched to this section's own buffer drop share, per rung
    q = BUF[BUF["sample"] == sample].set_index("buffer_pitch")
    rng = np.random.default_rng(seed_of(sample, "del"))
    for rung in (1, 2, 4):
        if rung not in q.index:
            continue
        frac = float(q.loc[rung, "frac_train_dropped"])
        vals = [decode_core(X, coord, folds4, xy, 0.0, rng=rng, drop_frac=frac)[0]
                for _ in range(N_DRAWS)]
        rows_del.append(dict(tissue=tissue, sample=sample, buffer_pitch=rung,
                             frac_dropped=round(frac, 4),
                             I_buffered=round(float(q.loc[rung, "I_bit"]), 4),
                             I_random_deletion=round(float(np.median(vals)), 4),
                             I_rd_lo=round(float(np.min(vals)), 4),
                             I_rd_hi=round(float(np.max(vals)), 4), n_draws=N_DRAWS))

    # (4) section-level covariates for S5d
    counts = np.asarray(Mx.sum(axis=1)).ravel()
    genes = np.asarray((Mx > 0).sum(axis=1)).ravel()
    ext = float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    rows_cov.append(dict(tissue=tissue, sample=sample, n=len(xy), pitch=round(pitch, 2),
                         extent=round(ext, 1), area_pitch2=round(len(xy) * pitch ** 2 / 1e6, 3),
                         median_counts=round(float(np.median(counts)), 1),
                         median_genes=round(float(np.median(genes)), 1),
                         n_features=int(X.shape[1])))
    print(f"{sample:38s} I(4x4)={I4:+.4f} rho={rho4:+.3f} pitch={pitch:.0f}", flush=True)

np.savez_compressed(f"{R}/fig5_pred_maps_all.npz", **maps)
pd.DataFrame(rows_bs).to_csv(f"{R}/fig5_blocksize_sensitivity.csv", index=False)
pd.DataFrame(rows_del).to_csv(f"{R}/fig5_deletion_control.csv", index=False)
pd.DataFrame(rows_cov).to_csv(f"{R}/fig5_section_covariates.csv", index=False)

BS = pd.DataFrame(rows_bs).pivot_table(index="sample", columns="blocks", values="I_checker")
print("\nblock size: median |I(k) - I(4)| =",
      {k: round(float((BS[k] - BS[4]).abs().median()), 4) for k in BLOCKS})
DL = pd.DataFrame(rows_del)
print("deletion control: buffered below random-deletion in",
      int((DL.I_buffered < DL.I_random_deletion).sum()), "of", len(DL), "section-rungs")
print("WROTE 4 files")
