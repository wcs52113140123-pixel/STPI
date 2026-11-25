#!/usr/bin/env python3
"""Coordinate field vs decoded field, for the meninges (S1) and the tendon (S4).

Figure 5b draws, for each exemplar section, the geometric coordinate beside what expression
recovers of it. This pairing appears nowhere else in the paper: the meninges figures and the
tendon figures (fig4/figS4) draw the coordinate alone, with "decoded" appearing only in axis
labels and prose. This script produces the decoded field for both tissues so that it can be
added to the panel.

The loaders, coordinate, pseudospot binning, depth thinning, normalisation, checkerboard folds
and decoder are imported or replayed verbatim from multiGW_scan.py (meninges) and
fig4_coord_transfer.py (tendon). Only the held-out prediction vector, which those scripts
compute and discard, is kept; no additional tuning is applied.

Cell choice: equal settings across the panels of one figure.

  meninges   Two cell choices are exported, because they serve different purposes:
               <stage>_*        each stage at its own best cell, the cell whose reading the
                                main developmental figure reports (GW6 50/512, GW8 50/256,
                                GW10 200/512, GW12 200/512, GW16 800/512). This is what
                                figure S1f draws, so the pattern shown corresponds to the
                                number the paper quotes. The cells differ between columns,
                                so the panel is not a cross-stage comparison; that comparison
                                appears in the main figure under matched settings, and the
                                caption should say so.
               <stage>_common_* all five at 50 um / depth 512, the deepest cell feasible in
                                all five (they share only six feasible cells, all at 50 um).
                                Equal settings, kept as a check rather than drawn: at 50 um
                                the band is one to three pseudospots wide in the later stages
                                and GW12 keeps 140 pseudospots against 322 at its own best
                                cell.

  tendon     200 um pseudospots, depth 2048, the common cell of the tendon null
             (fig4_tendon_null.py), feasible in all four sections. Coordinate: the transferred
             meninges centreline, the paper's main tendon axis.

Validity check, written to the output rather than asserted here: for every map the reading
recomputed from the kept prediction vector is compared against the value the published grid
holds for the same cell, and the difference is written to the CSV. The meninges scan draws its
multinomial thinning from one process-wide generator consumed in grid order, which cannot be
reproduced from a single cell, so a small deviation is expected there and is reported rather
than hidden. It is within 0.007 bit at the common cell but reaches +0.069 bit at GW12's own
best cell (0.552 here against 0.482 in the grid), the paper's headline stage: quote the grid
value, never a number recomputed from these maps. The tendon path seeds per section exactly as
fig4_coord_transfer.py does and lands within 0.049 bit.

Rendering resolution, separate from the decoding settings. The reading is computed on
pseudospots, as everywhere in the paper, and nothing about that changes. But one marker per
pseudospot gives 140-954 markers over a thin band, which reads as sparse next to figure 5b,
where thousands of native spots tile a blob. So each native spot is additionally given the
decoded value of the pseudospot it belongs to and exported as <unit>_spot_{xy,val}: the field
drawn from those is piecewise constant over each pseudospot, no value is invented and no
smoothing is applied, but it carries the tissue at full density and with its true outline.
Spots whose pseudospot failed the depth filter get NaN and are not drawn.

Output: results/decoded_maps_meninges.npz   <stage>_{xy,coord,decoded,cell,spot_xy,spot_val}
        results/decoded_maps_tendon.npz     <section>_{xy,coord,decoded,spot_xy,spot_val}
        results/decoded_maps_check.csv      per map: I recomputed, I in the published grid
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import importlib.util
import math
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

R = f"{RES_}"
STAGES = ["GW6", "GW6-S2", "GW8", "GW10", "GW12", "GW16"]
MEN_CELL = (50, 512)          # the deepest cell feasible in all five stages
TEN_CELL = (200, 2048)        # the common cell of the tendon null
UMPP = 0.5
N_MIN = 100


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def checker_decode(X, pos, cb):
    """Checkerboard decode, returning the held-out prediction values that the grid scripts
    compute internally but do not save."""
    yhat = np.zeros_like(pos)
    for b in np.unique(cb):
        te = cb == b
        tr = ~te
        if te.sum() < 5 or tr.sum() < 20:
            yhat[te] = pos[tr].mean()
            continue
        pc = PCA(n_components=min(30, X.shape[1], tr.sum() - 1), random_state=0).fit(X[tr])
        yhat[te] = Ridge(alpha=1.0).fit(pc.transform(X[tr]), pos[tr]).predict(pc.transform(X[te]))
    return yhat


def I_of(pos, yhat):
    mse = float(np.mean((pos - yhat) ** 2))
    var = float(np.var(pos))
    return 0.5 * math.log2(var / mse) if mse > 0 and var > 0 else 0.0


def unroll_meninges(x, y, nbin=60, ksm=5):
    """Computes both quantities produced inside radial_coord_full: the signed across-band
    offset it returns, and the along-band arclength, which is a local variable inside that
    function and is not returned.

    The code here mirrors radial_coord_full statement for statement so that both quantities
    are available. The returned offset is checked against radial_coord_full's output
    pointwise to confirm the two agree.
    """
    cx, cy = np.median(x), np.percentile(y, 15)
    theta = np.arctan2(y - cy, x - cx)
    tb = pd.cut(theta, nbin, labels=False)
    cl = []
    for b in range(nbin):
        mb = tb == b
        if mb.sum() < 15:
            continue
        cl.append((np.median(theta[mb]), np.median(x[mb]), np.median(y[mb])))
    cl = np.array(cl); cl = cl[np.argsort(cl[:, 0])]
    sm = lambda a: np.convolve(a, np.ones(ksm) / ksm, mode="same")
    clx, cly = sm(cl[:, 1]), sm(cl[:, 2])
    tang = np.gradient(np.column_stack([clx, cly]), axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    ro = np.column_stack([clx - cx, cly - cy])
    ro /= np.linalg.norm(ro, axis=1, keepdims=True) + 1e-9
    fl = np.sign((nrm * ro).sum(1)); fl[fl == 0] = 1
    nrm = nrm * fl[:, None]
    node = np.column_stack([clx, cly])
    from scipy.spatial import cKDTree as _KDT
    _, idx = _KDT(node).query(np.column_stack([x, y]))
    off = ((np.column_stack([x - clx[idx], y - cly[idx]])) * nrm[idx]).sum(1)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(node, axis=0), axis=1))])
    along = arc[idx] + ((np.column_stack([x - clx[idx], y - cly[idx]])) * tang[idx]).sum(1)
    assert np.allclose(off, radial_coord_full(x, y, nbin=nbin, ksm=ksm)), \
        "replayed offset differs from radial_coord_full"
    return along, off


def unroll_tendon(x, y, n_knots=60, smooth=5):
    """Computes the same quantities for the tendon: mirrors principal_curve_coord's body to
    obtain the along-backbone arclength, which that function does not return, and checks the
    offset against principal_curve_coord's output."""
    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra
    from scipy.spatial import cKDTree as _KDT
    P = np.c_[x, y].astype(float)
    k = min(n_knots, max(8, len(P) // 50))
    C = KMeans(n_clusters=k, n_init=4, random_state=0).fit(P).cluster_centers_
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    T_ = minimum_spanning_tree(csr_matrix(D)); Tsym = T_ + T_.T
    d0 = dijkstra(Tsym, indices=0)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(Tsym, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    bb = C[np.array(path[::-1])]
    if len(bb) >= smooth:
        w = np.ones(smooth) / smooth
        bb = np.c_[np.convolve(bb[:, 0], w, "valid"), np.convolve(bb[:, 1], w, "valid")]
    seg = np.diff(bb, axis=0)
    L = np.linalg.norm(seg, axis=1) + 1e-12
    tang = seg / L[:, None]
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    mid = (bb[:-1] + bb[1:]) / 2
    _, si = _KDT(mid).query(P)
    off = ((P - bb[:-1][si]) * nrm[si]).sum(1)
    arc = np.concatenate([[0.0], np.cumsum(L)])
    along = arc[si] + ((P - bb[:-1][si]) * tang[si]).sum(1)
    return along, off


# ───────────────────────── meninges ─────────────────────────
# multiGW_scan.py is a flat script: importing it would re-run the whole grid. Its two reusable
# functions are read out of it by exec'ing only their definitions.
_src = open(f"{HERE_}/multiGW_scan.py").read()
_head = _src[:_src.index("UMPP=0.5")]
_ns = {}
exec(compile(_head, "multiGW_scan_head", "exec"), _ns)
load_stage, radial_coord_full = _ns["load_stage"], _ns["radial_coord_full"]


def load_stage_ext(gw):
    """load_stage plus the second GW6 section, which multiGW_scan.py does not decode.

    multiGW_scan.py's GW6 branch reads GW6_Stereo_R1 only, so multiGW_grid_all.csv has one
    GW6 row, for R1. Figure 1b draws both GW6 sections, so the decoded panel does too. R2 is
    loaded here with the same files, the same cell-type filter and the same coordinate; no
    retuning is applied, but it has no published grid value: I_published is empty for it in
    decoded_maps_check.csv, and the caption should mark it as a section decoded for this panel,
    at R1's cell.
    """
    if gw != "GW6-S2":
        return load_stage(gw)
    import scipy.io as _sio
    d = f"{DATA_}/gw6_annotated/expr"
    m = pd.read_csv(f"{d}/GW6_Stereo_R2_meta.csv")
    counts = np.asarray(_sio.mmread(f"{d}/GW6_Stereo_R2_counts.mtx").todense()).T
    coords = m[["x", "y"]].values.astype(float)
    fib = np.isin(m["celltype"].values, ["Fibro-pia-pro", "Fibro-others"])
    return counts, coords, fib

G = pd.read_csv(f"{R}/multiGW_grid_all.csv")
GF = G[G.feasible]
BEST = {gw: (int(GF[GF.stage == gw].loc[GF[GF.stage == gw].I.idxmax()].scale_um),
             int(GF[GF.stage == gw].loc[GF[GF.stage == gw].I.idxmax()].depth))
        for gw in STAGES if gw in set(GF.stage)}
BEST["GW6-S2"] = BEST["GW6"]      # the second GW6 section is drawn at R1's cell

rows, men = [], {}
for gw in STAGES:
    counts, coords, fib = load_stage_ext(gw)
    r = radial_coord_full(coords[:, 0], coords[:, 1])
    lo, hi = np.percentile(r[fib], [3, 90])
    sel = fib & (r >= lo) & (r <= hi)
    xs, ys, Cs, rs = coords[sel, 0], coords[sel, 1], counts[sel], r[sel]
    along_s, off_s = unroll_meninges(xs, ys)        # band straightened, positions only

    for tag, (s_um, depth) in (("", BEST[gw]), ("common_", MEN_CELL)):
        step = s_um / UMPP
        gx = np.floor((xs - xs.min()) / step).astype(int)
        gy = np.floor((ys - ys.min()) / step).astype(int)
        _, inv = np.unique(gx * 100000 + gy, return_inverse=True)
        nps = inv.max() + 1
        agg = np.zeros((nps, Cs.shape[1]))
        Rp = np.zeros(nps); cx = np.zeros(nps); cy = np.zeros(nps)
        for i in range(nps):
            m = inv == i
            agg[i] = Cs[m].sum(0); Rp[i] = rs[m].mean()
            cx[i] = xs[m].mean(); cy[i] = ys[m].mean()
        umi = agg.sum(1)
        ok = umi >= depth
        assert ok.sum() >= N_MIN, f"{gw}: only {ok.sum()} pseudospots at {s_um}um/d{depth}"

        rng = np.random.default_rng(0)
        A = agg[ok]
        At = np.zeros_like(A)
        for i in range(A.shape[0]):
            tot = A[i].sum()
            At[i] = A[i] if tot <= depth else rng.multinomial(depth, A[i] / tot)
        lib = At.sum(1, keepdims=True); lib[lib == 0] = 1
        X = np.log1p(At / lib * np.median(lib))
        Rk, cxk, cyk = Rp[ok], cx[ok], cy[ok]
        pos = (Rk - Rk.min()) / (np.ptp(Rk) + 1e-9)
        bx = pd.cut(cxk, 4, labels=False).astype(int)
        by = pd.cut(cyk, 4, labels=False).astype(int)
        yhat = checker_decode(X, pos, (bx + by) % 2)

        men[f"{gw}_{tag}xy"] = np.column_stack([cxk, cyk]) * UMPP
        men[f"{gw}_{tag}coord"] = pos
        men[f"{gw}_{tag}decoded"] = yhat
        men[f"{gw}_{tag}cell"] = np.array([s_um, depth])
        # native-resolution rendering: every selected spot takes its pseudospot's value
        val = np.full(nps, np.nan)
        val[np.where(ok)[0]] = yhat
        men[f"{gw}_{tag}spot_xy"] = np.column_stack([xs, ys]) * UMPP
        men[f"{gw}_{tag}spot_val"] = val[inv]
        # unrolled band: along-band arclength and across-band offset per pseudospot (um)
        cnt_b = np.bincount(inv, minlength=nps)
        men[f"{gw}_{tag}unroll"] = np.column_stack([
            (np.bincount(inv, weights=along_s, minlength=nps) / np.maximum(cnt_b, 1))[ok],
            (np.bincount(inv, weights=off_s, minlength=nps) / np.maximum(cnt_b, 1))[ok]]) * UMPP
        pub = G[(G.stage == gw) & (G.scale_um == s_um) & (G.depth == depth)].I  # empty for GW6-S2
        rows.append(dict(tissue="meninges", unit=gw,
                         caliber=("own best" if tag == "" else "common"),
                         scale_um=s_um, depth=depth, n=int(ok.sum()),
                         I_here=round(I_of(pos, yhat), 4),
                         I_published=(round(float(pub.iloc[0]), 4) if len(pub) else None),
                         rho=round(float(spearmanr(pos, yhat).correlation), 4)))
        print(f"{gw} {s_um}um/d{depth}: n={int(ok.sum())} I={rows[-1]['I_here']:+.4f} "
              f"(grid {rows[-1]['I_published']}) rho={rows[-1]['rho']:+.3f}", flush=True)
np.savez_compressed(f"{R}/decoded_maps_meninges.npz", **men)

# ───────────────────────── tendon ─────────────────────────
T = _load("f4ct", f"{HERE_}/fig4_coord_transfer.py")
ten = {}
sc, dep = TEN_CELL
for section in T.SECTIONS:
    rng = np.random.default_rng(T._seed(section))
    Bz = np.load(f"{T.DD}/skin_{section}_markerfree.npz", allow_pickle=True)
    M = sp.csr_matrix((Bz["counts_data"], Bz["counts_indices"], Bz["counts_indptr"]),
                      shape=tuple(Bz["counts_shape"]))
    xy = Bz["xy_um"]
    _off_raw = T.principal_curve_coord(xy[:, 0], xy[:, 1])[0]
    y_spot = T.rank01(_off_raw)
    along_t, off_t = unroll_tendon(xy[:, 0], xy[:, 1])
    assert np.allclose(off_t, _off_raw), "recomputed tendon offset differs from the stored one"

    x0, y0 = xy[:, 0].min(), xy[:, 1].min()
    cbx = ((xy[:, 0] - x0) / sc).astype(np.int64)
    cby = ((xy[:, 1] - y0) / sc).astype(np.int64)
    uc, inv = np.unique(cbx * 100000 + cby, return_inverse=True)
    P = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uc), M.shape[0]))
    agg = (P @ M).tocsr()
    cnt = np.bincount(inv)
    cx = np.bincount(inv, weights=xy[:, 0]) / cnt
    cy = np.bincount(inv, weights=xy[:, 1]) / cnt
    ycoord = np.bincount(inv, weights=y_spot) / cnt
    umi = np.asarray(agg.sum(1)).ravel()
    idx = np.where(umi >= dep)[0]
    assert len(idx) >= T.N_MIN, f"{section}: only {len(idx)} pseudospots at {sc}um/d{dep}"

    X = T.lognorm(T.thin_to(agg[idx], dep, rng))
    yh = T.checker_oof(X, ycoord[idx], cx[idx], cy[idx])
    keep = ~np.isnan(yh)
    ten[f"{section}_xy"] = np.column_stack([cx[idx][keep], cy[idx][keep]])
    ten[f"{section}_coord"] = ycoord[idx][keep]
    ten[f"{section}_decoded"] = yh[keep]
    valS = np.full(len(uc), np.nan)
    valS[idx[~np.isnan(yh)]] = yh[~np.isnan(yh)]
    ten[f"{section}_spot_xy"] = xy
    ten[f"{section}_spot_val"] = valS[inv]
    ten[f"{section}_unroll"] = np.column_stack([
        (np.bincount(inv, weights=along_t) / cnt)[idx][~np.isnan(yh)],
        (np.bincount(inv, weights=off_t) / cnt)[idx][~np.isnan(yh)]])
    pubT = pd.read_csv(f"{R}/fig4_coord_transfer.csv")
    pq = pubT[(pubT.section == section) & (pubT.coord == "centreline")
              & (pubT.scale_um == sc) & (pubT.depth == dep)].I_checker
    rows.append(dict(tissue="tendon", unit=section, caliber="common", scale_um=sc, depth=dep, n=int(len(idx)),
                     I_here=round(I_of(ycoord[idx][keep], yh[keep]), 4),
                     I_published=(round(float(pq.iloc[0]), 4) if len(pq) else None),
                     rho=round(float(spearmanr(ycoord[idx][keep], yh[keep]).correlation), 4)))
    print(f"{section}: n={len(idx)} I={rows[-1]['I_here']:+.4f} "
          f"(grid {rows[-1]['I_published']}) rho={rows[-1]['rho']:+.3f}", flush=True)
np.savez_compressed(f"{R}/decoded_maps_tendon.npz", **ten)

chk = pd.DataFrame(rows)
chk["delta"] = (chk.I_here - chk.I_published.astype(float)).round(4)
chk.to_csv(f"{R}/decoded_maps_check.csv", index=False)
print("\nWROTE decoded_maps_{meninges,tendon}.npz and decoded_maps_check.csv")
print(chk.to_string(index=False))
