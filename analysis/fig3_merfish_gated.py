"""MERFISH stage comparison under the depth convention used for the Stereo pipeline, matched sample size AND
no synthetic oversampling.

Why this exists. merfish_strict_grid.downsample_depth draws `depth` counts multinomially
from every pseudospot regardless of its native total, so a pseudospot holding 40
transcripts still yields a vector summing to 256. The Stereo pipeline
(multiGW_scan.py) does the opposite: it keeps the vector unchanged when the native
total is at or below the target and gates feasibility on native total >= depth. At
50 um the native medians are GW6 34, GW8 99, GW10 265, GW12 152, GW16 172 transcripts,
so at depth 256 only 10% of GW12 pseudospots are genuinely that deep against 31% of
GW16, exactly where the GW12/GW16 leader flips. The flip may therefore be a property
of the resampling rule rather than of the tissue.

This script applies the Stereo rule to MERFISH: a pseudospot enters a depth cell only
if its native total >= depth, and it is thinned, never inflated. Sample size is matched
across stages at each rung, drawn without replacement, NREP repeats. A stage
is dropped from a depth cell when fewer than N_MIN gated pseudospots remain, so the
comparison simply stops where the data stop.

Everything else, loaders, geometry-only coordinate, binning, 5-block spatial CV with
buffer, I = 0.5*log2(Var/MSE), is imported unchanged from merfish_strict_grid.py.

Output: results/merfish_nmatched_gated.csv
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, pandas as pd, importlib.util

spec = importlib.util.spec_from_file_location("mg", f"{HERE_}/merfish_strict_grid.py")
mg = importlib.util.module_from_spec(spec); sys.modules["mg"] = mg
spec.loader.exec_module(mg)

SCALE = 50
DEPTHS = [16, 32, 64, 128, 256]
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]
NREP = 40
N_MIN = 100                      # same floor as the Stereo scan


def thin_gated(X, depth, rng):
    """Keeps a pseudospot's vector unchanged when its native total is at or below depth, otherwise multinomial-thins it down to depth."""
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        tot = X[i].sum()
        out[i] = X[i] if tot <= depth else rng.multinomial(int(depth), X[i] / tot)
    return out


def stage_pseudospots(stage, gene_layer):
    d = mg.load_stage(stage)
    genes = np.array(d["genes"])
    if gene_layer == "shared":
        shared = set(l.strip() for l in
                     open(f"{DER_}/stereo_merfish_shared_genes.txt"))
        gmask = np.isin(genes, list(shared))
    else:
        gmask = np.ones(len(genes), bool)
    X0 = d["X"][:, gmask]
    Rcell = mg.geometry_R(d["cx"], d["cy"])
    Xb, cxb, cyb, cnt, inv = mg.bin_pseudospots(d["cx"], d["cy"], X0, SCALE)
    keep = cnt >= 1
    Rb = np.zeros(len(cxb)); np.add.at(Rb, inv, Rcell); Rb /= cnt
    X = Xb[keep]
    return dict(X=X, R=Rb[keep], pts=np.c_[cxb[keep], cyb[keep]], tot=X.sum(1))


def main():
    rows, cover = [], []
    for gene_layer in ("native", "shared"):
        cache = {s: stage_pseudospots(s, gene_layer) for s in STAGES}
        for s in STAGES:
            for depth in DEPTHS:
                cover.append(dict(gene_layer=gene_layer, stage=s, depth=depth,
                                  n_total=len(cache[s]["tot"]),
                                  n_gated=int((cache[s]["tot"] >= depth).sum())))
        cov = pd.DataFrame(cover)
        for depth in DEPTHS:
            avail = {s: int((cache[s]["tot"] >= depth).sum()) for s in STAGES}
            elig = [s for s in STAGES if avail[s] >= N_MIN]
            if len(elig) < 2:
                print(f"{gene_layer} depth={depth}: only {elig} reach n>={N_MIN} - skipped",
                      flush=True)
                continue
            rung = min(avail[s] for s in elig)
            print(f"{gene_layer} depth={depth}: gated n={avail} -> stages {elig}, rung={rung}",
                  flush=True)
            for rep in range(NREP):
                rng = np.random.default_rng(7000 + 100 * depth + rep)
                for s in elig:
                    c = cache[s]
                    idx = np.flatnonzero(c["tot"] >= depth)
                    sel = rng.choice(idx, rung, replace=False)
                    Xd = thin_gated(c["X"][sel], depth, rng)
                    I, rho, nn = mg.decode_I(c["R"][sel], c["pts"][sel], Xd, "spatial",
                                             seed=rep)
                    rows.append(dict(gene_layer=gene_layer, depth=depth, rung_n=rung,
                                     rep=rep, stage=s, I=I, rho=rho))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES_}/merfish_nmatched_gated.csv", index=False)
    pd.DataFrame(cover).drop_duplicates().to_csv(
        f"{RES_}/merfish_depth_coverage.csv", index=False)
    print("WROTE merfish_nmatched_gated.csv", df.shape)
    for gl in ("native", "shared"):
        d = df[df.gene_layer == gl]
        if not len(d): continue
        print(f"\n=== {gl}: median I by stage x depth (gated, matched n) ===")
        print(d.pivot_table(index="depth", columns="stage", values="I",
                            aggfunc="median").round(3).to_string())
        piv = d.pivot_table(index=["rep", "depth"], columns="stage", values="I")
        if {"GW12", "GW16"} <= set(piv.columns):
            q = piv.dropna(subset=["GW12", "GW16"]).reset_index()
            print("GW12 > GW16 win rate by depth:")
            print(q.groupby("depth").apply(
                lambda t: f"{(t['GW12'] > t['GW16']).mean():.0%} (n={len(t)})").to_string())


if __name__ == "__main__":
    main()
