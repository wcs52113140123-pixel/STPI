"""Matches both the gene set and the unit count across the five developmental stages.

MERFISH decodes from a shared set of 265 genes, while Stereo-seq decodes from whatever
subset of those genes is present in its own expression matrix (90 at GW8, 125 at GW12,
144 at GW10, 255 at GW6, 265 at GW16; the gaston_multi matrices are pre-filtered to
5,000 highly variable genes). Here both datasets are restricted, per stage, to the genes
present in both, so the two technologies decode from the same features as well as the
same number of spatial units.

Binning, the R coordinate, the decoder, the parameters (SCALE=100, DEPTH=16, NB=5, B=300,
N_MIN=40) and the random seeds are reused from xtech_nmatched.py. GW6 is handled with two
dedicated loader branches that account for its different file layout and fibroblast
cell-type nomenclature.

Note: the depth gate (a unit is retained only if its total count over the retained genes
reaches DEPTH) is applied after the gene restriction, so the unit counts can differ from
a run without gene restriction as a result of this stricter requirement.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np, pandas as pd
from scipy.io import mmread

SRC = f"{HERE_}/xtech_nmatched.py"
src = open(SRC).read()
exec(compile(src[:src.index("rows=[]")], SRC, "exec"), globals())

def load_merfish_any(stage):
    if stage != "GW6":
        return load_merfish(stage)[:4] if False else load_merfish(stage)
    sk = "GW6_MERFISH_r0"
    M = mmread(f"{EXPR}/{sk}_counts.mtx").tocsr()
    genes = [l.strip() for l in open(f"{EXPR}/{sk}_genes.txt")]
    cells = np.array([l.strip() for l in open(f"{EXPR}/{sk}_cells.txt")])
    if M.shape[0] == len(genes): M = M.T
    M = M.tocsr(); cmap = {c: i for i, c in enumerate(cells)}
    meta = pd.read_csv(f"{EXPR}/{sk}_coords.csv"); meta["cell"] = meta["cell"].astype(str)
    meta = meta[meta["cell"].isin(cmap)]
    X = np.asarray(M[[cmap[c] for c in meta["cell"]]].todense())
    gmask = np.isin(genes, shared)
    return X[:, gmask], np.array(genes)[gmask], meta["x"].values, meta["y"].values, None

def load_stereo_any(stage):
    if stage != "GW6":
        return load_stereo(stage)
    m = pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv")
    counts = np.asarray(mmread(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_counts.mtx").todense()).T
    genes = np.array([l.strip() for l in open(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_genes.txt")])
    coords = m[["x", "y"]].values.astype(float)
    fib = np.isin(m["celltype"].values, ["Fibro-pia-pro", "Fibro-others"])
    gmask = np.isin(genes, shared)
    return counts[fib][:, gmask], genes[gmask], coords[fib, 0], coords[fib, 1]

rows = []
for stage in ["GW6", "GW8", "GW10", "GW12", "GW16"]:
    Xm, gm, cxm, cym = load_merfish_any(stage)[:4]
    Xs, gs, cxs, cys = load_stereo_any(stage)
    inter = np.array(sorted(set(map(str, gm)) & set(map(str, gs))))
    im_cols = [list(map(str, gm)).index(g) for g in inter]
    is_cols = [list(map(str, gs)).index(g) for g in inter]
    Xm, Xs = Xm[:, im_cols], Xs[:, is_cols]
    Xmb, cxmb, cymb, Rm = pseudospots(cxm, cym, Xm, SCALE, DEPTH)
    Xsb, cxsb, cysb, Rs = pseudospots(cxs, cys, Xs, SCALE_STEREO, DEPTH)
    nm, nsp = Xmb.shape[0], Xsb.shape[0]; nmatch = min(nm, nsp)
    print(f"{stage}: genes used {len(inter)} | pseudospots MER {nm} / Stereo {nsp} -> n {nmatch}", flush=True)
    if nmatch < N_MIN:
        rows.append(dict(stage=stage, n_genes_used=len(inter), n_mer=nm, n_stereo=nsp, n_match=nmatch,
                         I_mer_med=np.nan, I_stereo_med=np.nan, dI_med=np.nan, dI_lo=np.nan,
                         dI_hi=np.nan, pct_pos=np.nan, nboot=0))
        continue
    rng = np.random.default_rng(0); dIs, Ims, Iss = [], [], []
    for b in range(B):
        i_m = rng.choice(nm, nmatch, replace=False); i_s = rng.choice(nsp, nmatch, replace=False)
        Im = decode_I(Xmb[i_m], cxmb[i_m], cymb[i_m], Rm[i_m], DEPTH, b)
        Is = decode_I(Xsb[i_s], cxsb[i_s], cysb[i_s], Rs[i_s], DEPTH, b)
        if not (np.isnan(Im) or np.isnan(Is)): dIs.append(Im - Is); Ims.append(Im); Iss.append(Is)
    dIs, Ims, Iss = map(np.array, (dIs, Ims, Iss))
    rows.append(dict(stage=stage, n_genes_used=len(inter), n_mer=nm, n_stereo=nsp, n_match=nmatch,
                     I_mer_med=round(float(np.median(Ims)), 4), I_stereo_med=round(float(np.median(Iss)), 4),
                     dI_med=round(float(np.median(dIs)), 4), dI_lo=round(float(np.percentile(dIs, 2.5)), 4),
                     dI_hi=round(float(np.percentile(dIs, 97.5)), 4),
                     pct_pos=round(float((dIs > 0).mean()), 3), nboot=len(dIs)))
    print("   ", rows[-1], flush=True)

pd.DataFrame(rows).to_csv(f"{RES_}/xtech_nmatched_geneintersect.csv", index=False)
