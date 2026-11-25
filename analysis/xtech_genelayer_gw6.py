"""Computes decoded positional information (I) for GW6 using each platform's native (full) gene set, rather than a gene-matched subset.

fig3_xtech_genelayer.py covers GW8 through GW16 only, because neither the MERFISH nor the Stereo loader defines a GW6 branch. This script reuses the conditions, decoder, binning, seeds and bootstrap count (B) from xtech_nmatched.py by executing its source up to the point where results are accumulated; the only difference here is that no gene mask is applied, so each platform keeps its full (native) gene set.

Both gene counts are recorded because the native Stereo matrices are not uniform across stages: GW8-GW16 come from 5,000-HVG matrices, while GW6 carries the full transcriptome. GW6's native arm therefore sees far more genes than its siblings. This is a property of the input files and should be noted in the caption of any panel that shows this arm.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np, pandas as pd
from scipy.io import mmread

SRC = f"{HERE_}/xtech_nmatched.py"
src = open(SRC).read()
# run xtech_nmatched.py's setup (loaders, decoder, parameters) without its row-accumulation loop
exec(compile(src[:src.index("rows=[]")], SRC, "exec"), globals())

sk = "GW6_MERFISH_r0"
M = mmread(f"{EXPR}/{sk}_counts.mtx").tocsr()
genes_m = [l.strip() for l in open(f"{EXPR}/{sk}_genes.txt")]
cells = np.array([l.strip() for l in open(f"{EXPR}/{sk}_cells.txt")])
if M.shape[0] == len(genes_m): M = M.T  # orient matrix as cells x genes
M = M.tocsr(); cmap = {c: i for i, c in enumerate(cells)}
meta = pd.read_csv(f"{EXPR}/{sk}_coords.csv"); meta["cell"] = meta["cell"].astype(str)
meta = meta[meta["cell"].isin(cmap)]
Xm = np.asarray(M[[cmap[c] for c in meta["cell"]]].todense())      # native gene set: no mask applied
cxm, cym = meta["x"].values, meta["y"].values

ms = pd.read_csv(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_meta.csv")
counts = np.asarray(mmread(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_counts.mtx").todense()).T
genes_s = np.array([l.strip() for l in open(f"{DATA_}/gw6_annotated/expr/GW6_Stereo_R1_genes.txt")])
fib = np.isin(ms["celltype"].values, ["Fibro-pia-pro", "Fibro-others"])
Xs = counts[fib]                                                   # native gene set: no mask applied
cxs, cys = ms[["x", "y"]].values[fib, 0].astype(float), ms[["x", "y"]].values[fib, 1].astype(float)
print(f"native genes: MERFISH {Xm.shape[1]}, Stereo {Xs.shape[1]}", flush=True)

Xmb, cxmb, cymb, Rm = pseudospots(cxm, cym, Xm, SCALE, DEPTH)
Xsb, cxsb, cysb, Rs = pseudospots(cxs, cys, Xs, SCALE_STEREO, DEPTH)
nm, nsp = Xmb.shape[0], Xsb.shape[0]; nmatch = min(nm, nsp)
print(f"pseudospots at {SCALE:.0f} um / depth {DEPTH}: MER {nm} / Stereo {nsp} -> n {nmatch}", flush=True)

rng = np.random.default_rng(0); dIs, Ims, Iss = [], [], []
for b in range(B):
    im = rng.choice(nm, nmatch, replace=False); isx = rng.choice(nsp, nmatch, replace=False)
    Im = decode_I(Xmb[im], cxmb[im], cymb[im], Rm[im], DEPTH, b)
    Is = decode_I(Xsb[isx], cxsb[isx], cysb[isx], Rs[isx], DEPTH, b)
    if not (np.isnan(Im) or np.isnan(Is)): dIs.append(Im - Is); Ims.append(Im); Iss.append(Is)
    if b % 50 == 0: print(f"  draw {b}", flush=True)
dIs, Ims, Iss = map(np.array, (dIs, Ims, Iss))
row = dict(stage="GW6", n_mer=nm, n_stereo=nsp, n_match=nmatch,
           n_genes_mer=Xm.shape[1], n_genes_stereo=Xs.shape[1], gene_layer="native",
           I_mer_med=round(float(np.median(Ims)), 4), I_stereo_med=round(float(np.median(Iss)), 4),
           dI_med=round(float(np.median(dIs)), 4), dI_lo=round(float(np.percentile(dIs, 2.5)), 4),
           dI_hi=round(float(np.percentile(dIs, 97.5)), 4),
           pct_pos=round(float((dIs > 0).mean()), 3), nboot=len(dIs))
pd.DataFrame([row]).to_csv(f"{RES_}/fig3_xtech_genelayer_GW6.csv", index=False)
print(row, flush=True)
