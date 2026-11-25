"""Writes results/xtech_genelayer_all.csv.

Combines matched-gene and native-gene layer comparisons across cross-technology
stages into a single summary table.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import pandas as pd

ST = ["GW6", "GW8", "GW10", "GW12", "GW16"]

gi = pd.read_csv(f"{RES_}/xtech_nmatched_geneintersect.csv")
gl = pd.read_csv(f"{RES_}/fig3_xtech_genelayer.csv")
g6 = pd.read_csv(f"{RES_}/fig3_xtech_genelayer_GW6.csv")

m = gi.assign(gene_layer="matched", n_genes_mer=gi.n_genes_used, n_genes_stereo=gi.n_genes_used)
nat = pd.concat([gl[gl.gene_layer=="native"].assign(n_genes_mer=960, n_genes_stereo=5000), g6], ignore_index=True)
cols = ["stage","gene_layer","n_match","n_genes_mer","n_genes_stereo","I_mer_med","I_stereo_med",
        "dI_med","dI_lo","dI_hi","pct_pos","nboot"]
TAB = pd.concat([m[cols], nat[cols]], ignore_index=True)
TAB["stage"] = pd.Categorical(TAB.stage, ST, ordered=True)
TAB = TAB.sort_values(["gene_layer","stage"])
TAB.to_csv(f"{RES_}/xtech_genelayer_all.csv", index=False)
