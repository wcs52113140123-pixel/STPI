"""Writes results/depth_ceiling_triple_collinearity.csv.

Summarizes, for each developmental stage, the tissue size, the maximum feasible bin
count, the sequencing depth, and the predicted spatial resolution, to show how these
quantities co-vary across stages.
"""
import sys as _sys, pathlib as _pl
if "__file__" in globals():          # also runs when another script exec()s this one
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import DATA_, DER_, RES_, HERE_   # data, derived inputs, results, this folder
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

stages = ["GW6", "GW8", "GW10", "GW12", "GW16"]
T = np.array([637.74, 339.55, 664.47, 665.81, 1662.49])
sstar = np.array([50, 50, 100, 200, 800])

fc = pd.read_csv(f"{RES_}/stage_feasible_counts.csv").set_index("stage").reindex(stages)
grid = pd.read_csv(f"{RES_}/multiGW_grid_all.csv")

n50 = grid.groupby("stage")["n"].max().reindex(stages).values.astype(float)
medUMI = fc["median_umi"].values.astype(float)

tab = pd.DataFrame({
    "stage": stages,
    "T_um": T.astype(int),
    "n50": n50.astype(int),
    "median_umi_per_bin": medUMI.astype(int),
    "umi_per_um2": np.round(medUMI / 2500, 3),
    "s_pred_d512": np.round(50 * np.sqrt(512 / medUMI), 1),
    "reported_sstar": sstar,
    "whole_lib_M_umi": np.round(fc["total_bins"].values * medUMI / 1e6, 2)
})

tab.to_csv(f"{RES_}/depth_ceiling_triple_collinearity.csv", index=False)
