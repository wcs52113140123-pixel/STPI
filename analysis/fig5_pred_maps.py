#!/usr/bin/env python3
"""Writes results/fig5_pred_maps.npz: the decoded fields of the five sections drawn in Fig 5 B.

The maps are the out-of-fold predictions already stored for all 18 sections in
results/fig5_pred_maps_all.npz (written by figS5_support.py with the decoder);
this script copies out one section per tissue under the tissue's name.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import RES_
import numpy as np

SHOWN = {"kidney": "GSM8440999_No22", "intestine": "GSM7840104_V11U23-312_D",
         "cerebellum": "GSM8969382_CR_23_06187_TS_R_JSV_1_S8", "retina": "GSM7916286_078_A1",
         "finch_retina": "GSM9266973_rep1"}

A = np.load(f"{RES_}/fig5_pred_maps_all.npz")
out = {f"{t}_{k}": A[f"{s}_{k}"] for t, s in SHOWN.items() for k in ("xy", "true", "pred")}
np.savez_compressed(f"{RES_}/fig5_pred_maps.npz", **out)
print("wrote fig5_pred_maps.npz", sorted(out))
