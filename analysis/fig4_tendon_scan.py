#!/usr/bin/env python3
"""Fig 4 and Fig S4: size x depth scan on four injured mouse tendon sections
(Cherief et al. 2023, GSE245179), under four coordinates.

Writes results/fig4_tendon_cv{,_centreline,_along,_gaston}.csv

  surface      boundary geodesic distance stored with each section (R)
  centreline   distance across the backbone (stpi.coords.backbone_offset)
  along        arclength along the backbone (stpi.coords.along_backbone)
  gaston       GASTON isodepth fitted to expression (results/fig4_gaston_isodepth.npz),
               included as a published comparison

Inputs: <STPI_DATA>/derived/skin_<section>_markerfree.npz, built from GSE245179 by
skin_markerfree_coord.py. Pseudospots of 100-800 um, depths 16-2048 counts, 20 permuted
checkerboards per cell. Seed per section: SHA-256 of the section name. About 3 minutes
per coordinate on 4 cores.
"""
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from multiprocessing import Pool

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))   # works under python -P too
from _paths import RESULTS, DERIVED
import stpi
from stpi.coords import rank01, backbone_offset, along_backbone
from stpi.scan import size_depth_scan

SECTIONS = ["CTRL1", "CTRL2", "INMPP1", "INMPP1-2"]
SCALES = [100, 200, 400, 600, 800]
DEPTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]
SUFFIX = {"surface": "", "centreline": "_centreline", "along": "_along", "gaston": "_gaston"}


def coordinate(kind, xy, Bz):
    if kind == "surface":
        return rank01(np.asarray(Bz["R"], float))
    if kind == "centreline":
        return rank01(backbone_offset(xy[:, 0], xy[:, 1]))
    if kind == "along":
        return rank01(along_backbone(xy[:, 0], xy[:, 1]))
    G = np.load(RESULTS / "fig4_gaston_isodepth.npz")
    hit = [k[:-3] for k in G.files if k.endswith("_xy") and G[k].shape == xy.shape
           and np.allclose(G[k], xy)]
    assert len(hit) == 1, f"isodepth spot set does not match this section ({hit})"
    return rank01(np.asarray(G[f"{hit[0]}_isodepth_nl3"], float))


def one(args):
    kind, sec = args
    Bz = np.load(DERIVED / f"skin_{sec}_markerfree.npz", allow_pickle=True)
    M = sp.csr_matrix((Bz["counts_data"], Bz["counts_indices"], Bz["counts_indptr"]),
                      shape=tuple(Bz["counts_shape"]))
    xy = Bz["xy_um"]
    return size_depth_scan(M, xy, coordinate(kind, xy, Bz), SCALES, DEPTHS,
                           stpi.seed_of(sec), name=sec)


if __name__ == "__main__":
    kinds = sys.argv[1:] or list(SUFFIX)
    for kind in kinds:
        with Pool(len(SECTIONS)) as p:
            df = pd.concat(p.map(one, [(kind, s) for s in SECTIONS]), ignore_index=True)
        df["inflation"] = df.I_randlab_med - df.I_checker
        df.to_csv(RESULTS / f"fig4_tendon_cv{SUFFIX[kind]}.csv", index=False)
        print(f"wrote fig4_tendon_cv{SUFFIX[kind]}.csv", df.shape, flush=True)
