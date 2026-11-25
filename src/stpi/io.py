"""Readers for the Visium layouts of the public datasets used in the paper.

Each function returns (M, xy): M is units x genes (scipy sparse CSR), xy the pixel positions of
the in-tissue spots in (column, row) order.
"""
import glob
import gzip

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.sparse import csr_matrix


def _positions(path):
    """Reads tissue_positions_list.csv(.gz), spaceranger v1 format, headerless."""
    p = pd.read_csv(path, header=None,
                    names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"])
    p = p[p.in_tissue == 1]
    return p.set_index("barcode")[["pxl_col", "pxl_row"]].astype(float)


def read_geo_mtx(d, prefix):
    """Reads GEO supplementary files <prefix>*matrix.mtx.gz, *barcodes.tsv.gz,
    *tissue_positions_list.csv.gz (kidney, intestine, mouse retina)."""
    M = sio.mmread(glob.glob(f"{d}/{prefix}*matrix.mtx.gz")[0]).tocsr()      # genes x spots
    bc = [l.decode().split("\t")[0].strip()
          for l in gzip.open(glob.glob(f"{d}/{prefix}*barcodes.tsv.gz")[0])]
    P = _positions(glob.glob(f"{d}/{prefix}*tissue_positions_list.csv.gz")[0])
    keep = [i for i, b in enumerate(bc) if b in P.index]
    xy = P.loc[[bc[i] for i in keep]].values
    return M[:, keep].T.tocsr(), xy


def read_geo_h5(d, prefix):
    """Reads GEO supplementary files <prefix>*matrix.h5 and *tissue_positions_list.csv.gz
    (cerebellum)."""
    import h5py
    f = glob.glob(f"{d}/{prefix}*matrix.h5")[0]
    with h5py.File(f, "r") as h:
        g = h["matrix"]
        M = csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                       shape=g["shape"][:][::-1])                      # spots x genes
        bc = [b.decode() for b in g["barcodes"][:]]
    P = _positions(glob.glob(f"{d}/{prefix}*tissue_positions_list.csv.gz")[0])
    keep = [i for i, b in enumerate(bc) if b in P.index]
    return M[keep], P.loc[[bc[i] for i in keep]].values


def read_spaceranger(d, sub):
    """Reads a spaceranger output folder: <d>/<sub>/filtered_feature_bc_matrix and
    <d>/<sub>/spatial/tissue_positions.csv (finch retina)."""
    base = f"{d}/{sub}/filtered_feature_bc_matrix"
    M = sio.mmread(f"{base}/matrix.mtx.gz").tocsr()
    bc = [l.decode().strip() for l in gzip.open(f"{base}/barcodes.tsv.gz")]
    P = pd.read_csv(f"{d}/{sub}/spatial/tissue_positions.csv")
    P.columns = [c.strip() for c in P.columns]
    P = P.set_index(P.columns[0])
    xc = [c for c in P.columns if "pxl_col" in c][0]
    yc = [c for c in P.columns if "pxl_row" in c][0]
    keep = [i for i, b in enumerate(bc) if b in P.index]
    return M[:, keep].T.tocsr(), P.loc[[bc[i] for i in keep], [xc, yc]].values.astype(float)
