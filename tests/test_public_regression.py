"""Tests that recompute stored result tables from the public datasets, cell by cell.

Needs the public data under $STPI_DATA (layout in README). Each test recomputes a stored
table cell by cell; tolerance 1e-6 bit unless stated (the tables are rounded to 4 decimals,
so a difference of 0 is expected).

Run all 18 published sections with STPI_FULL=1 (about 15 minutes).
"""
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import stpi
from stpi import io
from stpi.coords import rank01, backbone_offset
from stpi.scan import size_depth_scan

from conftest import RESULTS, need_data

pytestmark = pytest.mark.public_data

DIRS = {"kidney": "kidney_GSE273978", "intestine": "intestine_GSE245274",
        "cerebellum": "cerebellum_GSE296375", "retina": "retina_GSE248517",
        "finch_retina": "retina_GSE309408_finch"}
AUD = pd.read_csv(RESULTS / "fieldwide_audit.csv").set_index("sample")
QUICK = ["GSM7916286_078_A1", "GSM8969380"]          # one mouse retina, one cerebellum
SECTIONS = list(AUD.index) if os.environ.get("STPI_FULL") else [
    s for s in AUD.index if any(s.startswith(q) for q in QUICK)]


def _load(sample):
    t = AUD.loc[sample, "tissue"]
    d = need_data(DIRS[t])
    if t == "finch_retina":
        return io.read_spaceranger(d, "25" if sample.endswith("rep1") else "26")
    if t == "cerebellum":
        return io.read_geo_h5(d, sample)
    return io.read_geo_mtx(d, sample)


@pytest.mark.parametrize("sample", SECTIONS)
def test_fig5_buffered_estimates(sample):
    """fig5_buffered_cv_all.csv: spatial block CV at buffers of 0-4 spot pitches."""
    M, xy = _load(sample)
    R = stpi.interiority(xy); X = stpi.log_hvg(M)
    f = stpi.checkerboard_folds(xy); p = stpi.spot_pitch(xy)
    B = pd.read_csv(RESULTS / "fig5_buffered_cv_all.csv")
    B = B[B["sample"] == sample]
    assert len(B) == 5
    for _, r in B.iterrows():
        I = stpi.recoverability(X, R, f, xy, r.buffer_pitch * p).I
        assert round(I, 4) == pytest.approx(r.I_bit, abs=1e-6), r.buffer_pitch
    # the two checks shown in Fig 5 C and D: random 5-fold CV (seed 0) and the checkerboard partition
    rnd = stpi.recoverability(X, R, stpi.random_folds(len(R), 5, 0)).I
    chk = stpi.recoverability(X, R, f).I
    assert round(rnd, 4) == pytest.approx(AUD.loc[sample, "I_random"], abs=1e-6)
    assert round(chk, 4) == pytest.approx(AUD.loc[sample, "I_checker"], abs=1e-6)


def test_fig5_null():
    """fig5_fieldwide_null_msr.csv: Moran spectral null on the smallest section (575 spots).

    The table is produced by analysis/fig5_null.py using this package.
    """
    sample = "GSM7916286_078_A1"
    M, xy = _load(sample)
    R = stpi.interiority(xy); X = stpi.log_hvg(M); f = stpi.checkerboard_folds(xy)
    S, _ = stpi.msr_surrogates(R, xy, n_draws=200, seed=stpi.seed_of(sample + "msr"))
    obs = stpi.recoverability(X, R, f).I
    null = np.array([stpi.recoverability(X, s, f).I for s in S])
    ref = pd.read_csv(RESULTS / "fig5_fieldwide_null_msr.csv").set_index("sample").loc[sample]
    assert round(obs, 4) == pytest.approx(ref.I_obs, abs=1e-6)
    assert round(float(np.median(null)), 4) == pytest.approx(ref.null_median, abs=1e-6)
    assert round(stpi.null_p(obs, null), 4) == pytest.approx(ref.P, abs=1e-6)


def _tendon(section):
    p = need_data("derived", f"skin_{section}_markerfree.npz")
    Bz = np.load(p, allow_pickle=True)
    M = sp.csr_matrix((Bz["counts_data"], Bz["counts_indices"], Bz["counts_indptr"]),
                      shape=tuple(Bz["counts_shape"]))
    return M, Bz["xy_um"]


@pytest.mark.parametrize("table,coord", [("fig4_tendon_cv_centreline.csv", "centreline"),
                                         ("fig4_tendon_cv.csv", "surface")])
def test_fig4_size_depth_scan(table, coord):
    """Fig 4 / Fig S4 scan on one tendon section: every (size, depth) cell."""
    sec = "CTRL2"
    M, xy = _tendon(sec)
    if coord == "surface":
        Bz = np.load(need_data("derived", f"skin_{sec}_markerfree.npz"), allow_pickle=True)
        c = rank01(np.asarray(Bz["R"], float))
    else:
        c = rank01(backbone_offset(xy[:, 0], xy[:, 1]))
    D = size_depth_scan(M, xy, c, [100, 200, 400, 600, 800],
                        [16, 32, 64, 128, 256, 512, 1024, 2048], stpi.seed_of(sec), name=sec)
    ref = pd.read_csv(RESULTS / table)
    m = D.merge(ref[ref.section == sec], on=["section", "scale_um", "depth"], suffixes=("", "_ref"))
    assert len(m) == (ref.section == sec).sum()
    for col in ("I_checker", "I_randlab_med"):
        assert np.allclose(m[col], m[col + "_ref"], atol=1e-5, equal_nan=True), col
