"""Checks on synthetic data; no downloads needed."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

import stpi
from stpi.coords import backbone_offset, harmonic, rank01


@pytest.fixture(scope="module")
def grid():
    g = np.stack(np.meshgrid(np.arange(40), np.arange(30)), -1).reshape(-1, 2).astype(float)
    return g, g[:, 0] / 39.0


def test_seed_is_stable():
    assert stpi.seed_of("CTRL2") == stpi.seed_of("CTRL2")
    assert stpi.seed_of("CTRL2") != stpi.seed_of("CTRL1")
    assert stpi.seed_of("abc") == 3128432319   # first 8 hex digits of SHA-256("abc")


def test_checkerboard_has_two_balanced_folds(grid):
    g, _ = grid
    f = stpi.checkerboard_folds(g)
    assert set(np.unique(f)) == {0, 1}
    assert abs(f.mean() - 0.5) < 0.05


def test_signal_is_recovered_and_noise_is_not(grid):
    g, coord = grid
    rng = np.random.default_rng(1)
    n, G = len(g), 50
    signal = np.outer(coord, rng.normal(size=G)) * 2 + rng.normal(size=(n, G))
    noise = rng.normal(size=(n, G))
    f = stpi.checkerboard_folds(g)
    assert stpi.recoverability(signal, coord, f).I > 1.0
    assert abs(stpi.recoverability(noise, coord, f).I) < 0.1


def test_random_cv_inflates_on_autocorrelated_expression(grid):
    """Smooth expression unrelated to the coordinate: random folds put neighbours on both
    sides of the split and read much more than spatial blocks do."""
    g, coord = grid
    rng = np.random.default_rng(1)
    F = np.array([gaussian_filter(rng.normal(size=(30, 40)), 4).ravel() for _ in range(50)]).T
    chk = stpi.recoverability(F, coord, stpi.checkerboard_folds(g)).I
    rnd = stpi.recoverability(F, coord, stpi.random_folds(len(g), 5, 0)).I
    assert rnd > chk + 0.5


def test_buffer_removes_training_units(grid):
    g, coord = grid
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(g), 20))
    f = stpi.checkerboard_folds(g)
    p = stpi.spot_pitch(g)
    fr = [stpi.recoverability(X, coord, f, g, k * p).frac_dropped for k in (0, 1, 2, 4)]
    assert fr[0] == 0 and fr[1] < fr[2] < fr[3]


def test_msr_surrogates_are_decorrelated_and_rank_matched(grid):
    g, coord = grid
    S, info = stpi.msr_surrogates(coord, g, n_draws=20, seed=0)
    assert S.shape == (20, len(g))
    assert info["rho"].max() < 0.30
    assert np.allclose(np.sort(S, axis=1), np.sort(coord)[None, :])


def test_null_p():
    assert stpi.null_p(1.0, [0.0] * 199) == pytest.approx(1 / 200)
    assert stpi.null_p(0.0, [1.0] * 199) == pytest.approx(1.0)


def test_interiority_is_zero_on_the_boundary_and_one_inside(grid):
    g, _ = grid
    R = stpi.interiority(g)
    assert R.min() == 0 and R.max() == pytest.approx(1, abs=1e-6)
    edge = (g[:, 0] == 0) | (g[:, 0] == 39) | (g[:, 1] == 0) | (g[:, 1] == 29)
    assert R[edge].max() == 0
    assert R[(g[:, 0] == 20) & (g[:, 1] == 15)][0] > 0.9


def test_across_band_follows_the_radius_away_from_the_ends():
    rng = np.random.default_rng(1)
    th = rng.uniform(0.02 * np.pi, 0.98 * np.pi, 4000)
    rad = rng.uniform(10, 12, 4000)
    r = stpi.across_band(rad * np.cos(th), rad * np.sin(th))
    mid = (th > 0.2 * np.pi) & (th < 0.8 * np.pi)
    assert abs(spearmanr(r[mid], rad[mid])[0]) > 0.99


def test_backbone_and_harmonic_follow_the_short_axis_of_a_band():
    g = np.stack(np.meshgrid(np.arange(100), np.arange(10)), -1).reshape(-1, 2).astype(float)
    assert abs(spearmanr(backbone_offset(g[:, 0], g[:, 1]), g[:, 1])[0]) > 0.9
    u, info = harmonic(g[:, 0], g[:, 1])
    assert info["ok"] and abs(spearmanr(u, g[:, 1])[0]) > 0.9


def test_rank01():
    assert np.allclose(rank01([3.0, 1.0, 2.0]), [1.0, 0.0, 0.5])


def test_moran_surrogates_do_not_depend_on_thread_count():
    """A regular spot grid gives repeated eigenvalues; the basis must not follow BLAS threads."""
    from threadpoolctl import threadpool_limits
    g = np.stack(np.meshgrid(np.arange(30), np.arange(20)), -1).reshape(-1, 2).astype(float)
    coord = g[:, 0] / 29.0
    out = []
    for n in (1, 4):
        with threadpool_limits(limits=n):
            S, _ = stpi.msr_surrogates(coord, g, n_draws=5, seed=7)
        out.append(S)
    assert np.array_equal(out[0], out[1])
