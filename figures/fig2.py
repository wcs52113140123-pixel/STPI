#!/usr/bin/env python3
"""Figure 2. Pseudospot size and depth on the human fetal meninges.

  A  Recoverability across pseudospot size and depth (STPI)
  B  Recoverability versus tissue thickness (50 um, depth 64)
  C  Depth gain per doubling (400 um)
  D  Recoverability at matched depth (depth 64)
  E  Recoverability versus depth per dimension (50 um, K = 30)
  F  Recoverability at matched number of pseudospots (n = 253)

Inputs (results/)
  multiGW_grid_all.csv                    estimate per stage x pseudospot size x depth
  multiGW_T.json                          radial thickness per stage (um)
  multiGW_effects_by_stage_block400.csv   depth gain per doubling, block bootstrap (400 um)
  depth_matched_sscan_5stage.csv          estimate at matched depth
  stage_matched_n_reps.csv                        estimate at a matched number of pseudospots
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import (RESULTS, STAGES, STAGE_C as SC, Layout, save_full, marks_dir,
                     px_figure, px_axes, axes_in_crop, colourbar_png)

FIG = "Fig2"
DPI = 600


def _grid():
    return pd.read_csv(RESULTS / "multiGW_grid_all.csv")


# =========================================================================== A
def a_data():
    grid = _grid()
    depths = sorted(grid.depth.unique()); scales = sorted(grid.scale_um.unique())
    surf = {}
    for s in STAGES:
        M = np.full((len(depths), len(scales)), np.nan)
        for _, r in grid[grid.stage == s].iterrows():
            if r.feasible and np.isfinite(r["I"]):
                M[depths.index(r.depth), scales.index(r.scale_um)] = r["I"]
        surf[s] = M
    amp = float(np.nanmax(np.abs(grid[grid.feasible]["I"])))   # colour scale symmetric about 0
    return surf, depths, scales, amp


def draw_a(ax, M, amp):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=-amp, vmax=amp,
                   interpolation="nearest")
    if np.isfinite(M).any():                     # star: highest cell of the surface
        yi, xi = np.unravel_index(np.nanargmax(M), M.shape)
        ax.scatter([xi], [yi], marker="*", s=300, color="#ee3b46", edgecolor="white",
                   linewidth=1.5, zorder=5)
    return im


A_IMG = (6341, 1466)              # five surfaces in one image, 24 px margin
A_AX_W, A_AX_H, A_STEP = 1090, 1418, 1264.25


def panel_a():
    surf, depths, scales, amp = a_data()
    fig = px_figure(*A_IMG)
    for k, s in enumerate(STAGES):
        ax = px_axes(fig, 24 + k * A_STEP, 24, A_AX_W, A_AX_H)
        im = draw_a(ax, surf[s], amp)
    cax = px_axes(fig, 6247, 23, 71, 1419)      # this axis is cropped out of the exported image
    cb = fig.colorbar(im, cax=cax)              # a separate colourbar image (Fig2_A_colourbar.png) is used in the figure layout instead
    cb.set_ticks([]); cb.outline.set_visible(False)
    fig.savefig(marks_dir(FIG) / "Fig2_A.png", dpi=DPI); plt.close(fig)
    colourbar_png(marks_dir(FIG) / "Fig2_A_colourbar.png", "viridis", 48, 727,
                  ((5, 42), (6, 722)), 6)

    d = Layout(2.98)
    fig, axs = plt.subplots(1, 5, figsize=(13.0, 3.6), gridspec_kw=dict(wspace=0.22))
    for k, (ax, s) in enumerate(zip(axs, STAGES)):
        im = draw_a(ax, surf[s], amp)
        d.axes(ax)
        ax.set_xticks(range(len(scales)))
        ax.set_xticklabels([int(c) for c in scales], rotation=51, ha="right")
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([int(i) for i in depths] if k == 0 else [])
        ax.set_title(s, fontsize=d.text)
        ax.set_xlabel("Pseudospot size (\u03bcm)", fontsize=d.text)
    axs[0].set_ylabel("Depth", fontsize=d.text)
    cb = fig.colorbar(im, ax=list(axs), fraction=0.012, pad=0.015)
    cb.outline.set_linewidth(d.line * 1.4); cb.ax.tick_params(labelsize=d.text, width=d.line)
    fig.suptitle("Recoverability across pseudospot size and depth (STPI)", x=0.125, ha="left",
                 y=1.04, fontsize=d.title)
    save_full(fig, FIG, "A")
    return {s: (round(float(np.nanmax(M)), 3) if np.isfinite(M).any() else None)
            for s, M in surf.items()}


# =========================================================================== B
def b_data():
    grid = _grid()
    T = json.load(open(RESULTS / "multiGW_T.json"))
    g = grid[(grid.feasible) & (grid.scale_um == 50) & (grid.depth == 64)]
    return T, {r.stage: float(r["I"]) for _, r in g.iterrows()}


def draw_b(ax, T, I):
    ax.axhline(0, color="#999", lw=1.0, ls=":", zorder=1)
    for s in STAGES:
        ax.scatter([T[s]], [I[s]], s=150, color=SC[s], edgecolor="white", linewidth=1.4, zorder=4)
    ax.set_xlim(150, 1900); ax.set_ylim(-0.06, 0.36)


def panel_b():
    T, I = b_data()
    fig = px_figure(1928, 1438)
    draw_b(px_axes(fig, *axes_in_crop(1928, 1438, pad_px=24)), T, I)
    fig.savefig(marks_dir(FIG) / "Fig2_B.png", dpi=DPI); plt.close(fig)

    d = Layout(3.70)
    fig, ax = plt.subplots(figsize=(3.13, 2.32)); draw_b(ax, T, I); d.axes(ax)
    off = {"GW16": (-14, 12), "GW12": (-4, 14), "GW10": (8, 10), "GW8": (8, 10), "GW6": (8, -8)}
    for s in STAGES:
        ax.annotate(s, (T[s], I[s]), textcoords="offset points", xytext=off[s],
                    ha="right" if off[s][0] < 0 else "left", va="center", fontsize=d.text)
    ax.set_xticks([500, 1000, 1500]); ax.set_xticklabels(["500", "1,000", "1,500"])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Tissue radial thickness (\u03bcm)", fontsize=d.text)
    ax.set_title("Recoverability versus tissue thickness\n(50 \u00b5m, depth 64)", loc="left",
                 fontsize=d.title)
    save_full(fig, FIG, "B")
    return {s: (round(T[s], 1), round(I[s], 3)) for s in STAGES}


# =========================================================================== C
def c_data():
    ef = pd.read_csv(RESULTS / "multiGW_effects_by_stage_block400.csv")
    return ef[ef.kind == "E_depth"].set_index("stage").reindex(STAGES)


def draw_c(ax, ed):
    yy = np.arange(len(STAGES))[::-1]
    ax.axvline(0, color="k", lw=1.2, ls=(0, (4, 3)), zorder=1)
    for y, s in zip(yy, STAGES):
        r = ed.loc[s]
        ax.plot([r.lo, r.hi], [y, y], color=SC[s], lw=1.9, solid_capstyle="butt", zorder=3)
        for xe in (r.lo, r.hi):
            ax.plot([xe, xe], [y - 0.15, y + 0.15], color=SC[s], lw=1.9, zorder=3)
        ax.plot([r["median"]], [y], "o", color=SC[s], ms=8, markeredgecolor="white",
                markeredgewidth=0.9, zorder=4)
    ax.set_xlim(-0.20, 0.25); ax.set_ylim(-0.75, len(STAGES) - 0.3)


C_DPI = 417.6                       # matches the resolution of the corresponding panel image in the figure layout


def panel_c():
    ed = c_data()
    fig = px_figure(1231, 1027, dpi=C_DPI)
    draw_c(px_axes(fig, 9.54, 15.41, 1207.3, 1001.2), ed)
    fig.savefig(marks_dir(FIG) / "Fig2_C.png", dpi=C_DPI); plt.close(fig)

    d = Layout(3.73)
    fig, ax = plt.subplots(figsize=(2.95, 2.46)); draw_c(ax, ed); d.axes(ax)
    ax.set_xlim(-0.2, 0.31)
    ax.set_yticks(np.arange(len(STAGES))[::-1]); ax.set_yticklabels(STAGES)
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    ax.set_xlabel("\u0394\u00ce (bit)", fontsize=d.text)
    ax.set_ylabel("Developmental stage", fontsize=d.text)
    ax.set_title("Depth gain per doubling (400 \u00b5m)", loc="left", fontsize=d.title)
    save_full(fig, FIG, "C")
    return ed[["lo", "median", "hi"]].round(4).to_dict("index")


# =========================================================================== D
def d_data():
    grid = _grid(); scales = sorted(grid.scale_um.unique())
    dm = pd.read_csv(RESULTS / "depth_matched_sscan_5stage.csv")
    M = np.full((len(STAGES), len(scales)), np.nan)
    for a, s in enumerate(STAGES):
        for _, r in dm[(dm.C == 64) & (dm.stage == s) & (dm.feasible)].iterrows():
            M[a, scales.index(r.scale_um)] = r["I"]
    return M, scales


def draw_d(ax, M):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.30, vmax=0.30, aspect="auto", origin="upper",
                   interpolation="nearest")
    ax.set_xlim(-0.5, M.shape[1] - 0.5); ax.set_ylim(M.shape[0] - 0.5, -0.6)
    return im


def panel_d():
    M, scales = d_data()
    fig = px_figure(1691, 1418)
    draw_d(px_axes(fig, *axes_in_crop(1691, 1418, expand=0.02)), M)
    fig.savefig(marks_dir(FIG) / "Fig2_D.png", dpi=DPI); plt.close(fig)
    # colourbar, matching the range shown in the figure (-0.3 to 0.3, no outline)
    colourbar_png(marks_dir(FIG) / "Fig2_D_colourbar.png", "RdBu_r", 118, 1438,
                  ((41, 93), (192, 1244)), 0)

    d = Layout(3.57)
    fig, ax = plt.subplots(figsize=(2.9, 2.4)); im = draw_d(ax, M)
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            v = M[a, b]
            if np.isfinite(v):
                ax.text(b, a, f"{v:.2f}".replace("-0.00", "0.00"), ha="center", va="center",
                        fontsize=d.text * 0.9,
                        color="white" if abs(v) > 0.22 else "#222")
            else:
                ax.text(b, a, "\u2014", ha="center", va="center", fontsize=d.text)
    ax.set_xticks(range(len(scales))); ax.set_xticklabels([int(c) for c in scales])
    ax.set_yticks(range(len(STAGES))); ax.set_yticklabels(STAGES)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0, labelsize=d.text)
    ax.set_xlabel("Pseudospot size (\u00b5m)", fontsize=d.text)
    ax.set_ylabel("Developmental stage", fontsize=d.text)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_ticks(np.round(np.arange(-0.3, 0.31, 0.1), 1))
    cb.ax.tick_params(labelsize=d.text, width=d.line); cb.outline.set_linewidth(d.line)
    ax.set_title("Recoverability at matched depth (depth 64)", loc="left", fontsize=d.title,
                 pad=d.text * 1.6)
    save_full(fig, FIG, "D")
    return np.round(M, 3)


# =========================================================================== E
def e_data():
    grid = _grid()
    st = grid[(grid.feasible == True) & (grid.scale_um == 50)][["stage", "depth", "I"]]  # noqa: E712
    pv = st.pivot(index="depth", columns="stage", values="I")[STAGES]
    return pv.index.values / 30.0, pv


def draw_e(ax, x, pv):
    ax.axhline(0, color="#999", lw=1.0, ls=":", zorder=2)
    for s in STAGES:
        ax.plot(x, pv[s], "-o", color=SC[s], lw=1.9, ms=5.5, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5 if s == "GW12" else 4)
    ax.set_xscale("log", base=2); ax.set_xlim(0.45, 64); ax.set_ylim(-0.09, 0.45)


def panel_e():
    x, pv = e_data()
    fig = px_figure(1917, 1418)
    draw_e(px_axes(fig, *axes_in_crop(1917, 1418, expand=0.02)), x, pv)
    fig.savefig(marks_dir(FIG) / "Fig2_E.png", dpi=DPI); plt.close(fig)

    d = Layout(3.69)
    fig, ax = plt.subplots(figsize=(3.13, 2.32)); draw_e(ax, x, pv); d.axes(ax)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16]); ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16"])
    ax.minorticks_off()
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    for s in STAGES:
        ax.annotate(s, (x[-1], pv[s].iloc[-1]), textcoords="offset points", xytext=(9, 0),
                    va="center", fontsize=d.text)
    ax.set_xlabel("Depth / K", fontsize=d.text)
    ax.set_title("Recoverability versus depth per dimension\n(50 \u00b5m, K = 30)", loc="left",
                 fontsize=d.title)
    save_full(fig, FIG, "E")
    return pv.round(3).to_dict()


# =========================================================================== F
def f_data():
    reps = pd.read_csv(RESULTS / "stage_matched_n_reps.csv")
    return reps[reps.rung_n == 253][[f"I_{s}" for s in STAGES]].values


def draw_f(ax, M):
    xs = np.arange(len(STAGES))
    ax.axhline(0, color="#999", lw=1.0, ls=":", zorder=1)
    for row in M:
        ax.plot(xs, row, color="#a8bccf", lw=0.5, alpha=0.5, zorder=2)
    ax.plot(xs, np.median(M, axis=0), color="#1f3b57", lw=1.8, zorder=4)
    for i, s in enumerate(STAGES):
        ax.plot([i], [np.median(M[:, i])], "o", color=SC[s], ms=9, markeredgecolor="white",
                markeredgewidth=1.0, zorder=5)
    ax.set_xlim(-0.4, len(STAGES) - 0.6); ax.set_ylim(-0.22, 0.52)


def panel_f():
    M = f_data()
    fig = px_figure(1928, 1438)
    draw_f(px_axes(fig, *axes_in_crop(1928, 1438, pad_px=24)), M)
    fig.savefig(marks_dir(FIG) / "Fig2_F.png", dpi=DPI); plt.close(fig)

    d = Layout(3.68)
    fig, ax = plt.subplots(figsize=(3.13, 2.32)); draw_f(ax, M); d.axes(ax)
    ax.set_xticks(range(len(STAGES))); ax.set_xticklabels(STAGES)
    ax.set_yticks(np.round(np.arange(-0.2, 0.51, 0.1), 1))
    ax.set_xlabel("Developmental stage", fontsize=d.text)
    ax.set_title("Recoverability at matched number of pseudospots\n(n = 253)", loc="left",
                 fontsize=d.title)
    save_full(fig, FIG, "F")
    top = int((M.argmax(1) == STAGES.index("GW12")).sum())
    return dict(zip(STAGES, np.round(np.median(M, 0), 3))), f"GW12 highest in {top}/{len(M)}"


if __name__ == "__main__":
    print("A maximum per stage:", panel_a())
    print("B (thickness um, estimate):", panel_b())
    print("C depth gain (lo, median, hi):", panel_c())
    print("D matched depth:\n", panel_d())
    panel_e()
    print("F medians:", panel_f())
