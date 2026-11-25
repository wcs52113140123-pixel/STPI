#!/usr/bin/env python3
"""Figure 1. STPI framework and the estimate across meningeal development.

Panel A (framework schematic) is drawn separately and is not produced here.

  B  Expression-independent coordinate across meningeal development (STPI)
  C  Recoverability under spatial block and random CV
  D  Recoverability against an autocorrelation-preserving null
  E  Recoverability across development

Inputs
  data_derived/gw6_two_sections.npz        GW6 sections S1 and S2: x, y, coordinate R
  data_derived/fivestage_normR.npz         GW8 to GW16: x, y, coordinate R
  results/fivestage_random_vs_checker_C.csv
  results/tangential_null_draws.npz
  results/multiGW_grid_all.csv

Stereo-seq coordinates are DNB units, 1 unit = 0.5 um.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import make_interp_spline

from _common import RESULTS, DERIVED, STAGES, Layout, save_marks, save_full, legend_key, marks_dir, colourbar_png

FIG = "Fig1"
UM_PER_UNIT = 0.5
C_RAND, C_CHK = "#dfa6a7", "#5a966c"
C_OBS, C_NULL, C_MED = "#F2C053", "#ED7439", "#2B9D81"
PINK, RAIL = "#e8637a", "#d9d9d9"

# layout factor k for each panel image (layout canvas width / rendered width)
K = {"B": 3.99, "C": 3.64, "D": 2.39, "E": 4.92}


# =========================================================================== B
def _rot(x, y, deg):
    t = np.deg2rad(deg)
    return x * np.cos(t) - y * np.sin(t), x * np.sin(t) + y * np.cos(t)


def _centre(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return x - (x.min() + x.max()) / 2, y - (y.min() + y.max()) / 2


def layout_b():
    """Sections in one row, growing with development.

    Display size is (true diagonal / GW8 diagonal) ** 0.55, with GW10 and the GW6 pair
    further scaled by 0.75, so the row fits the figure. Columns are therefore not on a
    common scale and each carries its own scale bar at true length.
    """
    g6 = np.load(DERIVED / "gw6_two_sections.npz")
    nz = np.load(DERIVED / "fivestage_normR.npz")
    x1, y1 = _centre(*_rot(g6["S1_xs"].astype(float), g6["S1_ys"].astype(float), 270))
    x2, y2 = _centre(*_rot(g6["S2_xs"].astype(float), g6["S2_ys"].astype(float), 270))
    shapes = {"GW6": [(x1, y1, g6["S1_Rn"]), (x2, y2, g6["S2_Rn"])]}
    diag = {}
    for st in STAGES[1:]:
        xx, yy = _centre(nz[f"{st}_x"].astype(float), nz[f"{st}_y"].astype(float))
        shapes[st] = [(xx, yy, nz[f"{st}_R"])]
        diag[st] = np.hypot(np.ptp(xx), np.ptp(yy))
    real = {"GW6": np.hypot(np.ptp(x1), np.ptp(y1)) / diag["GW8"] * 0.75, "GW8": 1.0,
            **{st: diag[st] / diag["GW8"] for st in ("GW10", "GW12", "GW16")}}
    size = {k: v ** 0.55 for k, v in real.items()}
    size["GW10"] *= 0.75
    cols, xcur, gap = {}, 0.0, 0.42
    for st in STAGES:
        parts = shapes[st]
        dx = np.ptp(np.concatenate([p[0] for p in parts]))
        dy = np.ptp(np.concatenate([p[1] for p in parts]))
        scale = size[st] / max(dx, dy)
        w = dx * scale
        cols[st] = dict(parts=parts, cx=xcur + w / 2, scale=scale, h=dy * scale,
                        ps=2.2 + 2.2 * size[st] / size["GW16"])
        xcur += w + gap
    return cols


def draw_b(ax, cols):
    sc = None
    for st in STAGES:
        c = cols[st]
        if st == "GW6":
            for (xx, yy, rr), yo in zip(c["parts"], [c["h"] * 1.05, -c["h"] * 1.05]):
                sc = ax.scatter(c["cx"] + xx * c["scale"], yo + yy * c["scale"], c=rr,
                                s=c["ps"], cmap="viridis", vmin=0, vmax=1, edgecolor="none")
        else:
            xx, yy, rr = c["parts"][0]
            sc = ax.scatter(c["cx"] + xx * c["scale"], yy * c["scale"], c=rr, s=c["ps"],
                            cmap="viridis", vmin=0, vmax=1, edgecolor="none")
    ax.set_aspect("equal"); ax.axis("off")
    return sc


# Annotation positions, in the data units of draw_b
B_SCALEBAR = {"GW6": (0.682, 1000), "GW8": (1.879, 1000), "GW10": (3.366, 2000),
              "GW12": (6.150, 5000), "GW16": (10.906, 10000)}   # left end, length (um)
B_BAR_Y, B_BARLABEL_Y, B_STAGE_Y, B_S2_Y = -1.713, -1.593, 1.593, -1.379
B_LEADERS = {  # label position, then the two sections it points to (S1, S2)
    "Brain\ncells": ((0.036, 0.003), [((-0.040, 0.251), (0.107, 0.748)),
                                      ((-0.034, -0.243), (0.209, -0.681))]),
    "Brain\noutside": ((0.549, -0.058), [((0.425, 0.182), (0.599, 0.588)),
                                         ((0.384, -0.321), (0.395, -0.581))]),
}


def panel_b():
    cols = layout_b()
    fig, ax = plt.subplots(figsize=(16, 4.2))
    draw_b(ax, cols)
    save_marks(fig, FIG, "Fig1_B", pad_inches=0.05)
    # colour bar positioned beside panel B (and S1 in panel A): 74 x 498 px, 3 px black frame
    colourbar_png(marks_dir(FIG) / "Fig1_B_colourbar.png", "viridis", 74, 498,
                  ((5, 67), (5, 491)), 3)

    d = Layout(K["B"])
    fig, ax = plt.subplots(figsize=(16, 4.9))
    sc = draw_b(ax, cols)
    for st in STAGES:
        ax.text(cols[st]["cx"], B_STAGE_Y, "GW6-S1" if st == "GW6" else st,
                ha="center", va="center", fontsize=d.text)
    ax.text(cols["GW6"]["cx"], B_S2_Y, "GW6-S2", ha="center", va="center", fontsize=d.text)
    for st, (x0, um) in B_SCALEBAR.items():
        L = um / UM_PER_UNIT * cols[st]["scale"]
        ax.plot([x0, x0 + L], [B_BAR_Y] * 2, color="k", lw=6.0 / d.k, solid_capstyle="butt")
        ax.text(x0 + L / 2, B_BARLABEL_Y, f"{um // 1000}mm", ha="center", va="center",
                fontsize=d.text)
    for lab, (xy, segs) in B_LEADERS.items():
        ax.text(*xy, lab, ha="center", va="center", fontsize=d.text, linespacing=1.0)
        for p, q in segs:
            ax.plot([p[0], q[0]], [p[1], q[1]], color="k", lw=d.line * 0.6)
    ax.set_ylim(-1.80, 1.70)
    ax.set_title("Expression-independent coordinate across meningeal development (STPI)",
                 loc="left", fontsize=d.title)
    cax = fig.add_axes([0.915, 0.30, 0.008, 0.42])
    cb = fig.colorbar(sc, cax=cax); cb.set_ticks(np.round(np.arange(0, 1.01, 0.2), 1))
    cb.outline.set_linewidth(d.line); cb.ax.tick_params(labelsize=d.text, width=d.line)
    cax.yaxis.set_label_position("left")
    cb.set_label("Coordinate R", fontsize=d.text, labelpad=d.text * 0.4)
    save_full(fig, FIG, "B")
    return {st: round(um / UM_PER_UNIT * cols[st]["scale"], 3) for st, (_, um) in B_SCALEBAR.items()}


# =========================================================================== C
def _c_layout():
    fig = plt.figure(figsize=(4.2, 3.6))
    gs = GridSpec(2, 1, height_ratios=[2.4, 1.0], hspace=0.12)
    return fig, fig.add_subplot(gs[0]), fig.add_subplot(gs[1])


def draw_c(ax, ax2, df):
    x = np.arange(len(STAGES)); w = 0.38
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylim(-0.08, 0.60)
    ax.bar(x - w / 2, df.I_random, w, color=C_RAND)
    ax.bar(x + w / 2, df.I_checker, w, color=C_CHK)
    frac = df.frac_artifact.values * 100
    m = np.isfinite(frac); xv, yv = x[m], frac[m]
    xs = np.linspace(xv.min(), xv.max(), 200)
    ax2.plot(xs, make_interp_spline(xv, yv, k=3)(xs), color="black", lw=1.4, zorder=1)
    ax2.scatter(xv, yv, s=22, color="black", zorder=2)
    ax2.set_xlim(ax.get_xlim()); ax2.set_ylim(0, 100)


def panel_c():
    df = pd.read_csv(RESULTS / "fivestage_random_vs_checker_C.csv").set_index("stage").loc[STAGES].reset_index()
    fig, ax, ax2 = _c_layout(); draw_c(ax, ax2, df)
    ax.axis("off"); ax2.axis("off")
    save_marks(fig, FIG, "Fig1_C")

    d = Layout(K["C"])
    fig, ax, ax2 = _c_layout(); draw_c(ax, ax2, df)
    x = np.arange(len(STAGES))
    d.axes(ax); d.axes(ax2)
    ax.set_xticks(x); ax.set_xticklabels([])
    ax.set_yticks(np.round(np.arange(0, 0.61, 0.1), 1))
    ax.legend(handles=[Patch(color=C_CHK), Patch(color=C_RAND)],
              labels=["Spatial block CV", "Random CV"], frameon=False, loc="upper left",
              fontsize=d.text, handlelength=1.6, handleheight=1.1)
    ax2.set_xticks(x); ax2.set_xticklabels(STAGES)
    ax2.set_yticks([50, 100])
    ax2.set_ylabel("Fraction (%)", fontsize=d.text)
    ax2.set_xlabel("Developmental stage", fontsize=d.text)
    ax.set_title("Recoverability under spatial block and random CV", loc="left",
                 fontsize=d.title, pad=d.text)
    save_full(fig, FIG, "C")
    return dict(zip(df.stage, df.frac_artifact.round(3)))


# =========================================================================== D
def _d_data():
    z = np.load(RESULTS / "tangential_null_draws.npz")
    return [z[f"{s}_null"] for s in STAGES], [float(z[f"{s}_iobs"]) for s in STAGES]


def draw_d(ax, nulls, iobs):
    xp = np.arange(len(STAGES))
    ax.axhline(0, color="#bbb", lw=1.0, ls=":", zorder=0)
    vp = ax.violinplot(nulls, positions=xp, showmeans=False, showextrema=False, widths=0.72)
    for b_ in vp["bodies"]:
        b_.set_facecolor(C_NULL); b_.set_alpha(0.32); b_.set_edgecolor(C_NULL); b_.set_linewidth(1.0)
    for i in xp:
        ax.hlines(np.median(nulls[i]), i - 0.18, i + 0.18, color=C_MED, lw=2.6, zorder=3)
        ax.hlines(np.percentile(nulls[i], 95), i - 0.18, i + 0.18, color=C_NULL, lw=2.2, zorder=3)
    ax.scatter(xp, iobs, marker="*", s=340, color=C_OBS, edgecolor="k", linewidth=0.8, zorder=5)
    ax.set_ylim(min(min(n.min() for n in nulls), min(iobs)) - 0.02, max(iobs) + 0.06)
    ax.set_xticks(xp); ax.set_xticklabels(STAGES)


def _d_keys():
    return [Patch(facecolor=C_NULL, alpha=0.32), Line2D([0], [0], color=C_NULL, lw=2.2),
            Line2D([0], [0], color=C_MED, lw=2.6),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=C_OBS,
                   markeredgecolor="k", ms=16)]


def panel_d():
    nulls, iobs = _d_data()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    draw_d(ax, nulls, iobs)
    ax.set_ylabel(r"$\hat{I}$ (bit)", fontsize=15); ax.tick_params(labelsize=13)
    ax.legend(handles=_d_keys(), labels=["", "", "", ""], frameon=False, loc="upper left",
              handlelength=1.8, labelspacing=0.5)
    for s in ax.spines.values():
        s.set_linewidth(1.3)
    fig.tight_layout()
    save_marks(fig, FIG, "Fig1_D")

    d = Layout(K["D"])
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    draw_d(ax, nulls, iobs)
    d.axes(ax)
    ax.set_yticks(np.round(np.arange(-0.2, 0.41, 0.1), 1))
    ax.legend(handles=_d_keys(), labels=["Null distribution", "Null 95th pct", "Null median", ""],
              frameon=False, loc="upper left", handlelength=1.8, labelspacing=0.5,
              fontsize=d.text)
    ax.set_xlabel("Developmental stage", fontsize=d.text)
    ax.set_title("Recoverability against an autocorrelation-preserving null", loc="left",
                 fontsize=d.title, pad=d.text)
    legend_key(ax, marks_dir(FIG) / f"{FIG}_D_key.png", scale=1.4)   # key placed beside the panel
    save_full(fig, FIG, "D")
    p = [(np.sum(n >= o) + 1) / (len(n) + 1) for n, o in zip(nulls, iobs)]
    return dict(zip(STAGES, np.round(p, 3)))


# =========================================================================== E
def draw_e(ax, rho):
    xp = np.arange(len(STAGES)); xs = np.linspace(0, 4, 300)
    for i in xp:
        ax.axvline(i, color=RAIL, lw=9, zorder=0, ymin=0.02, ymax=0.98)
    ax.plot(xs, make_interp_spline(xp, rho, k=3)(xs), color=PINK, lw=2.6, zorder=2,
            solid_capstyle="round")
    ax.scatter(xp, rho, s=260, facecolor="white", edgecolor=PINK, linewidths=3.2, zorder=3)
    ax.set_ylim(0, 0.95); ax.set_xlim(-0.5, 4.5)


def panel_e():
    grid = pd.read_csv(RESULTS / "multiGW_grid_all.csv")
    rho = [float(grid[grid.stage == g].loc[grid[grid.stage == g]["I"].idxmax()]["rho"])
           for g in STAGES]
    fig, ax = plt.subplots(figsize=(4.0, 3.8)); draw_e(ax, rho); ax.axis("off")
    save_marks(fig, FIG, "Fig1_E")

    d = Layout(K["E"])
    fig, ax = plt.subplots(figsize=(4.0, 3.8)); draw_e(ax, rho)
    d.axes(ax)
    ax.set_xticks(range(5)); ax.set_xticklabels(STAGES)
    ax.set_yticks(np.round(np.arange(0, 0.81, 0.2), 1))
    ax.set_ylabel(r"$\rho$", fontsize=d.text)
    ax.set_xlabel("Developmental stage", fontsize=d.text)
    ax.set_title("Recoverability across development", loc="left", fontsize=d.title, pad=d.text)
    save_full(fig, FIG, "E")
    return dict(zip(STAGES, np.round(rho, 3)))


if __name__ == "__main__":
    print("B scale bar lengths (data units):", panel_b())
    print("C fraction of the random-CV estimate that is inflation:", panel_c())
    print("D null P:", panel_d())
    print("E best decode rho:", panel_e())
