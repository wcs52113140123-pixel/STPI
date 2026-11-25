#!/usr/bin/env python3
"""Figure S1. The coordinate, the decoded field and the simulation checks.

  A  Expression-independent coordinate with a pia->dura normal field (STPI)
  B  Coordinates predicted by gene expression
  C  Changes in recoverability with signal strength beta
  D  Changes in recoverability with coordinate noise sigma
  E  Null permutation P ECDF

Inputs
  data_derived/gw6_two_sections.npz, data_derived/fivestage_normR.npz
  results/decoded_maps_meninges.npz
  results/sim_calibration_results.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib import gridspec
from matplotlib.lines import Line2D
from PIL import Image
from scipy.spatial import cKDTree

from _common import RESULTS, DERIVED, STAGES, Layout, save_marks, save_full, marks_dir, legend_key
from _fields import rank01, spot_field

FIG = "FigS1"
UM_PER_UNIT = 0.5
BAR_UM = {"GW6": 1000, "GW8": 1000, "GW10": 2000, "GW12": 5000, "GW16": 10000}

# width (cm) of each panel in the assembled figure layout; layout factor k = this / rendered width
FIG_CM = {"A": 121.1, "B": 119.8, "C": 33.8, "D": 34.2, "E": 32.7}


def layout_for(panel, dpi=600):
    """Figure layout size for a panel, from the width of its marks image."""
    w_px = Image.open(marks_dir(FIG) / f"{FIG}_{panel}.png").size[0]
    return Layout(FIG_CM[panel] / (w_px / dpi * 2.54))


# =========================================================================== A
def _rot(x, y, deg):
    t = np.deg2rad(deg)
    return x * np.cos(t) - y * np.sin(t), x * np.sin(t) + y * np.cos(t)


def _centre(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return x - (x.min() + x.max()) / 2, y - (y.min() + y.max()) / 2


def _sections():
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
    return shapes, size


def _poisson(X, Y, dmin, rng):
    order = rng.permutation(len(X)); sel = []; pts = np.c_[X, Y]; chosen = np.empty((0, 2))
    for i in order:
        if len(chosen) == 0 or np.min(np.sum((chosen - pts[i]) ** 2, axis=1)) >= dmin ** 2:
            sel.append(i); chosen = np.vstack([chosen, pts[i]])
    return np.array(sel)


def build_a(with_text=False, d=None):
    """Sections as in Fig 1B (same display sizes), with the pia->dura normal field drawn as
    arrows: the direction of the local gradient of R, fitted on the 11 nearest units at
    points sampled with a minimum spacing (Poisson disc, seed 3)."""
    shapes, size = _sections()
    MULT = {"GW6": 0.085, "GW8": 0.085, "GW10": 0.085, "GW12": 0.075, "GW16": 0.042}
    Lfac = 0.18
    at = [-1e9]
    heads = []          # every arrow head, so the saved image can be made to contain them

    def arrows(ax, X, Y, R, sel, L, lw, ms):
        xy = np.c_[X, Y]; tr = cKDTree(xy)
        for i in sel:
            _, ii = tr.query(xy[i], k=12); ii = ii[1:]
            A = np.c_[X[ii] - X[i], Y[ii] - Y[i]]; b = R[ii] - R[i]
            g = np.linalg.lstsq(A, b, rcond=None)[0]; nn = np.hypot(*g)
            if nn < 1e-6:
                continue
            gn = g / nn; ty = Y[i] + gn[1] * L; at[0] = max(at[0], ty)
            heads.append((X[i] + gn[0] * L, ty))
            # annotation_clip=False: an arrow whose head falls outside the axes box (the
            # GW16 hook, the right flank of GW16) is still drawn; by default matplotlib
            # drops it silently, which would leave eight of the 173 arrows out of the figure
            ax.annotate("", xy=(X[i] + gn[0] * L, ty), xytext=(X[i], Y[i]),
                        arrowprops=dict(arrowstyle="->", color="#c00000", lw=lw,
                                        mutation_scale=ms), zorder=5,
                        annotation_clip=False)

    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(16.5, 5.0))
    xcur, base_gap, prevL = 0.0, 0.34, 0.0
    cols = {}
    for st in STAGES:
        parts = shapes[st]
        dx = np.ptp(np.concatenate([p[0] for p in parts])); dy = np.ptp(np.concatenate([p[1] for p in parts]))
        scale = size[st] / max(dx, dy); w = dx * scale
        frac = size[st] / size["GW16"]; L, lw, ms = size[st] * Lfac, 0.8 + 1.0 * frac, 6 + 7 * frac
        if st != "GW6":
            xcur += base_gap + 0.65 * (prevL + L)
        cx = xcur + w / 2; ps = 2.2 + 2.2 * size[st] / size["GW16"]; dmin = size[st] * MULT[st]
        if st == "GW6":
            H = dy * scale
            for (xx, yy, rr), yo in zip(parts, [H * 1.55, -H * 1.10]):
                Xp = cx + xx * scale; Yp = yo + yy * scale
                sc = ax.scatter(Xp, Yp, c=rr, s=ps, cmap="viridis", vmin=0, vmax=1, edgecolor="none")
                arrows(ax, Xp, Yp, rr, _poisson(Xp, Yp, dmin, rng), L, lw, ms)
            gw6_bot = -H * 1.10 - H * 0.55
        else:
            xx, yy, rr = parts[0]; Xp = cx + xx * scale; Yp = yy * scale
            sc = ax.scatter(Xp, Yp, c=rr, s=ps, cmap="viridis", vmin=0, vmax=1, edgecolor="none")
            arrows(ax, Xp, Yp, rr, _poisson(Xp, Yp, dmin, rng), L, lw, ms)
        cols[st] = dict(cx=cx, scale=scale)
        xcur += w; prevL = L
    ax.set_aspect("equal"); ax.axis("off")
    yrange = at[0] - gw6_bot; title_y = at[0] + 0.06 * yrange
    ytop = title_y + 0.10 * yrange
    ybot = gw6_bot - abs(gw6_bot) * 0.08
    if with_text:
        for st in STAGES:
            ax.text(cols[st]["cx"], title_y, "GW6-S1" if st == "GW6" else st, ha="center",
                    va="bottom", fontsize=d.text)
        ax.text(cols["GW6"]["cx"], gw6_bot, "GW6-S2", ha="center", va="top", fontsize=d.text)
        yb = gw6_bot - 0.24 * yrange
        for st in STAGES:
            Lb = BAR_UM[st] / UM_PER_UNIT * cols[st]["scale"]; c = cols[st]["cx"]
            ax.plot([c - Lb / 2, c + Lb / 2], [yb, yb], color="k", lw=6.0 / d.k, solid_capstyle="butt")
            ax.text(c, yb + 0.02 * yrange, f"{BAR_UM[st] // 1000}mm", ha="center", va="bottom",
                    fontsize=d.text)
        ybot = yb - 0.05 * yrange
    ax.set_ylim(ybot, ytop)
    # The arrows at the GW16 hook point below the frame. A transparent box over the arrow
    # heads that lie outside it, not clipped, makes the tight bounding box of the saved image
    # reach them; the frame, the scale and every other mark stay where they were.
    hx, hy = np.array(heads).T; m = 0.03 * yrange
    out = hy < ybot
    if out.any():
        # add_artist, not add_patch: the box must not enter the data limits, or the
        # autoscaled x range and with it the scale of every section would change
        ax.add_artist(Rectangle((hx[out].min() - m, hy[out].min() - m),
                                hx[out].max() - hx[out].min() + 2 * m, ybot - hy[out].min() + m,
                                facecolor="none", edgecolor="none", clip_on=False))
    cax = fig.add_axes([0.92, 0.28, 0.013, 0.44]); cb = fig.colorbar(sc, cax=cax)
    if with_text:
        cb.set_ticks(np.round(np.arange(0, 1.01, 0.2), 1))
        cb.ax.tick_params(labelsize=d.text, width=d.line)
        cax.yaxis.set_label_position("left"); cb.set_label("Coordinate R", fontsize=d.text)
        cb.outline.set_linewidth(d.line)
    else:
        cb.set_ticks([]); cb.outline.set_linewidth(1.0)
    return fig, ax, cols


def panel_a():
    fig, _, _ = build_a()
    save_marks(fig, FIG, f"{FIG}_A", pad_inches=0.05)
    d = layout_for("A")
    fig, ax, cols = build_a(with_text=True, d=d)
    ax.set_title("Expression-independent coordinate with a pia\u2192dura normal field (STPI)",
                 loc="left", fontsize=d.title)
    save_full(fig, FIG, "A")
    return {st: round(BAR_UM[st] / UM_PER_UNIT * c["scale"], 4) for st, c in cols.items()}


# =========================================================================== B
DM_ORDER = ["GW6", "GW6-S2", "GW8", "GW10", "GW12", "GW16"]


def build_b(with_text=False, d=None):
    """Decoded value of each pseudospot (rank-scaled to [0, 1]) drawn on the tissue at native
    cell resolution; layout and orientation as in Fig 1B (GW6 rotated 270 degrees)."""
    DM = np.load(RESULTS / "decoded_maps_meninges.npz")
    CM = 1 / 2.54
    fig = plt.figure(figsize=(128.0 * CM, (26.0 if not with_text else 34.0) * CM))
    m = dict(left=0.010, right=0.990, top=0.985, bottom=0.015) if not with_text \
        else dict(left=0.010, right=0.905, top=0.83, bottom=0.17)
    gs = gridspec.GridSpec(2, 5, figure=fig, wspace=0.05, hspace=0.16, **m)
    SPAN = ["GW8", "GW10", "GW12", "GW16"]
    axes, widths = {}, {}
    for st, cellspec in [("GW6", gs[0, 0]), ("GW6-S2", gs[1, 0])] + \
                        [(s, gs[:, j + 1]) for j, s in enumerate(SPAN)]:
        sxy = np.asarray(DM[f"{st}_spot_xy"], float)
        sval = np.asarray(DM[f"{st}_spot_val"], float)
        keep = np.isfinite(sval)
        if st.startswith("GW6"):
            qx, qy = sxy[keep, 1], -sxy[keep, 0]          # rotate 270 degrees
        else:
            qx, qy = sxy[keep, 0], sxy[keep, 1]
        Q = np.c_[qx, qy]
        nat = float(np.median(cKDTree(Q).query(Q, k=2)[0][:, 1]))
        pitch = 3.0 * nat
        span = max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1]))
        up = float(np.clip(1500.0 * pitch / span, 2.0, 8.0))
        gx, gy, Z, Al = spot_field(Q, rank01(sval[keep]), pitch, up=up)
        ax = fig.add_subplot(cellspec)
        im = ax.imshow(Z, cmap="viridis", vmin=0, vmax=1, origin="lower", alpha=Al,
                       extent=[gx[0], gx[-1], gy[0], gy[-1]], interpolation="nearest",
                       rasterized=True)
        ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.02)
        axes[st] = ax
    if with_text:
        fig.canvas.draw()
        pos = {st: ax.get_position() for st, ax in axes.items()}
        y_top = max(pos[st].y1 for st in ["GW6"] + SPAN) + 0.015
        for st in ["GW6"] + SPAN:
            fig.text((pos[st].x0 + pos[st].x1) / 2, y_top, "GW6-S1" if st == "GW6" else st,
                     ha="center", va="bottom", fontsize=d.text)
        fig.text((pos["GW6-S2"].x0 + pos["GW6-S2"].x1) / 2, pos["GW6-S2"].y1 + 0.01, "GW6-S2",
                 ha="center", va="bottom", fontsize=d.text)
        for st in ["GW6"] + SPAN:
            ax = axes[st]; p = pos[st]
            x0, x1 = ax.get_xlim()
            um = BAR_UM[st] if st != "GW6" else BAR_UM["GW6"]
            frac = um / (x1 - x0) * p.width          # decoded-map coordinates are in um
            xc = (p.x0 + p.x1) / 2
            fig.add_artist(Line2D([xc - frac / 2, xc + frac / 2], [0.09, 0.09], color="k",
                                  lw=6.0 / d.k, transform=fig.transFigure, solid_capstyle="butt"))
            fig.text(xc, 0.11, f"{um // 1000}mm", ha="center", va="bottom", fontsize=d.text)
        cax = fig.add_axes([0.925, 0.35, 0.007, 0.40])
        cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=d.text, width=d.line); cb.outline.set_linewidth(d.line)
        cax.yaxis.set_label_position("left"); cb.set_label("Value", fontsize=d.text)
    return fig


def panel_b():
    fig = build_b()
    save_marks(fig, FIG, f"{FIG}_B", dpi=300, bbox_inches=None)
    d = layout_for("B", dpi=300)
    fig = build_b(with_text=True, d=d)
    fig.text(0.010, 0.995, "Coordinates predicted by gene expression", ha="left", va="top",
             fontsize=d.title)
    save_full(fig, FIG, "B", dpi=300)     # a very wide panel; 600 dpi gives a 200-megapixel PNG


# =========================================================================== C-E
def _sim():
    return pd.read_csv(RESULTS / "sim_calibration_results.csv")


def _lighten(c, f):
    r, g, b = mc.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


C_LINE = "#DE77AE"


def draw_c(ax, sim):
    s1 = sim[(sim.exp == "S1_signal") & (sim.cv == "spatial")].copy()
    agg = s1.groupby("beta_neigh").dI.mean()
    jit = np.random.default_rng(0).uniform(-0.015, 0.015, len(s1))
    ax.axhline(0, color="#888", lw=1.3, ls=":", zorder=0)
    ax.scatter(s1.beta_neigh + jit, s1.dI, s=34, color=_lighten(C_LINE, 0.45), alpha=0.55,
               edgecolor="white", linewidth=0.2, zorder=2)
    ax.plot(agg.index, agg.values, "-o", color=C_LINE, ms=11, lw=3.0, zorder=3)
    return agg


def panel_c():
    sim = _sim()
    fig, ax = plt.subplots(figsize=(6.4, 5.4)); draw_c(ax, sim); ax.axis("off"); fig.tight_layout()
    save_marks(fig, FIG, f"{FIG}_C")
    d = layout_for("C")
    fig, ax = plt.subplots(figsize=(6.4, 5.4)); agg = draw_c(ax, sim); d.axes(ax)
    ax.set_xticks(np.round(np.arange(0, 1.01, 0.2), 1))
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"])
    ax.legend(handles=[Line2D([0], [0], color=C_LINE, lw=3.0, marker="o", ms=11),
                       Line2D([0], [0], color="#888", lw=1.3, ls=":")],
              labels=["", "Zero (null)"], frameon=False, loc="upper left", fontsize=d.text)
    ax.set_xlabel("Signal strength \u03b2", fontsize=d.text)
    ax.set_title("Changes in recoverability with signal strength \u03b2", loc="left",
                 fontsize=d.title, pad=d.text)
    legend_key(ax, marks_dir(FIG) / f"{FIG}_C_key.png", scale=1.4)   # key placed beside the panel
    fig.tight_layout(); save_full(fig, FIG, "C")
    return agg.round(4).to_dict()


C_FILL, C_PT, C_MEAN, C_GRAND = "#6cbf84", "#2ca25f", "#0b5d2e", "#9aa7b5"
D_FIGSIZE = (6.2, 5.4)


def draw_d(ax, sim):
    s4 = sim[(sim.exp == "S4_coordnoise") & (sim.cv == "spatial")]
    sig = sorted(s4.sigma_coord.unique())
    data = [s4[s4.sigma_coord == s].dI.values for s in sig]
    grand = np.concatenate(data).mean()
    ax.axhline(grand, color=C_GRAND, lw=1.4, ls=(0, (5, 4)), zorder=0)
    n0 = len(ax.collections)
    sns.violinplot(data=data, cut=0, inner=None, linewidth=1.2, ax=ax, palette=[C_FILL] * len(sig),
                   saturation=1.0, width=0.72)
    for coll in ax.collections[n0:]:
        coll.set_alpha(0.40); coll.set_edgecolor(C_PT)
    rng = np.random.default_rng(2)
    for i, dd in enumerate(data):
        ax.scatter(i + rng.uniform(-.085, .085, len(dd)), dd, s=26, color=C_PT, alpha=0.55,
                   edgecolor="white", linewidth=0.25, zorder=3)
        ax.plot([i - 0.26, i + 0.26], [np.mean(dd)] * 2, color=C_MEAN, lw=4.2, zorder=4,
                solid_capstyle="round")
    ax.set_ylim(0, 0.31)
    return sig, [float(np.mean(x)) for x in data], float(grand)


def panel_d():
    sim = _sim()
    fig, ax = plt.subplots(figsize=D_FIGSIZE); draw_d(ax, sim); ax.axis("off"); fig.tight_layout()
    save_marks(fig, FIG, f"{FIG}_D")
    d = layout_for("D")
    fig, ax = plt.subplots(figsize=D_FIGSIZE); sig, means, grand = draw_d(ax, sim); d.axes(ax)
    ax.set_xticks(range(len(sig))); ax.set_xticklabels(["0.00", "0.02", "0.05", "0.1", "0.2"])
    ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_yticklabels(["0.05", "0.10", "0.15", "0.20", "0.25", "0.30"])
    ax.legend(handles=[Line2D([0], [0], color=C_MEAN, lw=4.2),
                       Line2D([0], [0], color=C_GRAND, lw=1.4, ls=(0, (5, 4)))],
              labels=["Per-\u03c3 mean", "Overall mean"], frameon=False, loc="upper left",
              fontsize=d.text)
    ax.set_xlabel("Coordinate noise \u03c3", fontsize=d.text)
    ax.set_title("Changes in recoverability with coordinate noise \u03c3", loc="left",
                 fontsize=d.title, pad=d.text)
    legend_key(ax, marks_dir(FIG) / f"{FIG}_D_key.png", scale=1.4)   # key placed beside the panel
    fig.tight_layout(); save_full(fig, FIG, "D")
    return dict(zip([f"{s:g}" for s in sig], np.round(means, 4))), round(grand, 4)


C_OBS, C_UNI = "#4c78a8", "#888"


def draw_e(ax, sim):
    tn = sim[(sim.exp == "typeI_null") & (sim.cv == "spatial")]
    pv = np.sort(tn.p_perm.dropna().values)
    ecdf = np.arange(1, len(pv) + 1) / len(pv)
    eps = np.sqrt(np.log(2 / 0.05) / (2 * len(pv)))        # DKW 95 % band
    ax.fill_between([0, 1], [-eps, 1 - eps], [eps, 1 + eps], color=C_OBS, alpha=0.12, zorder=0)
    ax.plot([0, 1], [0, 1], "--", color=C_UNI, lw=1.8, zorder=1)
    ax.step(pv, ecdf, where="post", color=C_OBS, lw=2.6, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return float((pv < 0.05).mean()), len(pv)


def panel_e():
    sim = _sim()
    fig, ax = plt.subplots(figsize=(6.0, 5.4)); draw_e(ax, sim); ax.axis("off"); fig.tight_layout()
    save_marks(fig, FIG, f"{FIG}_E")
    d = layout_for("E")
    fig, ax = plt.subplots(figsize=(6.0, 5.4)); t1, n = draw_e(ax, sim); d.axes(ax)
    ax.set_xticks(np.round(np.arange(0, 1.01, 0.2), 1))
    ax.set_yticks(np.round(np.arange(0.2, 1.01, 0.2), 1))
    ax.legend(handles=[Line2D([0], [0], color=C_UNI, lw=1.8, ls="--"),
                       Line2D([0], [0], color=C_OBS, lw=2.6)],
              labels=["Uniform", "Observed null"], frameon=False, loc="upper left",
              fontsize=d.text)
    ax.set_xlabel("Permutation P", fontsize=d.text); ax.set_ylabel("ECDF", fontsize=d.text)
    ax.set_title("Null permutation P ECDF", loc="left", fontsize=d.title, pad=d.text)
    legend_key(ax, marks_dir(FIG) / f"{FIG}_E_key.png", scale=1.4)   # key placed beside the panel
    fig.tight_layout(); save_full(fig, FIG, "E")
    return round(t1, 3), n


if __name__ == "__main__":
    print("A scale bar lengths (data units):", panel_a())
    panel_b()
    print("C mean dI per beta:", panel_c())
    print("D mean dI per sigma, overall:", panel_d())
    print("E type-I rate at 0.05, n:", panel_e())
