#!/usr/bin/env python3
"""Figure S2. Measurement conditions and robustness behind Figure 2.

  A  Pseudospot count and median UMI per pseudospot at each setting
  B  Fraction of draws yielding the highest recoverability (800 um, depth 512)
  C  Median depth gain across bootstrap block sizes
  D  Recoverability at two matched depths
  E  Recoverability versus depth per dimension at each pseudospot size (K = 30)
  F  Spread between stages against decoder dimensionality K

The sheet is composed once (compose); the marks images and the complete panels are
cut from it. B is drawn on its own, as in the figure.
panel_a ... panel_f each draw the panel of the same letter.

Inputs (results/)
  multiGW_grid_all.csv, depth_matched_sscan_5stage.csv, multiGW_bootstrap.csv,
  multiGW_effects_by_stage.csv, best_size_boot_raw.json,
  depth_ceiling_triple_collinearity.csv, stage_matched_n_reps.csv, matched_depth_bin_bias.csv,
  K_sensitivity.csv, and the block-size sweep read inside panel C.
"""
import sys
import json
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon
from matplotlib.collections import PolyCollection
from matplotlib.transforms import Bbox
from scipy.stats import spearmanr
import seaborn as sns

from _common import RESULTS, FONT, marks_dir, colourbar_png, OUT

FIG = "FigS2"
B = str(RESULTS.parent)          # panel C reads f"{B}/results/..."

STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]
SC = {"GW6": "#4c78a8", "GW8": "#59a14f", "GW10": "#9c755f",
      "GW12": "#e15759", "GW16": "#f2a53c"}
SCALES = [50, 100, 200, 400, 800]
FOCAL = "GW12"
BASE, ANNO, TICK = 11, 8.5, 8
plt.rcParams.update({
    "font.family": FONT, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
    "font.size": BASE, "axes.labelsize": BASE - 0.5, "axes.titlesize": BASE,
    "legend.fontsize": ANNO, "xtick.labelsize": TICK, "ytick.labelsize": TICK})

grid = pd.read_csv(RESULTS / "multiGW_grid_all.csv")
dm = pd.read_csv(RESULTS / "depth_matched_sscan_5stage.csv")
bs = pd.read_csv(RESULTS / "multiGW_bootstrap.csv")
ef = pd.read_csv(RESULTS / "multiGW_effects_by_stage.csv")
# exact per-draw winners (the 2-dp shares in best_size_summary.csv do not sum to 500)
_raw = json.load(open(RESULTS / "best_size_boot_raw.json"))
WINS = {s_: Counter(int(v) for v in _raw[s_]) for s_ in STAGES}
tc = pd.read_csv(RESULTS / "depth_ceiling_triple_collinearity.csv")
reps = pd.read_csv(RESULTS / "stage_matched_n_reps.csv")
bb = pd.read_csv(RESULTS / "matched_depth_bin_bias.csv")
tci = tc.set_index("stage").reindex(STAGES)
V = {"T": tci.T_um.values.astype(float), "n": tci.n50.values.astype(float),
     "1/depth": 1.0 / tci.median_umi_per_bin.values.astype(float)}
ks = list(V)
rr = [abs(spearmanr(V[a], V[b_])[0]) for a in ks for b_ in ks if a < b_]
RHO_LO, RHO_HI = min(rr), max(rr)
ksn = pd.read_csv(RESULTS / "K_sensitivity.csv")


def panel_a(fig, slot):
    """A  Pseudospot count (numbers) and median counts per pseudospot (colour) at each setting."""
    # pseudospots with at least 16 counts, and counts per pseudospot
    axa = fig.add_subplot(slot)
    nps = (grid[grid.depth == 16].pivot_table(index="stage", columns="scale_um", values="n")
           .reindex(index=STAGES, columns=SCALES))          # same n as the main response-surface headers
    umi = dm[dm.C == 64].pivot_table(index="stage", columns="scale_um",
                                     values="med_umi", aggfunc="first").reindex(STAGES)
    natfeas = (grid[grid.feasible].groupby(["stage", "scale_um"]).size()
               .unstack().reindex(index=STAGES, columns=SCALES))
    M = np.log10(umi[SCALES].values.astype(float))
    im = axa.imshow(M, cmap="YlGnBu", aspect="auto", interpolation="nearest")
    for i, s_ in enumerate(STAGES):
        for j, sc in enumerate(SCALES):
            n_ = nps[SCALES].values[i, j]
            if not np.isfinite(n_):
                continue
            ok = np.isfinite(natfeas.values[i, j])
            axa.text(j, i, f"{int(n_):,}", ha="center", va="center", fontsize=ANNO,
                     color="white" if M[i, j] > 3.4 else "#222",
                     fontweight="bold" if ok else "normal", zorder=3)
            if not ok:
                axa.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="none",
                                        edgecolor="#c0392b", lw=1.6, hatch="///", zorder=0.5))
    axa.set_xticks(range(5)); axa.set_xticklabels(SCALES)
    axa.set_yticks(range(5)); axa.set_ylabel("developmental stage")
    axa.set_yticklabels(STAGES)
    for t_, s_ in zip(axa.get_yticklabels(), STAGES):
        t_.set_color(SC[s_]); t_.set_fontweight("bold")
    axa.set_xlabel("pseudospot size s (µm)")
    axa.tick_params(length=0)
    for sp in axa.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=axa, fraction=0.030, pad=0.02)
    # colour encodes the median count per pseudospot in that cell, not a sum across pseudospots
    cb.set_label("median counts per pseudospot (log10)", fontsize=ANNO - 0.5, labelpad=2)
    cb.ax.tick_params(labelsize=TICK - 1)
    axa.set_title("a   Measurement conditions at each setting",
                  loc="left", fontsize=BASE)
    return [axa, cb.ax]


def b_matrix():
    feas = {s: sorted(grid[(grid.stage == s) & grid.feasible & (grid.depth == 512)].scale_um)
            for s in STAGES}
    W = np.full((len(STAGES), len(SCALES)), np.nan)
    for i, s in enumerate(STAGES):
        assert sum(WINS[s].values()) == 500, s
        for j, sc in enumerate(SCALES):
            if sc in feas[s]:
                W[i, j] = WINS[s].get(int(sc), 0) / 500
    return W


def draw_b(ax, W):
    ax.imshow(np.ma.masked_invalid(W), cmap="Blues", vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    for i_ in range(len(STAGES) + 1):
        ax.axhline(i_ - 0.5, color="white", linewidth=1.4)
    for j_ in range(len(SCALES) + 1):
        ax.axvline(j_ - 0.5, color="white", linewidth=1.4)
    ax.set_xlim(-0.5, len(SCALES) - 0.5); ax.set_ylim(len(STAGES) - 0.5, -0.5)


def panel_b(ax, bare=False):
    """B  Share of the 500 bootstrap draws in which each pseudospot size reads highest
    (depth 512, 800 um blocks), as a stage x size grid with the share printed in each cell."""
    W = b_matrix()
    draw_b(ax, W)
    if bare:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return W
    for i in range(len(STAGES)):
        for j in range(len(SCALES)):
            v = W[i, j]
            ax.text(j, i, "\u2014" if not np.isfinite(v) else f"{v * 100:.0f}%", ha="center",
                    va="center", fontsize=ANNO, color="white" if np.isfinite(v) and v > 0.55
                    else ("#aaa" if not np.isfinite(v) else "#222"))
    ax.set_xticks(range(len(SCALES))); ax.set_xticklabels(SCALES, fontsize=TICK)
    ax.set_yticks(range(len(STAGES))); ax.set_yticklabels(STAGES, fontsize=TICK)
    ax.set_xlabel("Pseudospot size (\u03bcm)", fontsize=ANNO + 0.5)
    ax.set_ylabel("Developmental stage", fontsize=ANNO + 0.5)
    return W


def panel_c(fig, slot):
    """C  Median depth gain across bootstrap block sizes."""
    # tests whether the choice of bootstrap block size affects the depth-gain estimates
    axc = fig.add_subplot(slot)
    bsum = pd.read_csv(f"{B}/results/block_size_extended_summary.csv")
    bsum = bsum[bsum.kind == "E_depth"]
    BLKS = sorted(bsum.block_um.unique())
    NBLK = {}                                    # independent blocks per stage x block size
    _nz = np.load(f"{B}/data_derived/fivestage_normR.npz")
    _g6 = np.load(f"{B}/data_derived/gw6_two_sections.npz")
    for s_ in STAGES:
        xs, ys = ((_g6["S1_xs"].astype(float), _g6["S1_ys"].astype(float)) if s_ == "GW6"
                  else (_nz[f"{s_}_x"].astype(float), _nz[f"{s_}_y"].astype(float)))
        NBLK[s_] = {}
        for bk in BLKS:
            st = bk / 0.5
            NBLK[s_][bk] = len(np.unique(np.floor((xs - xs.min()) / st).astype(int) * 10000
                                         + np.floor((ys - ys.min()) / st).astype(int)))
    xb = np.arange(len(BLKS))
    axc.axhline(0, color="#555", lw=1.2, ls=(0, (4, 3)), zorder=1)
    _i400 = list(BLKS).index(400)          # x axis is index-based, not value-based
    axc.axvline(_i400, color="#c0392b", lw=1.3, ls=(0, (5, 3)), zorder=2)
    # the dashed line marks the 400 um block size used in the main figure
    for s_ in STAGES:
        g = bsum[bsum.stage == s_].set_index("block_um").reindex(BLKS)
        foc = s_ in ("GW8", "GW12")
        _oy = {"GW6": -18, "GW8": 9, "GW10": 3, "GW12": 0, "GW16": -7}[s_]
        axc.plot(xb, g["median"].values, "-o", color=SC[s_], lw=1.9 if foc else 1.2,
                 ms=7 if foc else 5, markeredgecolor="white",
                 markeredgewidth=0.9 if foc else 0.6, zorder=5 if foc else 4)
        axc.annotate(s_, (xb[-1], g["median"].values[-1]), textcoords="offset points",
                     xytext=(10, _oy), va="center", fontsize=ANNO, color=SC[s_],
                     fontweight="bold" if foc else "normal")
    # At the 1,600 um block edge, GW6 retains 4 independent blocks and GW8 only 2 (23 and
    # 10 at 400 um), so the bootstrap resample at that edge is degenerate for those two
    # stages; this limitation is not annotated directly on the panel.
    axc.set_xticks(xb); axc.set_xticklabels([f"{b:,}" for b in BLKS])
                                                         # legitimate range (block < pseudospot)
    axc.set_xlim(-0.30, len(BLKS) - 0.28); axc.set_ylim(-0.075, 0.115)
    axc.set_xlabel("bootstrap block edge (µm)")
    axc.set_ylabel("median depth gain (bit)", labelpad=2)
    axc.set_title("c   Depth gain across bootstrap block sizes", loc="left", fontsize=BASE)
    return [axc]


def panel_d(fig, slot):
    """D  Recoverability at two further matched depths (C = 32 and 128)."""
    # depth-matched optimum at two further matched counts
    gsd = GridSpecFromSubplotSpec(1, 2, slot, wspace=0.24, width_ratios=[1, 1])
    AXD = []
    CS = [32, 128]          # C = 64 is the main-figure panel, not repeated here
    for k, C in enumerate(CS):
        ax = fig.add_subplot(gsd[k]); AXD.append(ax)
        Mm = np.full((len(STAGES), len(SCALES)), np.nan)
        d = dm[(dm.C == C) & (dm.feasible)]
        for _, r in d.iterrows():
            Mm[STAGES.index(r.stage), SCALES.index(r.scale_um)] = r["I"]
        im_d = ax.imshow(Mm, cmap="RdBu_r", vmin=-0.30, vmax=0.30, aspect="auto",
                         interpolation="nearest")
        if k == len(CS) - 1:
            # the colour bar clarifies that the cell values are in bits; the main figure's
            # matched-count map carries the same bar
            _cb = fig.colorbar(im_d, ax=ax, fraction=0.030, pad=0.014)
            _cb.set_label(r"$\hat{I}$ (bit)", fontsize=ANNO - 0.5)
            _cb.ax.tick_params(labelsize=TICK - 1)
        for i in range(len(STAGES)):
            row = Mm[i]
            top = np.nanmax(row) if np.isfinite(row).any() else np.nan
            for j in range(len(SCALES)):
                v = row[j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=ANNO - 1, color="#bbb")
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=ANNO - 1.5,
                        color="white" if abs(v) > 0.22 else "#222")
                # no outline is drawn for infeasible cells here; compare fig2.py panel d
        ax.set_xticks(range(len(SCALES)))
        ax.set_xticklabels(SCALES, rotation=45, fontsize=TICK - 1)
        ax.set_yticks(range(len(STAGES)))
        ax.set_yticklabels(STAGES if k == 0 else [], fontsize=TICK)
        if k == 0:
            ax.set_ylabel("developmental stage", fontsize=BASE - 1)
        if k == 0:
            for t_, s_ in zip(ax.get_yticklabels(), STAGES):
                t_.set_color(SC[s_]); t_.set_fontweight("bold")
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f"C = {C}", fontsize=BASE - 1)
        ax.set_xlabel("pseudospot size s (µm)", fontsize=BASE - 1)

    # Panel D is the two matched-count maps only; a further comparison of the coordinate
    # distribution of pseudospots excluded by the eligibility threshold (to check for bias)
    # is not included here. Those values are in results/matched_depth_bin_bias.csv.
    return AXD, _cb.ax


def panel_e(fig, gse):
    """E  Recoverability against depth per dimension at 100-800 um (K = 30); slots 1-4 of gse."""
    # non-collapse of recoverability at coarser pseudospot sizes
    # 50 um is panel e of the main figure, so this supplement starts at 100 um; the
    # decoder-dimensionality plot is panel F.
    AXE = []
    for k, s_um in enumerate([100, 200, 400, 800]):
        ax = fig.add_subplot(gse[k + 1]); AXE.append(ax)
        sub = grid[(grid.feasible) & (grid.scale_um == s_um)]
        pv = sub.pivot_table(index="depth", columns="stage", values="I")
        present = [s_ for s_ in STAGES if s_ in pv.columns]
        CK = pv.index.values / 30.0
        ax.axhline(0, color="#999", lw=1.0, ls=":", zorder=2)
        for s_ in present:
            foc = s_ == FOCAL
            ax.plot(CK, pv[s_], "-o", color=SC[s_], lw=1.6,
                    ms=4.5, markeredgecolor="white",
                    markeredgewidth=0.9 if foc else 0.5, zorder=5 if foc else 4)
        # data span 0.53-17.1; pad ~1.2x each side so the end points are not on the spine
        ax.set_xscale("log", base=2); ax.set_xlim(0.42, 22)
        ax.set_xticks([0.5, 2, 8, 16]); ax.set_xticklabels(["0.5", "2", "8", "16"])
        ax.set_ylim(-0.12, 0.62)
        ax.set_xlabel("depth / K")
        if k == 0:
            ax.set_ylabel(r"$\hat{I}$ (bit)")
        else:
            ax.set_yticklabels([])
        if len(present) >= 2:
            ax.set_title(f"{s_um} µm", fontsize=BASE - 1)
        else:
            # one feasible stage: no cross-stage spread is defined
            ax.set_title(f"{s_um} µm\nno comparison possible",
                         fontsize=BASE - 1, color="#a05a5a")
    return AXE


def panel_f(fig, slot):
    """F  Spread between stages against decoder dimensionality K (50 um)."""
    axk = fig.add_subplot(slot)
    pk = ksn[ksn.feasible].pivot_table(index=["K", "depth"], columns="stage", values="I")[STAGES]
    KS_ = sorted(ksn.K.unique())
    spr = [float((pk.loc[K].max(1) - pk.loc[K].min(1)).max()) for K in KS_]
    top = [int((pk.loc[K].idxmax(1) == FOCAL).sum()) for K in KS_]
    axk.plot(KS_, spr, "-o", color="#3f3f3f", lw=1.6, ms=4.5,
             markeredgecolor="white", markeredgewidth=1.1, zorder=3)
    axk.plot([30], [spr[KS_.index(30)]], "o", color="#c0392b", ms=7, zorder=4,
             markeredgecolor="white", markeredgewidth=1.2)
    axk.annotate("K = 30", (30, spr[KS_.index(30)]), textcoords="offset points",
                 xytext=(0, -22), ha="center", fontsize=ANNO - 0.5, color="#c0392b",
                 fontweight="bold")
    axk.set_xscale("log", base=2); axk.set_xticks(KS_); axk.set_xticklabels(KS_)
    axk.set_ylim(0, 0.60); axk.set_xlim(8, 62)
    axk.set_xlabel("decoder dimensionality K")
    axk.set_ylabel(r"$\Delta\hat{I}$ (bit)")   # this is the spread (difference) between
    # stages, not a single stage's own reading; Fig 2C uses the same symbol for a
    # per-doubling change, so each use needs its own qualifier in the panel title
    return [axk]


def compose():
    """The sheet of panels A, C, D, E and F, laid out as in the figure; returns the figure
    and the axes of each panel. Panel B is drawn on its own (panel_b)."""
    fig = plt.figure(figsize=(16.8, 12.8))
    gs = GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.34,
                  height_ratios=[1.0, 0.95, 1.0])
    A = panel_a(fig, gs[0, 0:2])
    C = panel_c(fig, gs[1, 0:1])
    D, D_cbar = panel_d(fig, gs[1, 1:4])
    gse = GridSpecFromSubplotSpec(1, 6, gs[2, 0:4], wspace=0.26,   # slot 0 is left empty; the
                                  width_ratios=[1, 1, 1, 1, 1, 0.95])  # sub-panel widths match the assembled figure layout
    E = panel_e(fig, gse)
    F = panel_f(fig, gse[5])
    return dict(fig=fig, A=A, C=C, D=D, D_cbar=D_cbar, E=E, F=F)


# ---------------------------------------------------------------------------- marks
def _strip_text(fig):
    for t in list(fig.texts):
        t.set_visible(False)
    for ax in fig.axes:
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
        for t in list(ax.texts):
            t.set_visible(False)
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()
        ax.set_xlabel(""); ax.set_ylabel(""); ax.axis("off")


def _crop(fig, keep, path, margin=0.04, dpi=600):
    for ax in fig.axes:
        ax.set_visible(ax in keep)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bx = Bbox.union([a.get_tightbbox(r) for a in keep]).transformed(fig.dpi_scale_trans.inverted())
    bx = Bbox.from_extents(bx.x0 - margin, bx.y0 - margin, bx.x1 + margin, bx.y1 + margin)
    fig.savefig(path, dpi=dpi, bbox_inches=bx)


def _no_shade(axes):
    for ax in axes:
        for p in list(ax.patches):
            if isinstance(p, (Polygon, Rectangle)) and p is not ax.patch:
                p.remove()
        for c_ in list(ax.collections):
            if isinstance(c_, PolyCollection):
                c_.remove()


def marks():
    d = marks_dir(FIG)
    H = compose(); fig = H["fig"]
    _strip_text(fig)
    _crop(fig, H["A"], d / "FigS2_A.png")
    _crop(fig, H["C"], d / "FigS2_C.png")
    # D: the two matrices without their colour bar, box expanded by 2 %
    for ax in fig.axes:
        ax.set_visible(ax in H["D"])
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bx = Bbox.union([a.get_tightbbox(r) for a in H["D"]])
    bx = bx.transformed(fig.dpi_scale_trans.inverted()).expanded(1.02, 1.02)
    fig.savefig(d / "FigS2_D.png", dpi=600, bbox_inches=bx)
    _no_shade(H["E"] + H["F"])
    for k, ax in enumerate(H["E"]):
        _crop(fig, [ax], d / f"FigS2_E{k + 1}.png")
    _crop(fig, H["F"], d / "FigS2_F.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    panel_b(ax, bare=True)
    fig.savefig(d / "FigS2_B.png", dpi=600, bbox_inches="tight"); plt.close(fig)

    fig, cax = plt.subplots(figsize=(0.32, 2.6))           # B colour ramp
    cax.imshow(np.linspace(1, 0, 256).reshape(-1, 1), cmap="Blues", aspect="auto",
               interpolation="bilinear")
    cax.set_xticks([]); cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.savefig(d / "FigS2_B_colourbar.png", dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    colourbar_png(d / "FigS2_A_colourbar.png", "YlGnBu", 22, 460, ((0, 21), (0, 459)), 0)
    colourbar_png(d / "FigS2_D_colourbar.png", "RdBu_r", 11, 227, ((0, 10), (0, 226)), 0)


# ---------------------------------------------------------------------------- full
TITLES = {
    "A": "Pseudospot count and median UMI per pseudospot at each setting",
    "B": "Fraction of draws yielding the highest recoverability (800 \u00b5m, depth 512)",
    "C": "Median depth gain across bootstrap block sizes",
    "D": "Recoverability at two matched depths",
    "E": "Recoverability versus depth per dimension at each pseudospot size (K = 30)",
    "F": "",
}


def _save_group(fig, keep, name, margin=0.15):   # text renders slightly wider at 600 dpi
    d = OUT / FIG; d.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.set_visible(ax in keep)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    ext_ = [a.get_tightbbox(r) for a in keep]
    ext_ += [t.get_window_extent(r) for a in keep for t in (a.title, a._left_title)
             if t.get_text()]
    bx = Bbox.union(ext_).transformed(fig.dpi_scale_trans.inverted())
    bx = Bbox.from_extents(bx.x0 - margin, bx.y0 - margin, bx.x1 + margin, bx.y1 + margin)
    for ext in (".png", ".pdf"):
        fig.savefig(d / f"{name}{ext}", dpi=600, bbox_inches=bx, facecolor="white")


def full():
    H = compose(); fig = H["fig"]
    for t in list(fig.texts):
        t.set_visible(False)
    axa, axc, axk = H["A"][0], H["C"][0], H["F"][0]
    axa.set_title(TITLES["A"], loc="left", fontsize=BASE)
    axa.set_xlabel("Pseudospot size (\u03bcm)"); axa.set_ylabel("Developmental stage")
    H["A"][1].set_ylabel("Median UMI  (log10)")
    axc.set_title(TITLES["C"], loc="left", fontsize=BASE)
    axc.set_xlabel("Bootstrap block size (\u03bcm)"); axc.set_ylabel("Median depth gain (bit)")
    for ax, C in zip(H["D"], (32, 128)):
        ax.set_title(f"Depth {C}", fontsize=BASE - 1)
        ax.set_xlabel("Pseudospot size (\u03bcm)", fontsize=BASE - 1)
    H["D"][0].set_title(TITLES["D"] + "\n", loc="left", fontsize=BASE)
    H["D"][0].set_ylabel("Developmental stage")
    for k, (ax, s_um) in enumerate(zip(H["E"], (100, 200, 400, 800))):
        ax.set_title(f"{s_um} \u00b5m", fontsize=BASE - 1)
        ax.set_xlabel("Depth / K" if k == 0 else "")
        ax.set_ylabel("")
    H["E"][0].set_title(TITLES["E"] + "\n", loc="left", fontsize=BASE)
    for ax in H["E"]:
        ax.title.set_color("k")
    axk.set_xlabel("K"); axk.set_ylabel("")
    _save_group(fig, H["A"], "FigS2_A")
    _save_group(fig, H["C"], "FigS2_C")
    _save_group(fig, H["D"] + [H["D_cbar"]], "FigS2_D")
    _save_group(fig, H["E"], "FigS2_E")
    _save_group(fig, H["F"], "FigS2_F")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    W = panel_b(ax)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03, ticks=[0, 50, 100])
    cb.set_label("Fraction (%)", fontsize=ANNO + 0.5); cb.ax.tick_params(labelsize=TICK)
    ax.set_title(TITLES["B"], loc="left", fontsize=BASE - 1)
    d = OUT / FIG; d.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(d / f"FigS2_B{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {s: dict(zip(SCALES, np.round(W[i] * 100, 1))) for i, s in enumerate(STAGES)}


if __name__ == "__main__":
    marks()
    print("B percentage of draws in which each size reads highest:", full())
