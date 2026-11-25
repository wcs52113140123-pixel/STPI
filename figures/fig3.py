#!/usr/bin/env python3
"""Figure 3. Coordinate construction, cross-validation and platform on the meninges.

  A  Recoverability across three coordinate constructions
  B  Changes in recoverability after adding a CV buffer
  C  Recoverability of spatial block and random CV on MERFISH (50 um, depth 32)
  D  Recoverability of Stereo-seq and MERFISH at each stage

Inputs (results/)
  fig3_coord_arbitration.csv, multiGW_grid_all.csv, fig3_bufferedcv_5stage.csv,
  merfish_grid_all.csv, xtech_nmatched_geneintersect.csv, merfish_nmatched_gated.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _common import RESULTS, FONT, marks_dir, legend_key, OUT

FIG = "Fig3"
plt.rcParams.update({"font.family": FONT, "font.size": 8, "pdf.fonttype": 42,
                     "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.9,
                     "xtick.major.width": 0.9, "ytick.major.width": 0.9})

ST = ["GW6", "GW8", "GW10", "GW12", "GW16"]
CS = {"GW6": "#4c78a8", "GW8": "#59a14f", "GW10": "#9c755f",
      "GW12": "#e15759", "GW16": "#f2a53c"}
GREY = "#8a8a8a"
DEP = [16, 32, 64, 128]

# ───────────────────────────────── data ─────────────────────────────────
A = pd.read_csv(RESULTS / "fig3_coord_arbitration.csv")
CO = ["polar", "curve", "harmonic"]

grid = pd.read_csv(RESULTS / "multiGW_grid_all.csv")
buf = pd.read_csv(RESULTS / "fig3_bufferedcv_5stage.csv")
PAIR = (buf[buf.feasible][["stage", "scale_um", "depth", "I", "frac_train_dropped"]]
        .rename(columns={"I": "I_buffered"})
        .merge(grid[grid.feasible][["stage", "scale_um", "depth", "I"]]
               .rename(columns={"I": "I_checker"}), on=["stage", "scale_um", "depth"]))
PAIR["d"] = PAIR.I_buffered - PAIR.I_checker

mg = pd.read_csv(RESULTS / "merfish_grid_all.csv")
_s = mg[(mg.feasible == 1) & (mg.scale_um == 50) & (mg.depth == 32) &
        (mg.gene_layer == "native")].pivot_table(index="stage", columns="cv", values="I")
MER = {s: (float(_s.loc[s, "random"]), float(_s.loc[s, "spatial"])) for s in ST}

# gene-matched run: both technologies are restricted, per stage, to the genes present in
# both, since letting MERFISH use more of the shared genes than Stereo-seq would confound
# technology with feature count; at GW8 a feature-count difference of that kind alone can
# move the MERFISH reading from 0.264 to 0.081 bit.
X3 = pd.read_csv(RESULTS / "xtech_nmatched_geneintersect.csv")
GT = pd.read_csv(RESULTS / "merfish_nmatched_gated.csv")
NAT = GT[GT.gene_layer == "native"]




# ───────────────────────────────── panel a ─────────────────────────────────
STY_A = {"polar":    dict(color="#e15759", lw=2.6, ls="-",           ms=6.5, mfc="#e15759", z=5),
         "curve":    dict(color="#7fa8c9", lw=1.5, ls=(0, (5, 2)),   ms=4.2, mfc="white",   z=3),
         "harmonic": dict(color="#d6a86a", lw=1.5, ls=(0, (5, 2)),     ms=4.2, mfc="white", z=2)}
# The polar coordinate and the two alternatives are implementations used in this study
# (not third-party software, unlike GASTON in Fig 4), so the legend names the concept each
# follows with "after": Hastie & Stuetzle 1989 (JASA 84:502-516) for principal curves,
# Jones et al. 2000 (Hum Brain Mapp 11:12-32) for the Laplace/harmonic depth coordinate.
LBL_A = {"polar": "polar centreline (STPI, this study)",
         "curve": "principal curve (this study, after Hastie & Stuetzle 1989)",
         "harmonic": "boundary harmonic (this study, after Jones et al. 2000)"}


def panel_a(ax, bare=False):
    x = np.arange(len(ST)); k = ST.index("GW12")
    ax.axvline(k, color="#e15759", lw=1.0, ls=(0, (4, 3)), alpha=0.35, zorder=0)
    ax.axhline(0, color=GREY, lw=0.8, ls=":", zorder=1)
    for c in CO:
        v = [A[(A.stage == s) & (A.coord == c)].I_max.iloc[0] for s in ST]
        st = STY_A[c]
        ax.plot(x, v, color=st["color"], lw=st["lw"], ls=st["ls"], marker="o", ms=st["ms"],
                mfc=st["mfc"], mec=st["color"], mew=1.2, zorder=st["z"], clip_on=False,
                label=LBL_A[c])
    ax.set_xticks(x); ax.set_xticklabels(ST); ax.set_xlim(-0.25, 4.25); ax.set_ylim(-0.09, 0.72)
    ax.set_ylabel(r"$\hat{I}$ (bit)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", labelspacing=0.32,
              handlelength=2.3, handletextpad=0.6, borderpad=0.1)
    if not bare:
        ax.set_title("Recoverability across three coordinate constructions", loc="left")


# ───────────────────────────────── panel b ─────────────────────────────────
def panel_b(ax, bare=False):
    """Per-stage counts (12 / 6 / 18 / 24 / 30 settings for GW6..GW16) and the reading note
    ("each point = one (scale, depth) setting; bar = stage median") are not shown on the
    panel; both appear in the caption."""
    rg = np.random.default_rng(3)
    ax.axhline(0, color="#444", lw=1.1, ls=(0, (5, 3)), zorder=2)
    for i, s in enumerate(ST):
        v = PAIR[PAIR.stage == s]["d"].values
        ax.plot(i + rg.uniform(-0.17, 0.17, len(v)), v, "o", ms=4.2, color=CS[s],
                alpha=0.65, mec="white", mew=0.5, zorder=3)
        m = np.median(v)
        ax.plot([i - 0.30, i + 0.30], [m, m], color=CS[s], lw=2.4, zorder=4, solid_capstyle="butt")
    ax.set_xticks(range(len(ST))); ax.set_xticklabels(ST)
    ax.set_xlim(-0.55, len(ST) - 0.45); ax.set_ylim(-0.098, 0.058)
    # the buffer width equals one pseudospot (buffer_um == scale_um in every row of
    # fig3_bufferedcv_5stage.csv); stated in the caption, not on the panel
    # the bare \Delta\hat{I} axis label needs the qualifier (buffered minus checkerboard)
    # and the buffer width, which appear in the title and caption. Fig 2c uses
    # \Delta\hat{I} for a per-doubling difference and Fig S2f for a between-stage
    # difference, so each panel using this symbol names its own difference to avoid
    # ambiguity.
    ax.set_ylabel(r"$\Delta\hat{I}$ (bit)")
    if not bare:
        ax.set_title("Change in recoverability when a\ncross-validation buffer is added", loc="left")
    # "Recoverability" alone would not describe this axis: it plots a difference between
    # two readings, not a stage's own level (same convention as Fig S2f)


# ───────────────────────────────── panel c ─────────────────────────────────
C_MER = "#4c78a8"


def panel_c(ax, bare=False):
    x = np.arange(len(ST))      # no pink GW12 reference line here, unlike panel a
    ax.axhline(0, color=GREY, lw=0.8, ls=":", zorder=1)
    ax.plot(x, [MER[s][0] for s in ST], color="#bfc4c8", lw=1.4, ls=(0, (1.6, 1.8)),
            marker="o", ms=4.0, mfc="white", mec="#bfc4c8", mew=1.2, zorder=2, clip_on=False,
            label="random CV")   # labelled by method only ("random CV"), avoiding
            # evaluative phrasing such as "spatially naive"
    ax.plot(x, [MER[s][1] for s in ST], color=C_MER, lw=2.6, marker="o", ms=6.5,
            mfc=C_MER, mec=C_MER, mew=1.2, zorder=5, clip_on=False,
            label="spatial block CV")
    ax.set_xticks(x); ax.set_xticklabels(ST); ax.set_xlim(-0.25, 4.62); ax.set_ylim(-0.05, 0.72)
    ax.set_ylabel(r"$\hat{I}$ (bit)")
    ax.set_xlabel("developmental stage")   # conditions moved to the caption
    ax.legend(frameon=False, fontsize=7, loc="upper left", labelspacing=0.32,
              handlelength=2.3, handletextpad=0.6, borderpad=0.1)
    for i, s in enumerate(ST):
        hi, lo = MER[s][0], MER[s][1]
        gap = hi - lo
        if abs(gap) >= 0.02:      # a degenerate arrow would render as a dot
            # head size tracks the gap, else a short arrow is mostly arrowhead
            ms = float(np.clip(abs(gap) * 95, 6.0, 12.0))
            ax.annotate("", xy=(i, hi), xytext=(i, lo),
                        arrowprops=dict(arrowstyle="<->", color="#333", lw=0.8,
                                        mutation_scale=ms, shrinkA=0, shrinkB=0))
        txt = f"{gap:+.2f}" if abs(gap) >= 0.005 else "0.00"
        ax.text(i + 0.12, (hi + lo) / 2, txt, fontsize=6.5,
                color="#333" if abs(gap) >= 0.02 else "#9a9a9a", ha="left", va="center")
    ax.text(0.0, -0.245, "arrows: random minus spatial (bit);  "
                         "GW16 gap is 0.00, no arrow drawn",
            transform=ax.transAxes, fontsize=6.5, color="#8a8a8a", ha="left", va="top")
    if not bare:
        ax.set_title("Random versus spatial cross-validation on MERFISH", loc="left")


# ───────────────────────────────── panel d ─────────────────────────────────
C_STE = "#e15759"


def panel_d(ax, bare=False):
    sg = list(X3.stage); y = np.arange(len(sg))[::-1]
    for yy, (_, r) in zip(y, X3.iterrows()):
        ax.plot([r.I_stereo_med, r.I_mer_med], [yy, yy], color="#c8ccd0", lw=2.0, zorder=1)
        ax.plot(r.I_stereo_med, yy, "o", ms=7.5, color=C_STE, mec="white", mew=1.0, zorder=3)
        ax.plot(r.I_mer_med, yy, "o", ms=7.5, color=C_MER, mec="white", mew=1.0, zorder=3)
        ax.text(r.I_mer_med + 0.035, yy, f"$\\Delta$ {r.dI_med:.2f}", fontsize=7,
                color="#333", va="center", ha="left")
    ax.axvline(0, color=GREY, lw=0.9, ls=":", zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(sg); ax.set_ylim(-0.7, len(sg) - 0.3)
    ax.set_xlim(-0.34, 0.92)
    ax.set_xlabel(r"$\hat{I}$ (bit)")
    ax.set_yticklabels([f"{s}\nn = {int(n)}, {int(g)} genes"
                        for s, n, g in zip(sg, X3.n_match, X3.n_genes_used)])
    h = [Line2D([0], [0], marker="o", color=C_STE, ls="none", ms=6.5, label="Stereo-seq"),
         Line2D([0], [0], marker="o", color=C_MER, ls="none", ms=6.5, label="MERFISH")]
    ax.legend(handles=h, frameon=False, fontsize=7, loc="lower right",
              labelspacing=0.32, handletextpad=0.5, borderpad=0.1)
    if not bare:
        # two-line title: the panel is narrow and a single line would run into the
        # adjacent panel's letter
        ax.set_title("Stereo-seq versus MERFISH\non the genes shared at each stage", loc="left")


# ────────── bare tier: strip text, keep the marks ──────────────────────────────────────
LEGLOC = {"a": "upper left", "c": "upper left", "d": "lower right", "e": "upper left"}


def strip_bare(ax, lab):
    """bare tier: no text, no axes. Data marks, reference lines, arrows and legend
    swatches survive; every string - title, letter, axis labels, tick labels, legend
    text, in-panel notes - is removed. Empty-text annotations are the arrows, so they
    are kept deliberately."""
    ax.set_title("", loc="left"); ax.set_xlabel(""); ax.set_ylabel("")
    for t in list(ax.texts):
        if t.get_text().strip():
            t.remove()
    lg = ax.get_legend()
    if lg is not None:
        hs = getattr(lg, "legend_handles", None) or getattr(lg, "legendHandles", [])
        hs = list(hs)
        lg.remove()
        if hs:
            ax.legend(handles=hs, labels=[""] * len(hs), frameon=False,
                      loc=LEGLOC.get(lab, "upper left"), handlelength=1.6,
                      labelspacing=0.3, borderpad=0.1)
    ax.axis("off")


PANELS = {"A": (panel_a, (4.3, 3.2)), "B": (panel_b, (4.5, 3.2)),
          "C": (panel_c, (4.3, 3.2)), "D": (panel_d, (4.3, 3.2))}


def marks():
    for lab, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, True)
        if lab == "A":                            # the key is placed beside the panel
            legend_key(ax, marks_dir(FIG) / "Fig3_A_key.png")
        strip_bare(ax, lab.lower())
        fig.savefig(marks_dir(FIG) / f"Fig3_{lab}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def complete_text(lab, ax):
    """Titles, axis names and keys as they read in the figure."""
    for t in list(ax.texts):                      # explanatory notes are not in the figure
        if t.get_text().startswith(("arrows:", "each point")):
            t.remove()
    if lab == "A":
        ax.set_title("Recoverability across three coordinate constructions", loc="left")
        for t, s in zip(ax.get_legend().get_texts(), ("STPI", "Hastie et al. 1989", "Jones et al. 2000")):
            t.set_text(s)
        ax.set_xlabel("Developmental stage"); ax.set_ylabel("")
    elif lab == "B":
        ax.set_title("Changes in recoverability after adding a CV buffer", loc="left")
        ax.set_xlabel("Developmental stage"); ax.set_ylabel("")
    elif lab == "C":
        ax.set_title("Recoverability of spatial block and random CV on MERFISH\n"
                     "(50 \u00b5m, depth 32)", loc="left")
        for t, s in zip(ax.get_legend().get_texts(), ("Random CV", "Spatial block CV")):
            t.set_text(s)
        ax.set_xlabel("Developmental stage"); ax.set_ylabel("")
    elif lab == "D":
        ax.set_title("Recoverability of Stereo-seq and MERFISH at each stage", loc="left")
        for t in ax.texts:
            t.set_text(t.get_text().replace("$\\Delta$ ", "\u0394"))
        ax.set_xlabel(r"$\hat{I}$ (bit)"); ax.set_ylabel("")


def full():
    d = OUT / FIG; d.mkdir(parents=True, exist_ok=True)
    for lab, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, False)
        complete_text(lab, ax)
        for ext in (".png", ".pdf"):
            fig.savefig(d / f"Fig3_{lab}{ext}", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    marks()
    full()
    print("A best recoverability per construction:",
          A.groupby("coord").I_max.apply(lambda s: [round(float(x), 3) for x in s]).to_dict())
    print("B median change per stage:", PAIR.groupby("stage").d.median().round(4).to_dict())
    print("C (random, spatial):", {s: tuple(round(v, 3) for v in MER[s]) for s in ST})
