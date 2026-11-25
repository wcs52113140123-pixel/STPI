#!/usr/bin/env python3
"""Figure S3. Supplementary checks behind Figure 3.

  A  Recoverability across three coordinate constructions (all settings)
  B  Changes in recoverability with the fraction of units removed by the buffer
  C  Highest-recoverability stage under each CV scheme
  D  Changes of recoverability on Stereo-seq and MERFISH at each stage (100 um, depth 16)

Inputs (results/)
  multiGW_grid_all.csv, fig3_bufferedcv_5stage.csv, merfish_grid_all.csv,
  fig3_coord_grids.csv, xtech_genelayer_all.csv
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

FIG = "FigS3"
plt.rcParams.update({"font.family": FONT, "font.size": 8, "pdf.fonttype": 42,
                     "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.9,
                     "xtick.major.width": 0.9, "ytick.major.width": 0.9})

ST = ["GW6", "GW8", "GW10", "GW12", "GW16"]

CS = {"GW6": "#4c78a8", "GW8": "#59a14f", "GW10": "#9c755f",
      "GW12": "#e15759", "GW16": "#f2a53c"}

CC = {"polar": "#e15759", "curve": "#7fa8c9", "harmonic": "#d6a86a"}

CL = {"polar": "polar centreline (STPI, this study)",
      "curve": "principal curve (this study, after Hastie & Stuetzle 1989)",
      "harmonic": "boundary harmonic (this study, after Jones et al. 2000)"}

GREY = "#8a8a8a"

GRID = pd.read_csv(RESULTS / "multiGW_grid_all.csv")

BUF = pd.read_csv(RESULTS / "fig3_bufferedcv_5stage.csv")

PAIR = (BUF[BUF.feasible][["stage", "scale_um", "depth", "I", "frac_train_dropped"]]
        .rename(columns={"I": "I_buffered"})
        .merge(GRID[GRID.feasible][["stage", "scale_um", "depth", "I"]]
               .rename(columns={"I": "I_checker"}), on=["stage", "scale_um", "depth"]))

PAIR["d"] = PAIR.I_buffered - PAIR.I_checker

MGRID = pd.read_csv(RESULTS / "merfish_grid_all.csv")

_m = MGRID[(MGRID.feasible == 1) & (MGRID.gene_layer == "native")]

MP = (_m.pivot_table(index=["stage", "scale_um", "depth"], columns="cv", values="I")
        .dropna().reset_index())



# ────────── panel a · recoverability across three coordinate constructions ──────────────
CG = pd.read_csv(RESULTS / "fig3_coord_grids.csv")

GRIDS = CG[CG.feasible]

def panel_a(ax, bare=False):
    """Every point in Figure 3a is a maximum over the whole (scale, depth) grid. This shows
    the grid behind each point: one marker per feasible setting, with the plotted maximum
    marked. GW12 is highest not only at its best setting but through the bulk of its
    distribution, under the two centreline constructions."""
    rg = np.random.default_rng(0)
    coords = ["polar", "curve", "harmonic"]
    ax.axhline(0, color=GREY, lw=0.8, ls=":", zorder=1)
    for j, s in enumerate(ST):
        for i, c in enumerate(coords):
            q = GRIDS[(GRIDS.stage == s) & (GRIDS.coord == c)]["I"].values
            if not len(q): continue
            x = j + (i - 1) * 0.26
            ax.plot(x + rg.uniform(-0.07, 0.07, len(q)), q, "o", ms=2.6,
                    color=CC[c], alpha=0.5, mec="none", zorder=3)
            ax.plot([x - 0.10, x + 0.10], [q.max()] * 2, color=CC[c], lw=2.0,
                    solid_capstyle="butt", zorder=5)
    ax.set_xticks(range(len(ST))); ax.set_xticklabels(ST)
    ax.set_xlim(-0.55, len(ST) - 0.45); ax.set_ylim(-0.34, 0.70)
    ax.set_ylabel(r"$\hat{I}$ (bit)")
    ax.set_xlabel("developmental stage")
    h = [Line2D([0], [0], marker="o", color=CC[c], ls="none", ms=4.5, alpha=0.6,
                label=CL[c]) for c in coords]
    h.append(Line2D([0], [0], color="#555", lw=2.0, label="maximum (value in Fig. 3a)"))
    ax.legend(handles=h, frameon=False, fontsize=6.2, loc="upper left",
              labelspacing=0.22, handletextpad=0.5, borderpad=0.0)
    if not bare:
        ax.set_title("Recoverability across three coordinate\n"
                 "constructions, at every feasible setting", loc="left")   # this longer heading spells out the full grid; Fig. 3a itself carries the short form


# ────────── panel b · buffer cost against buffer effect ─────────────────────────────────
def panel_b(ax, bare=False):
    """Two things the buffer does at once: it deletes training units (x) and it changes the
    reading (y). One point is one (scale, depth) setting; the six points of a vertical
    group are the six depths of one stage-by-size combination, which share a deletion
    fraction. The band marks |dI| <= 0.03 bit.

    The band and the two extreme settings below are not drawn in the panel; they are
    described here instead. They are different settings and should not be conflated:
      largest shift in the reading   GW16 at 800 um and depth 512, dI = -0.0808, deletion 16.9%
      largest deletion              GW6 at 100 um and depth 256, deletion 29.5%, dI = -0.0562"""
    ax.axhline(0, color="#444", lw=1.0, ls=(0, (5, 3)), zorder=1)
    for s in ST:
        q = PAIR[PAIR.stage == s]
        ax.plot(q.frac_train_dropped * 100, q.d, "o", ms=5, color=CS[s],
                alpha=0.9, mec="white", mew=0.5, zorder=3, label=s)
    ax.set_xlabel("training units deleted by the buffer (%)")
    ax.set_ylabel(r"$\Delta\hat{I}$  (buffered $-$ checkerboard, bit)")
    ax.set_xlim(-0.8, 31); ax.set_ylim(-0.098, 0.045)
    ax.legend(frameon=False, fontsize=6.6, loc="lower left", ncol=3,
              labelspacing=0.2, handletextpad=0.25, columnspacing=0.7, borderpad=0.0)
    if not bare:
        ax.set_title("Buffer cost against buffer effect", loc="left")


# ────────── panel c · highest-reading stage under each cross-validation scheme ──────────
def _peaks():
    rows = []
    for (sc, dp), g in MP.groupby(["scale_um", "depth"]):
        if len(g) < 3: continue
        rows.append(dict(scale=int(sc), depth=int(dp), k=len(g),
                         spatial=g.set_index("stage")["spatial"].idxmax(),
                         random=g.set_index("stage")["random"].idxmax()))
    return pd.DataFrame(rows).sort_values(["scale", "depth"]).reset_index(drop=True)

PK = _peaks()

def panel_c(ax, bare=False):
    """Main Fig 3c shows how much random CV inflates the reading at one setting. It does not
    show whether that inflation changes an answer. Here the highest stage is identified
    twice at every MERFISH setting where at least three stages are feasible, once under
    each scheme, so the consequence, not the magnitude, is what is plotted. Which stage is
    highest is a property of the setting (Fig. 3e); the point here is only whether the two
    schemes agree about it. Coarser pseudospot sizes are absent because fewer than three
    MERFISH stages are feasible at 200 um and none at 400 um."""
    y = np.arange(len(PK))[::-1]
    for k, r_ in PK.iterrows():
        same = r_["spatial"] == r_["random"]
        ax.plot([0, 1], [y[k]] * 2, color="#d8d8d8" if same else "#555",
                lw=1.0 if same else 1.6, ls="-" if not same else (0, (3, 3)), zorder=1)
        for xx, col in ((0, r_["spatial"]), (1, r_["random"])):
            ax.plot([xx], [y[k]], "s", ms=9, color=CS[col], mec="white", mew=1.0, zorder=3)
            ax.text(xx, y[k], col.replace("GW", ""), ha="center", va="center",
                    fontsize=5.6, color="white", zorder=4)
    ax.set_xlim(-0.45, 1.75); ax.set_ylim(-0.8, len(PK) - 0.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["spatial\nblock CV", "random\nCV"])
    # no x-axis name: the two tick labels already name the schemes, and a title there
    # collides with the key that sits under the axes
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r_['scale']} µm, {r_['depth']}" for _, r_ in PK.iterrows()],
                       fontsize=6.4)
    ax.set_ylabel("setting (size in \u00b5m, depth in transcripts)")   # the row labels give
    # bare numbers for the depth, so the unit is named here; "depth" matches Fig. 3e
    for k, r_ in PK.iterrows():
        g = MP[(MP.scale_um == r_["scale"]) & (MP.depth == r_["depth"])]
        who = " ".join(s.replace("GW", "") for s in ST if s in set(g.stage))
        ax.text(1.16, y[k], who, fontsize=5.9, color="#888", va="center", ha="left")
    # names which stages were in contention at each setting, so a missing stage can be
    # distinguished from one that lost
    ax.text(1.16, len(PK) - 0.62, "stages competing", fontsize=5.9, color="#888",
            va="bottom", ha="left")
    # the line style carries the panel's main message, so a key explains it here
    h = [Line2D([0], [0], color="#555", lw=1.6, label="the two schemes disagree"),
         Line2D([0], [0], color="#d8d8d8", lw=1.0, ls=(0, (3, 3)),
                label="same answer")]
    ax.legend(handles=h, frameon=False, fontsize=6.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.155), ncol=2, handlelength=2.2,
              handletextpad=0.5, columnspacing=1.4, borderpad=0.0)
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    ax.tick_params(axis="x", length=0)
    if not bare:
        ax.set_title("Highest-reading stage under each\ncross-validation scheme", loc="left")


# ────────── panel d · platform gap under two gene sets ──────────────────────────────────
GL = pd.read_csv(RESULTS / "xtech_genelayer_all.csv")

def panel_d(ax, bare=False):
    """Main Fig 3d compares the two platforms on the genes they share at each stage, where
    readings are negative. If that restriction explained the lower Stereo-seq readings,
    the gap would be an artefact of the gene set. The same matched-n comparison is rerun
    with each platform on its own native gene set, changing one variable while holding
    everything else constant."""
    sgs = [s for s in ST if s in set(GL.stage)]
    y = np.arange(len(sgs))[::-1]
    off = {"matched": 0.16, "native": -0.16}
    ax.axvline(0, color="#444", lw=1.0, ls=(0, (4, 3)), zorder=1)
    for lay, mfc in (("matched", None), ("native", "white")):
        g = GL[GL.gene_layer == lay].set_index("stage")
        for k, s in enumerate(sgs):
            if s not in g.index or not np.isfinite(g.loc[s, "dI_med"]): continue
            yy = y[k] + off[lay]
            ax.plot([g.loc[s, "dI_lo"], g.loc[s, "dI_hi"]], [yy, yy], color=CS[s],
                    lw=1.8, solid_capstyle="round", zorder=2)
            ax.plot([g.loc[s, "dI_med"]], [yy], "o", ms=6.5, color=CS[s],
                    mfc=mfc if mfc else CS[s], mec=CS[s], mew=1.6, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(sgs); ax.set_ylim(-0.75, len(sgs) - 0.3)
    ax.set_xlim(-0.30, 1.0)
    # the marker fill already encodes the gene set, so the platforms are named directly on
    # the axis label, otherwise the reader cannot tell which side is which
    ax.set_xlabel(r"$\Delta\hat{I}$ = MERFISH $-$ Stereo-seq (bit);"
                  "\nright of 0 = MERFISH higher")
    h = [Line2D([0], [0], marker="o", color="#777", ls="none", ms=6,
                label="gene set: shared at that stage"),
         Line2D([0], [0], marker="o", color="#777", mfc="white", mew=1.6, ls="none",
                ms=6, label="gene set: each platform's own")]
    ax.legend(handles=h, frameon=False, fontsize=6.3, loc="lower right",
              labelspacing=0.25, handletextpad=0.4, borderpad=0.0)
    ax.text(0.0, len(sgs) - 0.42, "  platforms agree", fontsize=6.2, color="#888",
            ha="left", va="top")
    if not bare:
        ax.set_title("Platform gap under two gene sets", loc="left")


LEGLOC = {"a": "upper left", "b": "lower left", "d": "lower right"}

def strip_bare(ax, lab):
    """bare tier: no text, no axes; data marks, reference lines and blank legend
    swatches survive."""
    ax.set_title("", loc="left"); ax.set_xlabel(""); ax.set_ylabel("")
    for t in list(ax.texts):
        if t.get_text().strip():
            t.remove()
    lg = ax.get_legend()
    if lg is not None:
        hs = list(getattr(lg, "legend_handles", None) or getattr(lg, "legendHandles", []))
        lg.remove()
        if hs:
            ax.legend(handles=hs, labels=[""] * len(hs), frameon=False,
                      loc=LEGLOC.get(lab, "upper left"), handlelength=1.6,
                      labelspacing=0.3, borderpad=0.1)
    # a colourbar lives in its own axes, which strip_bare would otherwise leave labelled;
    # axis("off") keeps the gradient drawn and removes only frame and ticks
    for other in ax.figure.axes:
        if other is ax: continue
        other.set_xlabel(""); other.set_ylabel(""); other.set_title("")
        for t in list(other.texts):
            if t.get_text().strip(): t.remove()
        other.axis("off")
    ax.axis("off")


PANELS = {"A": (panel_a, (4.8, 3.1)), "B": (panel_b, (4.3, 3.1)),
          "C": (panel_c, (4.0, 3.1)), "D": (panel_d, (4.3, 3.1))}


def marks():
    for lab, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, True)
        if lab in ("A", "D"):                     # these two keys are placed beside the panel
            legend_key(ax, marks_dir(FIG) / f"FigS3_{lab}_key.png", scale=2.0 if lab == "A" else 1.4)
        strip_bare(ax, lab.lower())
        fig.savefig(marks_dir(FIG) / f"FigS3_{lab}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def _relabel(ax, labels):
    for t, s in zip(ax.get_legend().get_texts(), labels):
        t.set_text(s)


def complete_text(lab, ax):
    if lab == "A":
        ax.set_title("Recoverability across three coordinate constructions", loc="left")
        _relabel(ax, ("STPI", "Hastie et al. 1989", "Jones et al. 2000",
                      "Maximum (value in Fig. 3A)"))
        ax.set_xlabel("Developmental stage"); ax.set_ylabel("")
    elif lab == "B":
        ax.set_title("Changes in recoverability with the fraction of units removed", loc="left")
        ax.set_xlabel("Fraction of units removed by the buffer (%)"); ax.set_ylabel("")
        ax.get_legend().remove()
        ax.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6.6,
                  loc="lower left", labelspacing=0.2, handletextpad=0.25, borderpad=0.0)
    elif lab == "C":
        ax.set_title("Highest-recoverability stage under each CV scheme", loc="left")
        ax.set_xticklabels(["spatial block CV", "random CV"]); ax.set_ylabel("Setting")
        ax.get_legend().remove()
    elif lab == "D":
        ax.set_title("Changes of recoverability on Stereo-seq and MERFISH at each stage\n"
                     "(100 \u00b5m, depth 16)", loc="left")
        for t in list(ax.texts):
            if "platforms agree" in t.get_text():
                t.remove()
        ax.set_xlabel(r"$\hat{I}$ (bit)"); ax.set_ylabel("Developmental stage")


def full():
    d = OUT / FIG; d.mkdir(parents=True, exist_ok=True)
    for lab, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, False)
        complete_text(lab, ax)
        for ext in (".png", ".pdf"):
            fig.savefig(d / f"FigS3_{lab}{ext}", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    marks()
    full()
    print("B extremes (fraction removed %, change in bit):",
          PAIR.loc[PAIR.d.idxmin(), ["stage", "scale_um", "depth"]].tolist(),
          round(float(PAIR.d.min()), 4), round(float(PAIR.frac_train_dropped.max() * 100), 1))
