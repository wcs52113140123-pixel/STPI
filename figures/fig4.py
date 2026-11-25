#!/usr/bin/env python3
"""Figure 4. STPI applied to injured mouse tendon (Cherief et al. 2023, GSE245179).

  A  Expression-independent coordinate in injured mouse tendon (STPI: across the band)
  B  Recoverability across pseudospot size and depth (STPI: across the band)
  C  Recoverability by pseudospot size and depths
  D  Recoverability versus tissue thickness (100 um, depth 2048)
  E  Difference in recoverability between spatial block and random CV (STPI: across the band)

Panel A shows the top row of the panel drawn here (the STPI coordinate); the second row
(GASTON isodepth, Chitra et al. 2025) is drawn for reference and is not part of the figure.

Inputs (results/)
  fig4_coord_transfer.csv, fig4_tendon_cv_centreline.csv, fig4_coord_maps.npz,
  fig4_coord_geometry.npz, fig4_gaston_isodepth.npz, fig4_gaston_agreement.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from PIL import Image

from _common import RESULTS, FONT, marks_dir, colourbar_png, OUT

FIG = "Fig4"
BASE_FS, ANNO_FS, TICK_FS = 8, 7, 6
plt.rcParams.update({
    "font.family": FONT, "pdf.fonttype": 42, "font.size": BASE_FS,
    "axes.titlesize": BASE_FS, "axes.labelsize": BASE_FS,
    "legend.fontsize": ANNO_FS, "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "legend.frameon": False, "savefig.dpi": 600})

SEC = ["CTRL1", "CTRL2", "INMPP1", "INMPP1-2"]

# Colour encodes condition rather than developmental stage: the hue family marks condition
# (CTRL blue, inhibited purple), and sections are shades within it
CC = {"CTRL1": "#1f5f82", "CTRL2": "#6aaed6", "INMPP1": "#6b3a75", "INMPP1-2": "#c994c7"}

# keys match the names used in the underlying files; these are the display labels used
# in the figure. GSE245179: Control_1/2 are TrkA wild type, 1NMPP1_1/2 are TrkA F592A
# given 1NM-PP1.
DISP = {"CTRL1": "Control-1", "CTRL2": "Control-2",
        "INMPP1": "1NMPP1-1", "INMPP1-2": "1NMPP1-2"}

DSEC = [DISP[s] for s in SEC]

GREY = "#8a8a8a"

SCALES = [100, 200, 400, 600, 800]

DEPTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]

TR = pd.read_csv(RESULTS / "fig4_coord_transfer.csv")

CVD = pd.read_csv(RESULTS / "fig4_tendon_cv_centreline.csv")

BEST = {}

for s in SEC:
    q = TR[(TR.section == s) & (TR.coord == "centreline")].dropna(subset=["I_checker"])
    BEST[s] = int(q.loc[q.I_checker.idxmax()].depth)

MAPS = np.load(RESULTS / "fig4_coord_maps.npz")

GEO = np.load(RESULTS / "fig4_coord_geometry.npz")

W_C = TR[TR.coord == "centreline"].dropna(subset=["I_checker"])

V_LO, V_HI = float(W_C.I_checker.min()), float(W_C.I_checker.max())

def aligned(s):
    """Rotate a section so its long axis is horizontal. Positions only: this is a display
    transform, so the four sections read as one series instead of four arbitrary poses."""
    xy = np.asarray(MAPS[f"{s}_xy"], float)
    c = xy - xy.mean(0)
    w, V = np.linalg.eigh(np.cov(c.T))
    Q = c @ V[:, ::-1]
    return Q, MAPS[f"{s}_centreline"]

# ───────────────────────────────── panel a ─────────────────────────────────
def _rank01(v):
    o = np.argsort(np.argsort(np.asarray(v, float)))
    return o / max(len(o) - 1, 1)

# comparison row: GASTON isodepth, a published method (Chitra et al.), trained on these
# same sections by analysis/fig4_gaston_tendon.py using the settings this paper uses for
# the meninges analysis
_G = np.load(RESULTS / "fig4_gaston_isodepth.npz")

GAS, GAGREE = {}, pd.read_csv(RESULTS / "fig4_gaston_agreement.csv").set_index("section")

for _s in SEC:
    assert np.allclose(_G[f"{_s}_xy"], MAPS[f"{_s}_xy"]), f"{_s}: isodepth spot set differs"
    GAS[_s] = _rank01(_G[f"{_s}_isodepth_nl3"])

def panel_a(ax, bare=False):
    """Reuses Fig 1b's layout: every section placed in one shared physical coordinate
    system with equal aspect, so the sections are shown at their true relative size;
    viridis colour map; section names above; a dense-tick colourbar at the right.

    The construction itself (MST backbone and signed normal offsets) is shown in Fig S4a
    for all four sections; this panel starts at the first section.

    Top row: the transferred meninges construction. Bottom row: GASTON isodepth (Chitra
    et al.), a published method, trained on these same spots using the settings this
    paper uses for the meninges (8 Pearson dims, [20,20]/[20,20], 8000 epochs, 8
    restarts), an external comparison, distinct from the tissue-surface field (an
    internal ablation shown in Fig S4). The number under each pair is the rank agreement
    between the two axes. The STPI coordinate uses positions only; GASTON's isodepth is
    fitted to expression.

    Absolute I is not compared between the two rows: the targets differ in kind (a
    geometric axis versus an axis distilled from the expression that the decoder then
    reads), and the rule in Fig 3d forbids comparing absolute I across targets. For
    reference, isodepth reads higher under the fixed decoder (0.979 / 1.130 / 1.321 /
    0.436 versus 0.340 / 0.430 / 0.325 / 0.159), and that gap persists under a gene split
    (GASTON trained on half the genes, decoded with the other half: 0.758-1.215 against
    0.147-0.326), so it is not due to shared-gene leakage. What matters for the figure is
    that the measurement-condition structure replicates under the published axis:
    coarse-scale penalty negative at 8/8 depths in CTRL1, CTRL2 and INMPP1 (medians
    -0.25 to -0.30), depth-monotone at 7/7 in 4/4 sections, optimum at 100-200 um;
    INMPP1-2 is the same exception as under the STPI coordinate
    (fig4_gaston_circularity.csv, fig4_coord_transfer.csv).

    The marker criterion is weak and inconsistent across sections
    (fig4_coord_validation.csv) and is not shown in the panel."""
    ax.axis("off"); ax.set_aspect("equal")
    Qs = {s: aligned(s) for s in SEC}
    span = max(np.ptp(Q[:, 0]) for Q, _ in Qs.values())
    GAP = 0.055 * span
    TOP = max(Q[:, 1].max() for Q, _ in Qs.values())
    BOT = min(Q[:, 1].min() for Q, _ in Qs.values())
    ROW = (TOP - BOT) * 1.22                       # vertical pitch between the two rows
    S0 = SEC[0]
    xy0 = np.asarray(GEO[f"{S0}_xy"], float)
    mu = xy0.mean(0)
    _, V = np.linalg.eigh(np.cov((xy0 - mu).T))
    rot = lambda P: (np.asarray(P, float) - mu) @ V[:, ::-1]

    cx = 0.0
    for s in SEC:
        Q, v = Qs[s]
        dx = np.ptp(Q[:, 0]); x0 = Q[:, 0].min()
        for r, vals in ((0, v), (1, GAS[s])):
            gx, gy, Z, A = spot_field(Q, vals)
            ax.imshow(Z, cmap="viridis", vmin=0, vmax=1, origin="lower", alpha=A,
                      extent=[cx + gx[0] - x0, cx + gx[-1] - x0,
                              gy[0] - r * ROW, gy[-1] - r * ROW],
                      interpolation="nearest", rasterized=True, zorder=2)
        if not bare:
            ax.text(cx + dx / 2, TOP + 0.02 * span, DISP[s], ha="center", va="bottom")
            ax.text(cx + dx / 2, BOT - ROW - 0.04 * span,
                    f"|$\\rho$| = {GAGREE.loc[s, 'rho_centreline_nl3']:.2f}",
                    ha="center", va="top", fontsize=ANNO_FS, color="#555")
        if s == SEC[0]:
            yb = BOT - ROW - 0.14 * span
            ax.plot([cx, cx + 1000], [yb, yb], color="black", lw=2.2,
                    solid_capstyle="butt")
            if not bare:
                ax.text(cx + 500, yb - 0.02 * span, "1 mm", ha="center", va="top",
                        fontsize=ANNO_FS)
                _data_note = yb - 0.02 * span      # y-position for the data-provenance label,
                                                    # placed the same way as the scale-bar label above
        cx += dx + GAP
    if not bare:
        # citation convention: the STPI coordinate is labelled "(this study)", the
        # external method is labelled first-author et al. (year)
        for r, lab in ((0, "STPI coordinate \u00b7 positions only\n(this study)"),
                       (1, "GASTON isodepth \u00b7 fitted to expression\nChitra et al. (2025)")):
            ax.text(-0.03 * span, (TOP + BOT) / 2 - r * ROW, lab, ha="right",
                    va="center", fontsize=ANNO_FS)
        ax.text(cx - GAP, _data_note, "data: Cherief et al. (2023)", ha="right",
                va="top", fontsize=ANNO_FS, color="#555")
        cax = ax.inset_axes([1.005, 0.30, 0.011, 0.42])
        cb = ax.figure.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(0, 1), cmap="viridis"), cax=cax)
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cb.ax.tick_params(direction="out", length=5, width=1.2,
                          labelsize=plt.rcParams["ytick.labelsize"])
        cb.outline.set_linewidth(1.0)
        cb.set_label("coordinate value", fontsize=ANNO_FS)
    ax.set_xlim(-0.21 * span, cx - GAP + 0.03 * span)
    ax.set_ylim(BOT - ROW - 0.24 * span, TOP + 0.13 * span)

# ───────────────────────────────── panel b ─────────────────────────────────
def panel_b(ax, bare=False):
    """Fig 1's recoverability-surface form: imshow with nearest interpolation, no grid lines
    between cells, viridis, colour range narrowed to the data. The four sections are laid
    out in one row in the same order as panel a, so a section's coordinate map and its
    response surface sit in the same column. Depth labels appear once, on the left tile.
    Zero bit is marked on the colourbar, because I is signed and viridis is sequential."""
    ax.axis("off")
    Wb = TR[TR.coord == "centreline"]
    vmin, vmax = float(np.nanmin(Wb.I_checker)), float(np.nanmax(Wb.I_checker))
    W_, GAP = 0.212, 0.026
    for k, s in enumerate(SEC):
        M = (Wb[Wb.section == s].pivot(index="depth", columns="scale_um",
                                       values="I_checker")
             .reindex(index=DEPTHS[::-1], columns=SCALES))
        sub = ax.inset_axes([k * (W_ + GAP) + 0.010, 0.10, W_, 0.80])
        sub.imshow(M.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto",
                   interpolation="nearest")
        # star = highest cell of this surface, same device and style as Fig 2a.
        # Descriptive: it locates each section's maximum, it is not a test that the
        # maximum differs from its neighbours.
        if np.isfinite(M.values).any():
            _yi, _xi = np.unravel_index(np.nanargmax(M.values), M.values.shape)
            sub.scatter([_xi], [_yi], marker="*", s=170, color="#ee3b46",
                        edgecolor="white", linewidth=1.2, zorder=5)
        sub.set_xticks(range(len(SCALES))); sub.set_yticks(range(len(DEPTHS)))
        sub.set_xticklabels([str(v) for v in SCALES] if not bare else [])
        sub.set_yticklabels([str(v) for v in DEPTHS[::-1]]
                            if (k == 0 and not bare) else [])
        sub.tick_params(length=2.0, width=0.7)
        for sp_ in sub.spines.values(): sp_.set_linewidth(0.7)
        if not bare:
            sub.set_title(DISP[s], loc="center", pad=2.5, fontsize=ANNO_FS)
            if k == 0:
                sub.set_ylabel("depth",   # Fig 2a's equivalent axis reads "depth (counts per pseudospot)"
                               fontsize=ANNO_FS, labelpad=1.5)
    if not bare:
        ax.text(0.44, 0.005, "pseudospot size s (µm)", transform=ax.transAxes,
                ha="center", va="top")
        cax = ax.inset_axes([4 * (W_ + GAP) + 0.012, 0.18, 0.013, 0.62])
        cb = ax.figure.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin, vmax), cmap="viridis"), cax=cax)
        cb.set_ticks(sorted({round(x, 1) for x in np.arange(-0.1, vmax + 0.01, 0.1)} | {0.0}))
        cb.ax.tick_params(direction="out", length=5, width=1.2,
                          labelsize=plt.rcParams["ytick.labelsize"])
        cb.outline.set_linewidth(1.0)
        # zero on this colourbar is located by the 0.0 tick, which set_ticks pins; no
        # separate marker line is drawn
        cb.set_label(r"$\hat{I}$ (bit)", fontsize=ANNO_FS)

# ───────────────────────────────── panel c ─────────────────────────────────
def panel_c(ax, bare=False):
    """Every feasible cell, not one depth slice: for each pseudospot size the 32 readings
    (four sections x eight depths) are shown as a half violin with the individual cells
    beside it, coloured by section, and a median tick. Coarsening moves the whole
    distribution down, not just the best cell.

    The fraction of cells above zero (100 / 88 / 78 / 66 / 53 % from 100 to 800 um) is a
    result stated in the caption rather than on the panel. Medians are 0.139 / 0.097 /
    0.052 / 0.049 / 0.006, computed from fig4_coord_transfer.csv."""
    ax.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    rg = np.random.default_rng(11)
    for i_, s0 in enumerate(SCALES):
        q = W_C[W_C.scale_um == s0]
        v = q.I_checker.values
        # gaussian KDE over the group's own support - a histogram outline is jagged and
        # trails a vertical line through the zero-density region
        kd = gaussian_kde(v, bw_method=0.45)
        yc = np.linspace(v.min(), v.max(), 200)
        dens = kd(yc); dens = dens / dens.max() * 0.34
        ax.fill_betweenx(yc, i_, i_ + dens, color="#c9d1d7", alpha=0.9, lw=0, zorder=2)
        ax.plot(i_ + dens, yc, color="#8d99a2", lw=0.8, zorder=3)
        for s in SEC:
            vs = q[q.section == s].I_checker.values
            ax.scatter(i_ - 0.10 - rg.uniform(0, 0.20, len(vs)), vs, s=9,
                       color=CC[s], alpha=0.9, linewidths=0, zorder=4)
        ax.plot([i_ - 0.34, i_ + 0.02], [np.median(v)] * 2, color="#2b3640", lw=2.2,
                zorder=6, solid_capstyle="butt")
    ax.set_xticks(range(len(SCALES)))
    ax.set_xticklabels(["100\n(native pitch)"] + [str(v) for v in SCALES[1:]])
    ax.set_xlim(-0.55, len(SCALES) - 0.40)
    ax.set_ylim(V_LO - 0.02, V_HI + 0.03)
    ax.set_xlabel("pseudospot size s (µm)")
    ax.set_ylabel(r"$\hat{I}$ (bit)")
    h = [Line2D([0], [0], color=CC[s], lw=0, marker="o", ms=4.4) for s in SEC] + \
        [Line2D([0], [0], color="#2b3640", lw=2.2)]
    ax.legend(handles=h, labels=[""] * 5 if bare else DSEC + ["median"], loc="upper right",
              fontsize=ANNO_FS, ncol=2, handlelength=1.4, handletextpad=0.45,
              labelspacing=0.24, columnspacing=0.9, borderpad=0.1)

# ──────────────── panel d · the reading against tissue thickness ────────────────
GEO_T = {s: float(np.percentile(GEO[f"{s}_off"], 95) - np.percentile(GEO[f"{s}_off"], 5))
         for s in SEC}                               # 5-95% span of the across-band offset

NSPOT = {s: int(np.asarray(GEO[f"{s}_xy"]).shape[0]) for s in SEC}

PITCH_UM = 100.0          # Visium centre-to-centre spot pitch

def spot_field(Q, vals, up=8.0, reach=0.62, soft_um=70.0):
    """Render spots as a filled field with a smooth outline.

    Drawing one marker per spot leaves the background visible between them, so the tissue
    reads as a dotted texture. Here every grid node takes the value of its nearest spot
    (no smoothing of the data, no value is invented), and the tissue boundary is carried
    by an alpha channel rather than by a hard mask: a hard distance cut would quantise
    the outline into steps of one grid cell plus the arc of each spot's disc, producing a
    serrated outline. The occupancy indicator is blurred with a kernel of `soft_um`
    micrometres, comparable to the pitch itself, because the serration has the period of
    the lattice and a kernel much smaller than the pitch cannot remove it, and then
    thresholded with a narrow ramp, which leaves a smooth antialiased outline.

    Grid step is pitch / up (12.5 um at the default), finer than print resolution.

    What the kernel costs, stated because it is a rendering choice and not data: interior
    gaps narrower than about soft_um close up (two single-spot gaps in Control-1 do), and
    an isolated island of one or two spots can disappear. Gaps and fragments three or more
    spots across survive. Nothing about the decoded values changes: this affects only
    where the drawn tissue ends; every coloured pixel still carries the coordinate value of
    its nearest spot. For an exact boundary, render at soft_um -> 0 (serrated) or use the
    bare tier with the raw lattice.
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter
    step = PITCH_UM / up
    pad = 2.0 * PITCH_UM
    xs = np.arange(Q[:, 0].min() - pad, Q[:, 0].max() + pad + step, step)
    ys = np.arange(Q[:, 1].min() - pad, Q[:, 1].max() + pad + step, step)
    Xg, Yg = np.meshgrid(xs, ys)
    d, idx = cKDTree(Q).query(np.c_[Xg.ravel(), Yg.ravel()])
    Z = np.asarray(vals, float)[idx].reshape(Xg.shape)
    occ = (d.reshape(Xg.shape) <= reach * PITCH_UM).astype(float)
    occ = gaussian_filter(occ, soft_um / step)
    A = np.clip((occ - 0.50) / 0.10, 0.0, 1.0)      # narrow ramp = smooth, crisp outline
    return xs, ys, Z, A

CELL_D = (100, 2048)   # the platform's native pitch and the deepest depth feasible in all

def panel_d(ax, bare=False):
    """Fig 2's question asked inside this dataset: is the difference between sections a
    property of the tissue, or of the measurement?

    One point per section: x is the across-band thickness (the 5-95 percentile span of the
    signed offset), y is the reading at one fixed setting, named on the y axis. Fixing the
    setting is what makes the four points comparable; the full surfaces are shown in
    panel b.

    Thickness rises 1.9-fold across the four sections and the readings do not follow it.
    The tick labels under each point carry the number of spots, which rises in exactly
    the same order (rank correlation with thickness 1.00 across these four sections), so
    in this sample the two cannot be told apart. Neither orders the readings.

    The reading, rather than the optimal size used in Fig 2, is plotted here because the
    optimum barely moves on this platform (23 of the 32 section x depth cells sit at
    100 um, the finest size the Visium pitch can form; per section 1/8, 8/8, 8/8, 6/8),
    leaving nothing for thickness to move against; that evidence is reported in the
    caption.

    The conditions do not separate at this cell: at 100 um / depth 2048 the order is
    Control-2 +0.393, 1NMPP1-1 +0.325, Control-1 +0.256, 1NMPP1-2 +0.159, so the two
    groups interleave. They do separate on each section's median over all 40 settings
    (Control 0.114 / 0.119 against 1NMPP1 0.027 / 0.049), a different quantity from the
    one plotted here. This figure does not compare the conditions: there are two sections
    per condition and no animal IDs in GEO.

    The absence of a thickness trend cannot be given a p-value either: with four points
    the smallest attainable two-sided Spearman p is 0.167, so the claim is the stated
    one, that thickness does not order the readings and cannot be separated from spot
    count, rather than a claim that a test failed to reject."""
    sc, dp = CELL_D
    xs, ys = [], []
    for s in SEC:
        q = W_C[(W_C.section == s) & (W_C.scale_um == sc) & (W_C.depth == dp)]
        assert len(q) == 1, f"{s}: cell {CELL_D} not unique in the grid"
        xs.append(GEO_T[s]); ys.append(float(q.I_checker.iloc[0]))
    # The panel shows one point per section, matching Fig 2b's form; each section's full
    # range over the 40 settings is reported in the caption instead: the within-section
    # range is 0.32-0.50 bit (CTRL1 -0.159..0.340, CTRL2 -0.028..0.430, INMPP1
    # -0.174..0.325, INMPP1-2 -0.165..0.159), while the spread between sections at the
    # plotted cell is only 0.234 bit, so the measurement setting moves the reading more
    # than the section does.
    ax.axhline(0, color=GREY, lw=0.9, ls=(0, (3, 3)), zorder=1)
    for s, x, y in zip(SEC, xs, ys):
        ax.plot([x], [y], "o", ms=11, color=CC[s], mec="white", mew=1.0, zorder=4)
        if not bare:
            # Labels show the section name only; the per-point spot counts are reported
            # in the caption instead (2,027 / 2,751 / 2,837 / 3,047 for Control-2 /
            # 1NMPP1-2 / 1NMPP1-1 / Control-1), with the confounding noted on the x axis
            # ("spot count rises in the same order").
            ax.annotate(DISP[s], (x, y), textcoords="offset points",
                        xytext=(13, 0), ha="left", va="center",
                        fontsize=ANNO_FS + 0.5, color=CC[s])
    # axis limits follow the points; zero stays in view because the dashed zero line is
    # the panel's reference
    lo_all, hi_all = min(min(ys), 0.0), max(ys)
    ax.set_xlim(1750, 5100)
    pad = 0.12 * (hi_all - lo_all)
    ax.set_ylim(lo_all - pad, hi_all + 2.0 * pad)
    ax.set_xticks([2000, 2500, 3000, 3500, 4000, 4500])
    ax.set_xlabel("tissue across-band thickness T (µm)\n"
                  "spot count rises in the same order (rank correlation 1.00)")
    ax.set_ylabel("$\\hat{I}$ (bit)\n" + f"at {sc} µm, {dp} counts")

# ───────────────────────────────── panel e ─────────────────────────────────
def panel_e(ax, bare=False):
    """Median inflation per section and depth, printed in every cell (32 cells). Fold count,
    fold sizes and the per-fold estimator are held fixed by construction, so the only thing
    the permutation changes is spatial structure."""
    M = (CVD.pivot_table(index="section", columns="depth", values="inflation",
                         aggfunc="median").reindex(index=SEC, columns=DEPTHS))
    v = float(np.nanmax(np.abs(M.values)))
    ax.imshow(M.values, cmap="PuOr_r", vmin=-v, vmax=v, aspect="auto",
              interpolation="nearest")
    ax.set_xticks(range(len(DEPTHS))); ax.set_yticks(range(len(SEC)))
    ax.set_xticklabels([str(d) for d in DEPTHS]); ax.set_yticklabels(DSEC)
    for sp_ in ax.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.6)
    ax.tick_params(length=1.8, width=0.6)
    if not bare:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}",
                            ha="center", va="center",
                            fontsize=plt.rcParams["xtick.labelsize"],
                            color="black" if abs(val) < 0.6 * v else "white")
        ax.set_xlabel("target depth (transcripts per pseudospot)")
        cax = ax.inset_axes([1.02, 0.08, 0.022, 0.84])
        cb = ax.figure.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(-v, v), cmap="PuOr_r"), cax=cax)
        cb.set_label(r"$\Delta\hat{I}$  permuted $-$ checkerboard (bit)",
                     fontsize=plt.rcParams["legend.fontsize"])
        cb.ax.tick_params(length=2.2, width=0.7); cb.outline.set_linewidth(0.7)


def strip_bare(ax, lab):
    """bare tier: no text, no axes; data marks and reference lines survive. Inset axes and
    colourbars carry their own text and are cleared too."""
    for other in ax.figure.axes:
        if other is ax: continue
        other.set_xlabel(""); other.set_ylabel(""); other.set_title("")
        for t in list(other.texts):
            if t.get_text().strip(): t.remove()
        other.axis("off")
    ax.set_title("", loc="left"); ax.set_xlabel(""); ax.set_ylabel("")
    for t in list(ax.texts):
        if t.get_text().strip(): t.remove()
    lg = ax.get_legend()
    if lg is not None: lg.remove()
    ax.axis("off")



PANELS = {"A": (panel_a, (10.4, 6.6)), "B": (panel_b, (7.3, 2.0)), "C": (panel_c, (4.8, 3.2)),
          "D": (panel_d, (4.6, 3.4)), "E": (panel_e, (5.4, 2.3))}
A_SHOWN = (0.0482, 0.5248)        # left and bottom fractions of panel A outside the figure


def marks():
    d = marks_dir(FIG)
    for lab, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, True)
        strip_bare(ax, lab.lower())
        fig.savefig(d / f"{FIG}_{lab}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    im = Image.open(d / "Fig4_A.png"); W, H = im.size
    im.crop((int(A_SHOWN[0] * W), 0, W, int(H * (1 - A_SHOWN[1])))).save(d / "Fig4_A.png")
    colourbar_png(d / "Fig4_A_colourbar.png", "viridis", 35, 1180, ((0, 34), (0, 1179)), 0)
    colourbar_png(d / "Fig4_B_colourbar.png", "viridis", 68, 1214, ((0, 67), (0, 1213)), 0)
    colourbar_png(d / "Fig4_E_colourbar.png", "PuOr_r", 35, 879, ((0, 34), (0, 878)), 0)


TITLES = {"A": "Expression-independent coordinate in injured mouse tendon (STPI: across the band)",
          "B": "Recoverability across pseudospot size and depth (STPI: across the band)",
          "C": "Recoverability by pseudospot size and depths",
          "D": "Recoverability versus tissue thickness\n(100 \u00b5m, depth 2048)",
          "E": "Difference in recoverability between spatial block\nand random CV (STPI: across the band)"}


def _a_top_row(ax):
    """Keeps the STPI row of panel A: removes the second row, its labels, and the
    associated notes."""
    ims = ax.get_images()
    top = ims[0::2]
    for im in ims[1::2]:
        im.remove()
    y_top0 = min(im.get_extent()[2] for im in top)
    span = max(im.get_extent()[1] for im in top) - min(im.get_extent()[0] for im in top)
    for t in list(ax.texts):
        if t.get_position()[1] < y_top0 or "this study" in t.get_text():
            t.remove()
    for ln in list(ax.lines):
        ln.remove()
    for im in top:                                    # 1 mm bar under each section
        x0, x1, y0, _ = im.get_extent()
        xm = x1 - 0.06 * (x1 - x0)
        ax.plot([xm - 1000, xm], [y_top0 - 0.03 * span] * 2, color="black", lw=1.6,
                solid_capstyle="butt")
        ax.text(xm - 500, y_top0 - 0.03 * span, "1mm", ha="center", va="bottom",
                fontsize=ANNO_FS)
    ax.text(0.5, 0.0, "Cherief et al. (2023)", transform=ax.transAxes, ha="center",
            va="top", fontsize=ANNO_FS)
    ax.set_ylim(y_top0 - 0.08 * span, ax.get_ylim()[1])
    for other in ax.child_axes:                       # the colour bar
        other.set_ylabel("Coordinate value", fontsize=ANNO_FS)


def complete_text(lab, ax):
    ax.set_title(TITLES[lab], loc="left")
    if lab == "A":
        _a_top_row(ax)
    elif lab == "B":
        if ax.get_xlabel():
            ax.set_xlabel("Pseudospot size (\u03bcm)")
        for tx in ax.texts:
            if tx.get_text().startswith("pseudospot size"):
                tx.set_text("Pseudospot size (\u03bcm)")
        for other in ax.child_axes:
            if other.get_xlabel():
                other.set_xlabel("Pseudospot size (\u03bcm)")
            if other.get_ylabel():
                other.set_ylabel("" if "bit" in other.get_ylabel() else "Depth")
    elif lab == "C":
        ax.set_xlabel("Pseudospot size (\u03bcm)"); ax.set_ylabel("")
        ax.set_xticklabels([t.get_text().split("\n")[0] for t in ax.get_xticklabels()])
    elif lab == "D":
        ax.set_xlabel("Tissue radial thickness (\u03bcm)"); ax.set_ylabel("")
    elif lab == "E":
        ax.set_xlabel("Depth"); ax.set_ylabel("")
        for other in ax.child_axes:
            other.set_ylabel("")


def full():
    d = OUT / FIG; d.mkdir(parents=True, exist_ok=True)
    for lab, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, False)
        complete_text(lab, ax)
        for ext in (".png", ".pdf"):
            fig.savefig(d / f"{FIG}_{lab}{ext}", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    marks()
    full()
    print("D (thickness um, estimate):",
          {DISP[s]: (round(GEO_T[s]), round(float(W_C[(W_C.section == s) &
                     (W_C.scale_um == CELL_D[0]) & (W_C.depth == CELL_D[1])].I_checker.iloc[0]), 3))
           for s in SEC})
