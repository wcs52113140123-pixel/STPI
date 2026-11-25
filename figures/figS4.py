#!/usr/bin/env python3
"""Figure S4. Supplementary analyses of the injured mouse tendon.

  A  Geometric construction of the injured mouse tendon (STPI: across the band)
  B  Coordinates predicted by gene expression (200 um, depth 2048)
  C  Recoverability against pseudospot size and depth under four coordinate constructions
  D  Recoverability of three axes against an autocorrelation-preserving null (200 um, depth 2048)
  E  Difference in recoverability between spatial block and random CV under two coordinates

Inputs (results/)
  fig4_tendon_null.npz/.csv, fig4_tendon_null_surface.npz/.csv, fig4_coord_transfer.csv,
  fig4_coord_geometry.npz, fig4_tendon_cv.csv, fig4_tendon_cv_along.csv,
  fig4_tendon_cv_gaston.csv, decoded_maps_tendon.npz
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from scipy.stats import gaussian_kde, spearmanr as _spearmanr
from PIL import Image

from _common import RESULTS, FONT, marks_dir, colourbar_png, OUT
from _fields import spot_field as _spot_field_nearest, rank01 as _rank01
from fig4 import spot_field as _spot_field_soft       # the same renderer Fig 4A uses

FIG = "FigS4"
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

# same palette as main Fig 4: condition picks the hue family, section samples within it
CC = {"CTRL1": "#1f5f82", "CTRL2": "#6aaed6", "INMPP1": "#6b3a75", "INMPP1-2": "#c994c7"}

GREY = "#8a8a8a"

# keys match the sample names on disk; these are the display labels used on the figure.
# GSE245179: Control_1/2 are TrkA wild type, 1NMPP1_1/2 are TrkA F592A given 1NM-PP1.
DISP = {"CTRL1": "Control-1", "CTRL2": "Control-2",
        "INMPP1": "1NMPP1-1", "INMPP1-2": "1NMPP1-2"}

DSEC = [DISP[s] for s in SEC]

# Column order: the three framework coordinates first (the main coordinate, then its two
# ablations), followed by the published method. STPI is the framework's name; the
# coordinate it uses on this tissue is the across-band offset from the transferred
# backbone, so the column is labelled as both.
# The second line of each column head states whether that coordinate was built from
# positions only or fitted to expression - GASTON is the only one of the four that used
# expression, so its absolute values are not directly comparable with the other three.
COORDS = [("centreline", "STPI (this study)"),
          ("along", "along the backbone (ours)"),
          ("surface", "surface distance (ours)"),
          ("gaston", "GASTON isodepth, Chitra et al. (2025)")]

KIND = {"centreline": "positions only", "along": "positions only",
        "surface": "positions only", "gaston": "fitted to expression"}

SHORT = {"centreline": "STPI\n(this study)", "along": "along the backbone\n(this study)",
         "surface": "surface distance\n(this study)",
         "gaston": "GASTON isodepth\nChitra et al. (2025)"}

NULLZ = np.load(RESULTS / "fig4_tendon_null.npz")

NULLT = pd.read_csv(RESULTS / "fig4_tendon_null.csv")

# The surface-distance arm is computed separately from the other two null arms. CTRL2 has
# no surrogate distribution for it: a boundary-distance field is near-radially symmetric,
# and on that section no rotation decorrelates it (0 of 1000 angles below |rho| = 0.30,
# median 0.48), so the device has no valid instantiation there and the cell is drawn as
# n.d. rather than relaxing the rule.
NULLZS = np.load(RESULTS / "fig4_tendon_null_surface.npz")

NULLTS = pd.read_csv(RESULTS / "fig4_tendon_null_surface.csv").set_index("section")

TR = pd.read_csv(RESULTS / "fig4_coord_transfer.csv")

CELL = (200, 2048)          # the cell drawn in panel d; all three are in the caption

SCALES = [100, 200, 400, 600, 800]

DEPTHS = [16, 32, 64, 128, 256, 512, 1024, 2048]

GEO = np.load(RESULTS / "fig4_coord_geometry.npz")



# ────────── panel a · construction: the coordinate being built, on every section ────────
def _aligned_geo(s, ref):
    """Every section rotated onto its own principal axes and expressed in the SAME physical
    coordinate system (µm), as in Fig 1b, so the four stand at their true relative size.
    `ref` fixes the rotation convention; each section uses its own principal frame."""
    xy = np.asarray(GEO[f"{s}_xy"], float)
    mu = xy.mean(0)
    _, V = np.linalg.eigh(np.cov((xy - mu).T))
    rot = lambda P: (np.asarray(P, float) - mu) @ V[:, ::-1]
    return rot(xy), rot(GEO[f"{s}_backbone"]), rot(GEO[f"{s}_arrow_p0"]), \
        rot(GEO[f"{s}_arrow_p1"])

def panel_a(ax, bare=False):
    """The coordinate construction, shown on all four sections.

    Grey: the spots. Black: the MST backbone (k-means knots, minimum spanning tree, tree
    diameter as the backbone, arclength smoothing). Red: the signed normal offset of a
    sample of spots, which is the coordinate itself. Positions only: no expression data
    enters the construction or these maps, since the axis read in Fig 4 is geometric and
    not a restatement of the expression the decoder sees.

    The sections share one physical scale (Fig 1b's convention), so their relative sizes
    are true; each is rotated onto its own principal axes so the backbone runs left to
    right and the four can be compared by eye.

    Backbone lengths and spot counts differ between sections and are reported in the
    caption rather than on the panel: CTRL1 3047 spots, CTRL2 2027, INMPP1 2837,
    INMPP1-2 2751."""
    G = {s: _aligned_geo(s, SEC[0]) for s in SEC}
    span = max(np.ptp(Q[:, 0]) for Q, _, _, _ in G.values())
    TOP = max(Q[:, 1].max() for Q, _, _, _ in G.values())
    BOT = min(Q[:, 1].min() for Q, _, _, _ in G.values())
    GAP = 0.055 * span
    ax.axis("off"); ax.set_aspect("equal")
    cx = 0.0
    for s in SEC:
        Q, bb, p0, p1 = G[s]
        dx = np.ptp(Q[:, 0]); x0 = Q[:, 0].min()
        ax.scatter(cx + Q[:, 0] - x0, Q[:, 1], s=1.6 + 2.0 * dx / span,
                   color="#d7dce0", linewidths=0, rasterized=True, zorder=2)
        for q0, q1 in zip(p0, p1):
            ax.annotate("", xy=(cx + q1[0] - x0, q1[1]), xytext=(cx + q0[0] - x0, q0[1]),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8,
                                        shrinkA=0, shrinkB=0, mutation_scale=6), zorder=4)
        ax.plot(cx + bb[:, 0] - x0, bb[:, 1], color="black", lw=1.6, zorder=5)
        if not bare:
            ax.text(cx + dx / 2, TOP + 0.02 * span, DISP[s], ha="center", va="bottom")
        if s == SEC[0]:
            yb = BOT - 0.10 * span
            ax.plot([cx, cx + 1000], [yb, yb], color="black", lw=2.2,
                    solid_capstyle="butt")
            if not bare:
                ax.text(cx + 500, yb - 0.02 * span, "1 mm", ha="center", va="top",
                        fontsize=ANNO_FS)
        cx += dx + GAP
    if not bare:
        h = [Line2D([0], [0], marker="o", ms=4, color="#d7dce0", ls="none"),
             Line2D([0], [0], color="black", lw=1.6),
             Line2D([0], [0], color="#c0392b", lw=1.0, marker=">", ms=4)]
        ax.legend(handles=h, labels=["spots (positions only)", "MST backbone",
                                     "signed normal offset = the coordinate"],
                  loc="upper left", fontsize=ANNO_FS, ncol=3, handlelength=1.4,
                  handletextpad=0.5, columnspacing=1.0, borderpad=0.1,
                  bbox_to_anchor=(0.0, -0.02))
    ax.set_xlim(-0.03 * span, cx - GAP + 0.03 * span)
    ax.set_ylim(BOT - 0.30 * span, TOP + 0.10 * span)


# ────────── panel b · coordinates predicted by gene expression ──────────────────────────
spot_field_f5, rank01_f5 = _spot_field_nearest, _rank01


spot_field_f4 = _spot_field_soft

DMT = np.load(RESULTS / "decoded_maps_tendon.npz")

def panel_b(ax, bare=False):
    """What expression recovers of the transferred coordinate, on all four sections.

    Figure 5b shows this for other groups' sections; S1 f is the meninges counterpart.

    Each map is the held-out prediction only: every test pseudospot is predicted by a
    model that never saw its spatial block. Colour is the rank-normalised predicted value,
    so the four maps share one scale and the question is whether the across-band gradient
    appears.

    Cell: 200 um pseudospots at depth 2048, the same cell as panel d and the cell feasible
    in all four sections, not each section's own best cell.

    rho(coordinate, decoded) is printed under each map and is descriptive, as in Fig 5b;
    the reported metric is I (mutual information). Values recomputed from the kept
    prediction vector sit within 0.049 bit of the grid in results/decoded_maps_check.csv.

    Sections keep their true relative size (Fig 1b's convention) and each is rotated onto
    its own principal axes, as in S4 a.
    """
    # Orientation uses the same approach as Fig 4a: each section is centred on its mean and
    # rotated onto its principal axes (same point set, so the same rotation matrix). The y
    # axis is not flipped, which keeps orientation consistent with Fig 4a (flipping, as in
    # Fig 5b's image-space convention, would mirror each section relative to Fig 4a).
    rotm = {}
    for s_ in SEC:
        xy = np.asarray(GEO[f"{s_}_xy"], float)
        mu = xy.mean(0)
        _, V = np.linalg.eigh(np.cov((xy - mu).T))
        rotm[s_] = (mu, V[:, ::-1])
    # Drawn at native spot resolution using main Fig 4a's renderer, so the silhouettes
    # match Fig 4a's. The underlying reading is still computed on 200 um pseudospots: each
    # spot takes the decoded value of the bin containing it, so the field is piecewise
    # constant over a bin and nothing is interpolated. Drawing the bin centres instead would
    # blockify the outline and erode the tissue edge by up to one bin width, which is why
    # spot positions are used instead. One marker per pseudospot (512 to 745 per section),
    # read as a mosaic.
    KEEP = {s_: np.isfinite(np.asarray(DMT[f"{s_}_spot_val"], float)) for s_ in SEC}
    Q = {s_: (np.asarray(DMT[f"{s_}_spot_xy"], float)[KEEP[s_]] - rotm[s_][0]) @ rotm[s_][1]
         for s_ in SEC}
    widths = {s_: np.ptp(q[:, 0]) for s_, q in Q.items()}
    span = max(widths.values())
    ax.axis("off")
    gap = 0.030
    wtot = sum(widths.values()) / span
    scale = (1.0 - gap * (len(SEC) - 1)) / wtot
    y0, y1 = 0.16, 0.92
    cx = 0.0
    for s_ in SEC:
        de = np.asarray(DMT[f"{s_}_decoded"], float)
        q = Q[s_]
        vals = np.asarray(DMT[f"{s_}_spot_val"], float)[KEEP[s_]]
        gx, gy, Z, Al = spot_field_f4(q, rank01_f5(vals))
        w = widths[s_] / span * scale
        sub = ax.inset_axes([cx, y0, w, y1 - y0])
        sub.imshow(Z, cmap="viridis", vmin=0, vmax=1, origin="lower", alpha=Al,
                   extent=[gx[0], gx[-1], gy[0], gy[-1]], interpolation="nearest",
                   rasterized=True)
        sub.set_aspect("equal"); sub.axis("off"); sub.margins(0.02)
        if not bare:
            ax.text(cx + w / 2, 0.985, DISP[s_], transform=ax.transAxes, ha="center",
                    va="top", fontsize=ANNO_FS, color=CC[s_])
            ax.text(cx + w / 2, 0.125,
                    f"$\\rho$ = {_spearmanr(DMT[f'{s_}_coord'], de).correlation:.2f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=TICK_FS,
                    color="#444")
            frac = 1000.0 / span * scale       # 1 mm in axes fraction
            bar = ax.inset_axes([cx + w / 2 - frac / 2, 0.055, frac, 0.010])
            bar.axis("off"); bar.plot([0, 1], [0.5, 0.5], color="black", lw=1.8,
                                      solid_capstyle="butt")
            bar.set_xlim(0, 1); bar.set_ylim(0, 1)
            # axis("off") hides the spines but the tick labels remain as text elements, so
            # they must be cleared explicitly to avoid stray text on the colour bar
            bar.set_xticks([]); bar.set_yticks([])
            ax.text(cx + w / 2, 0.045, "1 mm", transform=ax.transAxes, ha="center",
                    va="top", fontsize=TICK_FS, color=GREY)
        cx += w + gap
    if not bare:
        cax = ax.inset_axes([0.995, y0 + 0.08, 0.008, (y1 - y0) * 0.62])
        cb = ax.figure.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, 1), cmap="viridis"),
                                cax=cax)
        cb.set_ticks([0, 0.5, 1.0])
        cb.ax.tick_params(direction="out", length=2.4, width=0.8, labelsize=TICK_FS)
        cb.outline.set_linewidth(0.8)
        cb.set_label("decoded value (rank)", fontsize=TICK_FS)
# ────────── panel c · recoverability under four coordinate constructions ────────────────
def panel_c(ax, bare=False):
    """The two shapes main Fig 4 claims, drawn as curves under four coordinate constructions
    rather than reduced to a count, such as how many of 8 depths show the coarse-scale
    penalty or how many of 7 depth steps rise, which would ask the reader to reconstruct a
    shape from an integer.

    Columns, in order: the STPI coordinate main Fig 4 uses (the across-band offset from the
    transferred backbone), its two ablations (arclength along the same backbone, and
    geodesic distance from the tissue surface), and last the published GASTON isodepth.
    The first three are built from spot positions only; GASTON is fitted to the expression
    the decoder then reads, which is why the columns' absolute levels are not comparable.
    One line per section.

    Per-section counts move between columns: the coarse-scale penalty for 1NMPP1-2 is
    3/8 under STPI, 7/8 along the backbone, 6/8 on the surface field and 2/8 under GASTON.
    "The same shape" refers to the direction of each curve and where its optimum sits, not
    to agreement of the per-section counts (see caption).

    Top row: I against pseudospot size, each section at its own best depth. The curve is
    expected to fall to the right and peak near the platform's native pitch; the marker is
    each curve's maximum.
    Bottom row: I against molecular depth, each section at its own best size. The curve is
    expected to rise.

    y is scaled per column, and the columns are not meant to be read against each other:
    the targets differ in kind (a geometric axis versus one distilled from the expression
    the decoder then reads), and absolute I is not compared across them. What is compared
    is the shape, which is what main Fig 4 claims."""
    ax.axis("off")
    W_, H_, GX, GY = 0.190, 0.320, 0.078, 0.235
    for k, (c, lab) in enumerate(COORDS):
        q0 = TR[TR.coord == c].dropna(subset=["I_checker"])
        for row in (0, 1):
            sub = ax.inset_axes([k * (W_ + GX) + 0.055, 0.60 - row * (H_ + GY), W_, H_])
            lo, hi = [], []
            for sname in SEC:
                qs = q0[q0.section == sname]
                if not len(qs):
                    continue
                b = qs.loc[qs.I_checker.idxmax()]
                if row == 0:
                    v = qs[qs.depth == b.depth].sort_values("scale_um")
                    x, y = v.scale_um.values, v.I_checker.values
                else:
                    v = qs[qs.scale_um == b.scale_um].sort_values("depth")
                    x, y = v.depth.values, v.I_checker.values
                sub.plot(x, y, color=CC[sname], lw=1.3, zorder=3)
                sub.plot([x[np.argmax(y)]], [y.max()], "o", ms=4, color=CC[sname], zorder=4)
                lo.append(y.min()); hi.append(y.max())
            sub.set_xscale("log", base=2)
            sub.axhline(0, color="#cccccc", lw=0.7, zorder=1)
            pad = 0.08 * (max(hi) - min(lo))
            sub.set_ylim(min(lo) - pad, max(hi) + pad)
            if row == 0:
                sub.set_xticks(SCALES)
                sub.set_xticklabels([str(v) for v in SCALES] if not bare else [],
                                    fontsize=TICK_FS - 0.5)
            else:
                sub.set_xticks([16, 64, 256, 1024])
                sub.set_xticklabels(["16", "64", "256", "1024"] if not bare else [],
                                    fontsize=TICK_FS - 0.5)
            sub.tick_params(length=2.0, width=0.7, labelsize=TICK_FS - 0.5)
            for sp_ in sub.spines.values():
                sp_.set_linewidth(0.7)
            if bare:
                sub.set_yticklabels([])
            if not bare:
                if row == 0:
                    sub.set_title(SHORT[c] + "\n" + KIND[c], loc="center", pad=3.0,
                              fontsize=ANNO_FS - 0.5)
                sub.set_xlabel("size s (µm)" if row == 0
                               else "depth (per pseudospot)",
                               fontsize=ANNO_FS - 0.5, labelpad=1.5)
                if k == 0:
                    # the per-column scaling is a reading condition, so it is stated in the
                    # axis label rather than in a separate note
                    sub.set_ylabel("$\\hat{I}$ (bit)\nscaled per column"
                                   + ("\nat each section's best depth" if row == 0 else
                                      "\nat each section's best size"),
                                   fontsize=ANNO_FS - 0.5, labelpad=1.5)
    if not bare:
        h = [Line2D([0], [0], color=CC[sname], lw=1.3) for sname in SEC]
        ax.legend(handles=h, labels=DSEC, loc="upper center", fontsize=ANNO_FS, ncol=4,
                  handlelength=1.4, handletextpad=0.5, columnspacing=1.3, borderpad=0.1,
                  bbox_to_anchor=(0.5, 1.10))

W_CEN = TR[TR.coord == "centreline"].dropna(subset=["I_checker"])

BEST_C = {s_: int(W_CEN[W_CEN.section == s_]
                  .loc[W_CEN[W_CEN.section == s_].I_checker.idxmax()].depth) for s_ in SEC}


# ────────── panel d · guard (3): both axes against the null ─────────────────────────────
def panel_d(ax, bare=False):
    """The framework's third guard on the external tissue. The coordinate field is rigidly
    rotated about the section centroid and read back at the true spot positions; a draw
    counts only if it is decorrelated from the real axis (|rho| < 0.30), because on an
    elongated section a rotation by ~0 or ~180 degrees returns nearly the same field and
    would seed the null with re-labelled copies of the axis being tested.
    P = (#{null >= observed} + 1) / 201.

    Four tendon sections at 200 um / depth 2048. The other two preset cells (100 um /
    depth 2048 and 100 um / depth 1024) give the same verdict and are reported in the
    caption: across the band clears in 2 of 12 section-by-cell combinations, all of them
    CTRL2; along the backbone clears in 12 of 12 at P = 0.005, the floor at 200 draws.

    Sensitivity of the |rho| < 0.30 rejection rule, along
    the backbone at 200 um / depth 2048, P at cuts 0.20 / 0.30 / 0.40 / 0.50 / 1.00:
      CTRL1     0.005  0.005  0.005  0.030  0.209
      CTRL2     0.005  0.005  0.005  0.025  0.119
      INMPP1    0.005  0.005  0.005  0.005  0.174
      INMPP1-2  0.005  0.005  0.005  0.005  0.139
    The verdict does not depend on where the cut sits: P = 0.005 at every cut from
    0.20 to 0.40 in all four sections and stays at or below 0.030 at 0.50. Only
    removing the constraint entirely (1.00) collapses it to 0.119-0.209, the contamination
    the rule is designed to remove. Acceptance rate at 0.30 is 0.18 / 0.22 / 0.28 / 0.34.

    The meninges calibration is reported in the caption rather than in this panel: the two
    axes compared here are read off the same backbone, on the same sections, against the
    same null, and one clears it while the other does not, which demonstrates that the
    test discriminates without needing the external tissue. The caption reports the
    external check: the same null run unchanged on the meninges gives GW12 P = 0.005
    (I = +0.516 against a null median of +0.035) while GW8 (P = 0.552) and GW16 (P = 0.657)
    do not clear it, the same value the tangential null behind Fig 1d gives
    (fig4_null_calibration.csv)."""
    n0 = NULLT.query("scale_um == @CELL[0] and depth == @CELL[1]").set_index("section")
    xs = np.arange(len(SEC))
    # Two arms per section, side by side: left is the across-band coordinate the main
    # figure uses, right is the along-backbone arclength read off the same backbone. The
    # second arm is this panel's internal positive control: same sections, same backbone,
    # same null device, and one axis clearing (12/12 across the three cells) while the
    # other does not (2/12) shows that the test discriminates rather than rejecting
    # everything.
    # Both draw sets are stored in fig4_tendon_null.npz as *_spin and *_spin_along.
    DX = 0.25
    s0 = NULLTS.query("scale_um == @CELL[0] and depth == @CELL[1]")
    d_across = [NULLZ[f"{s}_{CELL[0]}um_d{CELL[1]}_spin"] for s in SEC]
    d_along = [NULLZ[f"{s}_{CELL[0]}um_d{CELL[1]}_spin_along"] for s in SEC]
    k_surf = {s: f"{s}_{CELL[0]}um_d{CELL[1]}_spin_surface" for s in SEC}
    d_surf = {s: NULLZS[k_surf[s]] for s in SEC if k_surf[s] in NULLZS.files}

    def _arm(data, pos, face):
        vp_ = ax.violinplot(data, positions=pos, widths=0.21, showextrema=False)
        for b_ in vp_["bodies"]:
            b_.set_facecolor(face); b_.set_alpha(0.95)
            b_.set_edgecolor("#7e8b94"); b_.set_linewidth(0.8)

    _arm(d_across, list(xs - DX), "#dfe6ea")
    _arm(d_along, list(xs), "#b3bec7")
    if d_surf:
        _arm([d_surf[s] for s in SEC if s in d_surf],
             [i + DX for i, s in enumerate(SEC) if s in d_surf], "#8d99a2")
    for i, s in enumerate(SEC):
        arms = [(d_across[i], -DX, n0.loc[s, "I_across"], n0.loc[s, "P_across"]),
                (d_along[i], 0.0, n0.loc[s, "I_along"], n0.loc[s, "P_along"])]
        if s in d_surf:
            arms.append((d_surf[s], DX, float(s0.loc[s, "I_surface"]),
                         float(s0.loc[s, "P_surface"])))
        for d_, off, obs, pv in arms:
            ax.hlines(np.percentile(d_, 95), i + off - 0.11, i + off + 0.11,
                      color="#5b6b76", lw=1.3, zorder=4)
            ax.plot([i + off], [obs], marker="o", ms=7.0, color=CC[s], mec="white",
                    mew=0.9, zorder=6)
            if not bare:
                up = obs > float(np.median(d_))
                ax.text(i + off, obs + (0.05 if up else -0.05),
                        f"{pv:.3f}".rstrip("0").rstrip("."), ha="center",
                        va="bottom" if up else "top", fontsize=ANNO_FS - 0.6,
                        color="#2b3640", zorder=7)
        if s not in d_surf and not bare:
            # no surrogate distribution exists for this axis on this section (see docstring)
            ax.text(i + DX, 0.02, "n.d.", ha="center", va="bottom",
                    fontsize=ANNO_FS - 0.6, color="#777", zorder=7)
    _hi_src = d_across + d_along + list(d_surf.values())
    ax.set_xticks(xs)
    ax.set_xticklabels(DSEC)
    ax.set_ylabel("$\\hat{I}$ (bit)")   # the cell is a condition and sits in the title
    if not bare:
        h = [Patch(facecolor="#dfe6ea", alpha=0.95, edgecolor="#7e8b94"),
             Patch(facecolor="#b3bec7", alpha=0.95, edgecolor="#7e8b94"),
             Patch(facecolor="#8d99a2", alpha=0.95, edgecolor="#7e8b94"),
             Line2D([0], [0], color="#5b6b76", lw=1.5),
             Line2D([0], [0], marker="o", ms=8, color="#5b6b76", mec="white", mew=1.0,
                    ls="none"),
             ]
        # draw count and the |rho| < 0.30 rejection rule are reported in the caption, not
        # in the legend
        ax.legend(handles=h, labels=["null, across the band (left)",
                                     "null, along the backbone (centre)",
                                     "null, surface distance (right)",
                                     "null 95th pct", "observed"],
                  loc="upper left", fontsize=ANNO_FS, ncol=3, handlelength=1.4,
                  handletextpad=0.5, columnspacing=1.1, borderpad=0.1,
                  bbox_to_anchor=(-0.015, -0.10))
        ax.text(0.995, 0.02, "numbers are P", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=ANNO_FS, color="#777")
    ax.set_xlim(-0.7, len(SEC) - 0.3)
    # the upper limit follows the drawn data: the STPI null reaches 1.199 at 1NMPP1-1, and
    # a fixed cut at 1.02 would truncate that violin, obscuring the null distribution it is
    # meant to show
    _hi = max(float(np.max(d)) for d in _hi_src)
    ax.set_ylim(-0.12, _hi + 0.12)


# ────────── panel e · is the cross-validation inflation particular to one coordinate? ────
# Main Fig 4e uses the STPI (across-band) coordinate, so that arm is not repeated here;
# this panel carries the three other coordinates. The gaston arm comes from
# analysis/fig4_tendon_scan.py run with the gaston target, which differs from the other
# three runs only in the target.
CVX = {"along": pd.read_csv(RESULTS / "fig4_tendon_cv_along.csv"),
       "surface": pd.read_csv(RESULTS / "fig4_tendon_cv.csv"),
       "gaston": pd.read_csv(RESULTS / "fig4_tendon_cv_gaston.csv")}

CVLBL = [("along", "along the backbone (ours)"),
         ("surface", "surface distance (ours)"),
         ("gaston", "GASTON isodepth, Chitra et al. (2025)")]

def _cv_grid(c):
    d = CVX[c].dropna(subset=["I_checker", "I_randlab_med"]).copy()
    d["inf"] = d.I_randlab_med - d.I_checker
    return (d.pivot_table(index="section", columns="depth", values="inf", aggfunc="median")
            .reindex(index=SEC, columns=DEPTHS))
def panel_e(ax, bare=False):
    """This panel extends the main Fig 4e comparison (cross-validation block permutation
    versus spatially contiguous blocks) to two further coordinates, to test whether the
    inflation of the random-CV estimate is specific to the across-band target or holds
    generally.

    Same device as main Fig 4e, so the two can be read side by side: a section by depth
    grid of medians, with the median over the five pseudospot sizes printed in every cell,
    one grid per coordinate. The STPI across-band coordinate is omitted here because it is
    the subject of main Fig 4e. Depth is the axis used on x in main Fig 4e, and it is the
    axis along which the inflation grows, which is why this panel keeps it: aggregating
    over depths would hide the trend reported in the main panel.

    All three grids share one diverging colour scale centred on zero, so colour is
    comparable across coordinates. Everything except the decoded target is held fixed
    between the random and spatial-block CV schemes (fold count, fold sizes, estimator,
    depth thinning, seeds); only the permutation changes which blocks are spatially
    contiguous."""
    GR = {c: _cv_grid(c) for c, _ in CVLBL}
    v = max(float(np.nanmax(np.abs(g.values))) for g in GR.values())
    ax.axis("off")
    W_, GAP = 0.300, 0.028
    for k, (c, lab) in enumerate(CVLBL):
        M = GR[c]
        sub = ax.inset_axes([k * (W_ + GAP), 0.06, W_, 0.84])
        sub.imshow(M.values, cmap="PuOr_r", vmin=-v, vmax=v, aspect="auto",
                   interpolation="nearest")
        sub.set_xticks(range(len(DEPTHS))); sub.set_yticks(range(len(SEC)))
        sub.set_xticklabels([str(d) for d in DEPTHS] if not bare else [], rotation=90)
        sub.set_yticklabels(DSEC if (k == 0 and not bare) else [])
        sub.tick_params(length=1.8, width=0.6)
        for sp_ in sub.spines.values():
            sp_.set_visible(True); sp_.set_linewidth(0.6)
        if not bare:
            for yy in range(M.shape[0]):
                for xx in range(M.shape[1]):
                    val = M.values[yy, xx]
                    if np.isfinite(val):
                        sub.text(xx, yy, f"{val:.2f}", ha="center", va="center",
                                 fontsize=ANNO_FS - 1.4,
                                 color="black" if abs(val) < 0.6 * v else "white")
            sub.set_title(lab, loc="center", pad=3.0, fontsize=ANNO_FS)
    if not bare:
        ax.text(1.5 * W_ + GAP, -0.10, "target depth (transcripts per pseudospot)",
                transform=ax.transAxes, ha="center", va="top", fontsize=ANNO_FS + 0.5)
        cax = ax.inset_axes([3 * (W_ + GAP) + 0.010, 0.14, 0.016, 0.66])
        cb = ax.figure.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(-v, v), cmap="PuOr_r"), cax=cax)
        cb.set_label(r"$\Delta\hat{I}$  permuted $-$ checkerboard (bit)", fontsize=ANNO_FS)
        cb.ax.tick_params(length=2.2, width=0.7, labelsize=ANNO_FS - 1)
        cb.outline.set_linewidth(0.7)
    ax.set_xlim(0, 3 * (W_ + GAP) + 0.075); ax.set_ylim(0, 1)


def strip_bare(ax, lab):
    """Bare tier: removes text and axis decorations while keeping data marks and reference
    lines. Inset axes and colourbars carry their own text and are cleared too."""
    for other in ax.figure.axes:
        if other is ax:
            continue
        other.set_xlabel(""); other.set_ylabel(""); other.set_title("")
        for t in list(other.texts):
            if t.get_text().strip():
                t.remove()
        other.axis("off")
    ax.set_title("", loc="left"); ax.set_xlabel(""); ax.set_ylabel("")
    for t in list(ax.texts):
        if t.get_text().strip():
            t.remove()
    lg = ax.get_legend()
    if lg is not None:
        lg.remove()
    ax.axis("off")



# letters as in the figure: A construction, B decoded field, C coordinate comparison,
# D autocorrelation-preserving null, E random-CV difference under two coordinates
PANELS = {"A": (panel_a, (10.4, 3.9)), "B": (panel_b, (10.8, 2.5)),
          "C": (panel_c, (9.2, 4.4)), "D": (panel_d, (5.2, 3.4)),
          "E": (panel_e, (11.6, 2.6))}
TITLES = {
    "A": "Geometric construction of the injured mouse tendon (STPI: across the band)",
    "B": "Coordinates predicted by gene expression (200 \u03bcm, depth 2048)",
    "C": "Recoverability against pseudospot size and depth under four coordinate constructions",
    "D": "Recoverability of three axes against an autocorrelation-preserving null\n"
         "(200 \u03bcm, depth 2048)",
    "E": "Difference in recoverability between spatial block and random CV under two coordinates"}


def _split_columns(path, out_stem, n, min_gap=40):
    """Cuts a row of n sections into n images at the widest white gaps."""
    im = Image.open(path).convert("RGB"); a = np.asarray(im)
    occ = (a.min(2) < 245).any(0)
    xs = np.where(occ)[0]
    gaps = [(xs[i + 1] - xs[i], xs[i], xs[i + 1]) for i in range(len(xs) - 1)
            if xs[i + 1] - xs[i] > min_gap]
    cuts = sorted(sorted(gaps, reverse=True)[:n - 1], key=lambda g: g[1])
    edges = [xs[0]] + [x for g in cuts for x in (g[1], g[2])] + [xs[-1]]
    for k in range(n):
        x0, x1 = edges[2 * k], edges[2 * k + 1] + 1
        sub = a[:, x0:x1]; rows = np.where((sub.min(2) < 245).any(1))[0]
        # the 1 mm scale bar under the first section is drawn directly in the figure, not cropped from this image
        br = np.where(np.diff(rows) > 60)[0]
        if len(br) and rows[-1] - rows[br[-1] + 1] < 40:
            rows = rows[:br[-1] + 1]
        sub = sub[rows[0]:rows[-1] + 1]
        cols = np.where((sub.min(2) < 245).any(0))[0]
        Image.fromarray(sub[:, cols[0]:cols[-1] + 1]).save(f"{out_stem}_{k + 1}.png")


def marks():
    d = marks_dir(FIG)
    for lab, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, True)
        strip_bare(ax, lab.lower())
        fig.savefig(d / f"{FIG}_{lab}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    # panel A is placed as one image per section (the scale bar is drawn directly in the figure)
    _split_columns(d / "FigS4_A.png", str(d / "FigS4_A_section"), 4)
    _a_keys(d)
    colourbar_png(d / "FigS4_B_colourbar.png", "viridis", 57, 899, ((0, 56), (0, 898)), 0)
    colourbar_png(d / "FigS4_E_colourbar.png", "PuOr_r", 68, 778, ((0, 67), (0, 777)), 0)


def _a_keys(d):
    """Draws the three legend marks for panel A (spot, backbone, coordinate), one image each."""
    from PIL import ImageDraw
    W, H = 136, 110
    im = Image.new("RGB", (W, H), "white"); g = ImageDraw.Draw(im)
    g.ellipse((51, 40, 91, 80), fill=(0xD7, 0xDC, 0xE0)); im.save(d / "FigS4_A_key_spot.png")
    im = Image.new("RGB", (W, H), "white"); g = ImageDraw.Draw(im)
    g.rectangle((15, 53, 111, 67), fill=(0, 0, 0)); im.save(d / "FigS4_A_key_backbone.png")
    im = Image.new("RGB", (W, H), "white"); g = ImageDraw.Draw(im)
    g.rectangle((24, 56, 113, 63), fill=(0xC0, 0x39, 0x2B))
    g.polygon([(48, 37), (48, 82), (86, 59.5)], fill=(0xC0, 0x39, 0x2B))
    im.save(d / "FigS4_A_key_coordinate.png")


def complete_text(lab, ax):
    """Sets the final text labels for the panel; the GASTON comparison columns in panels C
    and E are plotted but kept hidden in the assembled figure."""
    lg = ax.get_legend()
    if lab == "A" and lg is not None:
        for tx, s in zip(lg.get_texts(), ("Spot (position)", "Backbone", "Coordinate")):
            tx.set_text(s)
    if lab == "B":
        for tx in list(ax.texts):
            if tx.get_text().startswith("$\\rho$"):
                tx.remove()
        for c in ax.child_axes:
            if c.get_ylabel():
                c.set_ylabel("Decoded value")
    if lab == "C":
        names = ("STPI: across the band", "STPI: along the backbone", "STPI: surface distance")
        tops = [c for c in ax.child_axes if c.get_title()]
        for c, s in zip(tops, names):
            c.set_title(s)
        for c in ax.child_axes:
            if c.get_xlabel():
                c.set_xlabel("Pseudospot size (\u03bcm)" if "size" in c.get_xlabel() else "Depth")
            c.set_ylabel("")
        for c in ax.child_axes[6:8]:
            c.set_visible(False)
    if lab == "E":
        names = ("STPI: along the backbone", "STPI: surface distance")
        for c, s in zip(ax.child_axes[:2], names):
            c.set_title(s)
        ax.child_axes[2].set_visible(False)
        ax.child_axes[3].set_ylabel("")
        for tx in ax.texts:
            tx.set_text("Depth")


def full():
    d = OUT / FIG; d.mkdir(parents=True, exist_ok=True)
    for lab, (fn, size) in PANELS.items():
        fig, ax = plt.subplots(figsize=size)
        fn(ax, False)
        ax.set_title(TITLES[lab], loc="left")
        complete_text(lab, ax)
        for ext in (".png", ".pdf"):
            fig.savefig(d / f"{FIG}_{lab}{ext}", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    marks()
    full()
