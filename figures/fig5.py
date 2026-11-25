#!/usr/bin/env python3
"""Figure 5. STPI applied to 18 published sections from four tissues.

  A  Flow of published sections from four tissues through two guards (STPI)
  B  Expression-independent and dependent coordinates
  C  Recoverability under spatial block and random CV
  D  The random CV fraction of inflation against the recoverability and number of spots
  E  Recoverability under buffer space of spatial block CV

Inputs (results/)
  fieldwide_audit.csv, fig5_buffered_cv_all.csv, fig5_pred_maps.npz
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from PIL import Image

from _common import RESULTS, FONT, marks_dir, OUT

FIG = "Fig5"
BASE_FS, ANNO_FS, TICK_FS = 8, 7, 6
plt.rcParams.update({
    "font.family": FONT, "pdf.fonttype": 42, "font.size": BASE_FS,
    "axes.titlesize": BASE_FS, "axes.labelsize": BASE_FS,
    "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.fontsize": ANNO_FS, "figure.dpi": 110,
})
FAIL1_C = "#f5f012"     # sections whose spatial block reading is not positive (guard 1)
FAIL2_C = "#f20613"     # sections that do not stay positive under the widened gap (guard 2)

GREY = "#8a8a8a"; FAINT = "#d9d9d9"

# Tissue palette. Deliberately not the developmental-stage or tendon-section palettes.
# The two retinas share a hue family because they are the same organ in two species.
TIS = ["kidney", "intestine", "cerebellum", "retina", "finch_retina"]

TC = {"kidney": "#2d7f5e", "intestine": "#d1823a", "cerebellum": "#6b3a75",
      "retina": "#1f5f82", "finch_retina": "#6aaed6"}

# first-author-and-year credit for every external dataset, same convention as the other
# figures: nothing here is this study's own data (results/fig5_dataset_provenance.csv)
CRED = {'kidney': 'Yasui et al. (2024)', 'intestine': 'Mayassi et al. (2024)', 'retina': 'Schumann et al. (2025)', 'cerebellum': 'Ji et al. (2026)', 'finch_retina': 'Damsgaard et al. (2026)'}

# species are printed because four of the five datasets are mouse and one is zebra finch
# (results/fig5_sample_metadata.csv, organism field of every GSM)
TDISP = {"kidney": "Mouse kidney", "intestine": "Mouse intestine",
         "cerebellum": "Mouse cerebellum", "retina": "Mouse retina",
         "finch_retina": "Finch retina"}

CRED = {"kidney": "Yasui et al. (2024)", "intestine": "Mayassi et al. (2024)",
        "cerebellum": "Ji et al. (2026)", "retina": "Schumann et al. (2025)",
        "finch_retina": "Damsgaard et al. (2026)"}

A = pd.read_csv(RESULTS / "fieldwide_audit.csv")

A["tissue"] = pd.Categorical(A.tissue, TIS, ordered=True)

A = A.sort_values(["tissue", "sample"]).reset_index(drop=True)

NT = {t: int((A.tissue == t).sum()) for t in TIS}

BUF = pd.read_csv(RESULTS / "fig5_buffered_cv_all.csv")

MAPS = np.load(RESULTS / "fig5_pred_maps.npz")

def mode_of(r):
    if r.I_random <= 1e-6: return "coordinate-limited"
    return "collapse" if r.I_checker <= 0 else "survives"

A["mode"] = A.apply(mode_of, axis=1)

BUF4 = BUF[BUF.buffer_pitch == BUF.buffer_pitch.max()].set_index("sample").I_bit

A["I_buf4"] = A["sample"].map(BUF4)

A["stage2"] = np.where(A["mode"] != "survives", "-",
                       np.where(A.I_buf4 > 0, "holds under the buffer",
                                "falls under the buffer"))

SHARED_LEGEND = False      # set while composing the assembled figure; panels then drop their keys

def rank01(v):
    o = np.argsort(np.argsort(np.asarray(v, float)))
    return o / max(len(o) - 1, 1)

def tissue_handles(bare=False, with_n=True, with_cred=False):
    h = [Line2D([0], [0], marker="o", ls="none", ms=5, color=TC[t],
                mec="white", mew=0.5) for t in TIS]
    if bare:
        return h, [""] * len(TIS)
    lab = [TDISP[t] + (f"  (n = {NT[t]})" if with_n else "")
           + (f"\n{CRED[t]}" if with_cred else "") for t in TIS]
    return h, lab

def tissue_legend(ax, loc="lower right", bare=False, **kw):
    """Per-panel key. Suppressed when the assembled figure supplies one shared legend."""
    if SHARED_LEGEND and not bare:
        return
    h, lab = tissue_handles(bare=bare, **kw)
    ax.legend(handles=h, labels=lab, frameon=False, loc=loc, handletextpad=0.4,
              labelspacing=0.26, borderpad=0.1, fontsize=ANNO_FS)

# Visium is a 100 um lattice, but these sections carry coordinates in ARRAY units with a
# different scale each, so the pitch is measured per section and 1 mm = 10 pitches.
MM_IN_PITCHES = 10.0

def spot_field(Q, vals, pitch, up=8.0, reach=0.62, soft_frac=0.70):
    """Render spots as a FILLED field with a smooth outline.

    Drawing one marker per spot leaves the background visible between them, so the tissue
    reads as a dotted texture. Here every grid node takes the value of its NEAREST spot -
    no smoothing of the data, no value is invented - and the tissue boundary is carried by
    an alpha channel rather than by a hard mask: a hard distance cut would quantise the
    outline into steps of one grid cell plus the arc of each spot's disc, giving a
    serrated appearance. The occupancy indicator is blurred with a kernel of `soft_um`
    MICROMETRES - comparable to the pitch itself, because the serration has the period of
    the lattice, so a kernel much smaller than the pitch cannot remove it - and then
    thresholded with a narrow ramp, which leaves a smooth antialiased outline.

    Grid step is pitch / up, finer than print resolution. The pitch is MEASURED per
    section (median nearest-neighbour spacing) because these sections' coordinates are in
    array units, not microns, and the three differ: 131, 289 and 227 units. A single
    hard-coded pitch would leave two of the three maps entirely blank: every grid node
    would fall outside the occupancy radius.

    What the kernel costs, stated because it is a rendering choice and not data: interior
    gaps NARROWER than about soft_um close up (two single-spot gaps in Control-1 do), and
    an isolated island of one or two spots can disappear. Gaps and fragments three or more
    spots across survive. Nothing about the decoded values changes - this affects only
    where the drawn tissue ends; every coloured pixel still carries the coordinate value of
    its nearest spot. For an exact boundary, render at soft_um -> 0 (serrated) or use the
    bare tier with the raw lattice.
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter
    step = pitch / up
    pad = 2.0 * pitch
    xs = np.arange(Q[:, 0].min() - pad, Q[:, 0].max() + pad + step, step)
    ys = np.arange(Q[:, 1].min() - pad, Q[:, 1].max() + pad + step, step)
    Xg, Yg = np.meshgrid(xs, ys)
    d, idx = cKDTree(Q).query(np.c_[Xg.ravel(), Yg.ravel()])
    Z = np.asarray(vals, float)[idx].reshape(Xg.shape)
    occ = (d.reshape(Xg.shape) <= reach * pitch).astype(float)
    occ = gaussian_filter(occ, soft_frac * pitch / step)
    A = np.clip((occ - 0.50) / 0.10, 0.0, 1.0)      # narrow ramp = smooth, crisp outline
    return xs, ys, Z, A



# ────────── panel a · the audit as a flow (the spine of the figure) ─────────────────────
ORDER_MODE = ["survives", "collapse", "coordinate-limited"]

MLAB = {"survives": "survives\nspatial block CV",
        "collapse": "collapses",
        "coordinate-limited": "coordinate carries\nno signal"}

ORDER_ST = ["holds under the buffer", "falls under the buffer"]

SLAB = {"holds under the buffer": "still reads\nabove zero",
        "falls under the buffer": "falls to or\nbelow zero"}

GATE_C = {"survives": "#2f6f8f", "collapse": FAIL1_C, "coordinate-limited": "#bdbdbd"}

def _lanes(groups, gap=1.5):
    """y centre of every lane, grouped with a gap; returns {key: [y, ...]} top-down."""
    out, y = {}, 0.0
    for k, items in groups:
        ys = [y + i for i in range(len(items))]
        out[k] = (items, ys)
        y += len(items) + gap
    total = y - gap
    return out, total

def _ribbon(ax, x0, x1, y0, y1, color, h=0.86, alpha=0.55):
    t = np.linspace(0, 1, 60)
    sm = t * t * (3 - 2 * t)                       # smoothstep, no overshoot
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * sm
    ax.fill_between(xs, ys - h / 2, ys + h / 2, color=color, alpha=alpha, lw=0,
                    zorder=2)

def panel_a(ax, bare=False):
    """Where the 18 published sections end up. Lane height is one section; colour is the
    tissue, so the composition of every outcome is visible rather than stated.

    Left column: the sections, grouped by tissue. Middle: the outcome under spatial block
    (4x4 checkerboard) scheme - guard 1. Right: of those that survived guard 1, the ones
    whose reading is still above zero once the training-test gap is widened to four spot
    pitches - guard 2, which deletes about two thirds of the training spots. The four
    sections that did not survive guard 1 are not carried into guard 2 and their ribbons
    stop in the middle column.

    This panel is the figure's spine: c, d and e are the three steps drawn here, measured.
    """
    ax.axis("off")
    src_groups = [(t, list(A[A.tissue == t].index)) for t in TIS]
    L_src, H = _lanes(src_groups)
    # the coordinate-limited class is empty once the boundary guard is in place;
    # empty groups are dropped so no zero-height block or label is drawn
    mid_groups = [(m, list(A[A["mode"] == m].index)) for m in ORDER_MODE
                  if (A["mode"] == m).any()]
    L_mid, Hm = _lanes(mid_groups, gap=2.2)
    sv = A[A["mode"] == "survives"]
    dst_groups = [(s, list(sv[sv.stage2 == s].index)) for s in ORDER_ST]
    L_dst, Hd = _lanes(dst_groups, gap=2.2)
    H = max(H, Hm, Hd)
    off = {k: (H - tot) / 2 for k, tot in
           (("src", _lanes(src_groups)[1]), ("mid", Hm), ("dst", Hd))}
    ypos = lambda L, key, i, o: L[key][1][i] + off[o]

    XS, XM, XD, W = 0.085, 0.470, 0.855, 0.038
    ymap_src, ymap_mid, ymap_dst = {}, {}, {}
    for t in TIS:
        for i, ix in enumerate(L_src[t][0]): ymap_src[ix] = ypos(L_src, t, i, "src")
    for m, _ in mid_groups:
        for i, ix in enumerate(L_mid[m][0]): ymap_mid[ix] = ypos(L_mid, m, i, "mid")
    for s_ in ORDER_ST:
        for i, ix in enumerate(L_dst[s_][0]): ymap_dst[ix] = ypos(L_dst, s_, i, "dst")

    for ix, r in A.iterrows():
        c = TC[r.tissue]
        _ribbon(ax, XS + W, XM, ymap_src[ix], ymap_mid[ix], c)
        if ix in ymap_dst:
            _ribbon(ax, XM + W, XD, ymap_mid[ix], ymap_dst[ix], c)
    # the three column blocks
    POS = dict(XS=XS, XM=XM, XD=XD, W=W, src={}, mid={}, dst={})   # label anchors
    for t in TIS:
        ys = [ypos(L_src, t, i, "src") for i in range(len(L_src[t][0]))]
        POS["src"][t] = (min(ys) - 0.43, max(ys) + 0.43)
        ax.add_patch(Rectangle((XS, min(ys) - 0.43), W, max(ys) - min(ys) + 0.86,
                               color=TC[t], lw=0, zorder=3))
        if not bare:
            # ONE line: a two-line label is taller than a 2-3 lane tissue block and would
            # collide with its neighbours in the assembled figure
            src = CRED[t].replace("(", "").replace(")", "")
            ax.text(XS - 0.012, np.mean(ys), f"{TDISP[t]}  {len(ys)}  ({src})",
                    ha="right", va="center", fontsize=ANNO_FS, color=TC[t])
    for m in [k for k, _ in mid_groups]:
        ys = [ypos(L_mid, m, i, "mid") for i in range(len(L_mid[m][0]))]
        POS["mid"][m] = (min(ys) - 0.43, max(ys) + 0.43)
        ax.add_patch(Rectangle((XM, min(ys) - 0.43), W, max(ys) - min(ys) + 0.86,
                               color=GATE_C[m], lw=0, zorder=3))
        if not bare:
            if m == "survives":        # its ribbons leave rightward: label above instead
                # short form: the guard header above already names the scheme
                ax.text(XM + W / 2, min(ys) - 0.15, f"survives   {len(ys)}",
                        ha="center", va="bottom", fontsize=ANNO_FS)
            else:
                ax.text(XM + W + 0.012, np.mean(ys), f"{MLAB[m]}   {len(ys)}",
                        ha="left", va="center", fontsize=ANNO_FS, linespacing=1.3)
    for s_ in ORDER_ST:
        ys = [ypos(L_dst, s_, i, "dst") for i in range(len(L_dst[s_][0]))]
        POS["dst"][s_] = (min(ys) - 0.43, max(ys) + 0.43)
        col = "#2f6f8f" if s_ == ORDER_ST[0] else FAIL2_C
        ax.add_patch(Rectangle((XD, min(ys) - 0.43), W, max(ys) - min(ys) + 0.86,
                               color=col, lw=0, zorder=3))
        if not bare:
            ax.text(XD + W + 0.012, np.mean(ys), f"{SLAB[s_]}   {len(ys)}", ha="left",
                    va="center", fontsize=ANNO_FS, linespacing=1.3)
    if not bare:
        # the y axis is inverted (lane 0 at the top), so the headers sit at NEGATIVE y
        ax.text((XS + W + XM) / 2, -0.95, "guard 1\nspatial block cross-validation",
                ha="center", va="bottom", fontsize=ANNO_FS, color="#444",
                linespacing=1.3)
        ax.text((XM + W + XD) / 2, -0.95, "guard 2\ngap widened to 4 spot pitches",
                ha="center", va="bottom", fontsize=ANNO_FS, color="#444",
                linespacing=1.3)

    ax.set_xlim(-0.02, 1.085); ax.set_ylim(H + 1.0, -3.2)
    return POS
# ────────── panel b · decoded fields of five example sections ───────────────────────────
EX_MAPS = [("kidney", "kidney", "survives", 0.792),
           ("cerebellum", "cerebellum", "survives, weakly", 0.459),
           ("finch_retina", "finch_retina", "survives, weakly", 0.470),
           ("intestine", "intestine", "collapses", 0.144),
           ("retina", "retina", "collapses", 0.075)]

def panel_b(ax, bare=False, subset=None, rule=True):
    """Six maps tiled in ONE row across the full figure width: three sections, each shown
    twice side by side - left the geometric coordinate, right the same coordinate decoded
    from expression under spatial block CV. Colour is the rank-normalised value, so the
    pair shares one scale and the only question is whether the pattern survives decoding.

    Tiled rather than stacked in two rows because a full-width banner gives each map about
    twice the area; with two rows the maps are height-limited and the panel reads as small.

    Five sections, chosen to cover the outcome classes: a representative survivor (kidney,
    8 of 8 survive), BOTH collapse samples (intestine and retina - the only two in the
    panel whose spatial-block-CV reading is at or below zero), and two weak survivors. The
    second weak survivor is the finch, included although its class is already represented:
    it is the only non-mammal in the panel, and its tissue keeps the smallest share of its
    random-CV reading of any tissue here (median 27%). The retina is a strip-cut section with
    575 spots - that is the section, not the rendering, which is why the spot count is
    printed beside each pair.

    The cerebellum pair illustrates a boundary-rule degeneracy that can make a coordinate
    appear to carry no signal (see geom_interiority in analysis/fig5_audit.py) even though the
    tissue's true coordinate decodes, weakly, once that degeneracy is guarded against. The
    pair shows what weak-but-real decoding looks like next to the collapses.

    Each pair carries its own 1 mm bar: the four sections differ in extent by more than a
    factor of two.

    Rendered with spot_field (the same function used in Figure 4a), not one marker per
    spot: with markers the white background shows between the spots and the tissue reads
    as a pale dotted texture, which washes the panel out. Every grid node takes the value
    of its NEAREST spot - no smoothing of the data - and the outline is carried by an alpha
    channel.
    """
    ax.axis("off")
    # subset: draw one tissue only. The five pairs read as a cluttered strip at reading
    # size, so each tissue is also exported on its own (Fig5_b_<tissue>_<tier>); the
    # assembled figure keeps the five-pair row. rule=False drops the classification
    # footnote on those single-tissue exports, since it is a statement about the whole
    # panel and does not fit a one-pair width; it is then carried by the caption.
    ex = EX_MAPS if subset is None else [e for e in EX_MAPS if e[0] in set(subset)]
    x_left, x_right = 0.022, 0.918
    gap_pair, gap_in = 0.022, 0.008
    wpair = (x_right - x_left - (len(ex) - 1) * gap_pair) / len(ex)
    wmap = (wpair - gap_in) / 2
    y0, y1 = 0.215, 0.855
    GEO = dict(y0=y0, y1=y1, pairs=[])           # label anchors for the per-tissue export tier
    for k, (key, tis, verdict, rho) in enumerate(ex):
        xy = np.asarray(MAPS[f"{key}_xy"], float)
        Q = np.c_[xy[:, 0], -xy[:, 1]]
        span = max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1]))
        pitch = float(np.median(cKDTree(Q).query(Q, k=2)[0][:, 1]))
        px0 = x_left + k * (wpair + gap_pair)
        for c, vals in enumerate((MAPS[f"{key}_true"], MAPS[f"{key}_pred"])):
            gx, gy, Z, Al = spot_field(Q, rank01(vals), pitch)
            sub = ax.inset_axes([px0 + c * (wmap + gap_in), y0, wmap, y1 - y0])
            sub.imshow(Z, cmap="viridis", vmin=0, vmax=1, origin="lower", alpha=Al,
                       extent=[gx[0], gx[-1], gy[0], gy[-1]],
                       interpolation="nearest", rasterized=True)
            sub.set_aspect("equal"); sub.axis("off"); sub.margins(0.02)
            if not bare:
                sub.set_title("coordinate\n(geometry only)" if c == 0
                              else "decoded from\nexpression",
                              fontsize=TICK_FS, color=GREY, pad=2.0, linespacing=1.3)
        frac = MM_IN_PITCHES * pitch / span   # 1 mm
        GEO["pairs"].append(dict(tis=tis, verdict=verdict, x0=px0, wpair=wpair, wmap=wmap,
                                 gap_in=gap_in))
        # the bar sits at y=0.028, with its label above it, so it does not collide with
        # the rho / spot-count block above it
        # 1 mm measured on the drawn map itself: both maps are fitted to their slots with an
        # equal aspect, so the length follows from the map's own data-to-screen scale rather
        # than from the span-based estimate above, which can be off by up to 7%. Right-aligned
        # under the right map.
        sub.apply_aspect()
        mm_ax = ((sub.transData.transform((MM_IN_PITCHES * pitch, 0))[0]
                  - sub.transData.transform((0, 0))[0]) / ax.bbox.width)
        right = (sub.bbox.x1 - ax.bbox.x0) / ax.bbox.width
        GEO["pairs"][-1].update(bar_x0=right - mm_ax, bar_x1=right)
        bar = ax.inset_axes([right - mm_ax, 0.028, mm_ax, 0.012])
        bar.axis("off"); bar.plot([0, 1], [0.5, 0.5], color="black", lw=1.8,
                                  solid_capstyle="butt")
        bar.set_xlim(0, 1); bar.set_ylim(0, 1)
        if not bare:
            ax.text(px0 + wpair / 2, 0.998, f"{TDISP[tis]} \u00b7 {verdict}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=BASE_FS, color=TC[tis])
            ax.text(px0 + wpair / 2, 0.185, f"$\\rho$(coordinate, decoded) = {rho:.2f}\n{len(xy):,} spots in this section",
                    transform=ax.transAxes, ha="center", va="top", fontsize=ANNO_FS)
            # label sits above the bar, which spans y 0.028-0.040
            ax.text(right - mm_ax / 2, 0.044, "1 mm", transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=TICK_FS, color=GREY)
    if not bare and rule:
        # The column headers name an outcome, so the rule behind it is stated on the panel.
        # Thresholds are on I, not on rho; the rho/I correspondence measured on all 18
        # sections is in results/fig5_rho_vs_I.csv (Spearman 0.96, classes separated).
        ax.text(0.0, -0.045,
                "classified on $\\hat{I}$ under spatial block CV:  survives $\\hat{I}$ > 0"
                "  ·  collapses $\\hat{I}$ $\\leq$ 0  ·  coordinate-limited "
                "$\\hat{I}$(random CV) $\\leq$ 0\n"
                "$\\rho$ is descriptive, not the criterion - across all 18 sections it tracks "
                "$\\hat{I}$ (Spearman 0.96) and the two classes do not overlap: "
                "$\\hat{I}$ $\\leq$ 0 at $\\rho$ $\\leq$ 0.14, $\\hat{I}$ > 0 at $\\rho$ "
                "$\\geq$ 0.23",
                transform=ax.transAxes, ha="left", va="top", fontsize=TICK_FS,
                color="#444", linespacing=1.45)
    cax = ax.inset_axes([0.940, y0, 0.012, y1 - y0])
    cb = ax.figure.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, 1), cmap="viridis"),
                            cax=cax)
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(direction="out", length=2.6, width=0.8, labelsize=TICK_FS)
    cb.outline.set_linewidth(0.8)
    if not bare:
        cb.set_label("coordinate value (rank)", fontsize=ANNO_FS)
    GEO["cbar"] = cb
    return GEO


# ────────── panel c · what survives spatial block cross-validation, section by section ────
# Wording: the two schemes are named after the methods used, not labelled "honest" and
# "naive", since calling one scheme "honest" editorialises and implies the scheme is novel
# here. Spatial block CV and its buffered variant are established methods (Roberts et al.
# 2017, Ecography; the same collapse was shown for forest-biomass models in Ploton et al.
# 2020, Nat Commun). What is novel is the audit: applying this method across 18 published
# sections, with an expression-independent geometric coordinate as the target and a
# bit-valued readout.
def panel_c(ax, bare=False):
    """One bar per section, grouped by tissue. Bar height is the reading a random 5-fold CV
    reports; the SOLID part is what spatial block CV still reports
    and the PALE part is what it removes. Two sections have no solid part and a bar that
    goes below zero - the spatial-block-CV reading is negative there, so everything reported
    was artifact.

    The n.d. branch (an open circle at the baseline, for a section whose coordinate is
    constant so that no share is defined) handles the case where the boundary-rule
    degeneracy collapses a coordinate; with the guard in place, no section in this panel
    falls into that branch.

    The panel combines the absolute reading and the removed share in one bar, read two
    ways, rather than splitting them into two panels: they are the same quantity seen from
    two angles, and separating them would make the reader hold one panel in mind while
    looking at the other.
    """
    x, xt, xl = 0.0, [], []
    ax.axhline(0, lw=0.9, color="#444", zorder=3)
    for t in TIS:
        g = A[A.tissue == t].sort_values("I_random", ascending=False)
        x0 = x
        for _, r in g.iterrows():
            c = TC[t]
            if r.I_random <= 1e-6:                       # coordinate carries no signal
                ax.plot(x, 0, "o", ms=4.5, mfc="white", mec=c, mew=1.3, zorder=4)
                if not bare:
                    ax.text(x, -0.055, "n.d.", ha="center", va="top",
                            fontsize=TICK_FS, color=GREY)
            else:
                ax.bar(x, r.I_random, width=0.78, color=c, alpha=0.28, lw=0, zorder=2)
                # a negative spatial reading is drawn as a solid bar extending below zero;
                # no extra marker is used
                ax.bar(x, r.I_checker, width=0.78, color=c, lw=0, zorder=2)
            x += 1
        xt.append((x0 + x - 1) / 2); xl.append(TDISP[t])
        x += 1.1
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    for lb, t in zip(ax.get_xticklabels(), TIS): lb.set_color(TC[t])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-1.0, x - 0.6); ax.set_ylim(-0.22, 0.95)
    # the two CV schemes define what "kept" and "removed" mean, so that is stated once, on
    # the y-axis label: an in-axes note would overlap the legend, and an x-axis label would
    # overlap this panel's cell in the assembled figure
    ax.set_ylabel("$\\hat{I}$ (bit)\nrandom 5-fold vs spatial block CV (4\u00d74)")
    ax.spines["bottom"].set_visible(False)
    if not bare:
        h = [Patch(facecolor="#6b6b6b"), Patch(facecolor="#6b6b6b", alpha=0.28)]
        ax.legend(handles=h, labels=["Spatial block CV", "Excess of random CV"],
                  frameon=False, loc="upper right", fontsize=ANNO_FS,
                  handlelength=1.2, handletextpad=0.5, labelspacing=0.3, borderpad=0.0)


# ────────── panel d · inflation against signal strength, and against section size ───────
def panel_d(ax, bare=False):
    """Two candidate explanations for the removed share, on one shared y axis, each half
    labelled with the explanation it tests, with the verdict printed under it.

    The y axis is a ratio (one abstraction above the reading itself), and the right half's
    x axis (the section's spot count) is a control: without it, the left half's correlation
    is equally consistent with "the weak sections are simply the small ones". The question
    each half asks is written above it and the answer below it, so the panel states its own
    logic.

    Sections whose coordinate is constant would have no removed share; after the boundary
    guard there are none, so all 18 sections appear in both halves and in both
    correlations.
    """
    ax.axis("off")
    dd = A[A.frac_artifact.notna()]
    rho_sig = spearmanr(dd.I_random, dd.frac_artifact).correlation
    rho_n = spearmanr(dd.n, dd.frac_artifact).correlation
    left = ax.inset_axes([0.0, 0.0, 0.46, 0.80])
    right = ax.inset_axes([0.54, 0.0, 0.46, 0.80], sharey=left)
    # the axis label omits the fold count; it is a fixed condition of the audit, documented
    # in this docstring and in the caption
    spec = [(left, dd.I_random, "$\\hat{I}$ (bit), random CV",
             [0, 0.2, 0.4, 0.6, 0.8], (-0.05, 0.92),
             "is the weak signal to blame?", f"$\\rho$ = {rho_sig:.2f}", True),
            (right, dd.n, "spots in the section (its sample size)",
             [0, 1000, 2000, 3000, 4000], (0, 4100),
             "or just a small section?", f"$\\rho$ = {rho_n:.2f}", False)]
    for sub, xv, xlab, ticks, lim, question, verdict, is_left in spec:
        # 1.0 = the whole random-CV reading is inflation (drawn as a dashed line in the figure)
        sub.axhline(1.0, lw=0.9, color="#444", zorder=1, ls=(0, (4, 2.5)))
        for (_, r), v in zip(dd.iterrows(), xv):
            sub.plot(v, r.frac_artifact, "o", ms=6.2, color=TC[r.tissue],
                     mec="white", mew=0.6, zorder=3)
        sub.set_xlim(*lim); sub.set_xticks(ticks)
        # the upper limit follows the data: a fixed 2.1 would clip GSM7916286_078_A1,
        # whose value is 2.34
        sub.set_ylim(-0.05, max(2.1, float(dd.frac_artifact.max()) + 0.1))
        if not bare:
            sub.set_xlabel(xlab, fontsize=ANNO_FS)
            sub.set_title(question, loc="center", fontsize=ANNO_FS, color="#333", pad=3.0)
            # the verdict text sits inside the axes, upper right, so it does not overlap
            # the x-axis label below it (which would collide with the figure-level tissue
            # key) or the highest data point (1.93) at upper left
            sub.text(0.97, 0.95, verdict, transform=sub.transAxes, ha="right", va="top",
                     fontsize=BASE_FS)
    right.set_xticklabels(["0", "1k", "2k", "3k", "4k"])
    right.tick_params(labelleft=False)
    left.set_yticks(np.arange(0, float(dd.frac_artifact.max()) + 0.1, 0.5))
    if not bare:
        # same quantity as FigS5 d, so the same name: "removed share". The definition is
        # printed here (the main-figure panel is tall enough for it) and carried by the
        # caption for the supplementary strip.
        # the panel reports only the rho values; the qualitative judgement they imply
        # belongs in the caption and the text
        left.set_ylabel("Share of the random-CV reading\nthat is inflation")
# ────────── panel e · the buffered cross-validation ─────────────────────────────────────
def panel_e(ax, bare=False):
    """Widening the gap between training and test blocks, in units of the spot pitch. A
    real gradient decays slowly; a reading held up by local autocorrelation crosses zero
    fast. Thin lines are sections, thick lines tissue medians; colour is read from the shared
    tissue key (a standalone export of this panel carries its own).
    """
    x = sorted(BUF.buffer_pitch.unique())
    for (t, _), q in BUF.groupby(["tissue", "sample"]):
        q = q.sort_values("buffer_pitch")
        ax.plot(q.buffer_pitch, q.I_bit, lw=0.7, color=TC[t], alpha=0.35, zorder=2)
    ends = {}
    for t in TIS:
        q = BUF[BUF.tissue == t].groupby("buffer_pitch").I_bit.median()
        ax.plot(np.array(q.index, float), q.values, lw=2.2, color=TC[t], zorder=4,
                solid_capstyle="round")
        ends[t] = float(q.loc[max(x)])
    ax.axhline(0, lw=0.9, color="#444", zorder=1)
    drop = BUF.groupby("buffer_pitch").frac_train_dropped.median()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n{drop[k]:.0%}" for k in x])
    ax.set_xlim(-0.18, max(x) + 0.22)
    # the axis label is kept short; the second tick row carries the percentages
    ax.set_xlabel("training\u2013test gap (spot pitches)\n"
                  "% = training spots dropped")
    ax.set_ylabel("$\\hat{I}$ (bit)\nspatial block CV")
    # End-of-line labels are omitted: the three tissue medians converge within 0.02 bit at the
    # last pitch, so an automatic label-placement pass would push labels away from their own
    # lines and move the lowest one off the axes. Colour follows the figure's shared tissue
    # key; a standalone export carries its own key instead.
    if not bare and not SHARED_LEGEND:
        h = [Line2D([0], [0], color=TC[t], lw=2.2) for t in TIS]
        # The key sits outside the axes, top right: inside, it covered the thin kidney
        # lines (upper right) or the crossing bundle (lower left). The figure is widened
        # to make room for the key rather than shrinking the data region.
        # line handles (not dots) so thickness reads as the thin/thick convention, with
        # the section counts the shared key carries
        lab_n = [TDISP[t] + f"  (n = {NT[t]})" for t in TIS]
        ax.legend(handles=h, labels=lab_n, frameon=False,
                  loc="upper left", bbox_to_anchor=(1.015, 1.0), borderaxespad=0.0,
                  fontsize=ANNO_FS, handlelength=1.6, handletextpad=0.5,
                  labelspacing=0.32, borderpad=0.0)
    if not bare:
        ax.text(0.02, 0.03, "thin: one section   ·   thick: tissue median",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=TICK_FS, color=GREY)


def strip_bare(ax):
    for other in ax.figure.axes:
        other.set_xlabel(""); other.set_ylabel(""); other.set_title("")
        for t in list(other.texts):
            if t.get_text().strip(): t.remove()
        lg = other.get_legend()
        if lg is not None:
            for txt in lg.get_texts(): txt.set_text("")
        other.axis("off")




PANELS = {"A": (panel_a, (9.4, 3.0)), "B": (panel_b, (13.0, 3.1)),
          "C": (panel_c, (5.4, 3.2)), "D": (panel_d, (5.2, 3.5)),
          "E": (panel_e, (5.9, 3.2))}
TITLES = {"A": "Flow of published sections from four tissues through two guards (STPI)",
          "B": "Expression-independent and dependent coordinates",
          "C": "Recoverability under spatial block and random CV",
          "D": "The random CV fraction of inflation against the recoverability and number of spots",
          "E": "Recoverability under buffer space of spatial block CV"}


def _tight(path, pad=28):
    im = Image.open(path).convert("RGB"); a = np.asarray(im)
    ys, xs = np.where(a.min(2) < 245)
    im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
             min(xs.max() + pad + 1, im.size[0]), min(ys.max() + pad + 1, im.size[1]))).save(path)


def strip_all(ax):
    """No text, axes, ticks or frame on the panel and on its inset axes."""
    for a in [ax] + list(ax.child_axes):
        a.set_xlabel(""); a.set_ylabel(""); a.set_title("")
        a.set_xticks([]); a.set_yticks([]); a.minorticks_off()
        a.tick_params(which="both", length=0, labelleft=False, labelbottom=False)
        for s in a.spines.values():
            s.set_visible(False)
        if a.get_legend() is not None:
            a.get_legend().remove()
        for t in list(a.texts):
            t.remove()


def marks():
    global SHARED_LEGEND
    SHARED_LEGEND = False
    d = marks_dir(FIG)
    for lab in ("A", "B", "C", "D"):
        fn, size = PANELS[lab]
        fig, ax = plt.subplots(figsize=size)
        fn(ax, True)
        strip_bare(ax)
        fig.savefig(d / f"{FIG}_{lab}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    _tight(d / "Fig5_A.png")
    fig = plt.figure(figsize=(6.6, 3.4))                 # E: marks only, transparent
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    panel_e(ax, bare=True)
    strip_all(ax)
    for t in list(fig.texts):
        t.remove()
    fig.savefig(d / "Fig5_E.png", dpi=600, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)


# ────────── complete panels in the wording and proportions of the assembled figure ──────────
# A, B and E are drawn from the marks tier and lettered as in the assembled figure. The layout
# factor k is the width the panel image occupies on the 129 cm layout canvas divided by the
# width at which the marks are rendered; text is set at 42/k pt (titles 44/k), bold where the
# figure layout is bold.
from _common import Layout
K_LAYOUT = {"A": 6.96, "B": 5.04, "C": 3.50, "D": 3.31, "E": 2.12}
KEY_LAYOUT = ["cerebellum", "retina", "finch_retina", "intestine", "kidney"]   # figure layout order
VERDICT = {"survives": "survive", "survives, weakly": "survive", "collapses": "collapse"}
KEY_TOP = ["cerebellum", "retina", "finch_retina"]         # Fig 5e key, as in the figure layout
KEY_BOTTOM = ["intestine", "kidney"]


def tissue_key(ax, tissues, size, ncol=1, **kw):
    """Dataset key as it appears in the assembled figure: a dot on a short line, the name bold
    in the tissue colour."""
    h = [Line2D([], [], color=TC[t], marker="o", ms=size * 0.70, lw=size * 0.26,
                mec="white", mew=size * 0.05) for t in tissues]
    lg = ax.legend(handles=h, labels=[TDISP[t] for t in tissues], frameon=False, ncol=ncol,
                   prop=dict(size=size, weight="bold"), handlelength=1.4, handletextpad=0.4,
                   labelspacing=0.25, columnspacing=1.0, borderpad=0.0, **kw)
    for tx, t in zip(lg.get_texts(), tissues):
        tx.set_color(TC[t])
    return lg


def complete_a():
    d = Layout(K_LAYOUT["A"])
    fig, ax = plt.subplots(figsize=PANELS["A"][1])
    P = panel_a(ax, bare=True)
    XS, XM, XD, W = P["XS"], P["XM"], P["XD"], P["W"]
    for t in TIS:
        y0, y1 = P["src"][t]
        ax.text(XS - 0.010, (y0 + y1) / 2, f"{TDISP[t]} (n = {NT[t]})\n{CRED[t]}",
                ha="right", va="center", fontsize=d.text, fontweight="bold", color=TC[t],
                linespacing=1.15)
    for m, (y0, y1) in P["mid"].items():
        if m == "survives":
            ax.text(XM + W / 2, y0 - 0.3, "Survive", ha="center", va="bottom", fontsize=d.text)
        else:
            ax.text(XM + W / 2, y1 + 0.3, "Collapse", ha="center", va="top", fontsize=d.text)
    for s_, (y0, y1) in P["dst"].items():
        ax.text(XD + W + 0.008, (y0 + y1) / 2, "Survive" if s_ == ORDER_ST[0] else "Fail",
                ha="left", va="center", fontsize=d.text)
    for x, lab in (((XS + W + XM) / 2, "Guard 1\nBuffer space: 0 spot"),
                   ((XM + W + XD) / 2, "Guard 2\nBuffer space: 4 spot")):
        ax.text(x, -1.3, lab, ha="center", va="bottom", fontsize=d.text, fontweight="bold",
                linespacing=1.15)
    ax.text((XS + XD + W) / 2, -4.4, TITLES["A"], ha="center", va="bottom", fontsize=d.title,
            fontweight="bold")
    return fig


def complete_b():
    d = Layout(K_LAYOUT["B"])
    fig, ax = plt.subplots(figsize=PANELS["B"][1])
    G = panel_b(ax, bare=True)
    for p in G["pairs"]:
        ax.text(p["x0"] + p["wpair"] / 2, G["y1"] + 0.215,
                f"{TDISP[p['tis']]}, {VERDICT[p['verdict']]}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=d.text, fontweight="bold", color=TC[p["tis"]])
        for c, lab in enumerate(("Expression-\nindependent", "Expression-\ndependent")):
            ax.text(p["x0"] + c * (p["wmap"] + p["gap_in"]) + p["wmap"] / 2, G["y1"] + 0.02,
                    lab, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=40 / d.k, color="#121212", linespacing=1.1)
        ax.text((p["bar_x0"] + p["bar_x1"]) / 2, 0.050, "1mm", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=d.text)
    cb = G["cbar"]
    cb.set_label("Coordinate value", fontsize=d.text, fontweight="bold")
    cb.ax.tick_params(labelsize=d.text, width=d.line, length=d.tick)
    cb.outline.set_linewidth(d.line)
    ax.set_title(TITLES["B"], fontsize=d.title, fontweight="bold", color="#121212", pad=0,
                 y=G["y1"] + 0.36)
    return fig


def complete_e():
    d = Layout(K_LAYOUT["E"])
    fig = plt.figure(figsize=(6.6, 3.4))                 # size at which the marks tier is rendered
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    panel_e(ax, bare=True)
    d.axes(ax)
    x = sorted(BUF.buffer_pitch.unique())
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in x])
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["-0.2", "0.0", "0.2", "0.4", "0.6"])
    ax.set_xlabel("Buffer space", fontsize=d.text, fontweight="bold")
    ax.set_ylabel("$\\mathbf{\\hat{I}}$ (bit)", fontsize=d.text, fontweight="bold")
    lg = tissue_key(ax, KEY_TOP, 40 / d.k, loc="upper right", bbox_to_anchor=(1.0, 1.06))
    ax.add_artist(lg)
    tissue_key(ax, KEY_BOTTOM, 40 / d.k, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    ax.set_title("Recoverability under buffer space of\nspatial block CV", fontsize=d.title,
                 fontweight="bold", loc="center", pad=d.text * 1.6)
    return fig


def complete_c():
    d = Layout(K_LAYOUT["C"])
    fig, ax = plt.subplots(figsize=PANELS["C"][1])
    panel_c(ax, bare=True)
    d.axes(ax); ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["-0.2", "0.0", "0.2", "0.4", "0.6", "0.8"])
    # every name but the kidney wraps onto two lines because it sits in a narrow box in the figure layout
    ax.set_xticklabels([TDISP[t_] if t_ == "kidney" else TDISP[t_].replace(" ", "\n", 1)
                        for t_ in TIS], linespacing=1.05)
    for lb, t_ in zip(ax.get_xticklabels(), TIS):
        lb.set_fontsize(36 / d.k); lb.set_fontweight("bold"); lb.set_color(TC[t_]); lb.set_va("top")
    ax.set_ylabel("$\\mathbf{\\hat{I}}$ (bit)", fontsize=d.text, fontweight="bold")
    h = [Patch(facecolor="#6b6b6b"), Patch(facecolor="#6b6b6b", alpha=0.28)]
    ax.legend(handles=h, labels=["Spatial block CV", "Excess of random CV"], frameon=False,
              loc="upper center", bbox_to_anchor=(0.55, 1.02), fontsize=40 / d.k,
              handlelength=1.2, handletextpad=0.5, labelspacing=0.3, borderpad=0.0)
    ax.set_title(TITLES["C"], fontsize=d.title, fontweight="bold", pad=d.text)
    return fig
def complete_d():
    d = Layout(K_LAYOUT["D"])
    fig, ax = plt.subplots(figsize=PANELS["D"][1])
    panel_d(ax, bare=True)
    left, right = ax.child_axes[0], ax.child_axes[1]
    for a in (left, right):
        d.axes(a)
    left.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    left.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8"])
    yt = left.get_yticks(); left.set_yticklabels([f"{v:.1f}" for v in yt])
    right.set_xticklabels(["0", "1k", "2k", "3k", "4k"]); right.tick_params(labelleft=False)
    left.set_ylabel("Fraction of the inflation", fontsize=d.text, fontweight="bold")
    right.set_xlabel("Spot number", fontsize=d.text, fontweight="bold")
    tissue_key(right, KEY_LAYOUT, 40 / d.k, loc="upper left", bbox_to_anchor=(1.03, 1.0))
    fig.canvas.draw()
    bl, br = left.get_position(), right.get_position()
    fig.text((bl.x0 + br.x1) / 2, bl.y1 + 0.05, TITLES["D"], ha="center", va="bottom",
             fontsize=d.title, fontweight="bold")
    return fig


def full():
    global SHARED_LEGEND
    SHARED_LEGEND = False
    dd = OUT / FIG; dd.mkdir(parents=True, exist_ok=True)
    complete_panels = {"A": complete_a, "B": complete_b, "C": complete_c, "D": complete_d, "E": complete_e}
    for lab, (fn, size) in PANELS.items():
        if lab in complete_panels:
            fig = complete_panels[lab]()
        else:
            fig, ax = plt.subplots(figsize=size)
            fn(ax, False)
            ax.set_title(TITLES[lab], loc="left")
        for ext in (".png", ".pdf"):
            fig.savefig(dd / f"{FIG}_{lab}{ext}", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    marks()
    full()
    print("sections per tissue:", NT)
    print("outcome counts:", A["mode"].value_counts().to_dict())
