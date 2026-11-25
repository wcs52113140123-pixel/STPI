#!/usr/bin/env python3
"""Figure S5. Supplementary analyses of the 18 published sections.

  A  Detailed information of each section
  B  Expression-independent and dependent coordinates in the remaining 13 sections
  C  Random CV inflation of each section at three block sizes
  D  Inflation and depth across the different tissue sections
  E  Recoverability under buffer space of spatial block CV

Inputs (results/)
  fieldwide_audit.csv, fig5_buffered_cv_all.csv, fig5_fieldwide_null_msr.csv,
  fig5_pred_maps_all.npz, fig5_blocksize_sensitivity.csv, fig5_section_covariates.csv,
  fig5_deletion_control.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from _common import RESULTS, marks_dir, OUT
import fig5 as F5

FIG = "FigS5"
TIS, TC, TDISP = F5.TIS, F5.TC, F5.TDISP
BASE_FS, ANNO_FS, TICK_FS, GREY = F5.BASE_FS, F5.ANNO_FS, F5.TICK_FS, F5.GREY
SCALEBAR = False        # B: the figure carries no scale bar on the decoded fields

AU = pd.read_csv(RESULTS / "fieldwide_audit.csv").set_index("sample")

BUF = pd.read_csv(RESULTS / "fig5_buffered_cv_all.csv")

MSR = pd.read_csv(RESULTS / "fig5_fieldwide_null_msr.csv").set_index("sample")

G = pd.DataFrame(dict(
    tissue=AU.tissue,
    I=AU.I_checker,
    g1=AU.I_checker > 0,
    g2=BUF[BUF.buffer_pitch == 4].set_index("sample").I_bit.reindex(AU.index) > 0,
    g3=MSR.P.reindex(AU.index) <= 0.05,
    P=MSR.P.reindex(AU.index)))

G["tissue"] = pd.Categorical(G.tissue, TIS, ordered=True)

G = G.sort_values(["tissue", "I"], ascending=[True, False])

G["all3"] = G.g1 & G.g2 & G.g3

GUARDS = [("g1", "(1) spatial\nblock CV"), ("g2", "(2) buffered\nfolds"),
          ("g3", "(3) autocorrelation\n null")]

# Which ribbon of the main flow each section belongs to. Fig 5a draws three terminal groups
# and gives only their widths; this column is what turns a ribbon back into named sections,
# so the counts at the foot of this panel are the widths drawn there.
LANE = {(True, True): ("holds", "#2f6f8f"), (True, False): ("falls", "#e08a3c"),
        (False, False): ("collapses", "#b9b9b9"), (False, True): ("collapses", "#b9b9b9")}

G["lane"] = [LANE[(bool(a_), bool(b_))][0] for a_, b_ in zip(G.g1, G.g2)]

def short(s):
    """Section id as printed: the accession's own suffix, which is how the sources name it."""
    tail = s.split("_", 1)[1]
    return tail.split("_TS_")[0].replace("_A1", "").replace("_D", "")

# Row order for panels A and C: tissue, then spatial block information from high to low,
# matching the order used in the assembled figure. The order is fixed explicitly rather than
# computed by sorting, because No22 and No18 differ by only 0.0005 bit and the section names
# are written out below in the order used in the figure.
ROW_ORDER = ["No22", "No18", "No24", "No16", "No21", "No23", "No19", "No13",
             "V11Y11-298", "V11Y11-297", "V11U23-312",
             "CR_23_06187", "CR_23_06184", "CR_23_06182",
             "079", "078", "rep1", "rep2"]

G = G.loc[sorted(G.index, key=lambda s: ROW_ORDER.index(short(s)))]

def panel_a(ax, bare=False, guards=None):
    """One row per section, one column per guard; filled = passes, open = does not.

    Drawn as a matrix and not as a flow because guards (2) and (3) are not nested, see the
    module docstring for the four sections that cross.
    """
    n = len(G)
    y = np.arange(n)[::-1]
    gsel = GUARDS if guards is None else guards
    xs = [0.0, 1.0, 2.0][:len(gsel)]
    for yy, (samp, r) in zip(y, G.iterrows()):
        c = TC[r.tissue]
        for x, (col, _) in zip(xs, gsel):
            ok = bool(r[col])
            ax.plot([x], [yy], marker="o", ms=7.5, mfc=c if ok else "white",
                    mec=c if ok else "#bdbdbd", mew=1.1, zorder=3)
        if not bare:
            ax.text(-0.42, yy, short(samp), ha="right", va="center",
                    fontsize=TICK_FS, color=c)
            ax.text(3.02, yy, f"{r.I:+.3f}", ha="right", va="center",
                    fontsize=TICK_FS, color="#444")
            ax.text(3.74, yy, f"{r.P:.3f}", ha="right", va="center",
                    fontsize=TICK_FS, color="#444")
    # tissue blocks
    row_of = {s: i for i, s in enumerate(G.index)}
    for t in TIS:
        idx = [row_of[s] for s in G.index[G.tissue == t]]
        y0, y1 = y[max(idx)] - 0.42, y[min(idx)] + 0.42
        ax.plot([-1.45, -1.45], [y0, y1], lw=2.6, color=TC[t], solid_capstyle="butt",
                zorder=2, clip_on=False)
        if not bare:
            ax.text(-1.58, (y0 + y1) / 2, TDISP[t], ha="right", va="center",
                    fontsize=ANNO_FS, color=TC[t], rotation=0)
    ax.set_xlim(-1.5, 4.92)
    ax.set_ylim(-2.7, n + 1.25)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    if not bare:
        for x, (col, lab) in zip(xs, GUARDS):
            ax.text(x, n - 0.30, lab, ha="center", va="bottom", fontsize=TICK_FS,
                    color="#333", linespacing=1.25)
            ax.text(x, -0.75, f"{int(G[col].sum())}/18", ha="center", va="center",
                    fontsize=ANNO_FS, color="#333")
        # In the assembled figure this cell is narrower than the standalone export, so the
        # guard-(3) header would overlap it; the numeric column labels sit further right.
        ax.text(3.02, n - 0.30, "$\\hat{I}$ (bit)\nguard (1)", ha="right", va="bottom",
                fontsize=TICK_FS, color="#333", linespacing=1.25)
        ax.text(3.74, n - 0.30, "P\nguard (3)", ha="right", va="bottom",
                fontsize=TICK_FS, color="#333", linespacing=1.25)
        ax.text(4.88, n - 0.30, "ribbon in\nFig 5a", ha="right", va="bottom",
                fontsize=TICK_FS, color="#333", linespacing=1.25)
        for yy, (samp, r) in zip(np.arange(len(G))[::-1], G.iterrows()):
            ax.text(4.88, yy, LANE[(bool(r.g1), bool(r.g2))][0], ha="right", va="center",
                    fontsize=TICK_FS, color=LANE[(bool(r.g1), bool(r.g2))][1])
        for lane_, lab_ in (("holds", "holds"), ("falls", "falls"), ("collapses", "collapses")):
            pass
        ax.text(-0.42, -0.75, "passing:", ha="right", va="center", fontsize=ANNO_FS,
                color="#333")
        ax.text(-0.42, -1.55, f"all three: {int(G.all3.sum())}/18", ha="right", va="center",
                fontsize=ANNO_FS, color="#000")
        ax.text(4.88, -0.75,
                "  ".join(f"{k} {int((G.lane == k).sum())}" for k in ("holds", "falls", "collapses")),
                ha="right", va="center", fontsize=TICK_FS, color="#333")
        # One line, below the "all three" line, so the two footers do not collide.
        h = [Line2D([0], [0], marker="o", ls="none", mfc="#555", mec="#555", ms=6.5),
             Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#bdbdbd", ms=6.5)]
        # Legend placed below the matrix: the only position clear of the guard-(3) column header.
        ax.legend(handles=h, labels=["passes", "does not pass"], frameon=False,
                  loc="upper left", bbox_to_anchor=(0.18, -0.01), ncol=2,
                  fontsize=TICK_FS, handletextpad=0.35, columnspacing=1.4, borderpad=0.0)

# ─────────────── b · every section's decoded field, not only the five in Fig 5b ───────────────
PM = np.load(RESULTS / "fig5_pred_maps_all.npz")

# The five sections drawn in main Fig 5b, identified by matching fig5_pred_maps.npz against
# the per-section table on spot count and rho, rather than inferred from the section name:
# retina section 078 (n = 575, rho = 0.075) is the one drawn in the main panel, not 079.
EX5 = {"GSM8440999_No22", "GSM8969382_CR_23_06187_TS_R_JSV_1_S8", "GSM9266973_rep1",
       "GSM7840104_V11U23-312_D", "GSM7916286_078_A1"}

def panel_b(ax, bare=False, subset=None):
    """Each of the 13 sections not drawn in Fig 5b, shown as the same pair Fig 5b draws for
    its five: the geometric coordinate and what expression recovers of it.

    Main Fig 5b shows five pairs chosen to span the outcome classes; this panel shows the
    remaining thirteen, so the two together cover the complete set of eighteen sections with
    no repeats. Both the coordinate and the decoded field are shown together because the
    coordinate alone does not indicate what the decode is supposed to recover on these
    sections, which is the point of the comparison.

    Four pairs per row, four rows. Within a pair: left = coordinate (geometry only, no
    expression enters it), right = decoded (held-out predictions, 4x4 checkerboard folds).
    Colour encodes the rank within each map, so maps are comparable in pattern, not in value.

    Uses the same decode as the audit, verified against decode_buffered to 1e-9 in
    analysis/figS5_support.py. The I printed here is recomputed through analysis/fig5_buffered_all.py,
    while panel a's I column comes from analysis/fig5_audit.py; they differ by a median of 0.002 bit,
    a worst case of 0.015 (retina 078, both values below zero), and no section changes sign.
    Quote the audit table for I, not these map values.
    """
    ax.axis("off")
    order = [x for x in G.index if x not in EX5]
    if subset is not None:                 # one section on its own (per-section export)
        order = [x for x in order if x in set(subset)]
    ncol, nrow = 4, 4
    wpair, hmap = 0.214, 0.182
    x_pitch, y_pitch, x_org, y_org, lab_dy = 0.244, 0.245, 0.030, 0.760, 0.040
    if len(order) == 1:                    # fill the axes with the single pair
        wpair, hmap = 0.92, 0.66
        x_pitch = y_pitch = 0.0
        x_org, y_org, lab_dy = 0.04, 0.045, 0.150   # name clears the I/rho line below it
    n_last = len(order) % ncol or ncol            # centre the final, partial row
    for i, samp in enumerate(order):
        r, c = divmod(i, ncol)
        shift = (ncol - n_last) / 2.0 * x_pitch if r == (len(order) - 1) // ncol else 0.0
        xy = np.asarray(PM[f"{samp}_xy"], float)
        tru = np.asarray(PM[f"{samp}_true"], float)
        pred = np.asarray(PM[f"{samp}_pred"], float)
        I, rho, n, pitch = PM[f"{samp}_meta"]
        Q = np.c_[xy[:, 0], -xy[:, 1]]
        x0 = x_org + c * x_pitch + shift
        y0 = y_org - r * y_pitch
        for j, val in enumerate((tru, pred)):
            gx, gy, Z, Al = F5.spot_field(Q, F5.rank01(val), float(pitch))
            sub = ax.inset_axes([x0 + j * (wpair / 2 + 0.006), y0, wpair / 2, hmap])
            sub.imshow(Z, cmap="viridis", vmin=0, vmax=1, origin="lower", alpha=Al,
                       extent=[gx[0], gx[-1], gy[0], gy[-1]], interpolation="nearest",
                       rasterized=True)
            sub.set_aspect("equal"); sub.axis("off"); sub.margins(0.02)
            if j == 1 and SCALEBAR:
                # Scale bar, 1 mm, on the decoded half of every pair. These are external
                # Visium sections and their coordinates are full-resolution image pixels, not
                # microns, so the only defensible calibration is the platform's own geometry:
                # the capture-area lattice pitch is 100 um centre-to-centre, so 1 mm
                # corresponds to 10 times the measured native pitch (131 px kidney, 289-290
                # intestine, 691 cerebellum, 225 retina, 291 finch). As an accuracy check
                # (a residual, not a claim that every section falls inside the capture area):
                # spans come out to x 4.46-6.46 mm, y 4.21-6.71 mm, i.e. 45-65 pitches by
                # 49-78 rows against the v1 array's nominal 64 x 78 (6.30 x 6.67 mm). Two
                # sections exceed the nominal array (No23 by 2 columns, both intestine
                # sections by 1 row), so this calibration is accurate to about 3% (0.21 mm),
                # not exact. Do not state that every section falls inside 6.5 x 6.5 mm, and
                # do not quote a pooled 4.5-6.7 mm range: the lower bound of 4.5 excludes
                # finch rep2 (4.21 mm) and the upper bound of 6.7 is itself above 6.5. Do not
                # convert via the deposited spot diameter: only one dataset ships
                # scalefactors, and its diameter implies a scale about 15% smaller.
                L = F5.MM_IN_PITCHES * float(pitch)   # shared with main Fig 5b
                bx0, bx1 = sub.get_xlim(); by0, by1 = sub.get_ylim()
                sub.set_ylim(by0 - 0.26 * (by1 - by0), by1)
                yb = sub.get_ylim()[0] + 0.045 * (by1 - by0)
                xb = bx1 - 0.05 * (bx1 - bx0)
                sub.plot([xb - L, xb], [yb, yb], color="k", lw=0.8,
                         solid_capstyle="butt", zorder=9)
                if not bare:
                    # Label sits above the bar, inside the band opened by the ylim extension.
                    sub.text((2 * xb - L) / 2, yb + 0.022 * (by1 - by0), "1 mm",
                             ha="center", va="bottom", fontsize=TICK_FS * 0.80, color="#444")
        if not bare:
            tis = G.loc[samp, "tissue"]
            ax.text(x0 + wpair / 2, y0 + hmap + lab_dy, short(samp), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=TICK_FS, color=TC[tis])
            ax.text(x0 + wpair / 2, y0 + hmap + 0.004,
                    f"$\\hat{{I}}$ = {I:+.3f}   $\\rho$ = {rho:.2f}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=TICK_FS,
                    color="#444")
    if not bare:
        # Without this label the two halves of a pair are not distinguishable.
        note = ("within each pair:  left  coordinate (geometry only)"
                "   \u00b7   right  decoded from expression" if len(order) > 1 else
                "left  coordinate (geometry only)   \u00b7   right  decoded from expression")
        ax.text(x_org, 0.012 if len(order) > 1 else -0.045, note,
                transform=ax.transAxes, ha="left", va="bottom", fontsize=TICK_FS, color=GREY)
# ─────────────── c · the removed amount holds under every block geometry ───────────────
BS = pd.read_csv(RESULTS / "fig5_blocksize_sensitivity.csv")

WBS = BS.pivot_table(index="sample", columns="blocks", values="I_checker")

# Same unit as Fig 5c's pale bar segment: bits removed, not a ratio. A ratio of
# I_removed / I_random would exceed 1 on the two sections whose spatial reading is
# negative, requiring off-scale markers; in bits nothing leaves the axis.
REMOVED = pd.DataFrame({k: AU.I_random.reindex(WBS.index) - WBS[k] for k in (3, 4, 5)})

def panel_c(ax, bare=False):
    """Fig 5c's pale segment, the reading removed by spatial block CV, at all three
    checkerboard geometries, one row per section.

    The three geometries share a single axis in the figure's own unit, and no geometry is
    privileged by the layout; 4x4 is filled because it is the one the main text reports.

      every marker right of zero   54 of 54 (18 sections x 3 geometries): random CV reads
                                   higher than spatial block CV on every published section,
                                   whatever the block layout.
      the ordering is systematic   3x3 is the rightmost of the three on 14 of 18 sections
                                   and 5x5 the leftmost on 11 of 18. The count is of the
                                   maximum and minimum of the row, not pairwise against
                                   4x4: a pairwise comparison against 4x4 gives 15 and 12,
                                   a different and weaker statement. Bigger blocks put
                                   training and test further apart, so more is removed.
                                   Medians 0.245 / 0.153 / 0.143 bit.
      the fixed choice              is the middle of the three on 11 of 18 sections.

    Rows are in the order of panel A (ROW_ORDER), so a section sits on the same line in both
    panels. Per-section values: results/fig5_blocksize_sensitivity.csv.
    """
    MK = {3: ("^", "3\u00d73  bigger blocks"), 4: ("o", "4\u00d74  used by the audit"),
          5: ("v", "5\u00d75  smaller blocks")}
    rows = [s_ for s_ in G.index if s_ in REMOVED.dropna().index]   # same rows as panel A
    y = {s_: len(rows) - 1 - i for i, s_ in enumerate(rows)}
    ax.axvline(0, lw=0.9, color="#444", ls=(0, (4, 2.5)), zorder=1)
    for s_ in rows:
        r = REMOVED.loc[s_]
        c_ = TC[AU.loc[s_, "tissue"]]
        ax.plot([r.min(), r.max()], [y[s_]] * 2, lw=0.9, color=c_, alpha=0.45, zorder=2)
        for k, (mk, _) in MK.items():
            ax.plot(r[k], y[s_], marker=mk, ms=5.4 if k == 4 else 4.8,
                    markerfacecolor=c_ if k == 4 else "white", markeredgecolor=c_,
                    markeredgewidth=1.0, ls="none", zorder=4)
    ax.set_yticks([y[s_] for s_ in rows])
    ax.set_yticklabels([short(s_) for s_ in rows], fontsize=TICK_FS)
    for lab_, s_ in zip(ax.get_yticklabels(), rows):
        lab_.set_color(TC[AU.loc[s_, "tissue"]])
    # Two empty rows of headroom keep the note clear of the data (it would otherwise sit
    # over the first row's 3x3 marker) and keep the legend clear of the finch rows.
    ax.set_ylim(-0.9, len(rows) + 1.9)
    ax.set_xlim(-0.02, 0.60)
    ax.set_xlabel("reading removed (bit)\n"
                  "$\\hat{I}$(random 5-fold) \u2212 $\\hat{I}$(spatial block)")
    ax.tick_params(axis="y", length=0)
    if not bare:
        h = [Line2D([0], [0], marker=MK[k][0], ls="none", ms=5.4,
                    markerfacecolor=("#777" if k == 4 else "white"), markeredgecolor="#777")
             for k in (3, 4, 5)]
        ax.legend(handles=h, labels=[MK[k][1] for k in (3, 4, 5)], frameon=False,
                  loc="center right", fontsize=TICK_FS, handletextpad=0.5,
                  labelspacing=0.3, borderpad=0.0, bbox_to_anchor=(1.01, 0.40))

# ─────────────── d · the covariate Fig 5d does not test, and why it cannot be ───────────────
COV = pd.read_csv(RESULTS / "fig5_section_covariates.csv").set_index("sample")

FA = ((AU.I_random - AU.I_checker) / AU.I_random).reindex(COV.index)

def panel_d(ax, bare=False, boxes=None, depth_lim=None):
    """Why the two covariates a reader might propose next cannot be tested on this sample.

    The x axis is the argument: the five datasets ordered by spot pitch, which is also
    the five tissues.

      upper strip   removed share, one dot per section, the quantity Fig 5d plots
      lower strip   sequencing depth, one dot per section, log scale
      x             the five datasets in order of spot pitch, 131 to 691, printed under
                    each column. These are distances in each dataset's own full-resolution
                    image pixels, not microns: only one of the five deposited a
                    scalefactors file, and its spot diameter does not match the standard
                    Visium geometry (ratio 1.54 against the expected 1.82), so no px-to-um
                    conversion is defensible here. The quantity is therefore a per-dataset
                    constant whose physical meaning is not recoverable, i.e. a dataset
                    label.

    Spot pitch takes one value per dataset, so ordering sections by resolution is ordering
    them by tissue; depth clusters by tissue too (only intestine and finch retina overlap)
    and runs the opposite way, Spearman -0.76 against pitch. So rho(share, pitch) = +0.62 and
    rho(share, depth) = -0.62 are one comparison seen twice, over five tissues, not eighteen
    independent sections.

    Not drawn, deliberately: the apparent reading and the spot count (Fig 5d already has
    them), section extent and genes per spot (Spearman 0.91 against pitch, 0.98 against
    depth). All six are in results/fig5_section_covariates.csv.
    """
    ax.axis("off")
    b1, b2 = boxes if boxes is not None else ([0.0, 0.40, 1.0, 0.60], [0.0, 0.0, 1.0, 0.26])
    a1 = ax.inset_axes(b1)
    a2 = ax.inset_axes(b2)
    d_ = COV.assign(tissue=AU.tissue.reindex(COV.index), y=FA)
    # Tissue order (kidney, intestine, cerebellum, retina, finch retina) differs from the
    # pitch order because retina sits at pitch 226, between kidney 131 and intestine 289.
    # The pitch of each dataset is printed under its column, so the covariate remains
    # readable even though it is not the sort key, and the axis label below does not
    # claim it is.
    order = [t for t in TIS if t in set(d_.tissue)]
    rng = np.random.default_rng(7)
    for i, t in enumerate(order):
        q = d_[d_.tissue == t]
        jit = rng.uniform(-0.16, 0.16, len(q))
        a1.scatter(i + jit, q.y, s=36, color=TC[t], alpha=0.9, edgecolor="white", linewidth=0.6)
        a1.plot([i - 0.3, i + 0.3], [q.y.median()] * 2, lw=2.2, color=TC[t], solid_capstyle="round")
        a2.scatter(i + jit, q.median_counts, s=28, color=TC[t], alpha=0.9, edgecolor="white",
                   linewidth=0.6)
        a2.plot([i - 0.3, i + 0.3], [q.median_counts.median()] * 2, lw=2.0, color=TC[t],
                solid_capstyle="round")
    a1.axhline(0, lw=0.9, color="#444", ls=(0, (4, 2.5)))
    for a_ in (a1, a2):
        a_.set_xlim(-0.6, len(order) - 0.4)
        a_.set_xticks(range(len(order)))
    a1.set_xticklabels([])
    a2.set_yscale("log")
    if depth_lim is not None:
        a2.set_ylim(*depth_lim)
    # Pitch first, tissue name below it: the number is what orders the axis.
    a2.set_xticklabels([f"{int(d_[d_.tissue == t].pitch.median())}\n{TDISP[t]}"
                        for t in order])
    a1.set_ylabel("removed share")   # defined on main Fig 5d and in the caption
    a2.set_ylabel("depth\n(counts/spot)")
    a2.set_xlabel("the five datasets (spot pitch under each, in their own image units)")
    if not bare:
        a1.text(0.985, 0.97, f"$\\rho$ with pitch {spearmanr(d_.y, d_.pitch).correlation:+.2f}\n"
                f"$\\rho$ with depth {spearmanr(d_.y, d_.median_counts).correlation:+.2f}",
                transform=a1.transAxes, ha="right", va="top", fontsize=TICK_FS, color=GREY,
                linespacing=1.5)

# ─────────────── e · the buffer's decline is not the deleted training data ───────────────
DEL = pd.read_csv(RESULTS / "fig5_deletion_control.csv")

def panel_e(ax, bare=False):
    """The control Fig 5e needs. A widening buffer does two things at once: it separates
    training from test spots, and it deletes training spots (5 / 35 / 65 % at the three
    rungs). If the decline in Fig 5e were the deletion, then removing the same share of
    training spots at random, with no spatial gap, would reproduce it.

    It does not. At every rung the buffered reading sits below the matched random deletion
    (50 of 54 section-rungs), and the gap widens with the rung: medians -0.005, -0.075,
    -0.124 bit. Random deletion leaves 16 of 18 sections above zero at all three rungs,
    while the buffer takes them to 16, 13 and 11. The decline reflects the spatial
    separation, not the lost training data.

    Matched per section to that section's own measured dropped share, not to the median,
    and averaged over 5 draws with a per-section SHA-256 seed (analysis/figS5_support.py).
    """
    # Rungs come from the table rather than a literal list, so the 3-pitch rung
    # (computed separately, analysis/fig5_buffer3.py) is not silently dropped.
    x = sorted(DEL.buffer_pitch.unique())
    jit = 0.07
    for i, rung in enumerate(x):
        q = DEL[DEL.buffer_pitch == rung]
        for _, r in q.iterrows():
            ax.plot([rung - jit, rung + jit], [r.I_random_deletion, r.I_buffered],
                    lw=0.6, color=TC[r.tissue], alpha=0.40, zorder=2)
            ax.plot(rung - jit, r.I_random_deletion, "o", ms=2.6, mfc="white",
                    mec=TC[r.tissue], mew=0.7, zorder=3)
            ax.plot(rung + jit, r.I_buffered, "o", ms=2.6, color=TC[r.tissue], zorder=3)
    md = DEL.groupby("buffer_pitch")[["I_random_deletion", "I_buffered"]].median()
    ax.plot([k - jit for k in x], md.I_random_deletion.values, lw=2.0, color="#555",
            ls=(0, (4, 2.4)), zorder=5)
    ax.plot([k + jit for k in x], md.I_buffered.values, lw=2.4, color="#111", zorder=5)
    ax.axhline(0, lw=0.9, color="#444", zorder=1)
    drop = DEL.groupby("buffer_pitch").frac_dropped.median()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n{drop[k]:.0%}" for k in x])
    ax.set_xlim(0.55, 4.6)
    ax.set_xlabel("training\u2013test gap (spot pitches)\n"
                  "% = training spots dropped")
    ax.set_ylabel("$\\hat{I}$ (bit)\nspatial block CV")
    if not bare:
        # Each handle combines a marker and a line style so the two summary lines
        # (dashed = random deletion, solid = buffered) are identified without doubling
        # the legend to four entries.
        h = [Line2D([0], [0], marker="o", ms=4, mfc="white", mec="#555", mew=0.9,
                    color="#555", ls=(0, (4, 2.4)), lw=2.0),
             Line2D([0], [0], marker="o", ms=4, color="#333", ls="-", lw=2.4)]
        # Placed upper right: upper left would overlap the rung-1 column, and rung 4
        # peaks at 0.55, leaving the upper right clear.
        ax.legend(handles=h,
                  labels=["removed at random (dashed median)",
                          "removed next to the blocks (solid median)"],
                  frameon=False, loc="upper right", bbox_to_anchor=(0.998, 1.005),
                  fontsize=TICK_FS, handletextpad=0.5, labelspacing=0.32, borderpad=0.0)

PANELS = {"A": (panel_a, (7.4, 5.4)), "B": (panel_b, (10.4, 8.2)),
          "C": (panel_c, (5.0, 3.6)), "D": (panel_d, (6.6, 4.2)),
          "E": (panel_e, (6.0, 3.6))}
TITLES = {"A": "Detailed information of each section",
          "B": "Expression-independent and dependent coordinates in the remaining 13 sections",
          "C": "Random CV inflation of each section at three block sizes",
          "D": "Inflation and depth across the different tissue sections",
          "E": "Recoverability under buffer space of spatial block CV"}
B_PX = (3139, 2475)          # size at which the B image is placed in the figure


# Depth axis of S5 D matches the assembled figure layout: 10^4, 10^3 and 10^2 fall where
# the layout prints them (one decade = 0.0131 of the layout height, 1.86 cm) and the axes
# reach down to the layout's x baseline. The upper strip keeps its size and position; only
# the depth strip is taller.
_D_AX = 3.4 * 0.96 + 204 / 600             # panel height in inches: 204 px (600 dpi) taller
D_FIG = (7.4, _D_AX / 0.96)
D_BOXES = ([0.0, 1 - 3.4 * 0.96 * 0.60 / _D_AX, 1.0, 3.4 * 0.96 * 0.60 / _D_AX],   # upper strip box
           [0.0, 0.0, 1.0, 1 - 3.4 * 0.96 * (1 - 0.26) / _D_AX])                   # depth strip box, top edge fixed
D_DEPTH_LIM = (10 ** 0.2031, 10 ** 5.0671)
D_MINOR = [m * 10.0 ** e for e in range(1, 5) for m in range(2, 10) if m * 10.0 ** e <= 7e4]


def marks():
    d = marks_dir(FIG)
    for lab in ("A", "B", "C"):
        fn, size = PANELS[lab]
        fig, ax = plt.subplots(figsize=size)
        fn(ax, True)
        F5.strip_bare(ax)
        fig.savefig(d / f"{FIG}_{lab}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    from PIL import Image
    im = Image.open(d / "FigS5_B.png").convert("RGB")
    im.resize(B_PX, Image.LANCZOS).save(d / "FigS5_B.png")
    for lab, size, fn, kw in (("D", D_FIG, panel_d,
                               {"margin": True, "args": dict(boxes=D_BOXES, depth_lim=D_DEPTH_LIM)}),
                              ("E", (6.0, 3.6), panel_e, {})):
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        fn(ax, bare=True, **kw.get("args", {}))
        F5.strip_all(ax)
        if kw.get("margin"):
            for a in [ax] + list(ax.child_axes):
                a.margins(y=0.09)
        for t in list(fig.texts):
            t.remove()
        fig.savefig(d / f"{FIG}_{lab}.png", dpi=600, bbox_inches="tight", transparent=True,
                    pad_inches=0.02)
        plt.close(fig)
# ────────── complete panels in the wording and proportions of the assembled figure ──────────
# A, B, D and E as laid out in the figure: full wording, bold titles and axis titles, the
# dataset key, and no analysis annotations. A, D and E are drawn at the size of their marks
# images, so marks and text keep the proportions of the figure layout (text at 42/k pt,
# k = layout factor).
from _common import Layout
from scipy.spatial import cKDTree as _KD
K_LAYOUT = {"A": 4.02, "B": 3.15, "C": 4.81, "D": 3.00, "E": 3.81}
KEY_LAYOUT = ["cerebellum", "retina", "finch_retina", "intestine", "kidney"]   # figure order
STATUS = {(True, True): ("Survive", "#1F5F82"), (True, False): ("Fail", "#FF0000"),
          (False, False): ("Collapse", "#808080"), (False, True): ("Collapse", "#808080")}
B0 = BUF[BUF.buffer_pitch == 0].set_index("sample").I_bit
B4 = BUF[BUF.buffer_pitch == 4].set_index("sample").I_bit


def complete_a():
    d = Layout(K_LAYOUT["A"])
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    panel_a(ax, bare=True, guards=GUARDS[:2])
    n = len(G); y = np.arange(n)[::-1]
    row = 32 / d.k; head = 40 / d.k
    XV1, XV2, XS_ = 2.30, 3.15, 3.62
    for yy, (samp, r) in zip(y, G.iterrows()):
        c = TC[r.tissue]
        ax.text(-0.42, yy, short(samp), ha="right", va="center", fontsize=row, color=c)
        ax.text(XV1, yy, f"{B0[samp]:+.3f}", ha="center", va="center", fontsize=row)
        ax.text(XV2, yy, f"{B4[samp]:+.3f}", ha="center", va="center", fontsize=row)
        lab, col = STATUS[(bool(r.g1), bool(r.g2))]
        ax.text(XS_, yy, lab, ha="left", va="center", fontsize=row, color=col)
    row_of = {s: k for k, s in enumerate(G.index)}
    for t_ in TIS:
        idx = [row_of[s] for s in G.index[G.tissue == t_]]
        ym = (y[max(idx)] + y[min(idx)]) / 2
        ax.text(-1.58, ym, TDISP[t_].replace(" ", "\n"), ha="right", va="center",
                fontsize=42 / d.k, fontweight="bold", color=TC[t_], linespacing=1.05)
    for x, lab in ((-0.10, "(1) Buffer space:\n0 spot"), (1.10, "(2) Buffer space:\n4 spot"),
                   (XV1, "$\\hat{I}$ (bit)\nguard (1)"), (XV2, "$\\hat{I}$ (bit)\nguard (2)")):
        ax.text(x, n - 0.30, lab, ha="center", va="bottom", fontsize=head, linespacing=1.15)
    ax.text(1.55, n + 1.55, TITLES["A"], ha="center", va="bottom", fontsize=d.title,
            fontweight="bold")
    h = [Line2D([0], [0], marker="o", ls="none", mfc="#555", mec="#555", ms=7.5),
         Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#bdbdbd", mew=1.1, ms=7.5)]
    ax.legend(handles=h, labels=["Passes", "Does not pass"], frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.50, 0.075), fontsize=head,
              handletextpad=0.3, columnspacing=2.0, borderpad=0.0)
    return fig


def _pair(fig, samp, x0, y0, wmap, hmap, gap, d):
    """Coordinate and decoded map of one section, with a 1 mm bar measured on the map."""
    xy = np.asarray(PM[f"{samp}_xy"], float)
    tru = np.asarray(PM[f"{samp}_true"], float); pred = np.asarray(PM[f"{samp}_pred"], float)
    pitch = float(PM[f"{samp}_meta"][3])
    Q = np.c_[xy[:, 0], -xy[:, 1]]
    subs = []
    for j, val in enumerate((tru, pred)):
        gx, gy, Z, Al = F5.spot_field(Q, F5.rank01(val), pitch)
        sub = fig.add_axes([x0 + j * (wmap + gap), y0, wmap, hmap])
        sub.imshow(Z, cmap="viridis", vmin=0, vmax=1, origin="lower", alpha=Al,
                   extent=[gx[0], gx[-1], gy[0], gy[-1]], interpolation="nearest",
                   rasterized=True)
        sub.set_aspect("equal"); sub.axis("off"); sub.margins(0.02)
        subs.append(sub)
    s = subs[1]; s.apply_aspect()
    mm = (s.transData.transform((F5.MM_IN_PITCHES * pitch, 0))[0]
          - s.transData.transform((0, 0))[0]) / fig.bbox.width
    xr = (s.get_position().x1); yb = y0 - 0.018
    fig.add_artist(Line2D([xr - mm, xr], [yb, yb], color="black", lw=d.line * 0.9,
                          solid_capstyle="butt", transform=fig.transFigure))
    fig.text(xr - mm / 2, yb + 0.006, "1mm", ha="center", va="bottom", fontsize=d.text)
    tis = G.loc[samp, "tissue"]
    fig.text(x0 + wmap + gap / 2, y0 + hmap + 0.012, short(samp), ha="center", va="bottom",
             fontsize=40 / d.k, color=TC[tis])


def complete_b():
    d = Layout(K_LAYOUT["B"])
    fig = plt.figure(figsize=(16.0, 7.0))
    order = [x for x in G.index if x not in EX5]
    wpair, pitch_x, hmap, gap = 0.168, 0.198, 0.164, 0.006
    bottoms = [0.66, 0.35, 0.06]
    for i, samp in enumerate(order):
        r, c = divmod(i, 5)
        _pair(fig, samp, 0.006 + c * pitch_x, bottoms[r], (wpair - gap) / 2, hmap, gap, d)
    cax = fig.add_axes([0.612, 0.035, 0.010, 0.235])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"), cax=cax)
    cb.set_ticks(np.round(np.arange(0, 1.01, 0.2), 1))
    cb.ax.tick_params(labelsize=d.text, width=d.line, length=d.tick)
    cb.outline.set_linewidth(d.line)
    cb.set_label("Coordinate value", fontsize=d.text, fontweight="bold")
    cb.ax.yaxis.set_label_position("left")
    fig.text(0.675, 0.27, "Left:  Expression-independent\nRight: Expression-dependent",
             ha="left", va="top", fontsize=d.text, fontweight="bold", color="#121212",
             linespacing=1.25)
    kax = fig.add_axes([0.675, 0.03, 0.20, 0.15]); kax.axis("off")
    F5.tissue_key(kax, KEY_LAYOUT, 40 / d.k, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.text(0.5, 0.965, TITLES["B"], ha="center", va="top", fontsize=d.title,
             fontweight="bold", color="#121212")
    return fig


def complete_c():
    import matplotlib.transforms as mtr
    d = Layout(K_LAYOUT["C"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    panel_c(ax, bare=True)
    d.axes(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("\u0394\u00ce (bit)", fontsize=d.text, fontweight="bold")
    ax.set_xticks(np.round(np.arange(0, 0.61, 0.1), 1))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 0.61, 0.1)])
    for lb in ax.get_yticklabels():
        lb.set_fontsize(42 / d.k)
    rows = [s_ for s_ in G.index if s_ in REMOVED.dropna().index]
    yy = {s_: len(rows) - 1 - i for i, s_ in enumerate(rows)}
    fig.canvas.draw()
    wmax = max(lb.get_window_extent().width for lb in ax.get_yticklabels())
    xbar = -(wmax + 10) / ax.bbox.width                  # just left of the section names
    tr = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    for t_ in TIS:
        ys = [yy[s_] for s_ in rows if AU.loc[s_, "tissue"] == t_]
        ax.plot([xbar, xbar], [min(ys) - 0.42, max(ys) + 0.42], lw=2.6, color=TC[t_],
                solid_capstyle="butt", transform=tr, clip_on=False)
        ax.text(xbar - 0.02, (min(ys) + max(ys)) / 2, TDISP[t_].replace(" ", "\n"), ha="right",
                va="center", fontsize=42 / d.k, fontweight="bold", color=TC[t_],
                linespacing=1.05, transform=tr)
    MKS = {3: "^", 4: "o", 5: "v"}
    h = [Line2D([0], [0], marker=MKS[k], ls="none", ms=5.4,
                markerfacecolor=("#777" if k == 4 else "white"), markeredgecolor="#777")
         for k in (3, 4, 5)]
    lg = ax.legend(handles=h, labels=["3\u00d73", "4\u00d74", "5\u00d75"], frameon=False,
                   loc="lower right", fontsize=40 / d.k, handletextpad=0.4, labelspacing=0.3,
                   borderpad=0.0)
    ax.add_artist(lg)
    F5.tissue_key(ax, KEY_LAYOUT, 40 / d.k, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title(TITLES["C"], fontsize=d.title, fontweight="bold", pad=d.text)
    return fig


def complete_d():
    d = Layout(K_LAYOUT["D"])
    fig = plt.figure(figsize=(9.6, 3.6 * D_FIG[1] / 3.4))   # same proportions as the marks
    ax = fig.add_axes([0.02, 0.02, 0.80, 0.96])
    panel_d(ax, bare=True, boxes=D_BOXES, depth_lim=D_DEPTH_LIM)
    a1, a2 = ax.child_axes[0], ax.child_axes[1]
    for a in [ax, a1, a2]:
        a.margins(y=0.09)
    for a in (a1, a2):
        d.axes(a); a.set_xlabel("")
    a1.set_yticks(np.arange(0, 2.51, 0.5))
    a1.set_yticklabels([f"{v:.2f}" for v in np.arange(0, 2.51, 0.5)])
    a1.tick_params(axis="x", length=d.tick, width=d.line)
    a1.set_ylabel("Fraction of the inflation", fontsize=d.text, fontweight="bold")
    a2.set_ylabel("Depth", fontsize=d.text, fontweight="bold")
    a2.set_yticks([1e2, 1e3, 1e4])
    a2.set_yticks(D_MINOR, minor=True)                  # minor ticks shown on the depth axis
    order = [t_ for t_ in TIS if t_ in set(AU.tissue)]
    a2.set_xticklabels([TDISP[t_] for t_ in order], fontsize=36 / d.k, fontweight="bold")
    for tx, t_ in zip(a2.get_xticklabels(), order):
        tx.set_color(TC[t_])
    a2.yaxis.set_minor_formatter(plt.NullFormatter())
    F5.tissue_key(a1, KEY_LAYOUT, 40 / d.k, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    a1.set_title(TITLES["D"], fontsize=d.title, fontweight="bold", pad=d.text * 1.2)
    return fig


def complete_e():
    d = Layout(K_LAYOUT["E"])
    fig = plt.figure(figsize=(6.0, 3.6))                 # size at which the marks are drawn
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])        # the key sits to the right of it
    panel_e(ax, bare=True)
    d.axes(ax)
    x = sorted(DEL.buffer_pitch.unique())
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in x])
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["-0.2", "0.0", "0.2", "0.4", "0.6"])
    ax.set_xlabel("Buffer space", fontsize=d.text, fontweight="bold")
    ax.set_ylabel("$\\mathbf{\\hat{I}}$ (bit)", fontsize=d.text, fontweight="bold")
    s_ = 40 / d.k
    h = [Line2D([0], [0], marker="o", ls="none", color="#333", ms=s_ * 0.55),
         Line2D([0], [0], marker="o", ls="none", mfc="white", mec="#555", mew=1.2, ms=s_ * 0.55),
         Line2D([0], [0], ls="-", lw=2.4, color="#111"),
         Line2D([0], [0], ls=(0, (4, 2.4)), lw=2.0, color="#555")]
    lg = ax.legend(handles=h, labels=["Removed next to the blocks", "Removed at random",
                                      "Median", "Median"],
                   ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0),
                   fontsize=s_, handlelength=1.6, handletextpad=0.3, columnspacing=1.0,
                   labelspacing=0.15, borderpad=0.0)
    ax.add_artist(lg)
    F5.tissue_key(ax, KEY_LAYOUT, s_, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(TITLES["E"], fontsize=d.title, fontweight="bold", pad=d.text * 0.8)
    return fig
def full():
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
    print("sections passing all three guards:", int(G.all3.sum()), "of", len(G))
