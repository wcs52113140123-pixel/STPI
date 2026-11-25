#!/usr/bin/env python3
"""Colour and mark keys placed beside the panels (no text: labels are set in the figure).

  key_tissue_col_bare_line   Fig 5 D/E and Fig S5 B-E, tissue colours (key order)
  key_fig5e_col_bare         Fig S5 E, the four marks of the deletion control
  key_fig1c_bare             Fig 1 C, random and spatial block CV
plus labelled versions of the tissue key and the individual Fig 5 E marks.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
import os
KEYS = Path(os.environ.get("STPI_FIGOUT", HERE / "out")) / "keys"
KEYS.mkdir(parents=True, exist_ok=True)
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

src5 = open(HERE / "fig5.py").read()
TC = eval(re.search(r"TC = (\{.*?\})", src5, re.S).group(1))
TDISP = eval(re.search(r"TDISP = (\{.*?\})", src5, re.S).group(1))
TIS = eval(re.search(r"TIS = (\[.*?\])", src5, re.S).group(1))
src1 = open(HERE / "fig1.py").read()
C_RAND, C_CHK = re.search(r'C_RAND, C_CHK = "(#\w+)", "(#\w+)"', src1).groups()

# same three-level font fallback the figure scripts use (Arial if licensed and present in
# assets/fonts, else metrically identical Liberation Sans, else matplotlib default), so a
# key pasted beside a panel is set in the same face
import glob as _glob
for _f in _glob.glob(str(HERE.parent / "assets" / "fonts" / "*.[tT][tT][fF]")):
    try: matplotlib.font_manager.fontManager.addfont(_f)
    except Exception: pass
_have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for fam in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"):
    if fam in _have:
        plt.rcParams["font.family"] = fam
        break

def save(fig, stem):
    for ext in (".png", ".pdf"):
        fig.savefig(KEYS / f"{stem}{ext}", dpi=600, bbox_inches="tight",
                    transparent=True, pad_inches=0.02)
    plt.close(fig)

# dataset key: a line WITH a marker, so one key serves both the dot panels (5d, S5 d) and
# the line panels (5e, S5 e)
# dot only, no line through it: the panels this key serves draw one dot per section, and
# the line stub would read as a series marker that no panel actually uses
handles = [Line2D([], [], color=TC[t], marker="o", ms=7.0, linestyle="none",
                  mec="white", mew=0.6, label=TDISP[t]) for t in TIS]
fig = plt.figure(figsize=(7.6, 0.30))
fig.legend(handles=handles, loc="center", ncol=5, frameon=False, fontsize=9,
           handlelength=0.9, handletextpad=0.45, columnspacing=1.6, borderpad=0.0)
save(fig, "key_tissue_row")

fig = plt.figure(figsize=(1.75, 1.15))
fig.legend(handles=handles, loc="center", ncol=1, frameon=False, fontsize=9,
           handlelength=0.9, handletextpad=0.45, labelspacing=0.55, borderpad=0.0)
save(fig, "key_tissue_col")

# Fig 1 c: bars, so the handle is a patch, not a line
h1c = [Patch(facecolor=C_RAND, edgecolor="none", label="random CV"),
       Patch(facecolor=C_CHK, edgecolor="none", label="checkerboard CV")]
fig = plt.figure(figsize=(2.6, 0.62))
fig.legend(handles=h1c, loc="center", ncol=1, frameon=False, fontsize=9,
           handlelength=1.3, handleheight=0.9, handletextpad=0.5, labelspacing=0.5,
           borderpad=0.0)
save(fig, "key_fig1c")
print("colours:", {TDISP[t]: TC[t] for t in TIS}, "| random", C_RAND, "| checkerboard", C_CHK)

# label-free tiers, for use where the names are typed beside the swatches in the assembled
# figure: the same handles with empty labels, so colour order and marker style stay identical
for stem, (fs, nc) in {"key_tissue_row_bare": ((2.2, 0.26), 5),
                       "key_tissue_col_bare": ((0.42, 1.15), 1)}.items():
    hb = [Line2D([], [], color=TC[t], marker="o", ms=7.0, linestyle="none", mec="white",
                 mew=0.6, label=" ") for t in TIS]
    fig = plt.figure(figsize=fs)
    fig.legend(handles=hb, loc="center", ncol=nc, frameon=False, fontsize=9,
               handlelength=0.9, handletextpad=0.0, columnspacing=0.9, labelspacing=0.55,
               borderpad=0.0)
    save(fig, stem)

hb1c = [Patch(facecolor=C_RAND, edgecolor="none", label=" "),
        Patch(facecolor=C_CHK, edgecolor="none", label=" ")]
fig = plt.figure(figsize=(0.45, 0.62))
fig.legend(handles=hb1c, loc="center", ncol=1, frameon=False, fontsize=9, handlelength=1.3,
           handleheight=0.9, handletextpad=0.0, labelspacing=0.5, borderpad=0.0)
save(fig, "key_fig1c_bare")

# vertical, label-free key using the same mark style as the line panels (line through the dot).
# key_tissue_col_bare above draws the dot alone, which suits the dot panels (5d, S5d); this
# key is placed beside the LINE panels (5e, S5e), where each section is a line with dots, so
# the handle keeps its line stub here. Both tiers are kept; neither carries text.
# key order (top to bottom): cerebellum, retina, finch retina, intestine, kidney.
# TIS is the figure's plotting order; the labels are typed beside this swatch in the
# assembled figure, so the order here must match the key order below, not the panel draw order.
KEY_ORDER = ["Mouse cerebellum", "Mouse retina", "Finch retina", "Mouse intestine",
        "Mouse kidney"]
TIS_KEY = sorted(TIS, key=lambda t: KEY_ORDER.index(TDISP[t]))
hl = [Line2D([], [], color=TC[t], marker="o", ms=7.0, lw=2.6, mec="white", mew=0.6,
             label=" ") for t in TIS_KEY]
fig = plt.figure(figsize=(0.62, 1.30))
fig.legend(handles=hl, loc="center", ncol=1, frameon=False, fontsize=9, handlelength=1.5,
           handletextpad=0.0, labelspacing=0.62, borderpad=0.0)
save(fig, "key_tissue_col_bare_line")

# Fig S5 e: the four marks keyed one by one.
# The panel itself keys the two conditions (marker and line combined in one handle each).
# Split out here into four separate swatches, each labelled individually in the assembled
# figure. Styles are copied from figS5.py panel_e: per-section dots ms 2.6 (open = random
# deletion, filled = removed next to the blocks) and the two medians lw 2.0 dashed / 2.4 solid.
# Neutral greys, because in the panel the per-section dots take the tissue colour and the
# colour meaning is carried by the tissue key above.
E_MARKS = {
    "key_fig5e_filled_circle": dict(marker="o", ms=8.0, color="#333", ls="none"),
    "key_fig5e_open_circle":   dict(marker="o", ms=8.0, mfc="white", mec="#555", mew=1.6,
                                    ls="none"),
    "key_fig5e_solid_line":    dict(ls="-", lw=2.6, color="#111"),
    "key_fig5e_dashed_line":   dict(ls=(0, (4, 2.4)), lw=2.4, color="#555"),
}
for stem, st in E_MARKS.items():
    # a line swatch needs room for more than one dash cycle, or the dashed mark reads as a
    # single short bar and is indistinguishable from the solid one
    is_line = st.get("ls") != "none"
    fig = plt.figure(figsize=((1.05, 0.26) if is_line else (0.60, 0.26)))
    fig.legend(handles=[Line2D([], [], label=" ", **st)], loc="center", frameon=False,
               fontsize=9, handlelength=(3.0 if is_line else 1.0), handletextpad=0.0,
               borderpad=0.0)
    save(fig, stem)

# the same four stacked, for pasting as one block
fig = plt.figure(figsize=(1.05, 1.05))
fig.legend(handles=[Line2D([], [], label=" ", **E_MARKS[s]) for s in
                    ("key_fig5e_filled_circle", "key_fig5e_open_circle",
                     "key_fig5e_solid_line", "key_fig5e_dashed_line")],
           loc="center", ncol=1, frameon=False, fontsize=9, handlelength=3.0,
           handletextpad=0.0, labelspacing=0.62, borderpad=0.0)
save(fig, "key_fig5e_col_bare")
print("E marks:", list(E_MARKS))


# Fig 1 C key as placed in the figure: two swatches drawn pixel by pixel. Its colours
# (#f39d9f, #47a162) are a little more saturated than the bars of the panel (C_RAND, C_CHK);
# they are reproduced as they appear in the figure.
import numpy as np
from PIL import Image
img = np.full((237, 153, 3), 255, np.uint8)
img[8:86, 10:134] = (0xF3, 0x9D, 0x9F)
img[146:225, 10:134] = (0x47, 0xA1, 0x62)
Image.fromarray(img).save(KEYS / "key_fig1c_flat.png")
