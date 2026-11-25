"""Shared settings for the figure scripts.

Each figure script reproduces the panels of the assembled figure. For every panel it
writes two files:

  out/<Fig>/marks/<Fig>_<L>.png    the data marks exactly as they appear in the figure
                                    (same figure size, dpi, colours, limits and mark sizes)
  out/<Fig>/<Fig>_<L>.pdf|png      the complete panel: the same marks plus the axes, ticks,
                                    labels, legend and title set as text in the figure layout

Text in the complete panel keeps the proportions of the figure layout. The layout sets all
panel text at 42 pt, titles at 44 pt, panel letters at 60 pt and axis lines at 5 pt, on a
canvas 129 cm wide. A panel image that occupies D cm of the canvas but was rendered at
S cm has layout factor k = D / S, so its text is drawn at 42 / k pt, and so on. The factor
of each panel is a constant in its script.

Inputs are read from results/ and data_derived/ only (override with STPI_RESULTS and
STPI_DERIVED); output goes to figures/out/ (override with STPI_FIGOUT).
"""
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("STPI_RESULTS", ROOT / "results"))
DERIVED = Path(os.environ.get("STPI_DERIVED", ROOT / "data_derived"))
OUT = Path(os.environ.get("STPI_FIGOUT", ROOT / "figures" / "out"))

# ---- font ------------------------------------------------------------------
# Arial is licensed and not distributed here. Put ARIAL*.TTF in assets/fonts/ for a
# byte-identical rebuild; otherwise Liberation Sans (same metrics) is used.
for _f in glob.glob(str(ROOT / "assets" / "fonts" / "*.[tT][tT][fF]")):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass
_names = {f.name for f in font_manager.fontManager.ttflist}
if "Arial" not in _names:
    for _f in glob.glob("/usr/share/fonts/**/LiberationSans-*.ttf", recursive=True):
        font_manager.fontManager.addfont(_f)
    _names = {f.name for f in font_manager.fontManager.ttflist}
FONT = next(n for n in ("Arial", "Liberation Sans", "DejaVu Sans") if n in _names)

plt.rcParams.update({
    "font.family": FONT, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
})

# ---- figure layout text and line sizes ------------------------------------------------
FIG_TEXT, FIG_TITLE, FIG_LETTER, FIG_LINE, FIG_TICK = 42.0, 44.0, 60.0, 5.0, 14.2


class Layout:
    """Sizes for a panel whose layout factor is k."""

    def __init__(self, k):
        self.k = k
        self.text = FIG_TEXT / k
        self.title = FIG_TITLE / k
        self.letter = FIG_LETTER / k
        self.line = FIG_LINE / k
        self.tick = FIG_TICK / k

    def axes(self, ax, x=True, y=True):
        """Axis lines, ticks and tick labels in figure layout proportions."""
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(self.line)
        ax.spines["left"].set_visible(y)
        ax.spines["bottom"].set_visible(x)
        ax.tick_params(width=self.line, length=self.tick, labelsize=self.text,
                       pad=self.tick * 0.5)
        if not x:
            ax.set_xticks([])
        if not y:
            ax.set_yticks([])

    def title_letter(self, fig, ax, letter, title, dy=0.02):
        """Panel letter and left-aligned title above the axes, as in the figure layout."""
        ax.set_title(title, loc="left", fontsize=self.title, pad=self.text * 1.4)
        fig.text(0.0, 1.0 + dy, letter, fontsize=self.letter, fontweight="bold",
                 ha="left", va="bottom", transform=ax.transAxes)


# ---- palettes ----------------------------------------------------------------
STAGES = ["GW6", "GW8", "GW10", "GW12", "GW16"]
STAGE_C = {"GW6": "#4c78a8", "GW8": "#59a14f", "GW10": "#9c755f",
           "GW12": "#e15759", "GW16": "#f2a53c"}
TISSUES = ["kidney", "intestine", "cerebellum", "retina", "finch_retina"]
TISSUE_C = {"kidney": "#2d7f5e", "intestine": "#d1823a", "cerebellum": "#6b3a75",
            "retina": "#1f5f82", "finch_retina": "#6aaed6"}


# ---- marks images placed in the figure -------------------------------------------
def px_figure(w_px, h_px, dpi=600):
    """Figure of exactly w_px x h_px pixels at dpi (save with bbox_inches=None)."""
    return plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)


def px_axes(fig, x0, y0_top, w, h):
    """Axes placed in pixel units of the figure, measured from the top-left corner."""
    W = fig.get_figwidth() * fig.dpi
    H = fig.get_figheight() * fig.dpi
    ax = fig.add_axes([x0 / W, 1 - (y0_top + h) / H, w / W, h / H])
    ax.set_axis_off()
    return ax


def axes_in_crop(w_px, h_px, pad_px=None, expand=None):
    """Pixel box of the axes inside a panel image that was cropped around its axes.

    Two croppings occur in the panel images: a fixed pad of pad_px pixels on every side,
    or the axes box expanded by a factor (1 + expand) about its centre.
    Returns (x0, y0_top, w, h).
    """
    if pad_px is not None:
        return pad_px, pad_px, w_px - 2 * pad_px, h_px - 2 * pad_px
    w = w_px / (1 + expand); h = h_px / (1 + expand)
    return (w_px - w) / 2, (h_px - h) / 2, w, h


def colourbar_png(path, cmap, w, h, frame, inset, frame_rgb=(0, 0, 0)):
    """A vertical colour bar written pixel by pixel (high values at the top).

    w, h      image size in pixels
    frame     ((x0, x1), (y0, y1)) outer edges of the frame, inclusive
    inset     frame thickness in pixels
    """
    import numpy as np
    from PIL import Image
    cm = plt.get_cmap(cmap)
    img = np.full((h, w, 3), 255, np.uint8)
    (x0, x1), (y0, y1) = frame
    img[y0:y1 + 1, x0:x1 + 1] = frame_rgb
    ix0, ix1, iy0, iy1 = x0 + inset, x1 - inset, y0 + inset, y1 - inset
    n = iy1 - iy0 + 1
    vals = 1 - (np.arange(n) + 0.5) / n
    rgb = (np.array([cm(v)[:3] for v in vals]) * 255).round().astype(np.uint8)
    img[iy0:iy1 + 1, ix0:ix1 + 1] = rgb[:, None, :]
    Image.fromarray(img).save(path)


# per-key marker scale, row spacing and handle length, fitted to the key images in the figure
KEY_STYLE = {
    "Fig1_D_key": dict(scale=1.0, spacing=0.5, handlelength=2.0),
    "FigS1_C_key": dict(scale=0.8, spacing=0.7, handlelength=2.4),
    "FigS1_D_key": dict(scale=1.4, spacing=1.2, handlelength=3.2),
    "FigS1_E_key": dict(scale=2.0, spacing=2.1, handlelength=4.5),
    "Fig3_A_key": dict(scale=1.2, spacing=0.3, handlelength=1.6),
    "FigS3_A_key": dict(scale=2.0, spacing=0.9, handlelength=2.4),
    "FigS3_D_key": dict(scale=1.2, spacing=0.3, handlelength=1.6),
}


def legend_key(ax, path, scale=1.0, spacing=0.9, handlelength=1.6):
    """Write the swatches of ax's legend, without text, as a separate key image."""
    st = KEY_STYLE.get(Path(str(path)).stem)
    if st:
        scale, spacing, handlelength = st["scale"], st["spacing"], st["handlelength"]
    import copy
    lg = ax.get_legend()
    hs = list(getattr(lg, "legend_handles", None) or getattr(lg, "legendHandles", []))
    fig, k = plt.subplots(figsize=(0.6, 0.3 * len(hs)))
    hs2 = []
    for h in hs:
        h2 = copy.copy(h)
        if hasattr(h2, "set_markersize"):
            h2.set_markersize(h.get_markersize() * scale)
            h2.set_linewidth(h.get_linewidth() * scale)
        hs2.append(h2)
    k.legend(handles=hs2, labels=[""] * len(hs2), frameon=False, loc="center",
             handlelength=handlelength, labelspacing=spacing, borderpad=0,
             fontsize=10)       # spacing is in font-size units; fixed so it does not
                                # depend on the calling script's rcParams
    k.axis("off")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# ---- output -----------------------------------------------------------------
def marks_dir(figname):
    d = OUT / figname / "marks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_marks(fig, figname, name, dpi=600, **kw):
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(marks_dir(figname) / f"{name}.png", dpi=dpi, **kw)
    plt.close(fig)


def save_full(fig, figname, panel, dpi=600):
    d = OUT / figname
    d.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(d / f"{figname}_{panel}{ext}", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
    plt.close(fig)
