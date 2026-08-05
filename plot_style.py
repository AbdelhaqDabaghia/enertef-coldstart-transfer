"""plot_style.py -- shared IEEE publication matplotlib style for the coldstart
figures.

STYLE ONLY. This module sets rcParams (fonts, sizes, grid/spines, vector PDF
output with embedded fonts) so every figure in analysis.py looks consistent and
prints cleanly in a two-column IEEE paper. It contains NO data, NO colour<->
condition mapping (that stays in analysis.py's `C`), and NO plotting logic.

Import for its side effect (applies rcParams at import time) and for the two
IEEE column-width constants:

    import plot_style
    from plot_style import COL_WIDTH, WIDE_WIDTH
"""
import matplotlib

# Headless/vector rendering; must be set before pyplot picks a backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEEtran two-column geometry (inches). A single text column is ~3.5 in; the
# full text width (for figure* / two-column-spanning plots) is ~7.16 in.
COL_WIDTH = 3.5
WIDE_WIDTH = 7.16

plt.rcParams.update({
    # --- vector output: embed TrueType (fonttype 42), never bitmap the PDF ---
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 150,
    "savefig.dpi": 600,          # only affects any raster fallback; PDF stays vector

    # --- serif / Times-like body to match IEEEtran text; STIX for math ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L",
                   "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    # --- default single-column canvas ---
    "figure.figsize": (COL_WIDTH, COL_WIDTH * 0.75),

    # --- consistent, print-legible sizes across all figures ---
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    # --- recessive grid + de-boxed axes (thin, light, behind the data) ---
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "lines.linewidth": 1.4,
    "legend.frameon": False,
})
