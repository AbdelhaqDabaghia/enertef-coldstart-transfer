"""
figstyle.py -- shared publication style for the manuscript figures.

Follows the conventions of A. Churkin's BeautifulFigures
(https://github.com/AndreyChurkin/BeautifulFigures), which the project has
adopted as its figure standard:

  * vector output (PDF + SVG); PNG only as a convenience preview
  * monospace type at 20 pt -- at 0.9 column width in an IEEE template this
    renders at about 8 pt, matching the caption text
  * two-level grid, major at alpha 0.25 and minor at 0.15, drawn BELOW the data
  * few colours, used deliberately, each with a darker edge tone
  * explicit axis limits rather than whatever the data happened to span
  * no tight_layout, which silently overrides a fixed aspect ratio

Import and call `apply()` once before plotting, then `save(fig, name)`.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FIGDIR = os.path.join("paper", "v4", "figures")

# Palette from the reference, extended with one warm tone for a fourth series.
# Fills first, matching darker edges second.
FILL = {"finetune": "#9671bd", "mas": "#7e7e7e", "replay": "#77b5b6",
        "reference": "#c8a26a", "leaky": "#9671bd", "causal": "#77b5b6"}
EDGE = {"finetune": "#6a408d", "mas": "#4e4e4e", "replay": "#378d94",
        "reference": "#8c6a34", "leaky": "#6a408d", "causal": "#378d94"}
NEUTRAL = "#8a8a8a"

LABEL = {"B3_warm": "Fine-tune", "MAS": "Output-sensitivity",
         "B5_replay": "Bounded rehearsal"}
KEY = {"B3_warm": "finetune", "MAS": "mas", "B5_replay": "replay"}


def apply(size: int = 20) -> None:
    """Set the global rcParams. Falls back if Courier New is unavailable."""
    have = {f.name for f in fm.fontManager.ttflist}
    family = "Courier New" if "Courier New" in have else "DejaVu Sans Mono"
    plt.rcParams.update({
        "font.family": family,
        "font.size": size,
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "figure.titlesize": size,
        "svg.fonttype": "none",       # keep text editable in the SVG
        "pdf.fonttype": 42,           # embed TrueType, not Type 3
        "axes.linewidth": 1.0,
    })
    return family


def grid(ax) -> None:
    """Two-level grid, drawn below the data."""
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)


def legend_above(ax, ncol: int, handles=None, labels=None, y: float = 1.10):
    """Legend centred above the axes, no frame -- keeps the data area clean."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    return ax.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


def limits(ax, xs, ys, pad: float = 0.10, square: bool = False) -> None:
    """Explicit limits with a proportional margin, rather than autoscale."""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    if square:
        r = max(dx, dy) * (1 + 2 * pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - r / 2, cx + r / 2)
        ax.set_ylim(cy - r / 2, cy + r / 2)
    else:
        ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
        ax.set_ylim(y0 - pad * dy, y1 + pad * dy)


def save(fig, name: str, outdir: str = FIGDIR) -> None:
    """Vector first. No tight_layout -- it overrides a fixed aspect ratio."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(outdir, "%s.%s" % (name, ext)),
                    bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "%s.png" % name), dpi=150,
                bbox_inches="tight")
    print("  wrote %s.{pdf,svg,png}" % os.path.join(outdir, name))
    plt.close(fig)
