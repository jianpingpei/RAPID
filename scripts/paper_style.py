"""Shared publication style for the paper figures.

All figures import this module so that fonts, line weights, colours, grid
density and panel labels stay consistent across the manuscript.
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"

# Colour-blind-safe palette, used consistently across every figure.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#B8B8B8"
EVENT_BAND = "#EDEDED"     # annotation window (injected load, slowdown)
DATA_BAND = "#D6E3EE"      # uncertainty band around a measured mean
GRID = "#DCDCDC"

# Typography. Figure text sits one tick above the manuscript body size so that
# it stays legible after the two-column reduction.
BASE_SIZE = 8.0
LABEL_SIZE = 8.0
TICK_SIZE = 7.2
LEGEND_SIZE = 7.2
PANEL_LABEL_SIZE = 8.6

LINE_WIDTH = 1.4
THIN_WIDTH = 1.0


def configure() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": BASE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.linewidth": 0.7,
        "axes.labelpad": 2.5,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": 3.2,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axis(axis: Axes, *, grid_axis: Literal["x", "y", "both"] = "y") -> None:
    """Remove the top and right spines and add one light grid direction."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.5, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(width=0.7, length=3.0, pad=2)


def panel_label(axis: Axes, text: str, *, x: float = -0.02,
                y: float = 1.06) -> None:
    """Place an (a)/(b)/... label above the upper-left corner of a panel."""
    axis.text(x, y, text, transform=axis.transAxes, fontsize=PANEL_LABEL_SIZE,
              fontweight="bold", ha="right", va="bottom")


def event_band(axis: Axes, lower: float, upper: float) -> None:
    """Shade an interval in which an external load or event is active."""
    axis.axvspan(lower, upper, color=EVENT_BAND, zorder=0)


def save(figure: Figure, name: str, output_dir: Path | None = None) -> None:
    """Write both the vector PDF used by LaTeX and a 300-dpi raster preview."""
    directory = Path(output_dir) if output_dir is not None else FIG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.pdf")
    figure.savefig(directory / f"{name}.png", dpi=300)
    print(f"wrote {directory / (name + '.pdf')} and .png")


def gaussian(values, window: int = 5):
    """Light Gaussian smoothing over a short window, edge-padded in place."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    if window <= 1 or array.size < window:
        return array
    half = window // 2
    positions = np.arange(window) - half
    weights = np.exp(-0.5 * (positions / 1.0) ** 2)
    weights /= weights.sum()
    return np.convolve(np.pad(array, half, mode="edge"), weights, mode="valid")


def rolling_mean(values, window: int):
    """Centred moving average that keeps the series length and end points.

    Per-step controller traces oscillate between the two draft levels that the
    controller alternates over, which reads as visual noise at print size.
    Averaging a short window exposes the trend; the caption states the window.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    if window <= 1 or array.size < window:
        return array
    half = window // 2
    padded = np.pad(array, (half, half), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")
