"""Thesis-standardized Matplotlib style for VESUVIO analysis plots.

Targets an A4 document with 2.0 cm margins → 17.0 cm text width.
The "1:1 Rule": a 12 pt font in the saved figure equals 12 pt in the
LaTeX document when the figure is included at its natural width.

Typical usage::

    from vesuvio_analysis.core_functions.plot_style import set_thesis_style, figure_factory

    set_thesis_style()           # apply rcParams once at module level
    fig, ax = figure_factory()   # full-width (17 cm) figure with 4:3 ratio
    fig, ax = figure_factory("half_width")  # half-width (8.25 cm) figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Any, Tuple


# ---------------------------------------------------------------------------
# Physical layout constants (A4 + 2 cm margins)
# ---------------------------------------------------------------------------
_CM_PER_INCH: float = 2.54
FULL_WIDTH_CM: float = 17.0   # 17.0 cm text width
HALF_WIDTH_CM: float = 8.25   # ~half column width with gutter

# Publication-standard, color-blind friendly palette (Wong 2011 / Seaborn)
COLORBLIND_PALETTE: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky-blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def cm_to_inches(cm: float) -> float:
    """Convert centimetres to inches."""
    return cm / _CM_PER_INCH


def set_thesis_style(width_cm: float = FULL_WIDTH_CM, fraction: float = 1.0) -> None:
    """Apply thesis-compliant rcParams to Matplotlib globally.

    Sets fonts to 12 pt so that a figure saved at ``width_cm * fraction``
    inches wide and included in LaTeX at ``\\linewidth`` renders with
    12 pt body text — no re-scaling needed.

    Args:
        width_cm: Target text width in centimetres (default 17.0 cm).
        fraction: Fraction of ``width_cm`` the figure occupies (default 1.0).
    """
    width_in = cm_to_inches(width_cm * fraction)

    plt.rcParams.update({
        # --- Figure geometry ---
        "figure.figsize": (width_in, width_in / (4 / 3)),  # 4:3 default

        # --- Typography (1:1 rule) ---
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 12,

        # --- Font family ---
        # DejaVu Serif is always available; switch to 'text.usetex: True'
        # for full Computer Modern if a LaTeX installation is present.
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],

        # --- Lines & markers ---
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "patch.linewidth": 1.0,

        # --- Axes ---
        "axes.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=COLORBLIND_PALETTE),

        # --- Ticks ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        # --- Layout ---
        "figure.constrained_layout.use": False,  # tight_layout used per-figure
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # --- PDF backend ---
        "pdf.fonttype": 42,   # embed fonts as Type 42 (TrueType) in PDF
        "ps.fonttype": 42,
    })


def figure_factory(
    size: str = "full_width",
    aspect_ratio: float | None = None,
    **subplots_kwargs: Any,
) -> Tuple[plt.Figure, Any]:
    """Create a thesis-sized Matplotlib figure.

    Args:
        size: ``"full_width"`` (17.0 cm) or ``"half_width"`` (8.25 cm).
        aspect_ratio: Height-to-width ratio.  Defaults to the golden ratio
            inverse (``1 / 1.618 ≈ 0.618``) giving a wide landscape figure.
            Pass ``3/4`` for the classic 4:3 portrait-ish proportion.
        **subplots_kwargs: Additional keyword arguments forwarded to
            ``plt.subplots()``.  ``figsize`` is set automatically and
            should not be passed.

    Returns:
        ``(fig, axes)`` — the Matplotlib Figure and its Axes (or array of
        Axes when ``nrows`` / ``ncols`` > 1).

    Examples::

        fig, ax = figure_factory()
        fig, axs = figure_factory("half_width", nrows=2)
        fig, ax  = figure_factory("full_width", aspect_ratio=3/4,
                                  subplot_kw={"projection": "mantid"})
    """
    _width_map = {
        "full_width": FULL_WIDTH_CM,
        "half_width": HALF_WIDTH_CM,
    }
    if size not in _width_map:
        raise ValueError(f"size must be 'full_width' or 'half_width', got {size!r}")

    width_in = cm_to_inches(_width_map[size])

    if aspect_ratio is None:
        aspect_ratio = 1.0 / 1.618  # golden ratio (landscape)

    height_in = width_in * aspect_ratio
    figsize = (width_in, height_in)

    fig, axes = plt.subplots(figsize=figsize, **subplots_kwargs)
    return fig, axes
