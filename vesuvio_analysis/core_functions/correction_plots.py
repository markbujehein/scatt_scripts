"""Automated comparative visualization for MS and Gamma corrections.

Generates publication-grade **Correction Dashboard** plots in both
Time-of-Flight (TOF) and y-space for the VESUVIO DINS analysis pipeline.

Three scenarios are handled automatically based on the correction flags set
in the initial-conditions object:

* **Scenario A** (``MSCorrectionFlag=True``, ``GammaCorrectionFlag=False``):
  plots ``[Uncorrected, C_{MS}(t), Corrected]``.
* **Scenario B** (``MSCorrectionFlag=False``, ``GammaCorrectionFlag=True``):
  plots ``[Uncorrected, C_{\\gamma}(t), Corrected]``.
* **Scenario C** (``MSCorrectionFlag=True``, ``GammaCorrectionFlag=True``):
  plots ``[Uncorrected, C_{MS}(t), C_{\\gamma}(t), Corrected]``.

The **Forensic Legend** in every figure annotates each correction component
with its total integrated area as a percentage of the uncorrected signal
area, quantifying the correction's physical impact.

All figures are saved as both PDF (vector) and PNG to ``ic.figSavePath``
(set by ``ICHelpers.completeICFromInputs``).  The module is designed to be
fully importable and testable without the Mantid framework — workspace
extraction is delegated to the caller via duck-typed objects that expose
``extractX()``, ``extractY()``, and ``extractE()`` methods.

Typical usage (from ``run_script.py``)::

    from vesuvio_analysis.core_functions.correction_plots import (
        dispatch_correction_plots,
    )

    saved = dispatch_correction_plots(ic, mtd)

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from vesuvio_analysis.core_functions.plot_style import COLORBLIND_PALETTE, EXPERIMENTAL_STYLE, THEORETICAL_STYLE, figure_factory, set_thesis_style


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

#: Maps role → (x, y, err) where *err* may be ``None``.
CorrectionData = Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]

# Keys used in CorrectionData:
_KEY_UNCORRECTED = "uncorrected"
_KEY_CORRECTED = "corrected"
_KEY_MS = "ms_correction"
_KEY_GAMMA = "gamma_correction"


# ---------------------------------------------------------------------------
# Internal helpers — purely numeric, no Mantid dependency
# ---------------------------------------------------------------------------


def _get_scenario(ms_flag: bool, gc_flag: bool) -> str:
    """Return scenario label ``'A'``, ``'B'``, or ``'C'``.

    Args:
        ms_flag: ``True`` when multiple-scattering correction is active.
        gc_flag: ``True`` when gamma-background correction is active.

    Returns:
        ``'A'`` (MS only), ``'B'`` (gamma only), or ``'C'`` (both).

    Raises:
        ValueError: If both flags are ``False``.
    """
    if ms_flag and not gc_flag:
        return "A"
    if gc_flag and not ms_flag:
        return "B"
    if ms_flag and gc_flag:
        return "C"
    raise ValueError("At least one correction flag must be True.")


def _integrate_area(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the absolute integrated area with the trapezoidal rule.

    Finite values only — ``NaN`` and ``Inf`` are masked before integration.

    Args:
        x: Bin centres, shape ``(n,)``.
        y: Signal values, shape ``(n,)``.

    Returns:
        Non-negative scalar area.
    """
    finite = np.isfinite(y)
    if not np.any(finite):
        return 0.0
    # Use np.trapezoid (NumPy >= 2.0 required)
    return float(np.abs(np.trapezoid(y[finite], x[finite])))


def _area_fraction_pct(
    corr_x: np.ndarray,
    corr_y: np.ndarray,
    sig_x: np.ndarray,
    sig_y: np.ndarray,
) -> float:
    """Express the correction area as a percentage of the signal area.

    Args:
        corr_x: Bin centres of the correction spectrum.
        corr_y: Y values of the correction spectrum.
        sig_x: Bin centres of the uncorrected signal.
        sig_y: Y values of the uncorrected signal.

    Returns:
        ``100 * |area(correction)| / |area(signal)|``, or ``0.0`` if
        the signal area is zero.
    """
    sig_area = _integrate_area(sig_x, sig_y)
    if sig_area == 0.0:
        return 0.0
    corr_area = _integrate_area(corr_x, corr_y)
    return 100.0 * corr_area / sig_area


def _extract_ws_data(
    ws: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract summed x, y, and error arrays from a workspace-like object.

    Supports both histogram workspaces (``n_bins + 1`` x-points) and
    point-data workspaces (``n_bins`` x-points).  When multiple spectra
    are present they are summed; errors are combined in quadrature.

    Args:
        ws: Object that exposes ``extractX()``, ``extractY()``, and
            ``extractE()`` returning ``(n_spectra, n_bins[+1])``
            NumPy arrays.  Mantid ``MatrixWorkspace`` and the mock
            objects used in tests both satisfy this interface.

    Returns:
        ``(x, y, err)`` — 1-D NumPy arrays of length ``n_bins``.
    """
    raw_x = ws.extractX()
    raw_y = ws.extractY()
    raw_e = ws.extractE()

    # Sum across spectra (rows)
    y_sum = np.nansum(raw_y, axis=0)
    e_sum = np.sqrt(np.nansum(raw_e**2, axis=0))

    # Histogram → point data: take bin-centre x from the first row
    if raw_x.shape[1] == raw_y.shape[1] + 1:
        x = 0.5 * (raw_x[0, :-1] + raw_x[0, 1:])
    else:
        x = raw_x[0]

    return x, y_sum, e_sum

# The _build_theoretical_style function creates a shallow copy of THEORETICAL_STYLE 
# which could lead to unintended mutations if THEORETICAL_STYLE contains nested dictionaries. 
# While the current implementation appears safe given the flat dictionary structure, 
# consider using copy.deepcopy() or {**THEORETICAL_STYLE, 'linestyle': linestyle} 
# for clarity and robustness.
def _build_theoretical_style(linestyle: str) -> Dict[str, Any]:
    """Return theoretical style with an explicit linestyle override."""
    style = dict(THEORETICAL_STYLE)
    style["linestyle"] = linestyle
    return style


# ---------------------------------------------------------------------------
# Core plot functions (Mantid-free)
# ---------------------------------------------------------------------------


def _render_dashboard(
    data: CorrectionData,
    scenario: str,
    title: str,
    x_label: str,
    y_label: str,
    save_stem: Path,
) -> List[Path]:
    """Render and save a single correction dashboard figure.

    Applies ``set_thesis_style()`` and saves to both PDF and PNG.
    The **Forensic Legend** annotates each correction line with its
    area as a percentage of the uncorrected signal.

    Args:
        data: ``CorrectionData`` dict with keys ``'uncorrected'``,
            ``'corrected'``, and optionally ``'ms_correction'``
            and/or ``'gamma_correction'``.
        scenario: ``'A'``, ``'B'``, or ``'C'`` (controls which
            correction lines are drawn).
        title: Figure title string.
        x_label: X-axis label (may contain LaTeX).
        y_label: Y-axis label (may contain LaTeX).
        save_stem: ``Path`` without suffix; ``.pdf`` and ``.png``
            are appended automatically.

    Returns:
        List of saved ``Path`` objects ``[stem.pdf, stem.png]``.
    """
    set_thesis_style()
    fig, ax = figure_factory()

    col = COLORBLIND_PALETTE

    ux, uy, ue = data[_KEY_UNCORRECTED]
    cx, cy, ce = data[_KEY_CORRECTED]

    # --- Uncorrected (experimental scattering spectrum) ---
    ax.errorbar(ux, uy, ue, color=col[7], label="Uncorrected", **EXPERIMENTAL_STYLE)

    # --- MS correction ($C_{MS}(t)$) — theoretical correction term ---
    if scenario in ("A", "C") and _KEY_MS in data:
        mx, my, me = data[_KEY_MS]
        frac = _area_fraction_pct(mx, my, ux, uy)
        ax.plot(
            mx, my,
            color=col[0],
            label=rf"$C_{{MS}}$ [{frac:.1f}% of signal]",
            **_build_theoretical_style("--"),
        )

    # --- Gamma correction ($C_{\gamma}(t)$) — theoretical correction term ---
    if scenario in ("B", "C") and _KEY_GAMMA in data:
        gx, gy, ge = data[_KEY_GAMMA]
        frac = _area_fraction_pct(gx, gy, ux, uy)
        ax.plot(
            gx, gy,
            color=col[1],
            label=rf"$C_{{\gamma}}$ [{frac:.1f}% of signal]",
            **_build_theoretical_style(":"),
        )

    # --- Corrected (experimental scattering spectrum) ---
    ax.errorbar(cx, cy, ce, color=col[2], label="Corrected", **EXPERIMENTAL_STYLE)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    saved: List[Path] = []
    for suffix in (".pdf", ".png"):
        out = save_stem.with_suffix(suffix)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        saved.append(out)

    plt.close(fig)
    return saved


def plot_tof_correction_dashboard(
    ic_name: str,
    masses: np.ndarray,
    ms_flag: bool,
    gc_flag: bool,
    data: CorrectionData,
    fig_save_path: Path,
    iteration: int,
) -> List[Path]:
    """Generate a TOF-space correction dashboard and save to disk.

    Renders the full-spectrum comparison
    ``[Uncorrected, C_{MS}(t), C_{\\gamma}(t), Corrected]``
    (only the active correction terms are included) and saves two
    files: a vector PDF and a raster PNG.

    Args:
        ic_name: Base IC name (e.g. ``'thymol_10K_Gauss1D_FORWARD_'``).
        masses: Array of fitted masses (a.m.u.).
        ms_flag: ``True`` if the MS correction was applied.
        gc_flag: ``True`` if the gamma-background correction was applied.
        data: ``CorrectionData`` dict with workspace data.
        fig_save_path: Directory in which figures are saved.
        iteration: Final MS-iteration index used for file naming.

    Returns:
        List of saved ``Path`` objects (PDF + PNG), or empty list
        when no corrections are active.
    """
    if not (ms_flag or gc_flag):
        return []

    scenario = _get_scenario(ms_flag, gc_flag)
    stem_name = ic_name.rstrip("_") + f"_TOF_correction_iter{iteration}"
    save_stem = fig_save_path / stem_name
    title = f"TOF Correction Dashboard — {ic_name.rstrip('_')} (iter {iteration})"

    return _render_dashboard(
        data=data,
        scenario=scenario,
        title=title,
        x_label=r"TOF ($\mu$s)",
        y_label="Counts",
        save_stem=save_stem,
    )


def plot_yspace_correction_dashboard(
    ic_name: str,
    masses: np.ndarray,
    ms_flag: bool,
    gc_flag: bool,
    data: CorrectionData,
    fig_save_path: Path,
    iteration: int,
) -> List[Path]:
    """Generate a y-space correction dashboard and save to disk.

    Renders the full-spectrum comparison in y-space using data that
    has already been converted to y-space via
    $J(y) = M / (\\hbar q) \\cdot (E - E_{\\text{recoil}})$.

    Args:
        ic_name: Base IC name.
        masses: Array of fitted masses (a.m.u.).
        ms_flag: ``True`` if the MS correction was applied.
        gc_flag: ``True`` if the gamma-background correction was applied.
        data: ``CorrectionData`` dict in y-space units.
        fig_save_path: Directory in which figures are saved.
        iteration: Final MS-iteration index used for file naming.

    Returns:
        List of saved ``Path`` objects (PDF + PNG), or empty list
        when no corrections are active.
    """
    if not (ms_flag or gc_flag):
        return []

    scenario = _get_scenario(ms_flag, gc_flag)
    stem_name = ic_name.rstrip("_") + f"_yspace_correction_iter{iteration}"
    save_stem = fig_save_path / stem_name
    title = (
        r"$y$-space Correction Dashboard — "
        f"{ic_name.rstrip('_')} (iter {iteration})"
    )

    return _render_dashboard(
        data=data,
        scenario=scenario,
        title=title,
        x_label=r"$y$ ($\AA^{-1}$)",
        y_label=r"$J(y)$ (a.u.)",
        save_stem=save_stem,
    )


# ---------------------------------------------------------------------------
# Workspace extraction helpers (duck-typed — work with real or mock ADS)
# ---------------------------------------------------------------------------


def _build_correction_data_from_mtd(
    mtd: Any,
    uncorrected_ws: str,
    corrected_ws: str,
    ms_ws: Optional[str],
    gc_ws: Optional[str],
) -> Optional[CorrectionData]:
    """Build a ``CorrectionData`` dict by extracting from the Mantid ADS.

    Falls back to ``None`` if extraction raises any exception, so that
    the absence of optional correction workspaces never crashes the run.

    Args:
        mtd: Object supporting ``__contains__`` and ``__getitem__``.
        uncorrected_ws: Name of the iteration-0 (pre-correction) workspace.
        corrected_ws: Name of the final corrected workspace.
        ms_ws: Name of the ``_MulScattering`` workspace, or ``None``.
        gc_ws: Name of the ``_Gamma_Background`` workspace, or ``None``.

    Returns:
        Populated ``CorrectionData`` dict, or ``None`` on failure.
    """
    try:
        data: CorrectionData = {}
        data[_KEY_UNCORRECTED] = _extract_ws_data(mtd[uncorrected_ws])
        data[_KEY_CORRECTED] = _extract_ws_data(mtd[corrected_ws])
        if ms_ws is not None and ms_ws in mtd:
            data[_KEY_MS] = _extract_ws_data(mtd[ms_ws])
        if gc_ws is not None and gc_ws in mtd:
            data[_KEY_GAMMA] = _extract_ws_data(mtd[gc_ws])
        return data
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"correction_plots: failed to extract TOF workspace data: {exc}"
        )
        return None


def _build_yspace_data_from_mtd(
    mtd: Any,
    uncorrected_ws: str,
    corrected_ws: str,
    ms_ws: Optional[str],
    gc_ws: Optional[str],
    convert_fn: Callable[[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    mass: float,
) -> Optional[CorrectionData]:
    """Build y-space ``CorrectionData`` by converting TOF workspaces.

    Args:
        mtd: Mantid ADS.
        uncorrected_ws: Name of the pre-correction workspace.
        corrected_ws: Name of the final corrected workspace.
        ms_ws: MS-correction workspace name, or ``None``.
        gc_ws: Gamma-correction workspace name, or ``None``.
        convert_fn: Callable ``(ws_name, mass) -> (x, y, err)``
            that converts a named TOF workspace to y-space and returns
            1-D NumPy arrays (summed across spectra).
        mass: Target mass (a.m.u.) for the y-space conversion.

    Returns:
        Populated ``CorrectionData`` dict in y-space units, or
        ``None`` on failure.
    """
    try:
        data: CorrectionData = {}
        data[_KEY_UNCORRECTED] = convert_fn(uncorrected_ws, mass)
        data[_KEY_CORRECTED] = convert_fn(corrected_ws, mass)
        if ms_ws is not None and ms_ws in mtd:
            data[_KEY_MS] = convert_fn(ms_ws, mass)
        if gc_ws is not None and gc_ws in mtd:
            data[_KEY_GAMMA] = convert_fn(gc_ws, mass)
        return data
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"correction_plots: y-space conversion failed: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def dispatch_correction_plots(
    ic: Any,
    mtd: Any,
    convert_to_yspace_fn: Optional[
        Callable[[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = None,
) -> List[Path]:
    """Dispatch TOF-space and y-space correction dashboard plots.

    Called after ``iterativeFitForDataReduction`` completes.  Checks
    ``ic.MSCorrectionFlag`` and ``ic.GammaCorrectionFlag``; skips
    silently when neither is ``True`` or when ``ic.noOfMSIterations``
    is zero.

    Correction workspace names are resolved from ``ic.name`` following
    the naming convention established by ``createMulScatWorkspaces``
    and ``createWorkspacesForGammaCorrection``:

    * MS correction  : ``ic.name + "_NCPMasked_MulScattering"``
    * Gamma correction: ``ic.name + "_NCPMasked_Gamma_Background"``
    * Pre-correction  : ``ic.name + "0"``
    * Post-correction : ``ic.name + str(ic.noOfMSIterations)``

    Figures are saved to ``ic.figSavePath`` in both PDF and PNG format.
    All generated paths are printed to stdout and returned.

    The MS correction workspace (``{ic.name}_NCPMasked_MulScattering``)
    and gamma-background workspace (``{ic.name}_NCPMasked_Gamma_Background``)
    persist in the Mantid ADS after ``iterativeFitForDataReduction``
    returns — neither ``createMulScatWorkspaces`` nor
    ``createWorkspacesForGammaCorrection`` delete them.  This function
    relies on that persistence.  The caller is responsible for any
    subsequent cleanup.

    Args:
        ic: Completed ``BackwardInitialConditions`` or
            ``ForwardInitialConditions`` object.  Required attributes:
            ``MSCorrectionFlag``, ``GammaCorrectionFlag``,
            ``noOfMSIterations``, ``figSavePath``, ``name``,
            ``masses``.
        mtd: Mantid AnalysisDataService (or a compatible mock that
            supports ``__contains__`` and ``__getitem__``).
        convert_to_yspace_fn: Optional callable with signature
            ``(ws_name: str, mass: float) -> (x, y, err)``
            that converts a TOF workspace to y-space and returns 1-D
            NumPy arrays (summed across spectra).  When ``None``,
            y-space plots are skipped.

    Returns:
        Sorted list of all saved figure ``Path`` objects (PDF + PNG).
        Returns an empty list when no corrections are active or when
        the required workspaces are not found in the ADS.
    """
    ms_flag = bool(getattr(ic, "MSCorrectionFlag", False))
    gc_flag = bool(getattr(ic, "GammaCorrectionFlag", False))

    if not (ms_flag or gc_flag):
        return []

    n_iter = int(getattr(ic, "noOfMSIterations", 0))
    if n_iter == 0:
        return []

    fig_save_path: Path = ic.figSavePath
    ic_name: str = ic.name
    masses: np.ndarray = ic.masses
    mass0 = float(masses.flat[0])

    # Resolve workspace names
    ncpm_base = ic_name + "_NCPMasked"
    uncorrected_ws = ic_name + "0"
    corrected_ws = ic_name + str(n_iter)
    ms_ws = ncpm_base + "_MulScattering"
    gc_ws = ncpm_base + "_Gamma_Background"

    # Guard: required workspaces must be present
    if uncorrected_ws not in mtd or corrected_ws not in mtd:
        warnings.warn(
            f"correction_plots: required workspaces '{uncorrected_ws}' or "
            f"'{corrected_ws}' not found in ADS — skipping correction plots."
        )
        return []

    saved: List[Path] = []

    # ------------------------------------------------------------------ TOF --
    tof_data = _build_correction_data_from_mtd(
        mtd,
        uncorrected_ws=uncorrected_ws,
        corrected_ws=corrected_ws,
        ms_ws=ms_ws if ms_flag else None,
        gc_ws=gc_ws if gc_flag else None,
    )
    if tof_data is not None:
        paths = plot_tof_correction_dashboard(
            ic_name=ic_name,
            masses=masses,
            ms_flag=ms_flag,
            gc_flag=gc_flag,
            data=tof_data,
            fig_save_path=fig_save_path,
            iteration=n_iter,
        )
        saved.extend(paths)

    # --------------------------------------------------------------- y-space --
    if convert_to_yspace_fn is not None:
        yspace_data = _build_yspace_data_from_mtd(
            mtd,
            uncorrected_ws=uncorrected_ws,
            corrected_ws=corrected_ws,
            ms_ws=ms_ws if ms_flag else None,
            gc_ws=gc_ws if gc_flag else None,
            convert_fn=convert_to_yspace_fn,
            mass=mass0,
        )
        if yspace_data is not None:
            paths = plot_yspace_correction_dashboard(
                ic_name=ic_name,
                masses=masses,
                ms_flag=ms_flag,
                gc_flag=gc_flag,
                data=yspace_data,
                fig_save_path=fig_save_path,
                iteration=n_iter,
            )
            saved.extend(paths)

    for p in saved:
        print(f"[CorrectionPlots] Saved: {p}")

    return sorted(saved)
