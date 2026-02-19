"""Nanoscience Explorer — VESUVIO Data Dashboard.

A Streamlit application for visualising the full-stack VESUVIO DINS
analysis output (L0–L3) across TOF, Q, and y-space domains.

Data is loaded directly from compressed ``.npz`` stream files produced
by :class:`~vesuvio_analysis.core_functions.stream_manager.StreamManager`,
so no physics recalculations are performed.  Pre-serialised coordinate
arrays (``dataX``, ``q``, ``y``) found inside the stream file are used
directly as plot axes.

**Air-gapped / local-first deployment:** the companion
``dashboard/.streamlit/config.toml`` binds the server to ``127.0.0.1``
and disables telemetry and external-IP lookup (``checkip.amazonaws.com``).

Run with::

    streamlit run dashboard/result_viewer.py

or from the repository root::

    python -m streamlit run dashboard/result_viewer.py
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Repository root on sys.path so project imports work regardless of how
# Streamlit is launched.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vesuvio_analysis.core_functions.stream_manager import StreamManager  # noqa: E402
from vesuvio_analysis.core_functions.plot_style import (  # noqa: E402
    COLORBLIND_PALETTE,
    FULL_WIDTH_CM,
    cm_to_inches,
)

# ---------------------------------------------------------------------------
# Publication-style constants translated to Plotly equivalents
# (mirrors the Matplotlib rcParams set by plot_style.set_thesis_style)
# ---------------------------------------------------------------------------
_DPI_SCREEN: int = 96
_PLOT_WIDTH_PX: int = int(cm_to_inches(FULL_WIDTH_CM) * _DPI_SCREEN)  # ≈ 643 px
_FONT_SIZE_PT: int = 12
_FONT_FAMILY: str = "DejaVu Serif, Times New Roman, serif"
_TICK_FONT_SIZE: int = 10
_LEGEND_FONT_SIZE: int = 10

# Default location for serialised stream files (relative to repo root)
_DEFAULT_OUTPUT_DIR: Path = _REPO_ROOT / "outputs"

# Human-readable labels for each DataLevel bucket
_LEVEL_DISPLAY: Dict[str, str] = {
    "L0": "L0 — Raw",
    "L1": "L1 — Correction Components",
    "L2": "L2 — Intermediate Corrected",
    "L3": "L3 — Final Physics",
}

# Plotly dash style per data level
_LEVEL_DASH: Dict[str, str] = {
    "L0": "solid",
    "L1": "dash",
    "L2": "dot",
    "L3": "solid",
}

# Approximate atomic symbols for common DINS masses (in atomic mass units)
_MASS_SYMBOL: Dict[float, str] = {
    1.008: "H",
    2.016: "D",
    12.0: "C",
    14.0: "N",
    16.0: "O",
    26.982: "Al",   # aluminium canister
    32.0: "S",
    40.0: "Ar",
    56.0: "Fe",
    137.0: "Ba",
}


# ---------------------------------------------------------------------------
# Plotly layout factory
# ---------------------------------------------------------------------------

def _pub_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    **extra,
) -> go.Layout:
    """Return a Plotly :class:`~plotly.graph_objects.Layout` configured for
    publication-grade output (17 cm width, 12 pt serif fonts, white background).
    """
    return go.Layout(
        title=dict(
            text=title,
            font=dict(size=_FONT_SIZE_PT, family=_FONT_FAMILY),
        ),
        font=dict(family=_FONT_FAMILY, size=_FONT_SIZE_PT),
        xaxis=dict(
            title=xaxis_title,
            showline=True,
            linewidth=0.8,
            linecolor="black",
            mirror=True,
            ticks="inside",
            tickfont=dict(size=_TICK_FONT_SIZE),
        ),
        yaxis=dict(
            title=yaxis_title,
            showline=True,
            linewidth=0.8,
            linecolor="black",
            mirror=True,
            ticks="inside",
            tickfont=dict(size=_TICK_FONT_SIZE),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(font=dict(size=_LEGEND_FONT_SIZE)),
        width=_PLOT_WIDTH_PX,
        **extra,
    )


# ---------------------------------------------------------------------------
# Stream-parsing helpers
# ---------------------------------------------------------------------------

def _split_keys(data: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
    """Bucket NPZ keys by top-level DataLevel prefix.

    Returns a dict with keys ``"L0"``, ``"L1"``, ``"L2"``, ``"L3"``, and
    ``"metadata"``, each mapping to the list of matching NPZ keys.
    """
    buckets: Dict[str, List[str]] = {
        "L0": [], "L1": [], "L2": [], "L3": [], "metadata": [],
    }
    for key in data:
        if key.startswith("L0_"):
            buckets["L0"].append(key)
        elif key.startswith("L1_"):
            buckets["L1"].append(key)
        elif key.startswith("L2_"):
            buckets["L2"].append(key)
        elif key.startswith("L3_"):
            buckets["L3"].append(key)
        elif key.startswith("metadata."):
            buckets["metadata"].append(key)
    return buckets


def _detect_masses(data: Dict[str, np.ndarray]) -> List[int]:
    """Extract unique mass indices from hierarchical stream keys.

    Scans for patterns such as ``.mass0.`` or ``.mass12`` and returns a
    sorted list of the integer indices found.
    """
    indices: set[int] = set()
    pattern = re.compile(r"\.mass(\d+)(?:\.|$)")
    for key in data:
        for match in pattern.finditer(key):
            indices.add(int(match.group(1)))
    return sorted(indices)


def _detect_detectors(data: Dict[str, np.ndarray]) -> List[int]:
    """Infer available detector count from 2-D array shapes.

    Prefers L0 arrays; falls back to any 2-D array if no L0 data is
    present.  Returns ``[0]`` when only 1-D data is found.
    """
    # Prefer L0 arrays which have shape (n_detectors, n_bins)
    for key in sorted(data):
        if key.startswith("L0_") and data[key].ndim == 2:
            return list(range(data[key].shape[0]))
    # Fall back to any 2-D array
    for key in sorted(data):
        arr = data[key]
        if arr.ndim == 2 and arr.shape[0] > 1:
            return list(range(arr.shape[0]))
    return [0]


def _available_levels(buckets: Dict[str, List[str]]) -> List[str]:
    """Return level labels (e.g. ``'L0'``) that have at least one key."""
    return [lvl for lvl in ("L0", "L1", "L2", "L3") if buckets[lvl]]


def _mass_label(index: int, masses_meta: Optional[np.ndarray]) -> str:
    """Return a human-readable label for mass *index*.

    Examples: ``"H (1.008 u)"``, ``"C (12.000 u)"``, ``"Mass 3"``.
    """
    if masses_meta is not None and index < len(masses_meta):
        m = float(masses_meta[index])
        symbol = _MASS_SYMBOL.get(round(m, 3), f"M{index}")
        return f"{symbol} ({m:.3f} u)"
    return f"Mass {index}"


def _resolve_x_axis(
    data: Dict[str, np.ndarray],
    domain: str,
    det_idx: int,
    n_bins: int,
) -> np.ndarray:
    """Return the best available x-axis array for *domain* and *det_idx*.

    Looks for pre-serialised coordinate arrays in order of preference:

    * TOF: ``L0_raw.tof.dataX`` (2-D per-detector or 1-D shared)
    * Q:   ``L3_final.q.q``, ``L3_final.q.x``, ``metadata.q_x``
    * y:   ``L3_final.y.y``, ``L3_final.y.x``, ``metadata.y_x``

    Falls back to ``np.arange(n_bins)`` if nothing is found.
    """
    if domain == "tof":
        candidates = ("L0_raw.tof.dataX",)
    elif domain == "q":
        candidates = ("L3_final.q.q", "L3_final.q.x", "metadata.q_x")
    else:  # y
        candidates = ("L3_final.y.y", "L3_final.y.x", "metadata.y_x")

    for key in candidates:
        if key in data:
            arr = data[key]
            if arr.ndim == 2 and det_idx < arr.shape[0]:
                return arr[det_idx]
            if arr.ndim == 1:
                return arr
    return np.arange(n_bins, dtype=float)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_tof_figure(
    data: Dict[str, np.ndarray],
    selected_detectors: List[int],
    show_corrections: bool,
    show_optimizer_compare: bool,
    iteration: int,
) -> go.Figure:
    """Build the TOF-domain Plotly figure.

    Layers (bottom to top):

    1. **L0** raw detector counts (solid lines).
    2. **L1** MS and Gamma correction shaded bands (orange / green, if
       *show_corrections* is ``True`` and L1 data is present).
    3. **L2** intermediate corrected signal (dotted lines, if present).
    4. **L3** NCP total fit (dashed lines, if present).
    5. **Optimizer comparison** — iMinuit vs Scipy NCP overlays (if
       *show_optimizer_compare* is ``True`` and the keys exist).
    """
    palette = COLORBLIND_PALETTE
    fig = go.Figure(layout=_pub_layout(
        "TOF Domain", "Time-of-Flight (μs)", "Counts",
    ))

    raw_y: Optional[np.ndarray] = data.get("L0_raw.tof.dataY")
    if raw_y is None:
        fig.add_annotation(
            text="No L0 raw data found in this stream file.",
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=_FONT_SIZE_PT),
        )
        return fig

    # ---- L0 raw data --------------------------------------------------------
    for det_idx in selected_detectors:
        if det_idx >= raw_y.shape[0]:
            continue
        y = raw_y[det_idx]
        x = _resolve_x_axis(data, "tof", det_idx, len(y))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            name=f"Det {det_idx} — Raw (L0)",
            line=dict(color=palette[det_idx % len(palette)], width=1.5, dash="solid"),
            legendgroup=f"det{det_idx}",
        ))

    # ---- L1 correction shaded bands ----------------------------------------
    if show_corrections:
        _add_correction_bands(fig, data, raw_y, selected_detectors, iteration, palette)

    # ---- L2 intermediate corrected signal -----------------------------------
    l2_key = f"L2_intermediate.tof.iter{iteration}.corrected"
    if l2_key in data:
        l2_arr = data[l2_key]
        for det_idx in selected_detectors:
            if det_idx >= l2_arr.shape[0]:
                continue
            y_l2 = l2_arr[det_idx]
            x = _resolve_x_axis(data, "tof", det_idx, len(y_l2))
            fig.add_trace(go.Scatter(
                x=x, y=y_l2, mode="lines",
                name=f"Det {det_idx} — Corrected (L2)",
                line=dict(
                    color=palette[det_idx % len(palette)],
                    width=1.5, dash="dot",
                ),
                legendgroup=f"det{det_idx}_l2",
            ))

    # ---- L3 NCP total fit ---------------------------------------------------
    ncp_key = f"L3_final.tof.iter{iteration}.ncp_total"
    if ncp_key in data:
        ncp_arr = data[ncp_key]
        for det_idx in selected_detectors:
            if det_idx >= ncp_arr.shape[0]:
                continue
            y_ncp = ncp_arr[det_idx]
            x = _resolve_x_axis(data, "tof", det_idx, len(y_ncp))
            fig.add_trace(go.Scatter(
                x=x, y=y_ncp, mode="lines",
                name=f"Det {det_idx} — NCP Fit (L3)",
                line=dict(
                    color=palette[(det_idx + 2) % len(palette)],
                    width=1.5, dash="dash",
                ),
                legendgroup=f"det{det_idx}_ncp",
            ))

    # ---- iMinuit–Scipy Numerical Agreement Check ----------------------------
    if show_optimizer_compare:
        for opt_name, opt_suffix, dash_style in (
            ("iMinuit", "iminuit", "longdash"),
            ("Scipy", "scipy", "dashdot"),
        ):
            key = f"L3_final.tof.iter{iteration}.ncp_{opt_suffix}"
            if key not in data:
                continue
            opt_arr = data[key]
            for det_idx in selected_detectors:
                if det_idx >= opt_arr.shape[0]:
                    continue
                y_opt = opt_arr[det_idx]
                x = _resolve_x_axis(data, "tof", det_idx, len(y_opt))
                fig.add_trace(go.Scatter(
                    x=x, y=y_opt, mode="lines",
                    name=f"Det {det_idx} — {opt_name}",
                    line=dict(width=1.0, dash=dash_style),
                ))

    return fig


def _add_correction_bands(
    fig: go.Figure,
    data: Dict[str, np.ndarray],
    raw_y: np.ndarray,
    selected_detectors: List[int],
    iteration: int,
    palette: List[str],
) -> None:
    """Add L1 MS and Gamma corrections as shaded bands to *fig*.

    Each shaded band spans from ``(raw − correction)`` up to ``raw``,
    visualising the magnitude of the correction that was subtracted from
    the raw TOF signal.
    """
    correction_specs = [
        (
            f"L1_corrections.tof.iter{iteration}.ms",
            "MS correction",
            "rgba(230,159,0,0.25)",   # Wong orange, 25% opacity
        ),
        (
            f"L1_corrections.tof.iter{iteration}.gamma",
            "Gamma correction",
            "rgba(0,158,115,0.25)",   # Wong green, 25% opacity
        ),
    ]
    for corr_key, corr_label, fill_color in correction_specs:
        if corr_key not in data:
            continue
        corr_arr = data[corr_key]
        first_shown = True
        for det_idx in selected_detectors:
            if det_idx >= corr_arr.shape[0] or det_idx >= raw_y.shape[0]:
                continue
            y_corr = corr_arr[det_idx]
            y_raw = raw_y[det_idx]
            x = _resolve_x_axis(data, "tof", det_idx, len(y_corr))
            lower = y_raw - y_corr
            # Closed polygon: top edge (raw) → bottom edge (raw - correction)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([y_raw, lower[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                mode="lines",
                name=f"{corr_label} (Det {det_idx})" if not first_shown else corr_label,
                legendgroup=f"corr_{corr_label}",
                showlegend=first_shown,
            ))
            first_shown = False


def _build_q_figure(
    data: Dict[str, np.ndarray],
    selected_detectors: List[int],
) -> go.Figure:
    """Build the Q-space domain Plotly figure.

    Uses pre-serialised Q coordinate arrays from L3 streams.
    No Q-transformation is recalculated here.
    """
    palette = COLORBLIND_PALETTE
    fig = go.Figure(layout=_pub_layout("Q Domain", "Q (Å⁻¹)", "S(Q)"))

    # Collect all signal keys under L3 q-domain (excluding coordinate arrays)
    coord_keys = {"L3_final.q.q", "L3_final.q.x", "L3_final.q.dataX"}
    q_signal_keys = [
        k for k in sorted(data)
        if k.startswith("L3_final.q.") and k not in coord_keys
    ]

    if not q_signal_keys:
        fig.add_annotation(
            text=(
                "No Q-domain data found in this stream.<br>"
                "Capture L3 arrays with domain=\"q\" to enable this view."
            ),
            showarrow=False,
            x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=_FONT_SIZE_PT),
        )
        return fig

    for color_idx, key in enumerate(q_signal_keys):
        arr = data[key]
        signal_name = key.split(".")[-1]
        if arr.ndim == 2:
            for det_idx in selected_detectors:
                if det_idx >= arr.shape[0]:
                    continue
                y = arr[det_idx]
                x = _resolve_x_axis(data, "q", det_idx, len(y))
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines",
                    name=f"Det {det_idx} — {signal_name}",
                    line=dict(
                        color=palette[(det_idx + color_idx) % len(palette)],
                        width=1.5,
                    ),
                ))
        elif arr.ndim == 1:
            x = _resolve_x_axis(data, "q", 0, len(arr))
            fig.add_trace(go.Scatter(
                x=x, y=arr, mode="lines",
                name=signal_name,
                line=dict(color=palette[color_idx % len(palette)], width=1.5),
            ))

    return fig


def _build_y_figure(
    data: Dict[str, np.ndarray],
    selected_mass_indices: List[int],
    masses_meta: Optional[np.ndarray],
    show_optimizer_compare: bool,
    show_recoil_markers: bool = False,
) -> go.Figure:
    """Build the y-space (J(y)) domain Plotly figure.

    Layers:

    1. **Averaged J(y)** with ±σ error band (from ``L3_final.y.joy_avg``
       and ``L3_final.y.joy_avg_err``).
    2. **Per-mass contributions** (Global Fit View) — individual atomic
       peaks as dashed sub-curves for each selected mass index.
    3. **Resolution** — normalised and overlaid as a dotted curve.
    4. **Recoil Shift Diagnostic** — vertical dashed lines at y=0 labelled
       by atomic symbol when *show_recoil_markers* is ``True``.  The
       expected recoil position for every mass in y-space is y=0 by
       construction; deviations in the empirical peak reveal a centering
       error.
    5. **Optimizer cross-validation** — iMinuit and Scipy fits overlaid
       when *show_optimizer_compare* is ``True``.

    The x-axis uses the pre-serialised y coordinate array from the stream;
    no y-transformation is recalculated.
    """
    palette = COLORBLIND_PALETTE
    fig = go.Figure(layout=_pub_layout(
        "y-Space Domain — J(y)", "y (Å⁻¹)", "J(y) (Å)",
    ))

    joy_avg: Optional[np.ndarray] = data.get("L3_final.y.joy_avg")
    joy_err: Optional[np.ndarray] = data.get("L3_final.y.joy_avg_err")

    if joy_avg is None:
        # Search for any available 1-D L3 y-domain signal to display
        fallback_keys = [
            k for k in sorted(data)
            if k.startswith("L3_final.y.") and data[k].ndim == 1
        ]
        if not fallback_keys:
            fig.add_annotation(
                text=(
                    "No y-space data found in this stream.<br>"
                    "Capture L3 arrays with domain=\"y\" to enable this view."
                ),
                showarrow=False,
                x=0.5, y=0.5, xref="paper", yref="paper",
                font=dict(size=_FONT_SIZE_PT),
            )
            return fig

        for color_idx, key in enumerate(fallback_keys):
            arr = data[key]
            x = _resolve_x_axis(data, "y", 0, len(arr))
            fig.add_trace(go.Scatter(
                x=x, y=arr, mode="lines",
                name=key.split(".")[-1],
                line=dict(color=palette[color_idx % len(palette)], width=1.5),
            ))
        return fig

    n = len(joy_avg)
    x_y = _resolve_x_axis(data, "y", 0, n)

    # ---- ±σ error band ------------------------------------------------------
    if joy_err is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_y, x_y[::-1]]),
            y=np.concatenate([joy_avg + joy_err, (joy_avg - joy_err)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,114,178,0.15)",
            line=dict(width=0),
            mode="lines",
            name="J(y) ±σ",
            showlegend=True,
        ))

    # ---- Averaged J(y) ------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=x_y, y=joy_avg, mode="lines",
        name="J(y) averaged",
        line=dict(color=palette[0], width=2.0, dash="solid"),
    ))

    # ---- Per-mass contributions (Global Fit View) ---------------------------
    for mass_idx in selected_mass_indices:
        for name_candidate in ("ncp", "joy", "fit", "profile"):
            key = f"L3_final.y.mass{mass_idx}.{name_candidate}"
            if key in data:
                arr = data[key]
                if arr.ndim != 1:
                    continue
                x = _resolve_x_axis(data, "y", 0, len(arr))
                label = _mass_label(mass_idx, masses_meta)
                fig.add_trace(go.Scatter(
                    x=x, y=arr, mode="lines",
                    name=label,
                    line=dict(
                        color=palette[(mass_idx + 1) % len(palette)],
                        width=1.5, dash="dash",
                    ),
                ))
                break

    # ---- Resolution (normalised overlay) ------------------------------------
    res_key = "L3_final.y.resolution"
    if res_key in data:
        res = data[res_key]
        if res.ndim == 1:
            x = _resolve_x_axis(data, "y", 0, len(res))
            # Scale to match J(y) amplitude for visual overlay
            peak = np.nanmax(np.abs(res))
            if peak > 0:
                scale = np.nanmax(np.abs(joy_avg)) / peak
                res_scaled = res * scale
            else:
                res_scaled = res
            fig.add_trace(go.Scatter(
                x=x, y=res_scaled, mode="lines",
                name="Resolution (scaled)",
                line=dict(color=palette[5], width=1.0, dash="dot"),
            ))

    # ---- Recoil Shift Diagnostic (y = 0 reference lines) -------------------
    if show_recoil_markers:
        _add_recoil_markers(fig, selected_mass_indices, masses_meta)

    # ---- iMinuit–Scipy Numerical Agreement Check ----------------------------
    if show_optimizer_compare:
        for opt_name, opt_suffix, dash_style in (
            ("iMinuit", "iminuit", "longdash"),
            ("Scipy", "scipy", "dashdot"),
        ):
            for key_candidate in (
                f"L3_final.y.{opt_suffix}",
                f"L3_final.y.ncp_{opt_suffix}",
                f"L3_final.y.fit_{opt_suffix}",
            ):
                if key_candidate in data:
                    arr = data[key_candidate]
                    if arr.ndim == 1:
                        x = _resolve_x_axis(data, "y", 0, len(arr))
                        fig.add_trace(go.Scatter(
                            x=x, y=arr, mode="lines",
                            name=f"{opt_name} fit",
                            line=dict(width=1.5, dash=dash_style),
                        ))
                    break

    return fig


# ---------------------------------------------------------------------------
# New forensic / diagnostic helpers (all pure functions, testable)
# ---------------------------------------------------------------------------

def _add_recoil_markers(
    fig: go.Figure,
    selected_mass_indices: List[int],
    masses_meta: Optional[np.ndarray],
) -> None:
    """Add vertical reference lines at y=0 for each selected mass.

    In the West y-scaling, the theoretical recoil position for every mass
    is y=0.  Deviations of the empirical NCP peak from this line indicate
    a centering or calibration issue.  Each mass gets a distinctly coloured
    dashed vertical line labelled with its atomic symbol.
    """
    palette = COLORBLIND_PALETTE
    if not selected_mass_indices:
        return
    for mass_idx in selected_mass_indices:
        label = _mass_label(mass_idx, masses_meta)
        color = palette[(mass_idx + 3) % len(palette)]
        fig.add_vline(
            x=0.0,
            line=dict(color=color, width=1.0, dash="dash"),
            annotation_text=f"y₀ {label}",
            annotation_position="top right",
            annotation_font=dict(size=9, color=color),
        )


def _compute_area_audit(
    data: Dict[str, np.ndarray],
    iteration: int = 0,
) -> List[Dict[str, object]]:
    """Compute integral-contribution statistics for the correction Forensic Table.

    For each correction type (MS, Gamma) and each detector, calculates:

    * **Integral** of the correction array (absolute area under the curve).
    * **Integral contribution (%)** relative to the raw signal integral for
      the same detector.

    Uses :func:`numpy.trapezoid` (NumPy ≥ 2.0) for all integration.

    Args:
        data: Stream dictionary from :func:`StreamManager.load`.
        iteration: MS/GC iteration index to use (default: 0).

    Returns:
        A list of dicts, each with keys:
        ``"correction"``, ``"detector"``, ``"integral"``,
        ``"raw_integral"``, ``"contribution_pct"``.
        Returns an empty list when neither L0 raw nor L1 correction data
        is available.
    """
    raw_key = "L0_raw.tof.dataY"
    ms_key = f"L1_corrections.tof.iter{iteration}.ms"
    gamma_key = f"L1_corrections.tof.iter{iteration}.gamma"

    if raw_key not in data:
        return []

    raw_y: np.ndarray = data[raw_key]
    rows: List[Dict[str, object]] = []

    for corr_name, corr_key in (("MS", ms_key), ("Gamma", gamma_key)):
        if corr_key not in data:
            continue
        corr_arr: np.ndarray = data[corr_key]
        n_det = min(raw_y.shape[0], corr_arr.shape[0])
        for det_idx in range(n_det):
            raw_integral = float(np.trapezoid(np.abs(raw_y[det_idx])))
            corr_integral = float(np.trapezoid(np.abs(corr_arr[det_idx])))
            pct = (
                100.0 * corr_integral / raw_integral
                if raw_integral > 0
                else 0.0
            )
            rows.append({
                "correction": corr_name,
                "detector": det_idx,
                "integral": round(corr_integral, 4),
                "raw_integral": round(raw_integral, 4),
                "contribution_pct": round(pct, 3),
            })

    return rows


def _build_residuals_figure(
    data: Dict[str, np.ndarray],
    selected_detectors: List[int],
    residual_modes: List[str],
    iteration: int = 0,
) -> go.Figure:
    """Build a multi-component residuals figure for forensic TOF inspection.

    Each *residual_mode* selects a different subtraction:

    * ``"Raw − MS"``      — raw signal after MS correction only.
    * ``"Raw − Gamma"``   — raw signal after Gamma correction only.
    * ``"Raw − MS − Gamma"`` — fully corrected signal.

    Missing correction arrays are silently skipped; only components that
    are present in *data* are plotted.

    Args:
        data: Stream dictionary.
        selected_detectors: Detector indices to plot.
        residual_modes: Subset of ``["Raw − MS", "Raw − Gamma",
            "Raw − MS − Gamma"]`` to include.
        iteration: MS/GC iteration to use (default 0).

    Returns:
        A :class:`plotly.graph_objects.Figure` with one trace per
        (detector, residual_mode) combination.
    """
    palette = COLORBLIND_PALETTE
    fig = go.Figure(layout=_pub_layout(
        "TOF Forensic Residuals", "Time-of-Flight (μs)", "Counts",
    ))

    raw_y: Optional[np.ndarray] = data.get("L0_raw.tof.dataY")
    if raw_y is None:
        fig.add_annotation(
            text="No L0 raw data found in this stream file.",
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=_FONT_SIZE_PT),
        )
        return fig

    ms_arr: Optional[np.ndarray] = data.get(
        f"L1_corrections.tof.iter{iteration}.ms"
    )
    gamma_arr: Optional[np.ndarray] = data.get(
        f"L1_corrections.tof.iter{iteration}.gamma"
    )

    mode_specs: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]] = [
        ("Raw − MS", ms_arr, None),
        ("Raw − Gamma", None, gamma_arr),
        ("Raw − MS − Gamma", ms_arr, gamma_arr),
    ]

    color_offset = 0
    for mode_label, ms, gc in mode_specs:
        if mode_label not in residual_modes:
            continue
        for det_idx in selected_detectors:
            if det_idx >= raw_y.shape[0]:
                continue
            residual = raw_y[det_idx].copy()
            if ms is not None and det_idx < ms.shape[0]:
                residual = residual - ms[det_idx]
            elif ms is None and "MS" in mode_label:
                # MS component requested but not available
                continue
            if gc is not None and det_idx < gc.shape[0]:
                residual = residual - gc[det_idx]
            elif gc is None and "Gamma" in mode_label:
                # Gamma component requested but not available in stream
                continue
            x = _resolve_x_axis(data, "tof", det_idx, len(residual))
            fig.add_trace(go.Scatter(
                x=x, y=residual, mode="lines",
                name=f"Det {det_idx} — {mode_label}",
                line=dict(
                    color=palette[(det_idx + color_offset) % len(palette)],
                    width=1.5,
                ),
            ))
        color_offset += 2

    return fig


def _build_optimizer_diff_figure(
    data: Dict[str, np.ndarray],
    masses_meta: Optional[np.ndarray],
) -> go.Figure:
    """Build the iMinuit−Scipy Difference Plot (Residual of Residuals).

    Computes ``fit_iMinuit(y) − fit_Scipy(y)`` for the y-space domain and
    plots it alongside a zero reference line.  This highlights exactly
    where the two optimizers diverge (peak vs wings).

    If both optimizer fit arrays are missing from *data*, returns an empty
    figure with an explanatory annotation.

    Args:
        data: Stream dictionary.
        masses_meta: Mass array from stream metadata (for axis annotation).

    Returns:
        A :class:`plotly.graph_objects.Figure` with the difference trace.
    """
    palette = COLORBLIND_PALETTE
    fig = go.Figure(layout=_pub_layout(
        "iMinuit − Scipy Difference Plot",
        "y (Å⁻¹)",
        "Δ fit (iMinuit − Scipy)",
    ))

    iminuit_arr: Optional[np.ndarray] = None
    scipy_arr: Optional[np.ndarray] = None

    for key_candidate in ("L3_final.y.iminuit", "L3_final.y.ncp_iminuit",
                          "L3_final.y.fit_iminuit"):
        if key_candidate in data and data[key_candidate].ndim == 1:
            iminuit_arr = data[key_candidate]
            break

    for key_candidate in ("L3_final.y.scipy", "L3_final.y.ncp_scipy",
                          "L3_final.y.fit_scipy"):
        if key_candidate in data and data[key_candidate].ndim == 1:
            scipy_arr = data[key_candidate]
            break

    if iminuit_arr is None or scipy_arr is None:
        fig.add_annotation(
            text=(
                "Optimizer comparison data not found.<br>"
                "Capture L3 iMinuit and Scipy fit arrays with keys<br>"
                "<tt>L3_final.y.ncp_iminuit</tt> / <tt>L3_final.y.ncp_scipy</tt>."
            ),
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=_FONT_SIZE_PT),
        )
        return fig

    n = min(len(iminuit_arr), len(scipy_arr))
    diff = iminuit_arr[:n] - scipy_arr[:n]
    x_y = _resolve_x_axis(data, "y", 0, n)[:n]

    # Zero reference
    fig.add_hline(
        y=0.0, line=dict(color="black", width=0.8, dash="dot"),
    )

    # Difference trace
    fig.add_trace(go.Scatter(
        x=x_y, y=diff, mode="lines",
        name="iMinuit − Scipy",
        line=dict(color=palette[3], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(213,94,0,0.12)",
    ))

    return fig


def _auto_pdf_filename(
    domain: str,
    masses_meta: Optional[np.ndarray],
    selected_mass_indices: List[int],
    selected_detectors: List[int],
    correction_active: bool,
) -> str:
    """Generate an auto-labelled PDF filename for LaTeX insertion.

    Format::

        VESUVIO_{masses}_{det_range}_{corrections}_{domain}.pdf

    Examples::

        VESUVIO_H_det0-4_MS_Gamma_J(y).pdf
        VESUVIO_H_C_det0_TOF.pdf
        VESUVIO_all_det0-99_y.pdf

    Args:
        domain: ``"tof"``, ``"q"``, or ``"y"``.
        masses_meta: Mass array from stream metadata.
        selected_mass_indices: Currently active mass indices.
        selected_detectors: Currently active detector indices.
        correction_active: Whether MS/Gamma corrections are overlaid.

    Returns:
        A safe filename string.
    """
    # Mass part
    if masses_meta is not None and selected_mass_indices:
        symbols = [
            _MASS_SYMBOL.get(round(float(masses_meta[i]), 3), f"M{i}")
            for i in selected_mass_indices
            if i < len(masses_meta)
        ]
        mass_part = "_".join(symbols) if symbols else "all"
    else:
        mass_part = "all"

    # Detector range part
    if selected_detectors:
        dmin, dmax = min(selected_detectors), max(selected_detectors)
        det_part = f"det{dmin}" if dmin == dmax else f"det{dmin}-{dmax}"
    else:
        det_part = "det_all"

    # Correction status
    corr_part = "MS_Gamma" if correction_active else ""

    # Domain label
    domain_map = {"tof": "TOF", "q": "Q", "y": "J(y)"}
    domain_label = domain_map.get(domain, domain.upper())

    parts = ["VESUVIO", mass_part, det_part]
    if corr_part:
        parts.append(corr_part)
    parts.append(domain_label)

    return "_".join(parts) + ".pdf"


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _fig_to_pdf_bytes(fig: go.Figure) -> Optional[bytes]:
    """Attempt to render *fig* as a true vector PDF via kaleido.

    Uses ``engine="kaleido"`` explicitly to produce publication-grade
    vector graphics suitable for direct LaTeX insertion.

    Returns ``None`` if kaleido is not installed or the render fails.
    """
    try:
        return fig.to_image(format="pdf", engine="kaleido")
    except Exception:
        return None


def _fig_to_html_bytes(fig: go.Figure) -> bytes:
    """Render *fig* as a self-contained interactive HTML file."""
    return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")


def _export_row(
    fig: go.Figure,
    *,
    key: str,
    filename: Optional[str] = None,
) -> None:
    """Render one-click export buttons below a Plotly figure.

    Attempts vector PDF export via kaleido (``engine="kaleido"``) first;
    falls back to interactive HTML if kaleido is unavailable.

    Args:
        fig: The Plotly figure to export.
        key: Unique Streamlit widget key suffix.
        filename: Auto-labelled filename (without extension).  Defaults to
            ``vesuvio_{key}_plot``.
    """
    base_name = filename or f"vesuvio_{key}_plot"
    col_pdf, col_html = st.columns(2)
    with col_pdf:
        pdf_bytes = _fig_to_pdf_bytes(fig)
        if pdf_bytes is not None:
            st.download_button(
                label="⬇️ Download as PDF",
                data=pdf_bytes,
                file_name=f"{base_name}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{key}",
            )
        else:
            st.caption(
                "💡 Install `kaleido` (`pip install kaleido`) for "
                "vector PDF export suitable for LaTeX."
            )
    with col_html:
        html_bytes = _fig_to_html_bytes(fig)
        st.download_button(
            label="⬇️ Download as HTML",
            data=html_bytes,
            file_name=f"{base_name}.html",
            mime="text/html",
            key=f"dl_html_{key}",
        )


def _batch_export_zip(
    data: Dict[str, np.ndarray],
    selected_mass_indices: List[int],
    masses_meta: Optional[np.ndarray],
) -> bytes:
    """Build a ZIP archive containing one HTML per mass y-space fit.

    Creates one self-contained interactive HTML for each selected mass
    whose per-mass NCP array is present in *data*.  Useful for appending
    to a thesis as a complete batch of y-space diagnostic plots.

    Args:
        data: Stream dictionary.
        selected_mass_indices: Mass indices to include in the report.
        masses_meta: Mass metadata array for labels.

    Returns:
        Raw bytes of the ZIP archive.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for mass_idx in selected_mass_indices:
            # Build a per-mass figure
            fig = _build_y_figure(
                data,
                selected_mass_indices=[mass_idx],
                masses_meta=masses_meta,
                show_optimizer_compare=False,
                show_recoil_markers=True,
            )
            label = _mass_label(mass_idx, masses_meta)
            # Safe filename
            safe_label = re.sub(r"[^\w.\-]", "_", label)
            html_bytes = _fig_to_html_bytes(fig)
            zf.writestr(f"y_space_{safe_label}.html", html_bytes)
    return buf.getvalue()


@st.cache_resource
def _cached_load_stream(path_str: str) -> Dict[str, np.ndarray]:
    """Load a StreamManager NPZ file and cache the result.

    Wrapping :func:`StreamManager.load` in ``@st.cache_resource`` ensures
    that repeated renders for 100+ detector runs do not re-read the
    compressed file from disk on every Streamlit re-run.

    The cache is keyed by the absolute file path string.

    Args:
        path_str: Absolute path to the ``.npz`` file.

    Returns:
        Dictionary mapping hierarchical stream keys to NumPy arrays.
    """
    return StreamManager.load(Path(path_str))


# ---------------------------------------------------------------------------
# Streamlit app entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Nanoscience Explorer Streamlit dashboard."""
    st.set_page_config(
        page_title="Nanoscience Explorer — VESUVIO",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Nanoscience Explorer")
    st.caption(
        "VESUVIO DINS Data Viewer — full-stack L0–L3 multi-domain analysis  |  "
        f"17 cm publication width · 12 pt serif · "
        f"[Wong (2011)](https://www.nature.com/articles/nmeth.1618) colour palette  |  "
        f"🔒 Local-first · air-gapped (127.0.0.1)"
    )

    # =========================================================================
    # Sidebar — file selection and interactive controls
    # =========================================================================
    with st.sidebar:
        st.header("📂 Data Stream")

        # ---- File selector targeting outputs/ --------------------------------
        npz_files: List[Path] = (
            sorted(_DEFAULT_OUTPUT_DIR.glob("*.npz"))
            if _DEFAULT_OUTPUT_DIR.exists()
            else []
        )
        file_options: Dict[str, Path] = {p.name: p for p in npz_files}

        chosen_path: Optional[Path] = None
        uploaded_tmp_path: Optional[Path] = None
        if file_options:
            chosen_name: str = st.selectbox(
                "Select stream file",
                options=list(file_options.keys()),
            )
            chosen_path = file_options[chosen_name]
        else:
            st.info(
                f"No `.npz` files found in "
                f"`{_DEFAULT_OUTPUT_DIR.relative_to(_REPO_ROOT)}/`.  "
                f"Run a VESUVIO analysis with a `StreamManager` to generate one, "
                f"or upload a file below."
            )

        uploaded = st.file_uploader(
            "Or upload a stream file (.npz)", type=["npz"],
        )
        if uploaded is not None:
            # Write to a temporary file so the cached loader can use a Path
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                tmp.write(uploaded.read())
                uploaded_tmp_path = Path(tmp.name)
                chosen_path = uploaded_tmp_path

        st.divider()

        # ---- Load stream via cached StreamManager.load() --------------------
        # @st.cache_resource avoids re-reading the compressed NPZ on every
        # Streamlit re-run — critical for 100+ detector runs.
        data: Dict[str, np.ndarray] = {}
        if chosen_path is not None:
            try:
                data = _cached_load_stream(str(chosen_path))
                st.success(f"✅ Loaded **{len(data)}** streams.")
            except Exception as exc:
                st.error(f"Failed to load stream: {exc}")
            finally:
                # Clean up temp file created from upload
                if uploaded_tmp_path is not None and uploaded_tmp_path.exists():
                    try:
                        os.unlink(uploaded_tmp_path)
                    except OSError:
                        pass

        # ---- Parse metadata for auto-population of controls -----------------
        buckets = _split_keys(data)
        available_levels = _available_levels(buckets)
        all_mass_indices = _detect_masses(data)
        all_detectors = _detect_detectors(data)
        masses_meta: Optional[np.ndarray] = data.get("metadata.masses")  # type: ignore[assignment]

        # ---- DataLevel availability indicator --------------------------------
        if data:
            st.subheader("📊 DataLevels")
            for lvl in ("L0", "L1", "L2", "L3"):
                icon = "✅" if lvl in available_levels else "⬜"
                st.markdown(f"{icon} **{_LEVEL_DISPLAY[lvl]}**")

            st.divider()

            # ---- Detector multiselect ----------------------------------------
            st.subheader("🔭 Detectors")
            selected_detectors: List[int] = st.multiselect(
                "Detectors",
                options=all_detectors,
                default=all_detectors[: min(3, len(all_detectors))],
                format_func=lambda i: f"Det {i}",
            )

            # ---- Mass multiselect --------------------------------------------
            st.subheader("⚛️ Masses")
            mass_labels = {i: _mass_label(i, masses_meta) for i in all_mass_indices}
            selected_mass_indices: List[int] = (
                st.multiselect(
                    "Masses",
                    options=all_mass_indices,
                    default=all_mass_indices,
                    format_func=lambda i: mass_labels.get(i, f"Mass {i}"),
                )
                if all_mass_indices
                else []
            )

            st.divider()

            # ---- Options toggles and sliders ---------------------------------
            st.subheader("⚙️ Options")
            show_corrections: bool = st.toggle(
                "Layer corrections (MS + Gamma)",
                value=True,
                disabled="L1" not in available_levels,
                help="Overlay Multiple-Scattering and Gamma-background correction "
                     "shaded bands behind the raw TOF signal.",
            )
            show_optimizer_compare: bool = st.toggle(
                "iMinuit–Scipy Comparison",
                value=False,
                help=(
                    "Overlay the iMinuit (MIGRAD) and Scipy (SLSQP) optimizer "
                    "results for cross-validation. Requires both optimizer "
                    "residuals to have been captured in the stream."
                ),
            )
            show_recoil_markers: bool = st.toggle(
                "Recoil Shift Diagnostic",
                value=False,
                help=(
                    "Add vertical y=0 reference lines to the y-Space plot for "
                    "each selected mass.  The theoretical recoil position for "
                    "all masses in West y-scaling is y=0; deviations reveal a "
                    "centering or calibration issue."
                ),
            )

            # Iteration slider (only shown if more than one iteration is stored)
            n_iter_meta = data.get("metadata.n_iterations")
            max_iter: int = max(int(n_iter_meta) - 1, 0) if n_iter_meta is not None else 0
            selected_iteration: int = (
                st.slider(
                    "MS/GC Iteration",
                    min_value=0,
                    max_value=max_iter,
                    value=max_iter,
                    help="Which MS/GC iteration to display for L1 and L2 data.",
                )
                if max_iter > 0
                else 0
            )

            # ---- Forensic residuals selector ---------------------------------
            st.subheader("🔬 Forensic Residuals")
            all_residual_modes = ["Raw − MS", "Raw − Gamma", "Raw − MS − Gamma"]
            selected_residual_modes: List[str] = st.multiselect(
                "Residual components",
                options=all_residual_modes,
                default=[],
                disabled="L1" not in available_levels,
                help=(
                    "Select correction components to isolate in the Forensic "
                    "Residuals sub-plot (TOF tab)."
                ),
            )

            if masses_meta is not None:
                st.divider()
                st.subheader("📋 Stream Metadata")
                st.markdown(
                    "**Masses:** "
                    + ", ".join(
                        _mass_label(i, masses_meta) for i in range(len(masses_meta))
                    )
                )
                if "metadata.fit_model" in data:
                    raw_val = data["metadata.fit_model"]
                    fit_model = (
                        raw_val.item()
                        if isinstance(raw_val, np.ndarray)
                        else str(raw_val)
                    )
                    st.markdown(f"**Fit model:** `{fit_model}`")
                if "metadata.n_iterations" in data:
                    st.markdown(
                        f"**MS/GC iterations:** {int(data['metadata.n_iterations'])}"
                    )

        else:
            selected_detectors = []
            selected_mass_indices = []
            show_corrections = False
            show_optimizer_compare = False
            show_recoil_markers = False
            selected_iteration = 0
            selected_residual_modes = []

    # =========================================================================
    # Main content — domain tabs
    # =========================================================================
    if not data:
        st.info("👈 Select or upload a stream file in the sidebar to begin.")
        _render_schema_help()
        return

    tab_tof, tab_q, tab_y = st.tabs(["⏱ TOF", "🔵 Q-Space", "📈 y-Space"])

    # ---- TOF tab -------------------------------------------------------------
    with tab_tof:
        st.subheader("Time-of-Flight Domain")
        if not selected_detectors:
            st.warning("Select at least one detector in the sidebar.")
        else:
            fig_tof = _build_tof_figure(
                data,
                selected_detectors=selected_detectors,
                show_corrections=show_corrections and "L1" in available_levels,
                show_optimizer_compare=show_optimizer_compare,
                iteration=selected_iteration,
            )
            tof_filename = _auto_pdf_filename(
                "tof", masses_meta, selected_mass_indices,
                selected_detectors, show_corrections,
            ).rstrip(".pdf")
            st.plotly_chart(fig_tof, use_container_width=True)
            _export_row(fig_tof, key="tof", filename=tof_filename)

            if "L1" in available_levels and show_corrections:
                with st.expander("ℹ️ Correction Stack Legend"):
                    st.markdown(
                        "**Shaded orange band** — Multiple-Scattering (MS) correction.  \n"
                        "**Shaded green band** — Gamma-background correction.  \n"
                        "Each band spans from `raw − correction` to `raw`, showing "
                        "the magnitude of the correction subtracted from the raw signal."
                    )

            # ---- Forensic Residuals sub-plot ---------------------------------
            if selected_residual_modes and "L1" in available_levels:
                st.subheader("🔬 Forensic Residuals")
                fig_res = _build_residuals_figure(
                    data, selected_detectors, selected_residual_modes,
                    iteration=selected_iteration,
                )
                st.plotly_chart(fig_res, use_container_width=True)
                _export_row(fig_res, key="tof_residuals")

            # ---- Area Audit table -------------------------------------------
            if "L1" in available_levels:
                audit_rows = _compute_area_audit(data, iteration=selected_iteration)
                if audit_rows:
                    # Filter to selected detectors only
                    filtered = [
                        r for r in audit_rows
                        if r["detector"] in selected_detectors
                    ]
                    if filtered:
                        with st.expander(
                            "📊 Area Audit — Integral Contribution of Corrections",
                            expanded=False,
                        ):
                            st.caption(
                                "Integral Contribution (%) = |∫correction| / |∫raw| × 100.  "
                                "Use these values in the Errors & Uncertainties chapter."
                            )
                            st.dataframe(
                                filtered,
                                column_config={
                                    "correction": "Correction",
                                    "detector": "Detector",
                                    "integral": st.column_config.NumberColumn(
                                        "∫|correction|", format="%.4f",
                                    ),
                                    "raw_integral": st.column_config.NumberColumn(
                                        "∫|raw|", format="%.4f",
                                    ),
                                    "contribution_pct": st.column_config.NumberColumn(
                                        "Contribution (%)", format="%.3f%%",
                                    ),
                                },
                                hide_index=True,
                                use_container_width=True,
                            )

    # ---- Q tab ---------------------------------------------------------------
    with tab_q:
        st.subheader("Q-Space Domain")
        fig_q = _build_q_figure(data, selected_detectors=selected_detectors)
        q_filename = _auto_pdf_filename(
            "q", masses_meta, selected_mass_indices, selected_detectors, False,
        ).rstrip(".pdf")
        st.plotly_chart(fig_q, use_container_width=True)
        _export_row(fig_q, key="q", filename=q_filename)

    # ---- y-space tab ---------------------------------------------------------
    with tab_y:
        st.subheader("y-Space Domain — J(y)")
        if masses_meta is not None:
            st.caption(
                "Masses: "
                + " · ".join(
                    _mass_label(i, masses_meta) for i in range(len(masses_meta))
                )
            )
        fig_y = _build_y_figure(
            data,
            selected_mass_indices=selected_mass_indices,
            masses_meta=masses_meta,
            show_optimizer_compare=show_optimizer_compare,
            show_recoil_markers=show_recoil_markers,
        )
        y_filename = _auto_pdf_filename(
            "y", masses_meta, selected_mass_indices, selected_detectors,
            show_corrections,
        ).rstrip(".pdf")
        st.plotly_chart(fig_y, use_container_width=True)
        _export_row(fig_y, key="y", filename=y_filename)

        # ---- iMinuit–Scipy Difference Plot ----------------------------------
        if show_optimizer_compare:
            with st.expander(
                "📉 iMinuit–Scipy Difference Plot (Residual of Residuals)",
                expanded=False,
            ):
                st.caption(
                    "Shows exactly where iMinuit (MIGRAD) and Scipy (SLSQP) diverge.  "
                    "Deviations > ~1% in peak or wings warrant further inspection."
                )
                fig_diff = _build_optimizer_diff_figure(data, masses_meta)
                st.plotly_chart(fig_diff, use_container_width=True)
                _export_row(fig_diff, key="optimizer_diff")

            with st.expander("ℹ️ iMinuit–Scipy Numerical Agreement"):
                st.markdown(
                    "The **iMinuit–Scipy Numerical Agreement Check** overlays "
                    "the iMinuit (MIGRAD) and Scipy (SLSQP) optimizer fits "
                    "from the saved streams.  Deviations larger than ~1% between "
                    "the two indicate a fit that may warrant further inspection."
                )

        # ---- Batch export for thesis appendix --------------------------------
        if selected_mass_indices:
            st.subheader("📦 Batch Export — y-Space Fits")
            st.caption(
                "Export y-Space fits for all selected masses as a ZIP of "
                "interactive HTML files — ready for the thesis appendix."
            )
            if st.button("Generate Batch ZIP", key="batch_zip_btn"):
                zip_bytes = _batch_export_zip(
                    data, selected_mass_indices, masses_meta,
                )
                st.download_button(
                    label="⬇️ Download Batch ZIP",
                    data=zip_bytes,
                    file_name="vesuvio_y_space_batch.zip",
                    mime="application/zip",
                    key="dl_batch_zip",
                )


def _render_schema_help() -> None:
    """Show an expandable guide to the NPZ key schema."""
    with st.expander("📖 Stream file key schema"):
        st.markdown(
            """
The `StreamManager` serialises all analysis arrays into a single
compressed `.npz` file using a hierarchical dot-separated key convention:

| Key pattern | Contents |
|---|---|
| `L0_raw.tof.dataY` | Raw detector counts (shape: n_det × n_bins) |
| `L0_raw.tof.dataX` | TOF axis (μs) |
| `L1_corrections.tof.iter{N}.ms` | Multiple-scattering correction at iteration N |
| `L1_corrections.tof.iter{N}.gamma` | Gamma-background correction at iteration N |
| `L2_intermediate.tof.iter{N}.corrected` | Corrected signal after MS/GC subtraction |
| `L3_final.tof.iter{N}.ncp_total` | Total NCP fit |
| `L3_final.y.joy_avg` | Averaged J(y) in y-space |
| `L3_final.y.resolution` | Resolution function |
| `L3_final.y.mass{N}.ncp` | Per-mass NCP contribution (Global Fit View) |
| `metadata.masses` | Array of atomic masses used in the fit |
| `metadata.fit_model` | Name of the y-space fit model |

Generate a stream file by passing a `StreamManager` instance to
`iterativeFitForDataReduction()` and/or `fitInYSpaceProcedure()`.
"""
        )


if __name__ == "__main__":
    main()
