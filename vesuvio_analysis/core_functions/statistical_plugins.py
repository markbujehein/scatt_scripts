"""Phase 6 — Transparency-First Diagnostic Suite for the VESUVIO pipeline.

Provides modular statistical diagnostics for the VESUVIO analysis pipeline
organized into five logical blocks:

1. **Summary-Feature Outlier Detection** (EllipticEnvelope): Extracts
   per-detector summary statistics (total counts, RMS, skewness,
   kurtosis) and flags anomalies via robust Mahalanobis distance.
2. **Cluster Analysis** (DBSCAN): Identify density-based groupings of
   detector responses in (L, θ) space.
3. **Classification** (Fisher/LDA): Categorize spectra based on
   fit-fidelity labels (Agreement < 1% vs > 5%).
4. **Physical Diagnostics** (Anisotropy Detection): Analyze residuals as a
   function of scattering angle to identify non-Gaussian behavior.
5. **Calibration & Residuals**: Calculate δ = (Y_obs − Y_fit)/Y_fit to
   highlight systematic detector biases.

PHILOSOPHY — DIAGNOSTIC ASSISTANCE, NOT AUTOMATED REMOVAL:
    Automated outlier masking is **deprecated**. Real samples can be
    anisotropic; automated removal risks discarding genuine physical signals
    (e.g. anisotropic broadening) by misidentifying them as hardware failures.
    Outlier detection now functions as an *Anisotropy & Health Monitor*:
    suspicious detectors are flagged for manual review, and only spectra
    explicitly listed in ``maskedSpecAllNo`` are excluded from the fit.

Classes:
    HardwareOutlierDetector: Summary-feature + robust-covariance anomaly
        detection for broken detectors (diagnostic only — no automated
        masking).
    PhysicsTrendClusterer: DBSCAN clustering of detector features (L, θ).
    BayesianBootstrap: Rubin-style Weighted Bayesian Bootstrap with
        Dirichlet weights for high-speed resampling of NCP residuals.

Notes:
    - scikit-learn DBSCAN labels noise points as -1; these are explicitly
      excluded from physics-trend groups.
    - Per-detector summary statistics (total counts, RMS, skewness,
      kurtosis) provide a compact, deterministic, and interpretable
      feature space for outlier detection without nonlinear
      dimensionality reduction.  EllipticEnvelope uses robust
      Mahalanobis distance in this space.
    - Dirichlet(1, 1, ..., 1) produces the uniform prior over the simplex
      (Rubin, 1981).  Each weight vector sums to 1.0 (up to
      floating-point rounding).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

from vesuvio_analysis.core_functions.plot_style import (
    COLORBLIND_PALETTE,
    EXPERIMENTAL_STYLE,
    FULL_WIDTH_CM,
    THEORETICAL_STYLE,
    cm_to_inches,
    figure_factory,
    set_thesis_style,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic Table & Anisotropy-Health Monitor
# ---------------------------------------------------------------------------


def format_diagnostic_table(
    outlier_indices: NDArray[np.intp],
    metadata_map: Dict[int, Dict[str, Any]],
    masked_spec_all_no: NDArray[np.intp],
    fisher_scores: Optional[NDArray[np.floating]] = None,
) -> str:
    """Format outlier detections into a human-readable diagnostic table.

    Implements the transparency-first philosophy: every flagged spectrum
    is printed with its physical identifiers and a recommendation that
    explicitly distinguishes between user-acknowledged masks and
    unrecognized anomalies requiring manual review.

    Args:
        outlier_indices: Array indices of flagged spectra.
        metadata_map: ``{array_index: {spec_no, angle, detector_id}}``.
        masked_spec_all_no: Spectrum IDs explicitly listed in
            ``BackwardInitialConditions.maskedSpecAllNo`` or
            ``ForwardInitialConditions.maskedSpecAllNo``.
        fisher_scores: Optional per-detector Fisher discriminant scores.

    Returns:
        Formatted multi-line string suitable for terminal output.
    """
    pre_masked = set(int(s) for s in masked_spec_all_no)
    header = (
        f"{'Index':>6} | {'Spectrum ID':>11} | {'Scatt. Angle':>12} | "
        f"{'Fisher Score':>12} | {'Recommendation'}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        "=" * len(header),
        "  DETECTOR ANISOTROPY & HEALTH MONITOR — Diagnostic Report",
        "=" * len(header),
        header,
        sep,
    ]
    n_unrecognized = 0
    for idx in outlier_indices:
        meta = metadata_map.get(int(idx), {})
        spec_no = meta.get("spec_no", idx)
        angle = meta.get("angle", float("nan"))
        f_score = (
            f"{float(fisher_scores[idx]):.4f}"
            if fisher_scores is not None and idx < len(fisher_scores)
            else "N/A"
        )
        if int(spec_no) in pre_masked:
            recommendation = "PRE-MASKED (user)"
        else:
            recommendation = "MANUAL REVIEW"
            n_unrecognized += 1
        lines.append(
            f"{int(idx):>6} | {int(spec_no):>11} | {angle:>11.2f}° | "
            f"{f_score:>12} | {recommendation}"
        )
    lines.append(sep)
    lines.append(
        f"  Total flagged: {len(outlier_indices)}  |  "
        f"Pre-masked: {len(outlier_indices) - n_unrecognized}  |  "
        f"Unrecognized (manual review): {n_unrecognized}"
    )
    if n_unrecognized > 0:
        lines.append(
            "  WARNING: Unrecognized outliers detected. Add their Spectrum IDs "
            "to maskedSpecAllNo if they are confirmed hardware failures, or "
            "consider a more complex model (e.g. Anisotropic Gaussian) if the "
            "deviations are physical."
        )
    lines.append("=" * len(header))
    return "\n".join(lines)


def compute_anisotropy_residuals(
    spectra: NDArray[np.floating],
    ncp_total: NDArray[np.floating],
    theta_deg: NDArray[np.floating],
) -> Dict[str, NDArray[np.floating]]:
    """Analyze residuals as a function of scattering angle for anisotropy.

    Computes the per-detector normalised residual
    δ = (Y_obs − Y_fit) / Y_fit and its angular dependence.  A systematic
    trend (e.g. residual increasing with θ) indicates anisotropic
    broadening that a simple isotropic Gaussian model cannot capture.

    Args:
        spectra: Observed detector spectra, shape ``(n_det, n_bins)``.
        ncp_total: Fitted NCP profiles, shape ``(n_det, n_bins)``.
        theta_deg: Scattering angles in degrees, shape ``(n_det,)``.

    Returns:
        Dict with keys:
            - ``theta``: scattering angles (sorted).
            - ``mean_residual``: mean normalised residual per detector.
            - ``std_residual``: standard deviation of residual per detector.
            - ``spearman_r``: Spearman correlation between θ and |residual|.
            - ``spearman_p``: p-value of Spearman test.
    """
    eps = 1e-10
    denom = np.where(np.abs(ncp_total) > eps, ncp_total, eps)
    delta = (spectra - ncp_total) / denom
    delta = np.where(np.isfinite(delta), delta, 0.0)

    mean_res = np.mean(delta, axis=1)
    std_res = np.std(delta, axis=1)

    # Spearman rank correlation of |residual| vs θ — detects monotonic
    # angular trends characteristic of anisotropy.
    abs_mean = np.abs(mean_res)
    finite = np.isfinite(abs_mean) & np.isfinite(theta_deg)
    if np.sum(finite) >= 5 and np.ptp(abs_mean[finite]) > 0:
        sr = stats.spearmanr(theta_deg[finite], abs_mean[finite])
        spearman_r = float(sr.statistic)
        spearman_p = float(sr.pvalue)
    else:
        spearman_r = float("nan")
        spearman_p = float("nan")

    return {
        "theta": theta_deg,
        "mean_residual": mean_res,
        "std_residual": std_res,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def generate_physical_interpretation_hint(
    n_clusters: int,
    spearman_r: float,
    spearman_p: float,
    n_outliers: int,
    n_total: int,
) -> str:
    """Generate a 'Show Your Work' interpretation hint for the terminal.

    Provides actionable physical interpretation of the diagnostic results
    so the user understands what the statistical analysis implies about
    the sample and instrument.

    Args:
        n_clusters: Number of DBSCAN clusters found.
        spearman_r: Spearman rank correlation |residual| vs θ.
        spearman_p: p-value of the Spearman test.
        n_outliers: Number of flagged outlier detectors.
        n_total: Total number of detectors.

    Returns:
        Formatted multi-line string with physical interpretation.
    """
    lines = ["", "[Diagnostic] Physical Interpretation Hints:"]

    if n_clusters > 1:
        lines.append(
            f"  • Clusters found: {n_clusters}. If clusters correlate with "
            "scattering angle, consider checking for instrument resolution "
            "gradients or sample anisotropy."
        )
    elif n_clusters == 1:
        lines.append(
            "  • Single cluster found — detector bank appears homogeneous."
        )

    if np.isfinite(spearman_r):
        if abs(spearman_r) > 0.5 and spearman_p < 0.05:
            lines.append(
                f"  • Strong angular trend in residuals (ρ = {spearman_r:.3f}, "
                f"p = {spearman_p:.2e}). This suggests anisotropic "
                "broadening — consider a non-isotropic model."
            )
        elif abs(spearman_r) > 0.3:
            lines.append(
                f"  • Moderate angular trend (ρ = {spearman_r:.3f}, "
                f"p = {spearman_p:.2e}). May indicate mild anisotropy or "
                "resolution gradients."
            )
        else:
            lines.append(
                f"  • No significant angular trend (ρ = {spearman_r:.3f}). "
                "Residuals appear isotropic."
            )

    if n_outliers > 0:
        pct = 100.0 * n_outliers / max(n_total, 1)
        lines.append(
            f"  • {n_outliers}/{n_total} detectors flagged ({pct:.1f}%). "
            "Review the diagnostic table to distinguish hardware failures "
            "from genuine anisotropy."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 6 Diagnostic Dashboard
# ---------------------------------------------------------------------------


def plot_phase6_diagnostic_dashboard(
    summary_features: NDArray[np.floating],
    labels: NDArray[np.intp],
    fisher_scores: Optional[NDArray[np.floating]],
    fidelity_labels: Optional[NDArray[np.intp]],
    theta_deg: NDArray[np.floating],
    anisotropy: Dict[str, NDArray[np.floating]],
    roc_data: Optional[Dict[str, Any]],
    metadata_map: Dict[int, Dict[str, Any]],
    ws_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate the unified Phase 6 Diagnostic Summary dashboard.

    Four-panel layout:
        Panel 1: Feature-space scatter color-coded by Fisher Score.
        Panel 2: Fisher-style 1D histogram of discriminant scores.
        Panel 3: Residuals vs Scattering Angle (anisotropy detection).
        Panel 4: ROC curve showing detector-bank trustworthiness.

    Args:
        summary_features: Summary features, shape ``(n_det, >=2)``.
        labels: Outlier labels (``-1`` = outlier, ``0`` = inlier).
        fisher_scores: LDA discriminant scores, shape ``(n_det,)``.
        fidelity_labels: Convergence labels (0=hi, 1=poor, -1=unlabeled).
        theta_deg: Scattering angles, shape ``(n_det,)``.
        anisotropy: Output from ``compute_anisotropy_residuals()``.
        roc_data: Output from ``fisher_lda_with_roc()`` or ``None``.
        metadata_map: ``{array_index: {spec_no, angle, detector_id}}``.
        ws_name: Workspace name for the title.
        save_path: File path for the PDF output.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig_w = cm_to_inches(FULL_WIDTH_CM)
    fig_h = cm_to_inches(FULL_WIDTH_CM * 1.2)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # --- Panel 1: Feature-space scatter ---
    ax1 = fig.add_subplot(gs[0, 0])
    if fisher_scores is not None:
        sc = ax1.scatter(
            summary_features[:, 0], summary_features[:, 1],
            c=fisher_scores, cmap="viridis", s=18, alpha=0.85,
        )
        cbar = fig.colorbar(sc, ax=ax1, shrink=0.8)
        cbar.set_label("Fisher Score", fontsize=8)
    else:
        inlier_m = labels == 0
        outlier_m = labels == -1
        ax1.scatter(
            summary_features[inlier_m, 0], summary_features[inlier_m, 1],
            color=COLORBLIND_PALETTE[0], s=18, label="Inlier",
        )
        if np.any(outlier_m):
            ax1.scatter(
                summary_features[outlier_m, 0], summary_features[outlier_m, 1],
                color="#D62728", marker="x", s=50, linewidths=1.3,
                label="Flagged",
            )
            # Label flagged detectors with Spectrum IDs
            for idx in np.where(outlier_m)[0]:
                meta = metadata_map.get(int(idx), {})
                spec_id = meta.get("spec_no", "?")
                ax1.annotate(
                    str(spec_id), (summary_features[idx, 0], summary_features[idx, 1]),
                    fontsize=5, alpha=0.7,
                )
        ax1.legend(fontsize=7)
    ax1.set_xlabel("Total Counts", fontsize=9)
    ax1.set_ylabel("RMS", fontsize=9)
    ax1.set_title("(a) Detector Feature Space", fontsize=9)

    # --- Panel 2: Fisher 1D histogram ---
    ax2 = fig.add_subplot(gs[0, 1])
    if fisher_scores is not None and fidelity_labels is not None:
        hi_mask = fidelity_labels == 0
        poor_mask = fidelity_labels == 1
        if np.any(hi_mask):
            ax2.hist(
                fisher_scores[hi_mask], bins=20, alpha=0.6,
                color=COLORBLIND_PALETTE[0], label="Hi-Fidelity (<1%)",
                density=True,
            )
        if np.any(poor_mask):
            ax2.hist(
                fisher_scores[poor_mask], bins=20, alpha=0.6,
                color=COLORBLIND_PALETTE[3], label="Poor-Fidelity (>5%)",
                density=True,
            )
        if roc_data is not None:
            ax2.text(
                0.98, 0.98,
                f"AUC = {float(roc_data['auc']):.3f}",
                transform=ax2.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )
        ax2.legend(fontsize=7)
    else:
        ax2.text(
            0.5, 0.5, "Fisher/LDA not available\n(no optimizer diagnostics)",
            ha="center", va="center", transform=ax2.transAxes, fontsize=9,
        )
    ax2.set_xlabel("Fisher Discriminant Score", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("(b) Fisher Discriminant Distribution", fontsize=9)

    # --- Panel 3: Residuals vs Scattering Angle ---
    ax3 = fig.add_subplot(gs[1, 0])
    theta = anisotropy["theta"]
    mean_res = anisotropy["mean_residual"]
    std_res = anisotropy["std_residual"]
    ax3.errorbar(
        theta, mean_res, yerr=std_res,
        fmt="o", markersize=3, capsize=2, elinewidth=0.5,
        color=COLORBLIND_PALETTE[0], alpha=0.7,
    )
    ax3.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
    sr = anisotropy["spearman_r"]
    sp = anisotropy["spearman_p"]
    if np.isfinite(sr):
        ax3.text(
            0.02, 0.98,
            f"Spearman ρ = {sr:.3f}\np = {sp:.2e}",
            transform=ax3.transAxes, fontsize=7,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )
    ax3.set_xlabel(r"Scattering Angle $\theta$ (°)", fontsize=9)
    ax3.set_ylabel(r"Mean Residual $\langle\delta\rangle$", fontsize=9)
    ax3.set_title("(c) Residuals vs Angle — Anisotropy Check", fontsize=9)

    # --- Panel 4: ROC Curve ---
    ax4 = fig.add_subplot(gs[1, 1])
    if roc_data is not None:
        ax4.plot(
            roc_data["fpr"], roc_data["tpr"],
            color=COLORBLIND_PALETTE[3],
            label=f"Fisher LDA (AUC={float(roc_data['auc']):.3f})",
        )
        ax4.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8,
                 label="Chance")
        ax4.legend(fontsize=7)
    else:
        ax4.text(
            0.5, 0.5, "ROC not available",
            ha="center", va="center", transform=ax4.transAxes, fontsize=9,
        )
    ax4.set_xlabel("False Positive Rate", fontsize=9)
    ax4.set_ylabel("True Positive Rate", fontsize=9)
    ax4.set_title("(d) ROC — Detector Trustworthiness", fontsize=9)

    fig.suptitle(
        f"Phase 6 Diagnostic Summary — {ws_name}",
        fontsize=11, fontweight="bold", y=0.98,
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1,
                    facecolor="white")
        plt.close(fig)
        logger.info("Phase 6 dashboard saved to %s", save_path)
    return fig


def plot_feature_annotated(
    summary_features: NDArray[np.floating],
    labels: NDArray[np.intp],
    metadata_map: Dict[int, Dict[str, Any]],
    cluster_labels: Optional[NDArray[np.intp]] = None,
    spearman_r: Optional[float] = None,
    spearman_p: Optional[float] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Feature-space scatter with cluster angle annotations and anomaly labels.

    Clusters are annotated with their average scattering angle.
    Isolated anomaly detectors are labeled with their Spectrum ID.  When
    Spearman ρ is provided it is displayed as a *Physics Metric* inset box.

    Args:
        summary_features: Summary features, shape ``(n_det, >=2)``.
        labels: Outlier labels (``-1`` = outlier, ``0`` = inlier).
        metadata_map: Physical detector metadata per array index.
        cluster_labels: Optional DBSCAN cluster labels for annotation.
        spearman_r: Spearman rank correlation between residuals and angle
            (from ``compute_anisotropy_residuals``).  Displayed as a
            Physics Metric inset when finite.
        spearman_p: Two-sided p-value for the Spearman correlation.
        save_path: File path for the output PDF.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, ax = figure_factory()

    inlier_m = labels == 0
    outlier_m = labels == -1

    # Scatter inliers
    ax.scatter(
        summary_features[inlier_m, 0], summary_features[inlier_m, 1],
        color=COLORBLIND_PALETTE[0], s=20, alpha=0.7, label="Inlier",
    )

    # Scatter outliers with Spectrum ID annotations
    if np.any(outlier_m):
        ax.scatter(
            summary_features[outlier_m, 0], summary_features[outlier_m, 1],
            color="#D62728", marker="x", s=60, linewidths=1.5,
            label=f"Flagged (n={int(outlier_m.sum())})",
        )
        for idx in np.where(outlier_m)[0]:
            meta = metadata_map.get(int(idx), {})
            spec_id = meta.get("spec_no", idx)
            ax.annotate(
                f"Spec {spec_id}",
                (summary_features[idx, 0], summary_features[idx, 1]),
                fontsize=6, alpha=0.8,
                xytext=(5, 5), textcoords="offset points",
            )

    # Annotate clusters with average scattering angle
    if cluster_labels is not None:
        unique_clusters = sorted(set(cluster_labels) - {-1})
        for cl in unique_clusters:
            cl_mask = cluster_labels == cl
            if not np.any(cl_mask):
                continue
            cx = np.mean(summary_features[cl_mask, 0])
            cy = np.mean(summary_features[cl_mask, 1])
            angles = [
                metadata_map.get(int(i), {}).get("angle", float("nan"))
                for i in np.where(cl_mask)[0]
            ]
            mean_angle = float(np.nanmean(angles))
            ax.annotate(
                f"Cluster {cl}\n⟨θ⟩={mean_angle:.1f}°",
                (cx, cy), fontsize=7, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                          alpha=0.8, edgecolor="#999999"),
            )

    # Physics Metric inset — Spearman ρ between NCP residuals and angle
    if spearman_r is not None and np.isfinite(float(spearman_r)):
        _p_str = f"p = {float(spearman_p):.2e}" if (spearman_p is not None and np.isfinite(float(spearman_p))) else ""
        ax.text(
            0.98, 0.02,
            f"Physics Metric\nSpearman ρ = {float(spearman_r):.3f}" + (f"\n{_p_str}" if _p_str else ""),
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      alpha=0.85, edgecolor="#AAAAAA"),
        )

    ax.set_xlabel("Total Counts")
    ax.set_ylabel("RMS")
    ax.set_title("Detector Health — Feature Space")
    ax.legend(fontsize=7)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_fisher_distribution(
    fisher_scores: NDArray[np.floating],
    fidelity_labels: NDArray[np.intp],
    feature_names: List[str],
    lda_model: Any,
    roc_auc: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Fisher Discriminant distribution plot (AppStat NBI style).

    Shows the 1D Fisher discriminant score distribution for both
    hi-fidelity and poor-fidelity populations, annotated with the AUC
    and the feature importances (LDA coefficients).

    Args:
        fisher_scores: LDA discriminant scores, shape ``(n_det,)``.
        fidelity_labels: Convergence labels (0=hi, 1=poor, -1=unlabeled).
        feature_names: Human-readable names for the input features.
        lda_model: Trained ``LinearDiscriminantAnalysis`` model.
        roc_auc: AUC value from the ROC curve.
        save_path: File path for the output PDF.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, (ax_hist, ax_imp) = figure_factory(ncols=2, aspect_ratio=0.55)

    # Left panel: Score distributions
    hi_mask = fidelity_labels == 0
    poor_mask = fidelity_labels == 1

    if np.any(hi_mask):
        ax_hist.hist(
            fisher_scores[hi_mask], bins=25, alpha=0.6, density=True,
            color=COLORBLIND_PALETTE[0],
            label=f"Hi-Fidelity (n={int(hi_mask.sum())})",
        )
    if np.any(poor_mask):
        ax_hist.hist(
            fisher_scores[poor_mask], bins=25, alpha=0.6, density=True,
            color=COLORBLIND_PALETTE[3],
            label=f"Poor-Fidelity (n={int(poor_mask.sum())})",
        )

    ax_hist.set_xlabel("Fisher Discriminant Score")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"Fisher LDA (AUC = {roc_auc:.3f})")
    ax_hist.legend(fontsize=7)

    # Right panel: Feature importance (LDA coefficients)
    if hasattr(lda_model, "coef_") and lda_model.coef_ is not None:
        coefs = np.abs(lda_model.coef_[0])
        n_coefs = len(coefs)
        names = feature_names[:n_coefs] if len(feature_names) >= n_coefs else (
            feature_names + [f"Feature {i}" for i in range(len(feature_names), n_coefs)]
        )
        sorted_idx = np.argsort(coefs)[::-1]
        ax_imp.barh(
            range(n_coefs),
            coefs[sorted_idx],
            color=COLORBLIND_PALETTE[2], alpha=0.8,
        )
        ax_imp.set_yticks(range(n_coefs))
        ax_imp.set_yticklabels([names[i] for i in sorted_idx], fontsize=8)
        ax_imp.set_xlabel("|LDA Coefficient|")
        ax_imp.set_title("Feature Importance")
        ax_imp.invert_yaxis()
    else:
        ax_imp.text(
            0.5, 0.5, "No coefficients available",
            ha="center", va="center", transform=ax_imp.transAxes,
        )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_residuals_vs_angle(
    anisotropy: Dict[str, NDArray[np.floating]],
    metadata_map: Dict[int, Dict[str, Any]],
    outlier_indices: Optional[NDArray[np.intp]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot normalised residuals vs scattering angle for anisotropy detection.

    Each detector is plotted as a point at its scattering angle vs its
    mean normalised residual.  Outlier detectors (if provided) are
    highlighted and labeled with their Spectrum ID.

    Args:
        anisotropy: Output from ``compute_anisotropy_residuals()``.
        metadata_map: Physical detector metadata per array index.
        outlier_indices: Indices of flagged detectors (optional).
        save_path: File path for the output PDF.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, ax = figure_factory()

    theta = anisotropy["theta"]
    mean_res = anisotropy["mean_residual"]
    std_res = anisotropy["std_residual"]

    ax.errorbar(
        theta, mean_res, yerr=std_res,
        fmt="o", markersize=4, capsize=2, elinewidth=0.5,
        color=COLORBLIND_PALETTE[0], alpha=0.7, label="Detectors",
    )
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)

    if outlier_indices is not None and len(outlier_indices) > 0:
        ax.scatter(
            theta[outlier_indices], mean_res[outlier_indices],
            color="#D62728", marker="x", s=60, linewidths=1.5, zorder=5,
            label="Flagged",
        )
        for idx in outlier_indices:
            meta = metadata_map.get(int(idx), {})
            spec_id = meta.get("spec_no", idx)
            ax.annotate(
                f"Spec {spec_id}",
                (theta[idx], mean_res[idx]),
                fontsize=6, xytext=(4, 4), textcoords="offset points",
            )

    sr = anisotropy["spearman_r"]
    sp = anisotropy["spearman_p"]
    if np.isfinite(sr):
        ax.text(
            0.02, 0.98,
            f"Spearman ρ = {sr:.3f}\np = {sp:.2e}",
            transform=ax.transAxes, fontsize=8, va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )

    ax.set_xlabel(r"Scattering Angle $\theta$ (°)")
    ax.set_ylabel(r"Mean Normalised Residual $\langle\delta\rangle$")
    ax.set_title("Residuals vs Angle — Anisotropy Detection")
    ax.legend(fontsize=7)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# NCP publication-plot helpers
# ---------------------------------------------------------------------------

#: Maps raw parameter names to their LaTeX equivalents for legend rendering.
_PARAM_LABEL_MAP: Dict[str, str] = {
    "sigma": r"$\sigma_p$",
    "x0": r"$y_\mathrm{center}$",
}


def _apply_latex_labels(label: str) -> str:
    r"""Replace raw parameter names in *label* with LaTeX equivalents.

    Substitutions are driven by :data:`_PARAM_LABEL_MAP`, which maps
    ``'sigma'`` → ``r'$\sigma_p$'`` and ``'x0'`` → ``r'$y_\mathrm{center}$'``
    so that any legend entry containing these tokens follows Nanoscience
    conventions when rendered by Matplotlib's mathtext engine.

    Args:
        label: Raw legend label string.

    Returns:
        Label string with parameter tokens replaced by their LaTeX forms.
    """
    for raw, latex in _PARAM_LABEL_MAP.items():
        label = label.replace(raw, latex)
    return label


def _parse_script_name_components(ic_name: str) -> Tuple[str, str, str]:
    """Parse ``IC.name`` into ``(sample, temp_k, model)`` components.

    ``IC.name`` follows the convention
    ``'{scriptName}_{modeRunning}_'`` where *scriptName* is structured
    as ``'{sample}_{tempK}_{model}'`` with the temperature token matching
    the pattern ``r'\\d+[Kk]'`` (e.g. ``'10K'`` or ``'300K'``).

    Args:
        ic_name: The ``IC.name`` attribute, e.g.
            ``'thymol_10K_Gauss1D_FORWARD_'``.

    Returns:
        A ``(sample, temp_k, model)`` tuple, where *temp_k* is the
        numeric part only (e.g. ``'10'`` from ``'10K'``).  Returns
        ``(raw_name, '?', '?')`` when no temperature token is found.
    """
    # Strip optional trailing underscore, then remove the mode suffix.
    name = ic_name.rstrip("_")
    for mode in ("_FORWARD", "_BACKWARD", "_JOINT"):
        if name.upper().endswith(mode.upper()):
            name = name[: -len(mode)]
            break

    tokens = name.split("_")
    temp_idx: Optional[int] = None
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"\d+[Kk]", tok):
            temp_idx = i
            break

    if temp_idx is None:
        return name, "?", "?"

    sample = "_".join(tokens[:temp_idx]) or "Unknown"
    temp_val = tokens[temp_idx][:-1]          # strip trailing 'K'
    model = "_".join(tokens[temp_idx + 1 :]) or "?"
    return sample, temp_val, model


# ---------------------------------------------------------------------------
# Hardware Outlier Detection
# ---------------------------------------------------------------------------

class HardwareOutlierDetector:
    """Summary-feature robust covariance.

    Each row of the input matrix is a detector spectrum.  Per-detector
    summary statistics (total counts, RMS, skewness, kurtosis) are
    extracted to build a compact, interpretable feature space.
    ``EllipticEnvelope`` then flags anomalies via robust Mahalanobis
    distance in this space — no dimensionality reduction required.

    This approach is deterministic and fast: the four summary features
    capture both intensity anomalies (total counts), shape anomalies
    (skewness, kurtosis), and noise-level anomalies (RMS) without
    relying on stochastic manifold embeddings.

    Attributes:
        contamination: Expected fraction of outlier spectra
            (0 < contamination < 0.5).

    Example::

        detector = HardwareOutlierDetector(contamination=0.1)
        labels = detector.fit_predict(spectra_matrix)
        outlier_indices = np.where(labels == -1)[0]
    """

    def __init__(
        self,
        contamination: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.contamination = contamination
        self._detector = EllipticEnvelope(
            contamination=contamination, random_state=0,
        )

    @staticmethod
    def _extract_summary_features(
        spectra: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Extract per-detector summary statistics as outlier features.

        Features:
            0. Total counts (row sum)
            1. Row RMS
            2. Row skewness (Fisher–Pearson)
            3. Row excess kurtosis

        Args:
            spectra: Detector spectra, shape ``(n_spectra, n_bins)``.

        Returns:
            Feature matrix, shape ``(n_spectra, 4)``.
        """
        total = np.sum(spectra, axis=1)
        rms = np.sqrt(np.mean(np.square(spectra), axis=1))
        skew = stats.skew(spectra, axis=1, nan_policy="omit")
        kurt = stats.kurtosis(spectra, axis=1, nan_policy="omit")
        features = np.column_stack([total, rms, skew, kurt])
        return np.where(np.isfinite(features), features, 0.0)

    def fit_predict(self, spectra: NDArray[np.floating]) -> NDArray[np.intp]:
        """Fit the detector and return outlier labels.

        Extracts summary features then applies ``EllipticEnvelope``
        robust covariance outlier detection.

        Args:
            spectra: Raw detector spectra, shape ``(n_spectra, n_bins)``.

        Returns:
            Labels array, shape ``(n_spectra,)``. ``-1`` for outlier,
            ``0`` for inlier (mapped from EllipticEnvelope's +1 to
            align with DBSCAN convention).
        """
        features = self._extract_summary_features(spectra)
        self.summary_features_ = features

        raw_labels = self._detector.fit_predict(features)
        # EllipticEnvelope: -1 = outlier, +1 = inlier
        # Map +1 -> 0 (normal) to align with DBSCAN convention
        labels = np.where(raw_labels == 1, 0, -1)
        return labels


# ---------------------------------------------------------------------------
# Physics Trend Clustering
# ---------------------------------------------------------------------------

class PhysicsTrendClusterer:
    """Groups detectors by physical trends using DBSCAN.

    Input features are typically (flight-path-length L, scattering
    angle theta) pairs extracted from instrument parameter files.
    Features are standardised internally before clustering.

    Noise points (DBSCAN label ``-1``) are explicitly excluded from
    the returned cluster groups dictionary.

    Attributes:
        eps: Maximum distance between two samples for one to be
            considered in the neighbourhood of the other (in
            standardised space).
        min_samples: Minimum number of core points required to form
            a cluster.

    Example::

        clusterer = PhysicsTrendClusterer(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(features)
        groups = clusterer.get_cluster_groups(labels)
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self._scaler = StandardScaler()

    def fit_predict(self, features: NDArray[np.floating]) -> NDArray[np.intp]:
        """Clusters detector features and returns labels.

        Args:
            features: Detector feature matrix (e.g. columns [L, theta]),
                shape (n_spectra, n_features).

        Returns:
            Cluster label per spectrum, shape (n_spectra,). ``-1``
            denotes noise.
        """
        scaled = self._scaler.fit_transform(features)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = db.fit_predict(scaled)
        return labels

    @staticmethod
    def get_cluster_groups(
        labels: NDArray[np.intp],
    ) -> Dict[int, List[int]]:
        """Returns a dict mapping cluster IDs to member indices.

        Noise labels (``-1``) are excluded from the output.

        Args:
            labels: Cluster labels produced by ``fit_predict``,
                shape (n_spectra,).

        Returns:
            ``{cluster_id: [spectrum_indices]}``.
        """
        groups: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            groups.setdefault(int(label), []).append(idx)
        return groups


# ---------------------------------------------------------------------------
# Bayesian Bootstrap (Dirichlet Weights)
# ---------------------------------------------------------------------------

class BayesianBootstrap:
    """Rubin-style Weighted Bayesian Bootstrap using Dirichlet weights.

    Generates ``n_samples`` weight vectors drawn from a symmetric
    Dirichlet(1, 1, ..., 1) distribution (the uniform prior over the
    simplex).  Each weight vector has length ``n_spectra`` and sums
    to 1.0 (up to floating-point rounding).

    These weights can be applied to per-spectrum NCP residuals for
    high-speed resampling without re-fitting.

    Attributes:
        n_samples: Number of bootstrap replicas.

    References:
        Rubin, D. B. (1981). "The Bayesian Bootstrap". *Ann. Statist.*
        9(1), 130–134.

    Example::

        bootstrap = BayesianBootstrap(n_samples=1000, seed=42)
        weights = bootstrap.generate_weights(n_spectra=50)
        assert weights.shape == (1000, 50)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    """

    def __init__(self, n_samples: int = 1000,
                 seed: Optional[int] = None) -> None:
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    def generate_weights(
        self, n_spectra: int,
    ) -> NDArray[np.floating]:
        """Draws a Dirichlet weight matrix.

        Args:
            n_spectra: Number of detector spectra (simplex dimension).

        Returns:
            Weight matrix, shape (n_samples, n_spectra). Each row is a
            Dirichlet-distributed weight vector summing to 1.0 (up to
            floating-point rounding).
        """
        alpha = np.ones(n_spectra)
        weights = self._rng.dirichlet(alpha, size=self.n_samples)
        return weights

    def compute_weighted_residuals(
        self, residuals: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Computes weighted-sum residual profiles for each bootstrap sample.

        Before multiplying, rows where the residual is entirely NaN are
        zeroed out and the Dirichlet weights for those rows are zeroed and
        renormalized over the remaining valid spectra.  This prevents a
        single masked or failed spectrum from propagating NaN through the
        entire matrix multiply (``0 * NaN == NaN`` in IEEE 754).

        Args:
            residuals: Per-spectrum residuals from the NCP fit,
                shape (n_spectra, n_bins).

        Returns:
            Weighted residuals, shape (n_samples, n_bins). Each row is
            the weighted sum ``w @ residuals`` where ``w`` is a
            Dirichlet weight vector renormalized over valid spectra.
        """
        n_spectra, n_bins = residuals.shape

        # Identify rows that are entirely non-finite (masked / failed fits).
        bad_rows = ~np.any(np.isfinite(residuals), axis=1)   # (n_spectra,)
        n_bad = int(np.sum(bad_rows))
        if n_bad > 0:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "BayesianBootstrap: %d/%d residual spectrum row(s) are "
                "all-NaN (masked or failed fit).  Zeroing these rows "
                "before weighting to prevent NaN propagation.",
                n_bad, n_spectra,
            )

        # Work on a sanitized copy: replace bad rows with 0.
        clean_residuals = residuals.copy()
        clean_residuals[bad_rows, :] = 0.0
        # Replace any remaining isolated NaNs (partial-bin masking) with 0.
        np.nan_to_num(clean_residuals, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        weights = self.generate_weights(n_spectra)   # (n_samples, n_spectra)

        # Zero out weight elements for bad rows and renormalize each sample
        # so the remaining valid-spectrum weights still sum to 1.
        if n_bad > 0:
            weights[:, bad_rows] = 0.0
            row_sums = weights.sum(axis=1, keepdims=True)
            # Avoid divide-by-zero if somehow ALL rows are bad.
            safe_sums = np.where(row_sums > 0.0, row_sums, 1.0)
            weights = weights / safe_sums

        # Matrix multiply: (n_samples, n_spectra) @ (n_spectra, n_bins)
        return weights @ clean_residuals


def detector_relative_difference_metrics(
    y_obs: NDArray[np.floating],
    y_fit: NDArray[np.floating],
    eps: float = 1e-10,
) -> Dict[str, NDArray[np.floating]]:
    """Compute detector-wise calibration diagnostics from relative residuals.

    Implements the AppStat-style relative-difference observable
    ``delta = (Y_obs - Y_fit) / Y_fit`` with a small denominator floor for
    numerical stability.

    Returns detector-wise bias (mean delta) and RMS (sqrt(mean(delta^2))).
    """
    denom = np.where(np.abs(y_fit) > eps, y_fit, np.sign(y_fit) * eps + (y_fit == 0) * eps)
    delta = (y_obs - y_fit) / denom
    delta = np.where(np.isfinite(delta), delta, 0.0)
    bias = np.mean(delta, axis=1)
    rms = np.sqrt(np.mean(np.square(delta), axis=1))
    return {"delta": delta, "bias": bias, "rms": rms}


def apply_detector_intensity_calibration(
    y_obs: NDArray[np.floating],
    y_err: NDArray[np.floating],
    detector_bias: NDArray[np.floating],
    detector_mask: NDArray[np.bool_],
    max_abs_bias: float = 0.5,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Apply multiplicative detector calibration ``d_calib = d / (1 + f(x))``.

    The detector-level correction function ``f(x)`` is taken as clipped bias.
    """
    bias = detector_bias.copy()
    bias = np.clip(bias, -max_abs_bias, max_abs_bias)
    corr = np.ones_like(bias)
    corr[detector_mask] = 1.0 + bias[detector_mask]
    corr = np.where(corr <= 0.05, 0.05, corr)

    y_new = y_obs.copy()
    e_new = y_err.copy()
    y_new[detector_mask] = y_new[detector_mask] / corr[detector_mask, np.newaxis]
    e_new[detector_mask] = e_new[detector_mask] / corr[detector_mask, np.newaxis]
    return y_new, e_new, corr


def build_fidelity_labels(
    agreement: NDArray[np.floating],
    migrad_valid: NDArray[np.bool_],
    hi_fidelity_thr: float = 0.01,
    poor_fidelity_thr: float = 0.05,
) -> NDArray[np.intp]:
    """Build convergence-based labels for Fisher discriminant training.

    Labels:
      - ``0`` high-fidelity: agreement < 1% and valid MIGRAD
      - ``1`` poor-fidelity: agreement > 5% or invalid MIGRAD
      - ``-1`` unlabeled ambiguity region
    """
    labels = np.full(agreement.shape[0], -1, dtype=np.intp)
    hi = (agreement < hi_fidelity_thr) & migrad_valid
    poor = (agreement > poor_fidelity_thr) | (~migrad_valid)
    labels[hi] = 0
    labels[poor] = 1
    return labels


def build_detector_feature_matrix(
    spectra: NDArray[np.floating],
    theta_deg: NDArray[np.floating],
    width_proxy: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Assemble detector features for Fisher/LDA classification.

    Features: total counts, spectrum RMS, fitted width, scattering angle.
    """
    total_counts = np.sum(spectra, axis=1)
    row_std = np.std(spectra, axis=1)
    cols = [
        total_counts[:, np.newaxis],
        row_std[:, np.newaxis],
        width_proxy[:, np.newaxis],
        theta_deg[:, np.newaxis],
    ]
    features = np.hstack(cols)
    return np.where(np.isfinite(features), features, 0.0)


def fisher_lda_with_roc(
    features: NDArray[np.floating],
    labels: NDArray[np.intp],
) -> Optional[Dict[str, NDArray[np.floating] | float | LinearDiscriminantAnalysis]]:
    """Train Fisher/LDA on labeled detectors and compute ROC diagnostics.

    Methodology follows the AppStat Week-5 discriminator/ROC workflow:
    Fisher linear discriminant for compression + ROC/AUC for separability.
    """
    mask = labels >= 0
    if np.sum(mask) < 8:
        return None
    y = labels[mask]
    if len(np.unique(y)) < 2:
        return None

    model = LinearDiscriminantAnalysis(solver="svd")
    model.fit(features[mask], y)

    score_all = model.decision_function(features)
    score_lab = score_all[mask]

    fpr, tpr, thr = roc_curve(y, score_lab)
    roc_auc = float(auc(fpr, tpr))

    if hasattr(model, "predict_proba"):
        p_fail_all = model.predict_proba(features)[:, 1]
    else:
        p_fail_all = 1.0 / (1.0 + np.exp(-score_all))

    return {
        "model": model,
        "scores": score_all,
        "p_fail": p_fail_all,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr,
        "auc": roc_auc,
    }


def centroid_distance_scores(
    summary_features: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Standardised Euclidean distance from the bank centroid.

    Fallback Fisher-proxy used when ``fisher_lda_with_roc`` returns ``None``
    because all detectors achieve hi-fidelity convergence (no Poor-Fidelity
    class — typically the Forward bank in well-resolved samples).

    Larger values indicate detectors that deviate more from the typical bank
    behaviour, analogous to a Mahalanobis distance proxy in whitened space.

    Args:
        summary_features: Shape ``(n_det, n_features)`` summary feature matrix.

    Returns:
        1-D array of shape ``(n_det,)`` with per-detector standardised
        distances.  Suitable as a drop-in replacement for ``fisher_scores``
        in the diagnostic dashboard.
    """
    mean = np.nanmean(summary_features, axis=0)
    std = np.nanstd(summary_features, axis=0) + 1e-12
    dist = np.sqrt(np.nansum(((summary_features - mean) / std) ** 2, axis=1))
    return dist.astype(np.float64)


def detector_quality_weights(
    rms: NDArray[np.floating],
    p_fail: Optional[NDArray[np.floating]] = None,
    min_w: float = 0.05,
) -> NDArray[np.floating]:
    """Map calibration and LDA failure probability into detector weights.

    Lower RMS and lower failure probability yield larger weights.
    """
    rms_scale = float(np.nanmedian(rms) + np.nanstd(rms) + 1e-8)
    w_rms = np.exp(-np.square(rms / rms_scale))
    if p_fail is None:
        w = w_rms
    else:
        w = w_rms * (1.0 - np.clip(p_fail, 0.0, 1.0))
    w = np.clip(w, min_w, 1.0)
    return w


def plot_detector_calibration_distribution(
    bias: NDArray[np.floating],
    rms: NDArray[np.floating],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot detector bias and RMS distributions for calibration QA."""
    set_thesis_style()
    fig, (ax0, ax1) = figure_factory(ncols=2, aspect_ratio=0.55)

    ax0.hist(bias, bins=30, color=COLORBLIND_PALETTE[0], alpha=0.75)
    ax0.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax0.set_xlabel(r"Bias $\langle\delta\rangle$")
    ax0.set_ylabel("Detectors")
    ax0.set_title("Detector Bias Distribution")

    ax1.hist(rms, bins=30, color=COLORBLIND_PALETTE[2], alpha=0.75)
    ax1.set_xlabel(r"RMS$(\delta)$")
    ax1.set_ylabel("Detectors")
    ax1.set_title("Detector RMS Distribution")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_fisher_roc(
    fpr: NDArray[np.floating],
    tpr: NDArray[np.floating],
    roc_auc: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot ROC curve for Fisher/LDA detector-failure prediction."""
    set_thesis_style()
    fig, ax = figure_factory()
    ax.plot(fpr, tpr, color=COLORBLIND_PALETTE[3], label=f"Fisher LDA (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Convergence-Failure Prediction")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_feature_lda_overlay(
    summary_features: NDArray[np.floating],
    p_fail: NDArray[np.floating],
    labels: Optional[NDArray[np.intp]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay LDA failure probability onto the summary-feature scatter.

    Uses the first two summary features (total counts vs RMS) as axes.
    """
    set_thesis_style()
    fig, ax = figure_factory()
    sc = ax.scatter(
        summary_features[:, 0],
        summary_features[:, 1],
        c=p_fail,
        cmap="viridis",
        s=24,
        alpha=0.9,
    )
    if labels is not None:
        poor = labels == 1
        if np.any(poor):
            ax.scatter(
                summary_features[poor, 0],
                summary_features[poor, 1],
                facecolors="none",
                edgecolors=COLORBLIND_PALETTE[3],
                s=60,
                linewidths=1.0,
                label="Poor-Fidelity",
            )
            ax.legend(fontsize=8)
    ax.set_xlabel("Total Counts")
    ax.set_ylabel("RMS")
    ax.set_title("Feature Space + Fisher/LDA Failure Probability")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Predicted failure probability")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Diagnostic Visualisation
# ---------------------------------------------------------------------------


def plot_outlier_scatter(
    summary_features: NDArray[np.floating],
    labels: NDArray[np.intp],
    save_path: Optional[Path] = None,
    summary_stats: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """Scatter plot in summary-feature space highlighting outlier detectors.

    Renders the first two summary features (total counts vs RMS).
    Inlier detectors are drawn in the first colour of the
    COLORBLIND_PALETTE; outliers (``label == -1``) are drawn in red.

    When *summary_stats* is provided, a metadata table is rendered on
    the figure showing Total Detectors, Outliers Flagged, and DBSCAN
    Clusters for transparent traceability.

    Args:
        summary_features: 2-D array of summary features, shape
            ``(n_spectra, >=2)``.  At least 2 columns are required.
        labels: Outlier labels produced by
            :meth:`HardwareOutlierDetector.fit_predict`, shape
            ``(n_spectra,)``.  ``-1`` = outlier, ``0`` = inlier.
        save_path: Optional file path.  When provided the figure is
            saved and closed; otherwise it is returned open.
        summary_stats: Optional dict with keys ``n_total``,
            ``n_outliers``, and optionally ``n_clusters``.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, ax = figure_factory()

    inlier_mask = labels == 0
    outlier_mask = labels == -1

    ax.scatter(
        summary_features[inlier_mask, 0], summary_features[inlier_mask, 1],
        color=COLORBLIND_PALETTE[0], s=20, label="Inlier",
    )
    ax.scatter(
        summary_features[outlier_mask, 0], summary_features[outlier_mask, 1],
        color="#D62728", marker="x", s=60, linewidths=1.5,
        label=f"Outlier (n={int(outlier_mask.sum())})",
    )
    ax.set_xlabel("Total Counts")
    ax.set_ylabel("RMS")
    ax.set_title("Hardware Outlier Detection — Feature Space")
    ax.legend()

    # --- Summary metadata table ---
    if summary_stats is not None:
        n_total = summary_stats.get("n_total", len(labels))
        n_outliers = summary_stats.get("n_outliers", int(outlier_mask.sum()))
        pct = 100.0 * n_outliers / max(n_total, 1)
        n_clusters = summary_stats.get("n_clusters", "—")
        table_text = (
            f"Total Detectors: {n_total}\n"
            f"Outliers Flagged: {n_outliers} ({pct:.1f}%)\n"
            f"DBSCAN Clusters: {n_clusters}"
        )
        ax.text(
            0.02, 0.02, table_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_outlier_before_after(
    features_before: NDArray[np.floating],
    labels_before: NDArray[np.intp],
    features_after: NDArray[np.floating],
    labels_after: NDArray[np.intp],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Side-by-side feature-space scatter: before and after outlier removal.

    Left panel shows the original feature scatter with outliers
    highlighted.  Right panel shows the clean spectra after masking.

    Args:
        features_before: Summary features before masking, shape
            ``(n_spectra, >=2)``.
        labels_before: Outlier labels before masking (``-1`` = outlier).
        features_after: Summary features after masking, shape
            ``(n_clean, >=2)``.
        labels_after: Outlier labels after masking.
        save_path: Optional file path for saving.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, (ax_l, ax_r) = figure_factory(ncols=2, aspect_ratio=0.5)

    # --- Left panel: before ---
    inlier_b = labels_before == 0
    outlier_b = labels_before == -1
    ax_l.scatter(
        features_before[inlier_b, 0], features_before[inlier_b, 1],
        color=COLORBLIND_PALETTE[0], s=20, label="Inlier",
    )
    ax_l.scatter(
        features_before[outlier_b, 0], features_before[outlier_b, 1],
        color="#D62728", marker="x", s=60, linewidths=1.5,
        label=f"Outlier (n={int(outlier_b.sum())})",
    )
    ax_l.set_xlabel("Total Counts")
    ax_l.set_ylabel("RMS")
    ax_l.set_title("Before Outlier Removal")
    ax_l.legend(fontsize=7)

    # --- Right panel: after ---
    inlier_a = labels_after == 0
    outlier_a = labels_after == -1
    ax_r.scatter(
        features_after[inlier_a, 0], features_after[inlier_a, 1],
        color=COLORBLIND_PALETTE[0], s=20, label="Inlier",
    )
    if outlier_a.any():
        ax_r.scatter(
            features_after[outlier_a, 0], features_after[outlier_a, 1],
            color="#D62728", marker="x", s=60, linewidths=1.5,
            label=f"Outlier (n={int(outlier_a.sum())})",
        )
    ax_r.set_xlabel("Total Counts")
    ax_r.set_ylabel("")
    ax_r.set_title("After Outlier Removal")
    ax_r.legend(fontsize=7)

    fig.suptitle("Hardware Outlier Detection — Feature Space", fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_cluster_ltheta(
    features: NDArray[np.floating],
    labels: NDArray[np.intp],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """L vs theta scatter plot with DBSCAN cluster colouring.

    Each cluster receives a distinct colour from COLORBLIND_PALETTE.
    Noise points (``label == -1``) are rendered as grey crosses and
    labelled "Noise".

    Args:
        features: Detector feature matrix with columns ``[L, theta]``,
            shape ``(n_spectra, 2)``.
        labels: Cluster labels from
            :meth:`PhysicsTrendClusterer.fit_predict`, shape
            ``(n_spectra,)``.  ``-1`` = noise.
        save_path: Optional file path for saving.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, ax = figure_factory()

    unique_clusters = sorted(set(labels) - {-1})
    for k, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        colour = COLORBLIND_PALETTE[k % len(COLORBLIND_PALETTE)]
        ax.scatter(
            features[mask, 0], features[mask, 1],
            color=colour, s=20,
            label=f"Cluster {cluster_id} (n={int(mask.sum())})",
        )

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            features[noise_mask, 0], features[noise_mask, 1],
            color="grey", marker="x", s=40, linewidths=1.2,
            label=f"Noise (n={int(noise_mask.sum())})",
        )

    ax.set_xlabel("Flight-path length L (m)")
    ax.set_ylabel(r"Scattering angle $\theta$ (°)")
    ax.set_title("Detector Clustering — L vs θ")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_bayesian_corner(
    samples: NDArray[np.floating],
    param_names: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Corner plot of pairwise parameter correlations from Bayesian Bootstrap.

    Produces a lower-triangular grid.  Diagonal panels show marginal
    histograms; off-diagonal panels show pairwise scatter plots.

    Args:
        samples: Parameter samples, shape ``(n_samples, n_params)``.
        param_names: Human-readable parameter names, length ``n_params``.
        save_path: Optional file path for saving.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    n_params = samples.shape[1]
    set_thesis_style()
    fig, axes = figure_factory(
        aspect_ratio=1.0,
        nrows=n_params, ncols=n_params,
    )
    # Ensure axes is always 2-D
    if n_params == 1:
        axes = np.array([[axes]])

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if row == col:
                ax.hist(
                    samples[:, row], bins=30,
                    color=COLORBLIND_PALETTE[row % len(COLORBLIND_PALETTE)],
                    density=True,
                )
            else:
                ax.scatter(
                    samples[:, col], samples[:, row],
                    s=4, alpha=0.3,
                    color=COLORBLIND_PALETTE[col % len(COLORBLIND_PALETTE)],
                )
            if col == 0:
                ax.set_ylabel(param_names[row], fontsize=9)
            if row == n_params - 1:
                ax.set_xlabel(param_names[col], fontsize=9)

    fig.suptitle("Bayesian Bootstrap — Parameter Corner Plot")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_posterior_kde(
    samples: NDArray[np.floating],
    point_estimates: NDArray[np.floating],
    param_names: List[str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """KDE of Bayesian Bootstrap posteriors alongside frequentist estimates.

    Each parameter receives one subplot showing the KDE curve (from the
    bootstrap sample column-means) and a vertical line at the
    frequentist point estimate.

    Args:
        samples: Bootstrap posterior samples, shape
            ``(n_samples, n_params)``.  Each column is one parameter.
        point_estimates: Frequentist point-estimate values, shape
            ``(n_params,)``.
        param_names: Human-readable parameter names, length ``n_params``.
        save_path: Optional file path for saving.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    n_params = samples.shape[1]
    set_thesis_style()
    fig, axes = figure_factory(
        aspect_ratio=0.5,
        ncols=n_params,
    )
    if n_params == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        col = samples[:, k]
        kde = stats.gaussian_kde(col)
        x_grid = np.linspace(col.min(), col.max(), 200)
        colour = COLORBLIND_PALETTE[k % len(COLORBLIND_PALETTE)]
        ax.plot(x_grid, kde(x_grid), color=colour, label="Posterior KDE")
        ax.axvline(
            point_estimates[k], color="#D62728", linestyle="--",
            label=f"Frequentist\n{point_estimates[k]:.4g}",
        )
        ax.set_xlabel(param_names[k], fontsize=9)
        ax.set_ylabel("Density" if k == 0 else "")
        ax.legend(fontsize=7)

    fig.suptitle("Posterior Distributions — Bayesian Bootstrap")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_optimizer_residuals(
    x: NDArray[np.floating],
    scipy_fit: NDArray[np.floating],
    iminuit_fit: NDArray[np.floating],
    rel_diff_pct: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Residuals comparison between iMinuit and Scipy fits.

    The upper panel shows both fit curves overlaid on the same axes.
    The lower panel shows the point-wise difference
    ``iminuit_fit - scipy_fit``.  The maximum relative difference (%)
    is included in the legend.

    Args:
        x: Common x-axis values (e.g. y-space bins), shape
            ``(n_bins,)``.
        scipy_fit: Best-fit model evaluated at ``x`` by Scipy,
            shape ``(n_bins,)``.
        iminuit_fit: Best-fit model evaluated at ``x`` by iMinuit,
            shape ``(n_bins,)``.
        rel_diff_pct: Pre-computed maximum relative difference
            between the two fits (%), included in the legend.
        save_path: Optional file path for saving.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, (ax_top, ax_bot) = figure_factory(nrows=2, aspect_ratio=0.8)

    ax_top.plot(x, scipy_fit, color=COLORBLIND_PALETTE[0], label="Scipy fit")
    ax_top.plot(
        x, iminuit_fit, color=COLORBLIND_PALETTE[3], linestyle="--",
        label=f"iMinuit fit  (Δ_max={rel_diff_pct:.2f} %)",
    )
    ax_top.set_ylabel("Model value")
    ax_top.legend()

    residuals = iminuit_fit - scipy_fit
    ax_bot.plot(x, residuals, color=COLORBLIND_PALETTE[1])
    ax_bot.axhline(0, color="k", linewidth=0.8, linestyle=":")
    ax_bot.set_xlabel("y-space (Å⁻¹)")
    ax_bot.set_ylabel("iMinuit − Scipy")
    ax_top.set_title(
        f"Optimizer Cross-Check — Residuals  (max rel. diff = {rel_diff_pct:.2f} %)"
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_bootstrap_convergence(
    weighted_residuals: NDArray[np.floating],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bootstrap convergence diagnostic: histogram + KDE of replica means.

    Computes the mean of each bootstrap replica's weighted-residual
    profile, then overlays a normalised histogram with a Gaussian KDE
    to assess distributional convergence.  The bootstrap mean and
    standard deviation are annotated.

    Args:
        weighted_residuals: Weighted residual profiles from
            :meth:`BayesianBootstrap.compute_weighted_residuals`,
            shape ``(n_samples, n_bins)``.
        save_path: Optional file path for saving.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, ax = figure_factory()

    replica_means = np.mean(weighted_residuals, axis=1)

    # NaN-safety: drop non-finite values before histogram/KDE
    finite_mask = np.isfinite(replica_means)
    n_dropped = int(np.sum(~finite_mask))
    if n_dropped > 0:
        logger.warning(
            "plot_bootstrap_convergence: dropped %d non-finite replica means",
            n_dropped,
        )
    replica_means = replica_means[finite_mask]

    if len(replica_means) < 2:
        ax.text(0.5, 0.5, "Insufficient finite replicas",
                ha="center", va="center", transform=ax.transAxes)
        if save_path is not None:
            fig.savefig(save_path)
            plt.close(fig)
        return fig

    n_samples = len(replica_means)
    mu = float(np.mean(replica_means))
    sigma = float(np.std(replica_means))

    ax.hist(
        replica_means, bins=min(50, max(10, n_samples // 20)),
        density=True, color=COLORBLIND_PALETTE[0], alpha=0.6,
        edgecolor="white", linewidth=0.5, label="Histogram",
    )

    kde = stats.gaussian_kde(replica_means)
    x_grid = np.linspace(replica_means.min(), replica_means.max(), 300)
    ax.plot(x_grid, kde(x_grid), color=COLORBLIND_PALETTE[3],
            linewidth=1.8, label="KDE")

    ax.axvline(mu, color="#D62728", linestyle="--", linewidth=1.2,
               label=rf"$\mu = {mu:.4g}$")
    ax.axvspan(mu - sigma, mu + sigma, alpha=0.10, color=COLORBLIND_PALETTE[1],
               label=rf"$\pm 1\sigma = {sigma:.4g}$")

    ax.set_xlabel("Bootstrap replica mean residual")
    ax.set_ylabel("Density")
    ax.set_title(f"Bayesian Bootstrap Convergence  ($n = {n_samples}$)")
    ax.legend(fontsize=7)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Publication-quality NCP fit figure
# ---------------------------------------------------------------------------


def plot_sum_ncp_fits_publication(
    tof: NDArray[np.floating],
    data_y: NDArray[np.floating],
    data_e: NDArray[np.floating],
    total_ncp: NDArray[np.floating],
    mass_ncps: List[NDArray[np.floating]],
    masses: List[float],
    title: str,
    metadata: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Publication-quality two-panel NCP fit figure with residuals subplot.

    Renders the summed data alongside fitted NCP profiles in the upper
    panel (height ratio 3) and the normalised fit residuals in the lower
    panel (height ratio 1).  All five thesis-publication requirements are
    addressed:

    - **Title**: pre-formatted string passed in as *title*.
    - **Metadata box**: anchored ``'upper left'`` showing included spectra,
      outlier count, and :math:`\\chi^2/\\mathrm{ndof}`.
    - **Legend labels**: parameter names transformed via
      :data:`_PARAM_LABEL_MAP` (``'sigma'`` → ``$\\sigma_p$``,
      ``'x0'`` → ``$y_\\mathrm{center}$``).
    - **Residuals panel**: :math:`(d_i - f_i)/\\sigma_i` with 1-:math:`\\sigma`
      error bars and a dashed zero-line.
    - **Visual consistency**: white background, inward ticks on all axes.

    This function is Mantid-free; call it after extracting NumPy arrays
    from Mantid workspaces in ``plotSumNCPFits`` (``analysis_functions``).

    Args:
        tof: 1-D TOF x-axis, shape ``(n_bins,)``.
        data_y: Summed-spectra data counts, shape ``(n_bins,)``.
        data_e: 1-:math:`\\sigma` data errors, shape ``(n_bins,)``.
        total_ncp: Total NCP fit profile, shape ``(n_bins,)``.
        mass_ncps: Per-mass NCP components, each shape ``(n_bins,)``.
        masses: Atomic masses (u) corresponding to *mass_ncps*.
        title: Pre-formatted figure title string.
        metadata: Optional display metadata.  Recognised keys:

            - ``'included_spec_ids'``: ``list[int]`` of spectrum IDs
              used in the sum.
            - ``'n_outliers'``: ``int`` count of masked spectra.
            - ``'chi2'``: ``float`` :math:`\\chi^2` value.
            - ``'ndof'``: ``int`` degrees of freedom.

        save_path: When provided the figure is saved (300 dpi, white
            background) and closed; otherwise it is returned open.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()

    fig_w = cm_to_inches(FULL_WIDTH_CM)
    fig_h = cm_to_inches(FULL_WIDTH_CM * 0.85)   # slightly taller for two panels
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.06)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)

    # --- Upper panel: data + NCP components ---
    ax_main.errorbar(
        tof, data_y, yerr=data_e,
        color="black",
        linestyle="None", marker="o", markersize=2,
        capsize=0, elinewidth=0.6, alpha=1.0, zorder=3,
        label=_apply_latex_labels("Data"),
    )
    ax_main.plot(
        tof, total_ncp,
        color="#D62728",
        linestyle="-", linewidth=1.5, alpha=0.85, zorder=2,
        label=_apply_latex_labels("Total NCP"),
    )
    for k, (m, ncp_y) in enumerate(zip(masses, mass_ncps)):
        raw_label = f"NCP  $m = {m:.4g}$ u"
        ax_main.plot(
            tof, ncp_y,
            color=COLORBLIND_PALETTE[k % len(COLORBLIND_PALETTE)],
            linestyle="--", linewidth=0.9, alpha=0.85, zorder=2,
            label=_apply_latex_labels(raw_label),
        )

    ax_main.set_ylabel(r"Counts (a.u.)")
    ax_main.set_title(title, fontsize=10, pad=4)
    ax_main.set_facecolor("white")
    ax_main.tick_params(axis="both", direction="in", which="both",
                        top=True, right=True)
    ax_main.legend(fontsize=7, framealpha=0.85, loc="upper right")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # --- Metadata textbox (upper left, anchored inside main panel) ---
    meta_lines: List[str] = []
    if "included_spec_ids" in metadata:
        spec_ids: List[int] = list(metadata["included_spec_ids"])
        if len(spec_ids) > 8:
            meta_lines.append(
                f"Included Spectra: {spec_ids[0]}\u2013{spec_ids[-1]}"
                f"  ({len(spec_ids)} total)"
            )
        else:
            meta_lines.append(f"Included Spectra: {spec_ids}")
    if "n_outliers" in metadata:
        meta_lines.append(f"Outliers Masked: {int(metadata['n_outliers'])}")
    if "chi2" in metadata and "ndof" in metadata:
        chi2_v = float(metadata["chi2"])
        ndof_v = int(metadata["ndof"])
        meta_lines.append(
            rf"$\chi^2/\mathrm{{ndof}}$: {chi2_v:.1f}/{ndof_v} = {chi2_v / ndof_v:.2f}"
        )
    if meta_lines:
        ax_main.text(
            0.02, 0.98,
            "\n".join(meta_lines),
            transform=ax_main.transAxes,
            fontsize=7.5,
            verticalalignment="top",
            horizontalalignment="left",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.88,
                edgecolor="#AAAAAA",
                linewidth=0.6,
            ),
        )

    # --- Lower panel: normalised residuals ---
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = np.where(data_e > 0, (data_y - total_ncp) / data_e, np.nan)
    valid = np.isfinite(pull)
    ax_res.errorbar(
        tof[valid], pull[valid],
        yerr=np.ones(int(valid.sum())),
        color="black",
        linestyle="None", marker="o", markersize=2,
        capsize=2, elinewidth=0.6, zorder=3,
    )
    ax_res.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9)
    ax_res.set_xlabel(r"TOF ($\mu$s)")
    ax_res.set_ylabel(r"Residuals ($\sigma$)", fontsize=9)
    ax_res.set_facecolor("white")
    ax_res.tick_params(axis="both", direction="in", which="both",
                       top=True, right=True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path, bbox_inches="tight", pad_inches=0.05, facecolor="white",
        )
        plt.close(fig)
        logger.info("plot_sum_ncp_fits_publication: saved to %s", save_path)
    return fig
