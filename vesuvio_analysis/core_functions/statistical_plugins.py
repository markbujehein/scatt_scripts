"""Phase 6 — Full-Stack Statistical Workflow for the VESUVIO pipeline.

Provides statistical post-processing for the VESUVIO analysis pipeline:
hardware outlier identification, density-based detector clustering, and
probabilistic uncertainty quantification via Bayesian Bootstrap.

Classes:
    HardwareOutlierDetector: UMAP + robust-covariance anomaly detection
        for broken detectors.
    PhysicsTrendClusterer: DBSCAN clustering of detector features
        (L, theta).
    BayesianBootstrap: Rubin-style Weighted Bayesian Bootstrap with
        Dirichlet weights for high-speed resampling of NCP residuals.

Notes:
    - scikit-learn DBSCAN labels noise points as -1; these are explicitly
      excluded from physics-trend groups.
    - UMAP preserves local topological structure via a fuzzy simplicial
      set construction (McInnes, Healy & Melville, 2018,
      arXiv:1802.03426).  This is critical for spectroscopic data where
      non-linear variance arises from kinematic TOF broadening and
      $J(y)$ scaling — linear PCA cannot capture these manifold
      curvatures.
    - Dirichlet(1, 1, ..., 1) produces the uniform prior over the simplex
      (Rubin, 1981).  Each weight vector sums to 1.0 (up to
      floating-point rounding).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

from vesuvio_analysis.core_functions.plot_style import COLORBLIND_PALETTE, figure_factory, set_thesis_style

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware Outlier Detection
# ---------------------------------------------------------------------------

class HardwareOutlierDetector:
    """Identifies broken detectors via UMAP + robust covariance scoring.

    Each row of the input matrix is a detector spectrum.  The spectra
    are standardised, then projected onto a low-dimensional manifold
    using UMAP (Uniform Manifold Approximation and Projection).
    Outliers are detected in the reduced space using a robust
    covariance estimator (``EllipticEnvelope``).

    UMAP is preferred over linear PCA because TOF spectroscopic data
    exhibit non-linear variance from kinematic broadening and the
    $J(y)$ scaling relation.  UMAP preserves local topological
    structure via a fuzzy simplicial set construction, faithfully
    representing the detector manifold curvature that PCA collapses
    (McInnes, Healy & Melville, 2018, arXiv:1802.03426).

    Attributes:
        n_components: Number of UMAP embedding dimensions.
        n_neighbors: Size of the local neighbourhood used by UMAP
            for manifold approximation.  Controls the balance between
            local and global structure preservation.
        min_dist: Minimum distance between embedded points in UMAP.
            Smaller values produce tighter clusters.
        contamination: Expected fraction of outlier spectra
            (0 < contamination < 0.5).

    Example::

        detector = HardwareOutlierDetector(
            n_components=2, n_neighbors=15, min_dist=0.1,
        )
        labels = detector.fit_predict(spectra_matrix)
        outlier_indices = np.where(labels == -1)[0]
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        contamination: float = 0.1,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.contamination = contamination
        self._scaler = StandardScaler()
        self._detector = EllipticEnvelope(
            contamination=contamination, random_state=0,
        )

    def fit_predict(self, spectra: NDArray[np.floating]) -> NDArray[np.intp]:
        """Fit the detector and return outlier labels.

        Applies UMAP dimensionality reduction followed by
        ``EllipticEnvelope`` robust covariance outlier detection.

        Args:
            spectra: Raw detector spectra, shape ``(n_spectra, n_bins)``.

        Returns:
            Labels array, shape ``(n_spectra,)``. ``-1`` for outlier,
            ``0`` for inlier (mapped from EllipticEnvelope's +1 to
            align with DBSCAN convention).
        """
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ImportError(
                "umap-learn is required for UMAP-based outlier detection.  "
                "Install with: pip install umap-learn"
            ) from exc

        scaled = self._scaler.fit_transform(spectra)

        # UMAP preserves local topological structure of the spectral
        # manifold (McInnes et al., 2018).  n_neighbors controls the
        # trade-off between local vs global structure; min_dist controls
        # cluster compactness in the embedding.
        reducer = UMAP(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, spectra.shape[0] - 1),
            min_dist=self.min_dist,
            random_state=0,
            n_jobs=1,
        )
        reduced = reducer.fit_transform(scaled)
        self.embedding_coords_ = reduced

        raw_labels = self._detector.fit_predict(reduced)
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
    umap_embedding: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Assemble detector features for Fisher/LDA classification.

    Features include total counts, spectrum RMS shape proxy, fitted width,
    scattering angle, and optional UMAP coordinates.
    """
    total_counts = np.sum(spectra, axis=1)
    row_std = np.std(spectra, axis=1)
    cols = [
        total_counts[:, np.newaxis],
        row_std[:, np.newaxis],
        width_proxy[:, np.newaxis],
        theta_deg[:, np.newaxis],
    ]
    if umap_embedding is not None and umap_embedding.shape[0] == spectra.shape[0]:
        cols.append(umap_embedding[:, :2])
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


def plot_umap_lda_overlay(
    embedding_coords: NDArray[np.floating],
    p_fail: NDArray[np.floating],
    labels: Optional[NDArray[np.intp]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay LDA failure probability onto the UMAP detector embedding."""
    set_thesis_style()
    fig, ax = figure_factory()
    sc = ax.scatter(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        c=p_fail,
        cmap="viridis",
        s=24,
        alpha=0.9,
    )
    if labels is not None:
        poor = labels == 1
        if np.any(poor):
            ax.scatter(
                embedding_coords[poor, 0],
                embedding_coords[poor, 1],
                facecolors="none",
                edgecolors=COLORBLIND_PALETTE[3],
                s=60,
                linewidths=1.0,
                label="Poor-Fidelity",
            )
            ax.legend(fontsize=8)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP + Fisher/LDA Failure Probability")
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
    embedding_coords: NDArray[np.floating],
    labels: NDArray[np.intp],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot in UMAP embedding space highlighting outlier detectors.

    Renders the first two UMAP embedding dimensions.  Inlier detectors
    are drawn in the first colour of the COLORBLIND_PALETTE; outliers
    (``label == -1``) are drawn in red.

    Args:
        embedding_coords: 2-D array of UMAP projections, shape
            ``(n_spectra, n_components)``.  At least 2 columns are
            required.
        labels: Outlier labels produced by
            :meth:`HardwareOutlierDetector.fit_predict`, shape
            ``(n_spectra,)``.  ``-1`` = outlier, ``0`` = inlier.
        save_path: Optional file path.  When provided the figure is
            saved and closed; otherwise it is returned open.

    Returns:
        The Matplotlib :class:`~matplotlib.figure.Figure`.
    """
    set_thesis_style()
    fig, ax = figure_factory()

    inlier_mask = labels == 0
    outlier_mask = labels == -1

    ax.scatter(
        embedding_coords[inlier_mask, 0], embedding_coords[inlier_mask, 1],
        color=COLORBLIND_PALETTE[0], s=20, label="Inlier",
    )
    ax.scatter(
        embedding_coords[outlier_mask, 0], embedding_coords[outlier_mask, 1],
        color="#D62728", marker="x", s=60, linewidths=1.5,
        label=f"Outlier (n={int(outlier_mask.sum())})",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Hardware Outlier Detection — UMAP Embedding")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_outlier_before_after(
    embedding_before: NDArray[np.floating],
    labels_before: NDArray[np.intp],
    embedding_after: NDArray[np.floating],
    labels_after: NDArray[np.intp],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Side-by-side UMAP scatter: before and after outlier removal.

    Left panel shows the original UMAP embedding with outliers
    highlighted.  Right panel shows the re-embedded clean spectra
    after masking.

    Args:
        embedding_before: UMAP projections before masking, shape
            ``(n_spectra, >=2)``.
        labels_before: Outlier labels before masking (``-1`` = outlier).
        embedding_after: UMAP projections after masking, shape
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
        embedding_before[inlier_b, 0], embedding_before[inlier_b, 1],
        color=COLORBLIND_PALETTE[0], s=20, label="Inlier",
    )
    ax_l.scatter(
        embedding_before[outlier_b, 0], embedding_before[outlier_b, 1],
        color="#D62728", marker="x", s=60, linewidths=1.5,
        label=f"Outlier (n={int(outlier_b.sum())})",
    )
    ax_l.set_xlabel("UMAP 1")
    ax_l.set_ylabel("UMAP 2")
    ax_l.set_title("Before Outlier Removal")
    ax_l.legend(fontsize=7)

    # --- Right panel: after ---
    inlier_a = labels_after == 0
    outlier_a = labels_after == -1
    ax_r.scatter(
        embedding_after[inlier_a, 0], embedding_after[inlier_a, 1],
        color=COLORBLIND_PALETTE[0], s=20, label="Inlier",
    )
    if outlier_a.any():
        ax_r.scatter(
            embedding_after[outlier_a, 0], embedding_after[outlier_a, 1],
            color="#D62728", marker="x", s=60, linewidths=1.5,
            label=f"Outlier (n={int(outlier_a.sum())})",
        )
    ax_r.set_xlabel("UMAP 1")
    ax_r.set_ylabel("")
    ax_r.set_title("After Outlier Removal")
    ax_r.legend(fontsize=7)

    fig.suptitle("UMAP Hardware Outlier Detection", fontsize=11)
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
