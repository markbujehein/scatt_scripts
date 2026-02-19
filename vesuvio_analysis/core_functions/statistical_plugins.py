"""Phase 6 — Full-Stack Statistical Workflow for the VESUVIO pipeline.

Provides statistical post-processing for the VESUVIO analysis pipeline:
hardware outlier identification, density-based detector clustering, and
probabilistic uncertainty quantification via Bayesian Bootstrap.

Classes:
    HardwareOutlierDetector: PCA-based anomaly detection for broken
        detectors.
    PhysicsTrendClusterer: DBSCAN clustering of detector features
        (L, theta).
    BayesianBootstrap: Rubin-style Weighted Bayesian Bootstrap with
        Dirichlet weights for high-speed resampling of NCP residuals.

Notes:
    - scikit-learn DBSCAN labels noise points as -1; these are explicitly
      excluded from physics-trend groups.
    - PCA requires standardised (zero-mean, unit-variance) input; this is
      handled internally by ``HardwareOutlierDetector``.
    - Dirichlet(1, 1, ..., 1) produces the uniform prior over the simplex
      (Rubin, 1981).  Each weight vector sums to 1.0 (up to
      floating-point rounding).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .plot_style import COLORBLIND_PALETTE, figure_factory, set_thesis_style


# ---------------------------------------------------------------------------
# Hardware Outlier Detection
# ---------------------------------------------------------------------------

class HardwareOutlierDetector:
    """Identifies broken detectors via PCA + robust covariance scoring.

    Each row of the input matrix is a detector spectrum.  The spectra
    are standardised and projected onto ``n_components`` principal
    components.  Outliers are detected in the reduced space using a
    robust covariance estimator (``EllipticEnvelope``).

    Attributes:
        n_components: Number of PCA components to retain.
        contamination: Expected fraction of outlier spectra
            (0 < contamination < 0.5).

    Example::

        detector = HardwareOutlierDetector(n_components=5, contamination=0.1)
        labels = detector.fit_predict(spectra_matrix)
        outlier_indices = np.where(labels == -1)[0]
    """

    def __init__(self, n_components: int = 5,
                 contamination: float = 0.1) -> None:
        self.n_components = n_components
        self.contamination = contamination
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=n_components)
        self._detector = EllipticEnvelope(
            contamination=contamination, random_state=0,
        )

    def fit_predict(self, spectra: NDArray[np.floating]) -> NDArray[np.intp]:
        """Fits the detector and returns outlier labels.

        Args:
            spectra: Raw detector spectra, shape (n_spectra, n_bins).

        Returns:
            Labels array, shape (n_spectra,). ``-1`` for outlier, ``0``
            for inlier (mapped from EllipticEnvelope's +1 to align with
            DBSCAN convention).
        """
        scaled = self._scaler.fit_transform(spectra)
        reduced = self._pca.fit_transform(scaled)
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

        Args:
            residuals: Per-spectrum residuals from the NCP fit,
                shape (n_spectra, n_bins).

        Returns:
            Weighted residuals, shape (n_samples, n_bins). Each row is
            the weighted sum ``w @ residuals`` where ``w`` is a
            Dirichlet weight vector.
        """
        n_spectra = residuals.shape[0]
        weights = self.generate_weights(n_spectra)
        # Matrix multiply: (n_samples, n_spectra) @ (n_spectra, n_bins)
        return weights @ residuals


# ---------------------------------------------------------------------------
# Diagnostic Visualisation
# ---------------------------------------------------------------------------


def plot_outlier_scatter(
    pca_coords: NDArray[np.floating],
    labels: NDArray[np.intp],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot in PCA space highlighting outlier detectors in red.

    Renders the first two principal components.  Inlier detectors are
    drawn in the first colour of the COLORBLIND_PALETTE; outliers
    (``label == -1``) are drawn in red.

    Args:
        pca_coords: 2-D array of PCA projections, shape
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
        pca_coords[inlier_mask, 0], pca_coords[inlier_mask, 1],
        color=COLORBLIND_PALETTE[0], s=20, label="Inlier",
    )
    ax.scatter(
        pca_coords[outlier_mask, 0], pca_coords[outlier_mask, 1],
        color="#D62728", marker="x", s=60, linewidths=1.5,
        label=f"Outlier (n={int(outlier_mask.sum())})",
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Hardware Outlier Detection — PCA Space")
    ax.legend()
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
    fig, axes = plt.subplots(
        n_params, n_params,
        figsize=(
            n_params * 2.5,
            n_params * 2.5,
        ),
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
    fig, axes = plt.subplots(
        1, n_params,
        figsize=(max(6.0, n_params * 2.5), 3.5),
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
