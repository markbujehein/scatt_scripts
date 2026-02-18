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
      (Rubin, 1981).  Each weight vector sums exactly to 1.0.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    exactly to 1.0.

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
            Dirichlet-distributed weight vector summing to 1.0.
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
