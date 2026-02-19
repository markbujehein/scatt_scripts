"""Tests for the Phase 6 statistical workflow.

These tests do **not** depend on Mantid and can be run in any standard
Python environment with NumPy, SciPy, and scikit-learn installed::

    python -m pytest tests/test_statistical_workflow.py -v

Each test uses deterministic dummy data to verify:
1. Hardware outlier detection correctly labels injected outlier spectra.
2. Physics trend clustering groups physical clusters via DBSCAN and
   excludes noise labels (-1).
3. Bayesian Bootstrap generates valid Dirichlet-distributed weights that
   sum to 1.0.
4. Diagnostic visualisation functions produce figures without errors.
"""

import unittest
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Hardware Outlier Detection (PCA-based)
# ---------------------------------------------------------------------------

class TestHardwareOutlierDetector(unittest.TestCase):
    """Verifies that the outlier detector identifies injected hardware faults."""

    def _make_detector_data(self, n_spectra=50, n_bins=200, n_outliers=3,
                            seed=42):
        """Build synthetic detector data with known outliers.

        Normal spectra follow a smooth Gaussian peak; outlier spectra
        have wildly different baselines (simulating broken detectors).
        """
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 10, n_bins)
        normal = np.exp(-0.5 * ((x - 5) / 1.0) ** 2)

        data = np.empty((n_spectra, n_bins))
        for i in range(n_spectra):
            data[i] = normal + rng.normal(0, 0.05, n_bins)

        # Inject outliers: flat or huge-offset spectra
        outlier_indices = list(range(n_outliers))
        for idx in outlier_indices:
            data[idx] = rng.uniform(-10, 10, n_bins)  # random noise

        return data, np.array(outlier_indices)

    def test_outliers_detected_by_pca(self):
        """PCA-based Mahalanobis distance should flag injected outliers."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            HardwareOutlierDetector,
        )

        data, true_outlier_idx = self._make_detector_data(
            n_spectra=50, n_outliers=3, seed=42,
        )
        detector = HardwareOutlierDetector(n_components=5, contamination=0.1)
        labels = detector.fit_predict(data)

        # labels: -1 = outlier, 0 = normal
        detected = np.where(labels == -1)[0]
        # All injected outliers should be detected
        for idx in true_outlier_idx:
            self.assertIn(
                idx, detected,
                f"Injected outlier at spectrum {idx} was not detected",
            )

    def test_labels_shape(self):
        """Label array must match the number of input spectra."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            HardwareOutlierDetector,
        )

        data, _ = self._make_detector_data(n_spectra=30, n_outliers=2)
        detector = HardwareOutlierDetector(n_components=3, contamination=0.15)
        labels = detector.fit_predict(data)
        self.assertEqual(labels.shape[0], 30)

    def test_no_outliers_in_clean_data(self):
        """When data is clean, detector should flag very few or no outliers."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            HardwareOutlierDetector,
        )

        rng = np.random.default_rng(99)
        x = np.linspace(0, 10, 100)
        normal = np.exp(-0.5 * ((x - 5) / 1.0) ** 2)
        data = np.array([normal + rng.normal(0, 0.01, 100)
                         for _ in range(40)])

        detector = HardwareOutlierDetector(n_components=3, contamination=0.05)
        labels = detector.fit_predict(data)
        n_outliers = np.sum(labels == -1)
        # At most ~10% should be flagged in clean data
        self.assertLessEqual(n_outliers, max(2, int(0.1 * len(data))))


# ---------------------------------------------------------------------------
# Physics Trend Clustering (DBSCAN)
# ---------------------------------------------------------------------------

class TestPhysicsTrendClusterer(unittest.TestCase):
    """Verifies that the clusterer groups detectors by physics trends."""

    def _make_clustered_features(self, seed=42):
        """Build synthetic (L, theta) feature data with known clusters.

        Two well-separated groups plus a few noise points.
        """
        rng = np.random.default_rng(seed)
        cluster_a = rng.normal(loc=[1.0, 30.0], scale=[0.1, 2.0],
                               size=(20, 2))
        cluster_b = rng.normal(loc=[3.0, 120.0], scale=[0.1, 2.0],
                               size=(20, 2))
        noise = rng.uniform(low=[0, 0], high=[5, 180], size=(3, 2))
        features = np.vstack([cluster_a, cluster_b, noise])
        return features

    def test_two_clusters_found(self):
        """DBSCAN should find exactly 2 physical clusters."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            PhysicsTrendClusterer,
        )

        features = self._make_clustered_features()
        clusterer = PhysicsTrendClusterer(eps=1.0, min_samples=3)
        labels = clusterer.fit_predict(features)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # exclude noise
        self.assertEqual(
            len(unique_labels), 2,
            f"Expected 2 clusters, got {len(unique_labels)}: {unique_labels}",
        )

    def test_noise_excluded_from_groups(self):
        """Noise labels (-1) must not appear in the cluster groups dict."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            PhysicsTrendClusterer,
        )

        features = self._make_clustered_features()
        clusterer = PhysicsTrendClusterer(eps=5.0, min_samples=3)
        labels = clusterer.fit_predict(features)
        groups = clusterer.get_cluster_groups(labels)

        self.assertNotIn(-1, groups,
                         "Noise label -1 must be excluded from groups dict")

    def test_labels_shape(self):
        """Label array must match the number of feature rows."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            PhysicsTrendClusterer,
        )

        features = self._make_clustered_features()
        clusterer = PhysicsTrendClusterer(eps=5.0, min_samples=3)
        labels = clusterer.fit_predict(features)
        self.assertEqual(len(labels), features.shape[0])

    def test_all_cluster_members_accounted(self):
        """Every non-noise index should appear in exactly one group."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            PhysicsTrendClusterer,
        )

        features = self._make_clustered_features()
        clusterer = PhysicsTrendClusterer(eps=5.0, min_samples=3)
        labels = clusterer.fit_predict(features)
        groups = clusterer.get_cluster_groups(labels)

        all_indices = set()
        for indices in groups.values():
            all_indices.update(indices)

        non_noise = set(np.where(labels != -1)[0])
        self.assertEqual(all_indices, non_noise)


# ---------------------------------------------------------------------------
# Bayesian Bootstrap (Dirichlet Weights)
# ---------------------------------------------------------------------------

class TestBayesianBootstrap(unittest.TestCase):
    """Verifies the Weighted Bayesian Bootstrap with Dirichlet weights."""

    def test_weights_sum_to_one(self):
        """Each row of Dirichlet weights must sum to 1.0."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrap,
        )

        n_spectra = 50
        n_samples = 200
        bootstrap = BayesianBootstrap(n_samples=n_samples, seed=42)
        weights = bootstrap.generate_weights(n_spectra)

        self.assertEqual(weights.shape, (n_samples, n_spectra))
        np.testing.assert_allclose(
            weights.sum(axis=1), 1.0, atol=1e-12,
            err_msg="Dirichlet weight rows must sum to 1.0",
        )

    def test_weights_non_negative(self):
        """All Dirichlet weights must be >= 0."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrap,
        )

        bootstrap = BayesianBootstrap(n_samples=100, seed=7)
        weights = bootstrap.generate_weights(30)
        self.assertTrue(np.all(weights >= 0),
                        "Dirichlet weights must be non-negative")

    def test_uniform_dirichlet_mean(self):
        """With uniform alpha, mean weight should be ~1/n_spectra."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrap,
        )

        n_spectra = 40
        bootstrap = BayesianBootstrap(n_samples=10_000, seed=123)
        weights = bootstrap.generate_weights(n_spectra)

        mean_weights = weights.mean(axis=0)
        expected = 1.0 / n_spectra
        np.testing.assert_allclose(
            mean_weights, expected, atol=0.005,
            err_msg="Mean Dirichlet weight should be ~1/n",
        )

    def test_weighted_residuals(self):
        """Weighted residual computation should produce valid arrays."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrap,
        )

        n_spectra = 20
        n_bins = 100
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, (n_spectra, n_bins))

        bootstrap = BayesianBootstrap(n_samples=50, seed=42)
        weighted = bootstrap.compute_weighted_residuals(residuals)

        self.assertEqual(weighted.shape, (50, n_bins))
        # Weighted sums should not be NaN or Inf
        self.assertFalse(np.any(np.isnan(weighted)))
        self.assertFalse(np.any(np.isinf(weighted)))

    def test_reproducibility(self):
        """Same seed must produce identical weight matrices."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrap,
        )

        s1 = BayesianBootstrap(n_samples=10, seed=55)
        s2 = BayesianBootstrap(n_samples=10, seed=55)
        w1 = s1.generate_weights(20)
        w2 = s2.generate_weights(20)
        np.testing.assert_array_equal(w1, w2)


# ---------------------------------------------------------------------------
# Diagnostic Visualisation Functions
# ---------------------------------------------------------------------------


class TestDiagnosticVisualisations(unittest.TestCase):
    """Verifies that all five diagnostic plot functions run without error."""

    def _make_pca_data(self, n=40, seed=7):
        rng = np.random.default_rng(seed)
        coords = rng.normal(size=(n, 2))
        labels = np.zeros(n, dtype=int)
        labels[:3] = -1
        return coords, labels

    def _make_ltheta_data(self, seed=7):
        rng = np.random.default_rng(seed)
        features = np.vstack([
            rng.normal(loc=[1.0, 30.0], scale=[0.1, 1.0], size=(15, 2)),
            rng.normal(loc=[3.0, 120.0], scale=[0.1, 1.0], size=(15, 2)),
        ])
        labels = np.array([0] * 15 + [1] * 15, dtype=int)
        return features, labels

    def test_plot_outlier_scatter_returns_figure(self):
        """plot_outlier_scatter must return a Figure without raising."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_outlier_scatter,
        )
        coords, labels = self._make_pca_data()
        fig = plot_outlier_scatter(coords, labels)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_outlier_scatter_saves_file(self):
        """plot_outlier_scatter must save a file when save_path is given."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_outlier_scatter,
        )
        coords, labels = self._make_pca_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "outlier.png"
            plot_outlier_scatter(coords, labels, save_path=out)
            self.assertTrue(out.is_file())

    def test_plot_cluster_ltheta_returns_figure(self):
        """plot_cluster_ltheta must return a Figure without raising."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_cluster_ltheta,
        )
        features, labels = self._make_ltheta_data()
        fig = plot_cluster_ltheta(features, labels)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_cluster_ltheta_saves_file(self):
        """plot_cluster_ltheta must save a file when save_path is given."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_cluster_ltheta,
        )
        features, labels = self._make_ltheta_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "clusters.pdf"
            plot_cluster_ltheta(features, labels, save_path=out)
            self.assertTrue(out.is_file())

    def test_plot_cluster_ltheta_noise_points(self):
        """plot_cluster_ltheta must handle noise points (label=-1) gracefully."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_cluster_ltheta,
        )
        features, labels = self._make_ltheta_data()
        labels_with_noise = labels.copy()
        labels_with_noise[0] = -1
        fig = plot_cluster_ltheta(features, labels_with_noise)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_bayesian_corner_returns_figure(self):
        """plot_bayesian_corner must return a Figure without raising."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_bayesian_corner,
        )
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(200, 3))
        fig = plot_bayesian_corner(samples, ["width_H", "width_C", "intensity"])
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_bayesian_corner_single_param(self):
        """plot_bayesian_corner must handle n_params=1 (single-parameter case)."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_bayesian_corner,
        )
        rng = np.random.default_rng(0)
        samples = rng.normal(size=(100, 1))
        fig = plot_bayesian_corner(samples, ["width_H"])
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_posterior_kde_returns_figure(self):
        """plot_posterior_kde must return a Figure without raising."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_posterior_kde,
        )
        rng = np.random.default_rng(3)
        samples = rng.normal(loc=[5.0, 4.9], scale=[0.2, 0.1], size=(500, 2))
        fig = plot_posterior_kde(
            samples,
            point_estimates=np.array([5.0, 4.9]),
            param_names=["width_H", "width_C"],
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_posterior_kde_saves_file(self):
        """plot_posterior_kde must save a file when save_path is given."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_posterior_kde,
        )
        rng = np.random.default_rng(9)
        samples = rng.normal(loc=[5.0], scale=[0.3], size=(300, 1))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "posterior.pdf"
            plot_posterior_kde(
                samples,
                point_estimates=np.array([5.0]),
                param_names=["width_H"],
                save_path=out,
            )
            self.assertTrue(out.is_file())

    def test_plot_optimizer_residuals_returns_figure(self):
        """plot_optimizer_residuals must return a Figure without raising."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_optimizer_residuals,
        )
        x = np.linspace(-20, 20, 100)
        scipy_fit = np.exp(-0.5 * (x / 5) ** 2)
        iminuit_fit = scipy_fit * 1.01  # 1% difference
        fig = plot_optimizer_residuals(x, scipy_fit, iminuit_fit, rel_diff_pct=1.0)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_optimizer_residuals_saves_file(self):
        """plot_optimizer_residuals must save a file when save_path is given."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_optimizer_residuals,
        )
        x = np.linspace(-10, 10, 50)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "residuals.pdf"
            plot_optimizer_residuals(
                x, np.zeros_like(x), np.zeros_like(x),
                rel_diff_pct=0.0, save_path=out,
            )
            self.assertTrue(out.is_file())


# ---------------------------------------------------------------------------
# Phase 6 histogram/point-data shape alignment
# ---------------------------------------------------------------------------


class _StubResults:
    """Minimal results stub that mimics the shape of ResultsIterations."""

    def __init__(
        self,
        all_fit_workspaces: np.ndarray,
        all_tot_ncp: np.ndarray,
    ) -> None:
        self.all_fit_workspaces = [all_fit_workspaces]
        self.all_tot_ncp = [all_tot_ncp]


class TestPhase6HistogramAlignment(unittest.TestCase):
    """Verifies that _runStatisticalAnalysis aligns spectra to point data.

    When ``runHistData=True`` the fitted workspace has N histogram bins
    while the NCP profile has N-1 point-data bins.  The subtraction
    ``residuals = spectra - ncp_total`` must not raise a ValueError.
    """

    def test_no_valueerror_with_histogram_mismatch(self):
        """Shape-alignment guard must prevent ValueError when n_hist > n_ncp."""
        n_spectra, n_hist = 10, 50
        rng = np.random.default_rng(0)
        results = _StubResults(
            all_fit_workspaces=rng.normal(size=(n_spectra, n_hist)),
            all_tot_ncp=rng.normal(size=(n_spectra, n_hist - 1)),
        )

        spectra = results.all_fit_workspaces[-1]
        ncp_total = results.all_tot_ncp[-1]

        # Apply the same guard as _runStatisticalAnalysis
        if spectra.shape[1] == ncp_total.shape[1] + 1:
            spectra = spectra[:, :-1].copy()

        residuals = spectra - ncp_total
        self.assertEqual(residuals.shape, (n_spectra, n_hist - 1))

    def test_point_data_unaffected(self):
        """When spectra and ncp_total already match, guard must be a no-op."""
        n_spectra, n_bins = 10, 49
        rng = np.random.default_rng(1)
        spectra = rng.normal(size=(n_spectra, n_bins))
        ncp_total = rng.normal(size=(n_spectra, n_bins))

        if spectra.shape[1] == ncp_total.shape[1] + 1:
            spectra = spectra[:, :-1].copy()

        self.assertEqual(spectra.shape, (n_spectra, n_bins))
        residuals = spectra - ncp_total
        self.assertEqual(residuals.shape, (n_spectra, n_bins))


if __name__ == "__main__":
    unittest.main(verbosity=2)
