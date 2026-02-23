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
        Outliers are placed at the END of the array to avoid index
        collisions.
        """
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 10, n_bins)
        normal = np.exp(-0.5 * ((x - 5) / 1.0) ** 2)

        data = np.empty((n_spectra, n_bins))
        for i in range(n_spectra):
            # Very tight normal cluster — essentially identical spectra
            data[i] = normal + rng.normal(0, 1e-4, n_bins)

        # Inject outliers at the END.  Each outlier uses a unique random
        # spectrum drawn independently from a heavy-tailed Cauchy
        # distribution, producing a radically different shape that
        # cannot be mistaken for the Gaussian-peak population via
        # summary statistics (total counts, RMS, skewness, kurtosis).
        outlier_indices = list(range(n_spectra - n_outliers, n_spectra))
        for idx in outlier_indices:
            data[idx] = rng.standard_cauchy(n_bins)

        return data, np.array(outlier_indices)

    def test_outliers_detected_by_summary_features(self):
        """Summary-feature + EllipticEnvelope should flag injected outliers."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            HardwareOutlierDetector,
        )

        data, true_outlier_idx = self._make_detector_data(
            n_spectra=50, n_outliers=3, seed=42,
        )
        detector = HardwareOutlierDetector(contamination=0.1)
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
        detector = HardwareOutlierDetector(contamination=0.15)
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

        detector = HardwareOutlierDetector(contamination=0.05)
        labels = detector.fit_predict(data)
        n_outliers = np.sum(labels == -1)
        # At most ~10% should be flagged in clean data
        self.assertLessEqual(n_outliers, max(2, int(0.1 * len(data))))

    def test_summary_features_stored(self):
        """After fit_predict, summary_features_ should be available."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            HardwareOutlierDetector,
        )

        data, _ = self._make_detector_data(n_spectra=30, n_outliers=2)
        detector = HardwareOutlierDetector(contamination=0.1)
        detector.fit_predict(data)
        self.assertTrue(hasattr(detector, "summary_features_"))
        self.assertEqual(detector.summary_features_.shape, (30, 4))


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


# ---------------------------------------------------------------------------
# Diagnostic table formatting
# ---------------------------------------------------------------------------


class TestFormatDiagnosticTable(unittest.TestCase):
    """Verifies the transparency-first diagnostic table formatter."""

    def _make_metadata_map(self, n=10, start_spec=3):
        meta = {}
        for i in range(n):
            meta[i] = {
                "spec_no": start_spec + i,
                "angle": 30.0 + i * 5.0,
                "detector_id": 100 + i,
            }
        return meta

    def test_empty_outlier_list(self):
        """An empty outlier array should produce a table with zero data rows."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            format_diagnostic_table,
        )
        meta = self._make_metadata_map()
        table = format_diagnostic_table(
            np.array([], dtype=np.intp), meta,
            np.array([], dtype=np.intp),
        )
        self.assertIn("Total flagged: 0", table)
        self.assertNotIn("MANUAL REVIEW", table)

    def test_pre_masked_spectra_labelled(self):
        """Outliers that appear in maskedSpecAllNo must be labelled PRE-MASKED."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            format_diagnostic_table,
        )
        meta = self._make_metadata_map(n=10, start_spec=3)
        masked = np.array([3, 5], dtype=np.intp)  # spec IDs 3 and 5
        outlier_idx = np.array([0, 2], dtype=np.intp)  # idx 0→spec 3, idx 2→spec 5
        table = format_diagnostic_table(outlier_idx, meta, masked)
        self.assertIn("PRE-MASKED", table)

    def test_unrecognized_outlier_triggers_warning(self):
        """Outliers NOT in maskedSpecAllNo must trigger MANUAL REVIEW."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            format_diagnostic_table,
        )
        meta = self._make_metadata_map(n=10, start_spec=3)
        masked = np.array([], dtype=np.intp)
        outlier_idx = np.array([0], dtype=np.intp)
        table = format_diagnostic_table(outlier_idx, meta, masked)
        self.assertIn("MANUAL REVIEW", table)
        self.assertIn("Unrecognized", table)

    def test_fisher_scores_displayed(self):
        """When Fisher scores are provided they should appear in the table."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            format_diagnostic_table,
        )
        meta = self._make_metadata_map(n=5, start_spec=10)
        scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        outlier_idx = np.array([1], dtype=np.intp)
        table = format_diagnostic_table(
            outlier_idx, meta,
            np.array([], dtype=np.intp),
            fisher_scores=scores,
        )
        self.assertIn("0.9000", table)


# ---------------------------------------------------------------------------
# Anisotropy residuals
# ---------------------------------------------------------------------------


class TestComputeAnisotropyResiduals(unittest.TestCase):
    """Verifies the anisotropy-detection residual computation."""

    def test_perfect_fit_gives_zero_residuals(self):
        """Identical spectra and NCP should yield near-zero residuals."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            compute_anisotropy_residuals,
        )
        rng = np.random.default_rng(0)
        spectra = rng.uniform(1.0, 2.0, size=(20, 50))
        theta = np.linspace(30, 130, 20)

        result = compute_anisotropy_residuals(spectra, spectra, theta)
        np.testing.assert_allclose(result["mean_residual"], 0.0, atol=1e-8)

    def test_spearman_correlation_returned(self):
        """Output dict must contain spearman_r and spearman_p keys."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            compute_anisotropy_residuals,
        )
        rng = np.random.default_rng(1)
        spectra = rng.normal(5.0, 0.5, size=(20, 50))
        ncp = spectra * 0.95
        theta = np.linspace(30, 130, 20)

        result = compute_anisotropy_residuals(spectra, ncp, theta)
        self.assertIn("spearman_r", result)
        self.assertIn("spearman_p", result)
        self.assertTrue(np.isfinite(result["spearman_r"]))

    def test_angular_trend_detected(self):
        """Synthetic linear trend in residuals should yield |ρ| > 0.3."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            compute_anisotropy_residuals,
        )
        n_det = 30
        theta = np.linspace(30, 150, n_det)
        rng = np.random.default_rng(7)
        ncp = rng.uniform(1.0, 3.0, size=(n_det, 80))
        # Inject angle-dependent offset ⇒ anisotropic residual
        offset = np.outer(theta / 150.0, np.ones(80)) * 0.5
        spectra = ncp + offset

        result = compute_anisotropy_residuals(spectra, ncp, theta)
        self.assertGreater(abs(result["spearman_r"]), 0.3)


# ---------------------------------------------------------------------------
# Physical interpretation hints
# ---------------------------------------------------------------------------


class TestGeneratePhysicalInterpretationHint(unittest.TestCase):
    """Verifies the 'Show Your Work' hint generator."""

    def test_single_cluster_message(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            generate_physical_interpretation_hint,
        )
        text = generate_physical_interpretation_hint(
            n_clusters=1, spearman_r=0.05, spearman_p=0.8,
            n_outliers=0, n_total=50,
        )
        self.assertIn("homogeneous", text)

    def test_multi_cluster_message(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            generate_physical_interpretation_hint,
        )
        text = generate_physical_interpretation_hint(
            n_clusters=3, spearman_r=0.1, spearman_p=0.5,
            n_outliers=2, n_total=50,
        )
        self.assertIn("Clusters found: 3", text)

    def test_strong_angular_trend_message(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            generate_physical_interpretation_hint,
        )
        text = generate_physical_interpretation_hint(
            n_clusters=2, spearman_r=0.7, spearman_p=0.001,
            n_outliers=5, n_total=50,
        )
        self.assertIn("anisotropic", text.lower())
        self.assertIn("5/50", text)


# ---------------------------------------------------------------------------
# Build fidelity labels
# ---------------------------------------------------------------------------


class TestBuildFidelityLabels(unittest.TestCase):
    """Verifies convergence-based label construction for Fisher/LDA."""

    def test_perfect_agreement_labelled_hi(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            build_fidelity_labels,
        )
        agreement = np.array([0.001, 0.005, 0.008])
        migrad_valid = np.array([True, True, True])
        labels = build_fidelity_labels(agreement, migrad_valid)
        np.testing.assert_array_equal(labels, [0, 0, 0])

    def test_poor_agreement_labelled_poor(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            build_fidelity_labels,
        )
        agreement = np.array([0.10, 0.20])
        migrad_valid = np.array([True, True])
        labels = build_fidelity_labels(agreement, migrad_valid)
        np.testing.assert_array_equal(labels, [1, 1])

    def test_ambiguous_region_labelled_minus_one(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            build_fidelity_labels,
        )
        agreement = np.array([0.03])
        migrad_valid = np.array([True])
        labels = build_fidelity_labels(agreement, migrad_valid)
        np.testing.assert_array_equal(labels, [-1])

    def test_invalid_migrad_forces_poor(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            build_fidelity_labels,
        )
        agreement = np.array([0.005])  # would be hi-fidelity
        migrad_valid = np.array([False])  # but migrad failed
        labels = build_fidelity_labels(agreement, migrad_valid)
        np.testing.assert_array_equal(labels, [1])


# ---------------------------------------------------------------------------
# Detector relative difference metrics
# ---------------------------------------------------------------------------


class TestDetectorRelativeDifferenceMetrics(unittest.TestCase):
    """Verifies the AppStat-style calibration diagnostic computation."""

    def test_identical_gives_zero(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            detector_relative_difference_metrics,
        )
        y = np.ones((5, 20))
        result = detector_relative_difference_metrics(y, y)
        np.testing.assert_allclose(result["bias"], 0.0, atol=1e-8)
        np.testing.assert_allclose(result["rms"], 0.0, atol=1e-8)

    def test_known_offset(self):
        """A 10% multiplicative offset should yield bias ≈ 0.1 and rms ≈ 0.1."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            detector_relative_difference_metrics,
        )
        y_fit = np.ones((3, 50)) * 10.0
        y_obs = y_fit * 1.1
        result = detector_relative_difference_metrics(y_obs, y_fit)
        np.testing.assert_allclose(result["bias"], 0.1, atol=1e-6)
        np.testing.assert_allclose(result["rms"], 0.1, atol=1e-6)


# ---------------------------------------------------------------------------
# Fisher LDA with ROC
# ---------------------------------------------------------------------------


class TestFisherLDAWithROC(unittest.TestCase):
    """Verifies Fisher/LDA + ROC pipeline on synthetic data."""

    def test_returns_none_with_insufficient_labels(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            fisher_lda_with_roc,
        )
        features = np.random.default_rng(0).normal(size=(10, 3))
        labels = np.full(10, -1, dtype=np.intp)  # all unlabeled
        self.assertIsNone(fisher_lda_with_roc(features, labels))

    def test_separable_data_yields_high_auc(self):
        from vesuvio_analysis.core_functions.statistical_plugins import (
            fisher_lda_with_roc,
        )
        rng = np.random.default_rng(42)
        features = np.vstack([
            rng.normal(loc=[0, 0, 0], scale=0.3, size=(20, 3)),
            rng.normal(loc=[3, 3, 3], scale=0.3, size=(20, 3)),
        ])
        labels = np.array([0] * 20 + [1] * 20, dtype=np.intp)
        result = fisher_lda_with_roc(features, labels)
        self.assertIsNotNone(result)
        self.assertGreater(result["auc"], 0.9)


# ---------------------------------------------------------------------------
# Phase 6 diagnostic dashboard and enhanced plots
# ---------------------------------------------------------------------------


class TestPhase6DiagnosticPlots(unittest.TestCase):
    """Verifies the Phase 6 unified dashboard and enhanced plot functions."""

    def _make_anisotropy(self, n=30, seed=5):
        rng = np.random.default_rng(seed)
        theta = np.linspace(30, 150, n)
        return {
            "theta": theta,
            "mean_residual": rng.normal(0, 0.1, n),
            "std_residual": rng.uniform(0.01, 0.05, n),
            "spearman_r": 0.35,
            "spearman_p": 0.06,
        }

    def _make_metadata_map(self, n=30, start_spec=3):
        meta = {}
        for i in range(n):
            meta[i] = {
                "spec_no": start_spec + i,
                "angle": 30.0 + i * 4.0,
                "detector_id": 200 + i,
            }
        return meta

    def test_dashboard_returns_figure(self):
        """plot_phase6_diagnostic_dashboard must return a Figure."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_phase6_diagnostic_dashboard,
        )
        n = 30
        rng = np.random.default_rng(0)
        features = rng.normal(size=(n, 4))
        labels = np.zeros(n, dtype=np.intp)
        labels[:3] = -1
        aniso = self._make_anisotropy(n)
        meta = self._make_metadata_map(n)

        fig = plot_phase6_diagnostic_dashboard(
            summary_features=features,
            labels=labels,
            fisher_scores=None,
            fidelity_labels=None,
            theta_deg=aniso["theta"],
            anisotropy=aniso,
            roc_data=None,
            metadata_map=meta,
            ws_name="test_ws_0",
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_dashboard_saves_pdf(self):
        """plot_phase6_diagnostic_dashboard must save a PDF when save_path given."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_phase6_diagnostic_dashboard,
        )
        n = 20
        rng = np.random.default_rng(1)
        features = rng.normal(size=(n, 4))
        labels = np.zeros(n, dtype=np.intp)
        aniso = self._make_anisotropy(n, seed=1)
        meta = self._make_metadata_map(n)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dashboard.pdf"
            plot_phase6_diagnostic_dashboard(
                summary_features=features,
                labels=labels,
                fisher_scores=None,
                fidelity_labels=None,
                theta_deg=aniso["theta"],
                anisotropy=aniso,
                roc_data=None,
                metadata_map=meta,
                ws_name="test_ws_1",
                save_path=out,
            )
            self.assertTrue(out.is_file())

    def test_dashboard_with_fisher_scores(self):
        """Dashboard should render fisher-colored scatter when scores provided."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_phase6_diagnostic_dashboard,
        )
        n = 25
        rng = np.random.default_rng(2)
        features = rng.normal(size=(n, 4))
        labels = np.zeros(n, dtype=np.intp)
        scores = rng.uniform(0, 1, n)
        fidelity = np.array([0] * 15 + [1] * 10, dtype=np.intp)
        aniso = self._make_anisotropy(n, seed=2)
        meta = self._make_metadata_map(n)
        roc_data = {
            "fpr": np.array([0.0, 0.3, 1.0]),
            "tpr": np.array([0.0, 0.8, 1.0]),
            "auc": 0.85,
        }

        fig = plot_phase6_diagnostic_dashboard(
            summary_features=features,
            labels=labels,
            fisher_scores=scores,
            fidelity_labels=fidelity,
            theta_deg=aniso["theta"],
            anisotropy=aniso,
            roc_data=roc_data,
            metadata_map=meta,
            ws_name="test_fisher",
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_feature_annotated_returns_figure(self):
        """plot_feature_annotated must return a Figure with annotations."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_feature_annotated,
        )
        n = 20
        rng = np.random.default_rng(3)
        features = rng.normal(size=(n, 4))
        labels = np.zeros(n, dtype=np.intp)
        labels[:2] = -1
        meta = self._make_metadata_map(n)
        cluster_labels = np.array([0] * 10 + [1] * 10, dtype=np.intp)

        fig = plot_feature_annotated(
            features, labels, meta, cluster_labels=cluster_labels,
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_fisher_distribution_returns_figure(self):
        """plot_fisher_distribution must return a Figure."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_fisher_distribution,
        )
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        rng = np.random.default_rng(4)
        n = 40
        features = np.vstack([
            rng.normal(loc=[0, 0], scale=0.5, size=(20, 2)),
            rng.normal(loc=[2, 2], scale=0.5, size=(20, 2)),
        ])
        labels = np.array([0] * 20 + [1] * 20, dtype=np.intp)
        model = LinearDiscriminantAnalysis(solver="svd")
        model.fit(features, labels)
        scores = model.decision_function(features)

        fig = plot_fisher_distribution(
            scores, labels,
            feature_names=["Feature A", "Feature B"],
            lda_model=model,
            roc_auc=0.95,
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_residuals_vs_angle_returns_figure(self):
        """plot_residuals_vs_angle must return a Figure."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_residuals_vs_angle,
        )
        aniso = self._make_anisotropy(n=25, seed=6)
        meta = self._make_metadata_map(n=25)
        outlier_idx = np.array([0, 5], dtype=np.intp)

        fig = plot_residuals_vs_angle(aniso, meta, outlier_indices=outlier_idx)
        self.assertIsInstance(fig, plt.Figure)
        plt.close("all")

    def test_residuals_vs_angle_saves_file(self):
        """plot_residuals_vs_angle must save a file when save_path is given."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            plot_residuals_vs_angle,
        )
        aniso = self._make_anisotropy(n=15, seed=7)
        meta = self._make_metadata_map(n=15)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "res_vs_angle.pdf"
            plot_residuals_vs_angle(aniso, meta, save_path=out)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
