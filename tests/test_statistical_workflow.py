"""Tests for the Phase 6 statistical-sieve workflow.

These tests do **not** depend on Mantid and can be run in any standard
Python environment with NumPy, SciPy, and scikit-learn installed::

    python -m pytest tests/test_statistical_workflow.py -v

Each test uses deterministic dummy data to verify:
1. Sieve 1 (Hardware Outliers) correctly labels injected outlier spectra.
2. Sieve 2 (Physics Trends) clusters physical groups via DBSCAN and
   excludes noise labels (-1).
3. Sieve 4 (Bayesian Bootstrap) generates valid Dirichlet-distributed
   weights that sum to 1.0.
"""

import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Sieve 1 — Hardware Outlier Detection (PCA-based)
# ---------------------------------------------------------------------------

class TestHardwareOutlierSieve(unittest.TestCase):
    """Verify that Sieve 1 identifies injected hardware outliers."""

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
            HardwareOutlierSieve,
        )

        data, true_outlier_idx = self._make_detector_data(
            n_spectra=50, n_outliers=3, seed=42,
        )
        sieve = HardwareOutlierSieve(n_components=5, contamination=0.1)
        labels = sieve.fit_predict(data)

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
            HardwareOutlierSieve,
        )

        data, _ = self._make_detector_data(n_spectra=30, n_outliers=2)
        sieve = HardwareOutlierSieve(n_components=3, contamination=0.15)
        labels = sieve.fit_predict(data)
        self.assertEqual(labels.shape[0], 30)

    def test_no_outliers_in_clean_data(self):
        """When data is clean, sieve should flag very few or no outliers."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            HardwareOutlierSieve,
        )

        rng = np.random.default_rng(99)
        x = np.linspace(0, 10, 100)
        normal = np.exp(-0.5 * ((x - 5) / 1.0) ** 2)
        data = np.array([normal + rng.normal(0, 0.01, 100)
                         for _ in range(40)])

        sieve = HardwareOutlierSieve(n_components=3, contamination=0.05)
        labels = sieve.fit_predict(data)
        n_outliers = np.sum(labels == -1)
        # At most ~5% should be flagged in clean data
        self.assertLessEqual(n_outliers, max(2, int(0.1 * len(data))))


# ---------------------------------------------------------------------------
# Sieve 2 — Physics Trend Clustering (DBSCAN)
# ---------------------------------------------------------------------------

class TestPhysicsTrendSieve(unittest.TestCase):
    """Verify that Sieve 2 groups detectors by physics trends."""

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
            PhysicsTrendSieve,
        )

        features = self._make_clustered_features()
        sieve = PhysicsTrendSieve(eps=1.0, min_samples=3)
        labels = sieve.fit_predict(features)

        unique_labels = set(labels)
        unique_labels.discard(-1)  # exclude noise
        self.assertEqual(
            len(unique_labels), 2,
            f"Expected 2 clusters, got {len(unique_labels)}: {unique_labels}",
        )

    def test_noise_excluded_from_groups(self):
        """Noise labels (-1) must not appear in the cluster groups dict."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            PhysicsTrendSieve,
        )

        features = self._make_clustered_features()
        sieve = PhysicsTrendSieve(eps=5.0, min_samples=3)
        labels = sieve.fit_predict(features)
        groups = sieve.get_cluster_groups(labels)

        self.assertNotIn(-1, groups,
                         "Noise label -1 must be excluded from groups dict")

    def test_labels_shape(self):
        """Label array must match the number of feature rows."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            PhysicsTrendSieve,
        )

        features = self._make_clustered_features()
        sieve = PhysicsTrendSieve(eps=5.0, min_samples=3)
        labels = sieve.fit_predict(features)
        self.assertEqual(len(labels), features.shape[0])

    def test_all_cluster_members_accounted(self):
        """Every non-noise index should appear in exactly one group."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            PhysicsTrendSieve,
        )

        features = self._make_clustered_features()
        sieve = PhysicsTrendSieve(eps=5.0, min_samples=3)
        labels = sieve.fit_predict(features)
        groups = sieve.get_cluster_groups(labels)

        all_indices = set()
        for indices in groups.values():
            all_indices.update(indices)

        non_noise = set(np.where(labels != -1)[0])
        self.assertEqual(all_indices, non_noise)


# ---------------------------------------------------------------------------
# Sieve 4 — Bayesian Bootstrap (Dirichlet Weights)
# ---------------------------------------------------------------------------

class TestBayesianBootstrapSieve(unittest.TestCase):
    """Verify the Weighted Bayesian Bootstrap with Dirichlet weights."""

    def test_weights_sum_to_one(self):
        """Each row of Dirichlet weights must sum to 1.0."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrapSieve,
        )

        n_spectra = 50
        n_samples = 200
        sieve = BayesianBootstrapSieve(n_samples=n_samples, seed=42)
        weights = sieve.generate_weights(n_spectra)

        self.assertEqual(weights.shape, (n_samples, n_spectra))
        np.testing.assert_allclose(
            weights.sum(axis=1), 1.0, atol=1e-12,
            err_msg="Dirichlet weight rows must sum to 1.0",
        )

    def test_weights_non_negative(self):
        """All Dirichlet weights must be >= 0."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrapSieve,
        )

        sieve = BayesianBootstrapSieve(n_samples=100, seed=7)
        weights = sieve.generate_weights(30)
        self.assertTrue(np.all(weights >= 0),
                        "Dirichlet weights must be non-negative")

    def test_uniform_dirichlet_mean(self):
        """With uniform alpha, mean weight should be ~1/n_spectra."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrapSieve,
        )

        n_spectra = 40
        sieve = BayesianBootstrapSieve(n_samples=10_000, seed=123)
        weights = sieve.generate_weights(n_spectra)

        mean_weights = weights.mean(axis=0)
        expected = 1.0 / n_spectra
        np.testing.assert_allclose(
            mean_weights, expected, atol=0.005,
            err_msg="Mean Dirichlet weight should be ~1/n",
        )

    def test_weighted_residuals(self):
        """Weighted residual computation should produce valid arrays."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrapSieve,
        )

        n_spectra = 20
        n_bins = 100
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, (n_spectra, n_bins))

        sieve = BayesianBootstrapSieve(n_samples=50, seed=42)
        weighted = sieve.compute_weighted_residuals(residuals)

        self.assertEqual(weighted.shape, (50, n_bins))
        # Weighted sums should not be NaN or Inf
        self.assertFalse(np.any(np.isnan(weighted)))
        self.assertFalse(np.any(np.isinf(weighted)))

    def test_reproducibility(self):
        """Same seed must produce identical weight matrices."""
        from vesuvio_analysis.core_functions.statistical_plugins import (
            BayesianBootstrapSieve,
        )

        s1 = BayesianBootstrapSieve(n_samples=10, seed=55)
        s2 = BayesianBootstrapSieve(n_samples=10, seed=55)
        w1 = s1.generate_weights(20)
        w2 = s2.generate_weights(20)
        np.testing.assert_array_equal(w1, w2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
