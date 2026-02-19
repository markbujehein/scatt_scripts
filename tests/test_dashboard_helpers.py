"""Tests for the Nanoscience Explorer dashboard helper functions.

These tests verify the pure-Python (non-Streamlit) utility functions in
``dashboard/result_viewer.py`` without launching a browser or Streamlit
server.  They depend only on NumPy and the project's StreamManager.

Run with::

    python -m pytest tests/test_dashboard_helpers.py -v
"""

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Import the helpers under test directly (avoids Streamlit runtime)
from dashboard.result_viewer import (
    _split_keys,
    _detect_masses,
    _detect_detectors,
    _available_levels,
    _mass_label,
    _resolve_x_axis,
    _build_tof_figure,
    _build_q_figure,
    _build_y_figure,
    _compute_area_audit,
    _build_residuals_figure,
    _build_optimizer_diff_figure,
    _auto_pdf_filename,
    _batch_export_zip,
    _MASS_SYMBOL,
)
from vesuvio_analysis.core_functions.stream_manager import DataLevel, StreamManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sample_data() -> dict:
    """Return a minimal stream dictionary with arrays at all four DataLevels."""
    rng = np.random.default_rng(42)
    n_det, n_bins = 5, 100

    data = {
        # L0 — raw
        "L0_raw.tof.dataX": rng.uniform(100, 600, (n_det, n_bins)),
        "L0_raw.tof.dataY": rng.random((n_det, n_bins)),
        "L0_raw.tof.dataE": rng.random((n_det, n_bins)) * 0.01,
        # L1 — corrections
        "L1_corrections.tof.iter0.ms": rng.random((n_det, n_bins)) * 0.05,
        "L1_corrections.tof.iter0.gamma": rng.random((n_det, n_bins)) * 0.03,
        # L2 — intermediate
        "L2_intermediate.tof.iter0.corrected": rng.random((n_det, n_bins)),
        # L3 — final physics
        "L3_final.tof.iter0.ncp_total": rng.random((n_det, n_bins)),
        "L3_final.y.joy_avg": rng.random(80),
        "L3_final.y.joy_avg_err": rng.random(80) * 0.01,
        "L3_final.y.resolution": rng.random(80),
        "L3_final.y.mass0.ncp": rng.random(80),
        "L3_final.y.mass1.ncp": rng.random(80),
        # metadata
        "metadata.masses": np.array([1.008, 12.0]),
        "metadata.n_iterations": np.array(1),
        "metadata.fit_model": np.array("ANSIO_GAUSSIAN"),
    }
    return data


# ---------------------------------------------------------------------------
# Tests for key-parsing helpers
# ---------------------------------------------------------------------------

class TestSplitKeys(unittest.TestCase):
    def test_bucketing(self):
        data = _make_sample_data()
        buckets = _split_keys(data)
        self.assertIn("L0_raw.tof.dataY", buckets["L0"])
        self.assertIn("L1_corrections.tof.iter0.ms", buckets["L1"])
        self.assertIn("L2_intermediate.tof.iter0.corrected", buckets["L2"])
        self.assertIn("L3_final.tof.iter0.ncp_total", buckets["L3"])
        self.assertIn("metadata.masses", buckets["metadata"])

    def test_empty_data(self):
        buckets = _split_keys({})
        for lvl in ("L0", "L1", "L2", "L3", "metadata"):
            self.assertEqual(buckets[lvl], [])

    def test_unknown_keys_ignored(self):
        data = {"unknown.key": np.array([1])}
        buckets = _split_keys(data)
        for keys in buckets.values():
            self.assertEqual(keys, [])


class TestDetectMasses(unittest.TestCase):
    def test_detects_mass_indices(self):
        data = _make_sample_data()
        masses = _detect_masses(data)
        self.assertEqual(masses, [0, 1])

    def test_no_masses_returns_empty(self):
        data = {"L0_raw.tof.dataY": np.ones((3, 50))}
        self.assertEqual(_detect_masses(data), [])

    def test_multiple_masses_sorted(self):
        data = {
            "L3_final.y.mass3.ncp": np.ones(10),
            "L3_final.y.mass1.ncp": np.ones(10),
            "L3_final.y.mass0.ncp": np.ones(10),
        }
        self.assertEqual(_detect_masses(data), [0, 1, 3])


class TestDetectDetectors(unittest.TestCase):
    def test_from_l0_array(self):
        data = {"L0_raw.tof.dataY": np.ones((7, 200))}
        dets = _detect_detectors(data)
        self.assertEqual(dets, list(range(7)))

    def test_fallback_1d_returns_zero(self):
        data = {"L3_final.y.joy_avg": np.ones(50)}
        dets = _detect_detectors(data)
        self.assertEqual(dets, [0])

    def test_prefers_l0_over_other_2d(self):
        data = {
            "L1_corrections.tof.iter0.ms": np.ones((10, 50)),
            "L0_raw.tof.dataY": np.ones((3, 50)),
        }
        # Should pick L0 shape (3), not L1 shape (10)
        dets = _detect_detectors(data)
        self.assertEqual(len(dets), 3)


class TestAvailableLevels(unittest.TestCase):
    def test_all_levels(self):
        buckets = _split_keys(_make_sample_data())
        levels = _available_levels(buckets)
        self.assertIn("L0", levels)
        self.assertIn("L1", levels)
        self.assertIn("L2", levels)
        self.assertIn("L3", levels)

    def test_only_l0(self):
        data = {"L0_raw.tof.dataY": np.ones((3, 50))}
        buckets = _split_keys(data)
        levels = _available_levels(buckets)
        self.assertEqual(levels, ["L0"])


# ---------------------------------------------------------------------------
# Tests for mass labelling
# ---------------------------------------------------------------------------

class TestMassLabel(unittest.TestCase):
    def test_hydrogen(self):
        masses = np.array([1.008, 12.0])
        self.assertEqual(_mass_label(0, masses), "H (1.008 u)")

    def test_carbon(self):
        masses = np.array([1.008, 12.0])
        self.assertEqual(_mass_label(1, masses), "C (12.000 u)")

    def test_unknown_mass(self):
        masses = np.array([99.0])
        label = _mass_label(0, masses)
        self.assertIn("99", label)

    def test_no_meta(self):
        label = _mass_label(2, None)
        self.assertEqual(label, "Mass 2")

    def test_out_of_bounds_index(self):
        masses = np.array([1.008])
        label = _mass_label(5, masses)
        self.assertEqual(label, "Mass 5")


# ---------------------------------------------------------------------------
# Tests for x-axis resolution
# ---------------------------------------------------------------------------

class TestResolveXAxis(unittest.TestCase):
    def test_tof_2d_per_detector(self):
        data = {"L0_raw.tof.dataX": np.arange(30).reshape(3, 10).astype(float)}
        x = _resolve_x_axis(data, "tof", det_idx=1, n_bins=10)
        np.testing.assert_array_equal(x, np.arange(10, 20, dtype=float))

    def test_tof_1d_shared(self):
        x_shared = np.linspace(100, 500, 50)
        data = {"L0_raw.tof.dataX": x_shared}
        x = _resolve_x_axis(data, "tof", det_idx=0, n_bins=50)
        np.testing.assert_array_equal(x, x_shared)

    def test_y_axis_from_metadata(self):
        y_ref = np.linspace(-20, 20, 80)
        data = {"metadata.y_x": y_ref}
        x = _resolve_x_axis(data, "y", det_idx=0, n_bins=80)
        np.testing.assert_array_equal(x, y_ref)

    def test_fallback_arange(self):
        x = _resolve_x_axis({}, "tof", det_idx=0, n_bins=30)
        np.testing.assert_array_equal(x, np.arange(30, dtype=float))


# ---------------------------------------------------------------------------
# Tests for figure builders (structural / smoke tests)
# ---------------------------------------------------------------------------

class TestBuildTofFigure(unittest.TestCase):
    def setUp(self):
        self.data = _make_sample_data()

    def test_returns_figure(self):
        import plotly.graph_objects as go
        fig = _build_tof_figure(
            self.data,
            selected_detectors=[0, 1],
            show_corrections=True,
            show_optimizer_compare=False,
            iteration=0,
        )
        self.assertIsInstance(fig, go.Figure)

    def test_raw_traces_present(self):
        fig = _build_tof_figure(
            self.data, selected_detectors=[0], show_corrections=False,
            show_optimizer_compare=False, iteration=0,
        )
        names = [t.name for t in fig.data]
        self.assertTrue(any("Raw" in n for n in names))

    def test_correction_bands_added(self):
        fig = _build_tof_figure(
            self.data, selected_detectors=[0], show_corrections=True,
            show_optimizer_compare=False, iteration=0,
        )
        # Correction bands are Scatter traces with fill="toself"
        filled = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
        self.assertGreater(len(filled), 0)

    def test_no_correction_bands_when_disabled(self):
        fig = _build_tof_figure(
            self.data, selected_detectors=[0], show_corrections=False,
            show_optimizer_compare=False, iteration=0,
        )
        filled = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
        self.assertEqual(len(filled), 0)

    def test_empty_detectors_shows_annotation(self):
        data_no_l0 = {k: v for k, v in self.data.items() if not k.startswith("L0_")}
        fig = _build_tof_figure(
            data_no_l0, selected_detectors=[0], show_corrections=False,
            show_optimizer_compare=False, iteration=0,
        )
        self.assertEqual(len(fig.data), 0)
        self.assertTrue(len(fig.layout.annotations) > 0)


class TestBuildQFigure(unittest.TestCase):
    def test_no_q_data_shows_annotation(self):
        import plotly.graph_objects as go
        data = _make_sample_data()
        fig = _build_q_figure(data, selected_detectors=[0])
        self.assertIsInstance(fig, go.Figure)
        # No Q data in sample → annotation expected
        self.assertTrue(
            len(fig.data) == 0 or len(fig.layout.annotations) > 0
        )

    def test_with_q_data(self):
        import plotly.graph_objects as go
        data = dict(_make_sample_data())
        rng = np.random.default_rng(0)
        data["L3_final.q.signal"] = rng.random((3, 60))
        data["L3_final.q.q"] = np.linspace(0.5, 15.0, 60)
        fig = _build_q_figure(data, selected_detectors=[0, 1])
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)


class TestBuildYFigure(unittest.TestCase):
    def setUp(self):
        self.data = _make_sample_data()

    def test_returns_figure(self):
        import plotly.graph_objects as go
        fig = _build_y_figure(
            self.data, selected_mass_indices=[0, 1],
            masses_meta=self.data["metadata.masses"],
            show_optimizer_compare=False,
        )
        self.assertIsInstance(fig, go.Figure)

    def test_joy_avg_trace_present(self):
        fig = _build_y_figure(
            self.data, selected_mass_indices=[0, 1],
            masses_meta=self.data["metadata.masses"],
            show_optimizer_compare=False,
        )
        names = [t.name for t in fig.data]
        self.assertTrue(any("J(y)" in n for n in names))

    def test_error_band_present(self):
        fig = _build_y_figure(
            self.data, selected_mass_indices=[],
            masses_meta=None, show_optimizer_compare=False,
        )
        filled = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
        self.assertGreater(len(filled), 0)

    def test_per_mass_traces_shown(self):
        fig = _build_y_figure(
            self.data, selected_mass_indices=[0, 1],
            masses_meta=self.data["metadata.masses"],
            show_optimizer_compare=False,
        )
        names = [t.name for t in fig.data]
        # H (mass 0) and C (mass 1) should appear
        self.assertTrue(any("H" in n for n in names))
        self.assertTrue(any("C" in n for n in names))

    def test_resolution_trace_present(self):
        fig = _build_y_figure(
            self.data, selected_mass_indices=[],
            masses_meta=None, show_optimizer_compare=False,
        )
        names = [t.name for t in fig.data]
        self.assertTrue(any("Resolution" in n for n in names))

    def test_no_y_data_shows_annotation(self):
        data_no_y = {k: v for k, v in self.data.items() if not k.startswith("L3_final.y")}
        fig = _build_y_figure(
            data_no_y, selected_mass_indices=[],
            masses_meta=None, show_optimizer_compare=False,
        )
        self.assertEqual(len(fig.data), 0)
        self.assertGreater(len(fig.layout.annotations), 0)

    def test_optimizer_compare_keys_searched(self):
        """When optimizer compare is on and keys present, extra traces added."""
        data = dict(self.data)
        rng = np.random.default_rng(7)
        data["L3_final.y.ncp_iminuit"] = rng.random(80)
        data["L3_final.y.ncp_scipy"] = rng.random(80)
        fig_with = _build_y_figure(
            data, selected_mass_indices=[],
            masses_meta=None, show_optimizer_compare=True,
        )
        fig_without = _build_y_figure(
            data, selected_mass_indices=[],
            masses_meta=None, show_optimizer_compare=False,
        )
        self.assertGreater(len(fig_with.data), len(fig_without.data))


# ---------------------------------------------------------------------------
# Round-trip: save stream → load → build all three figures
# ---------------------------------------------------------------------------

class TestDashboardRoundTrip(unittest.TestCase):
    """Verify that a round-tripped StreamManager file drives all figure builders."""

    def test_full_round_trip(self):
        import plotly.graph_objects as go

        rng = np.random.default_rng(123)
        n_det, n_bins = 4, 80

        with tempfile.TemporaryDirectory() as tmpdir:
            sm = StreamManager(
                output_dir=Path(tmpdir),
                script_name="test_dash",
                direction="BACKWARD",
            )
            sm.capture("dataX", rng.uniform(100, 600, (n_det, n_bins)),
                       DataLevel.RAW, domain="tof")
            sm.capture("dataY", rng.random((n_det, n_bins)),
                       DataLevel.RAW, domain="tof")
            sm.capture("ms", rng.random((n_det, n_bins)) * 0.05,
                       DataLevel.CORRECTION_COMPONENTS, domain="tof", iteration=0)
            sm.capture("corrected", rng.random((n_det, n_bins)),
                       DataLevel.INTERMEDIATE_CORRECTED, domain="tof", iteration=0)
            sm.capture("ncp_total", rng.random((n_det, n_bins)),
                       DataLevel.FINAL_PHYSICS, domain="tof", iteration=0)
            sm.capture("joy_avg", rng.random(60),
                       DataLevel.FINAL_PHYSICS, domain="y")
            sm.capture("joy_avg_err", rng.random(60) * 0.01,
                       DataLevel.FINAL_PHYSICS, domain="y")
            sm.set_metadata("masses", np.array([1.008, 12.0]))
            sm.set_metadata("n_iterations", 1)

            path = sm.save()
            data = StreamManager.load(path)

        all_dets = _detect_detectors(data)
        all_masses = _detect_masses(data)
        masses_meta = data.get("metadata.masses")

        fig_tof = _build_tof_figure(
            data, selected_detectors=all_dets[:2],
            show_corrections=True, show_optimizer_compare=False, iteration=0,
        )
        fig_q = _build_q_figure(data, selected_detectors=all_dets[:2])
        fig_y = _build_y_figure(
            data, selected_mass_indices=all_masses,
            masses_meta=masses_meta, show_optimizer_compare=False,
        )

        self.assertIsInstance(fig_tof, go.Figure)
        self.assertIsInstance(fig_q, go.Figure)
        self.assertIsInstance(fig_y, go.Figure)

        # TOF figure must have at least raw + L2 traces + correction bands
        self.assertGreaterEqual(len(fig_tof.data), 4)
        # y figure must have J(y) traces
        self.assertGreater(len(fig_y.data), 0)


# ---------------------------------------------------------------------------
# Tests for new forensic helpers (added in upgrade)
# ---------------------------------------------------------------------------

class TestAluminiumSymbol(unittest.TestCase):
    """Al (canister) must be present in _MASS_SYMBOL."""

    def test_aluminium_in_symbol_map(self):
        self.assertIn(26.982, _MASS_SYMBOL)
        self.assertEqual(_MASS_SYMBOL[26.982], "Al")

    def test_aluminium_label(self):
        masses = np.array([1.008, 26.982])
        self.assertEqual(_mass_label(1, masses), "Al (26.982 u)")


class TestComputeAreaAudit(unittest.TestCase):
    """_compute_area_audit returns correct integral-contribution stats."""

    def _make_data(self) -> dict:
        rng = np.random.default_rng(0)
        n_det, n_bins = 4, 100
        return {
            "L0_raw.tof.dataY": rng.random((n_det, n_bins)) * 500 + 1,
            "L1_corrections.tof.iter0.ms": rng.random((n_det, n_bins)) * 50,
            "L1_corrections.tof.iter0.gamma": rng.random((n_det, n_bins)) * 30,
        }

    def test_returns_rows_for_each_correction_and_detector(self):
        data = self._make_data()
        rows = _compute_area_audit(data, iteration=0)
        corrections = {r["correction"] for r in rows}
        self.assertIn("MS", corrections)
        self.assertIn("Gamma", corrections)
        # 4 detectors × 2 corrections = 8 rows
        self.assertEqual(len(rows), 8)

    def test_contribution_pct_between_0_and_100(self):
        data = self._make_data()
        rows = _compute_area_audit(data, iteration=0)
        for row in rows:
            self.assertGreaterEqual(row["contribution_pct"], 0.0)
            self.assertLessEqual(row["contribution_pct"], 100.0)

    def test_empty_when_no_raw_data(self):
        rows = _compute_area_audit({}, iteration=0)
        self.assertEqual(rows, [])

    def test_only_ms_when_gamma_absent(self):
        data = self._make_data()
        del data["L1_corrections.tof.iter0.gamma"]
        rows = _compute_area_audit(data, iteration=0)
        corrections = {r["correction"] for r in rows}
        self.assertIn("MS", corrections)
        self.assertNotIn("Gamma", corrections)

    def test_known_contribution(self):
        """With simple arrays, check numerical correctness."""
        # raw = [1, 2, 3], ms = [0.1, 0.2, 0.3]  → contribution = 10%
        raw = np.array([[1.0, 2.0, 3.0]])
        ms = np.array([[0.1, 0.2, 0.3]])
        data = {
            "L0_raw.tof.dataY": raw,
            "L1_corrections.tof.iter0.ms": ms,
        }
        rows = _compute_area_audit(data, iteration=0)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["contribution_pct"], 10.0, places=5)

    def test_uses_trapezoid_not_sum(self):
        """Integral should equal np.trapezoid, not np.sum."""
        rng = np.random.default_rng(99)
        raw = rng.random((1, 50)) + 1
        ms = rng.random((1, 50)) * 0.1
        data = {
            "L0_raw.tof.dataY": raw,
            "L1_corrections.tof.iter0.ms": ms,
        }
        rows = _compute_area_audit(data, iteration=0)
        expected_pct = 100.0 * float(np.trapezoid(np.abs(ms[0]))) / float(np.trapezoid(np.abs(raw[0])))
        # _compute_area_audit rounds to 3 decimal places; compare at same precision
        self.assertAlmostEqual(rows[0]["contribution_pct"], expected_pct, places=3)


class TestBuildResidualsFigure(unittest.TestCase):
    """_build_residuals_figure produces correct traces."""

    def _make_data(self) -> dict:
        rng = np.random.default_rng(1)
        n_det, n_bins = 3, 80
        return {
            "L0_raw.tof.dataY": rng.random((n_det, n_bins)) * 500,
            "L1_corrections.tof.iter0.ms": rng.random((n_det, n_bins)) * 50,
            "L1_corrections.tof.iter0.gamma": rng.random((n_det, n_bins)) * 30,
        }

    def test_returns_figure(self):
        import plotly.graph_objects as go
        data = self._make_data()
        fig = _build_residuals_figure(data, [0, 1], ["Raw − MS"], iteration=0)
        self.assertIsInstance(fig, go.Figure)

    def test_raw_minus_ms_traces(self):
        data = self._make_data()
        fig = _build_residuals_figure(data, [0, 1], ["Raw − MS"], iteration=0)
        names = [t.name for t in fig.data]
        self.assertTrue(any("Raw − MS" in n for n in names))

    def test_all_three_modes(self):
        data = self._make_data()
        modes = ["Raw − MS", "Raw − Gamma", "Raw − MS − Gamma"]
        fig = _build_residuals_figure(data, [0], modes, iteration=0)
        names = [t.name for t in fig.data]
        for mode in modes:
            self.assertTrue(any(mode in n for n in names))

    def test_no_l0_shows_annotation(self):
        fig = _build_residuals_figure({}, [0], ["Raw − MS"], iteration=0)
        self.assertEqual(len(fig.data), 0)
        self.assertGreater(len(fig.layout.annotations), 0)

    def test_missing_ms_skips_raw_minus_ms(self):
        data = self._make_data()
        del data["L1_corrections.tof.iter0.ms"]
        fig = _build_residuals_figure(data, [0], ["Raw − MS"], iteration=0)
        names = [t.name for t in fig.data]
        self.assertFalse(any("Raw − MS" in n for n in names))

    def test_missing_gamma_skips_raw_minus_gamma(self):
        """'Raw − Gamma' must not be plotted when gamma data is absent."""
        data = self._make_data()
        del data["L1_corrections.tof.iter0.gamma"]
        fig = _build_residuals_figure(data, [0], ["Raw − Gamma"], iteration=0)
        names = [t.name for t in fig.data]
        self.assertFalse(any("Raw − Gamma" in n for n in names))

    def test_empty_modes_produces_empty_figure(self):
        data = self._make_data()
        fig = _build_residuals_figure(data, [0, 1], [], iteration=0)
        self.assertEqual(len(fig.data), 0)

    def test_residual_values_correct(self):
        """Raw − MS values must equal raw[det] - ms[det] element-wise."""
        raw = np.array([[10.0, 20.0, 30.0]])
        ms = np.array([[1.0, 2.0, 3.0]])
        data = {
            "L0_raw.tof.dataY": raw,
            "L1_corrections.tof.iter0.ms": ms,
        }
        fig = _build_residuals_figure(data, [0], ["Raw − MS"], iteration=0)
        expected = raw[0] - ms[0]
        np.testing.assert_array_almost_equal(fig.data[0].y, expected)


class TestBuildOptimizerDiffFigure(unittest.TestCase):
    """_build_optimizer_diff_figure computes iMinuit − Scipy correctly."""

    def _make_data_with_opts(self) -> dict:
        rng = np.random.default_rng(2)
        return {
            "L3_final.y.ncp_iminuit": rng.random(80),
            "L3_final.y.ncp_scipy": rng.random(80),
        }

    def test_returns_figure(self):
        import plotly.graph_objects as go
        data = self._make_data_with_opts()
        fig = _build_optimizer_diff_figure(data, None)
        self.assertIsInstance(fig, go.Figure)

    def test_difference_trace_present(self):
        data = self._make_data_with_opts()
        fig = _build_optimizer_diff_figure(data, None)
        names = [t.name for t in fig.data]
        self.assertTrue(any("iMinuit" in n and "Scipy" in n for n in names))

    def test_difference_values_correct(self):
        iminuit = np.array([1.0, 2.0, 3.0])
        scipy = np.array([0.5, 1.5, 2.5])
        data = {
            "L3_final.y.ncp_iminuit": iminuit,
            "L3_final.y.ncp_scipy": scipy,
        }
        fig = _build_optimizer_diff_figure(data, None)
        expected = iminuit - scipy
        # Find the difference trace (not the hline)
        diff_traces = [t for t in fig.data if t.name and "iMinuit" in t.name]
        self.assertEqual(len(diff_traces), 1)
        np.testing.assert_array_almost_equal(diff_traces[0].y, expected)

    def test_missing_data_shows_annotation(self):
        fig = _build_optimizer_diff_figure({}, None)
        self.assertEqual(len(fig.data), 0)
        self.assertGreater(len(fig.layout.annotations), 0)

    def test_accepts_alternative_key_names(self):
        """Should also find fit_iminuit / fit_scipy key naming convention."""
        rng = np.random.default_rng(3)
        data = {
            "L3_final.y.fit_iminuit": rng.random(40),
            "L3_final.y.fit_scipy": rng.random(40),
        }
        fig = _build_optimizer_diff_figure(data, None)
        self.assertGreater(len(fig.data), 0)


class TestAutoPdfFilename(unittest.TestCase):
    """_auto_pdf_filename generates correctly structured filenames."""

    def test_hydrogen_single_det(self):
        masses = np.array([1.008])
        name = _auto_pdf_filename("y", masses, [0], [5], False)
        self.assertIn("H", name)
        self.assertIn("det5", name)
        self.assertIn("J(y)", name)
        self.assertTrue(name.endswith(".pdf"))

    def test_multiple_masses(self):
        masses = np.array([1.008, 12.0])
        name = _auto_pdf_filename("tof", masses, [0, 1], [0, 1, 2], True)
        self.assertIn("H", name)
        self.assertIn("C", name)
        self.assertIn("MS_Gamma", name)
        self.assertIn("TOF", name)

    def test_detector_range(self):
        masses = np.array([1.008])
        name = _auto_pdf_filename("y", masses, [0], [0, 1, 2, 3, 4], False)
        self.assertIn("det0-4", name)

    def test_no_correction(self):
        masses = np.array([1.008])
        name = _auto_pdf_filename("tof", masses, [0], [0], False)
        self.assertNotIn("MS_Gamma", name)

    def test_no_masses_meta(self):
        name = _auto_pdf_filename("q", None, [], [], False)
        self.assertIn("all", name)
        self.assertIn("Q", name)
        self.assertTrue(name.endswith(".pdf"))

    def test_aluminium_canister(self):
        masses = np.array([1.008, 26.982])
        name = _auto_pdf_filename("y", masses, [0, 1], [0], False)
        self.assertIn("Al", name)


class TestBuildYFigureRecoilMarkers(unittest.TestCase):
    """_build_y_figure with show_recoil_markers adds vline shapes."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = {
            "L3_final.y.joy_avg": rng.random(80),
            "L3_final.y.joy_avg_err": rng.random(80) * 0.01,
        }
        self.masses_meta = np.array([1.008, 12.0])

    def test_recoil_markers_added_when_enabled(self):
        fig = _build_y_figure(
            self.data, [0, 1], self.masses_meta,
            show_optimizer_compare=False, show_recoil_markers=True,
        )
        # add_vline creates shapes in fig.layout.shapes
        self.assertGreater(len(fig.layout.shapes), 0)

    def test_no_recoil_markers_when_disabled(self):
        fig = _build_y_figure(
            self.data, [0, 1], self.masses_meta,
            show_optimizer_compare=False, show_recoil_markers=False,
        )
        self.assertEqual(len(fig.layout.shapes), 0)

    def test_one_shape_per_selected_mass(self):
        fig = _build_y_figure(
            self.data, [0, 1], self.masses_meta,
            show_optimizer_compare=False, show_recoil_markers=True,
        )
        # One vline per mass
        self.assertEqual(len(fig.layout.shapes), 2)


class TestBatchExportZip(unittest.TestCase):
    """_batch_export_zip produces a valid ZIP with one HTML per mass."""

    def _make_data(self) -> dict:
        rng = np.random.default_rng(5)
        return {
            "L3_final.y.joy_avg": rng.random(60),
            "L3_final.y.joy_avg_err": rng.random(60) * 0.01,
            "L3_final.y.mass0.ncp": rng.random(60),
            "L3_final.y.mass1.ncp": rng.random(60),
            "metadata.masses": np.array([1.008, 12.0]),
        }

    def test_returns_bytes(self):
        import zipfile
        data = self._make_data()
        result = _batch_export_zip(data, [0, 1], data["metadata.masses"])
        self.assertIsInstance(result, bytes)
        # Must be a valid ZIP
        self.assertTrue(zipfile.is_zipfile(io.BytesIO(result)))

    def test_zip_contains_one_file_per_mass(self):
        import zipfile
        data = self._make_data()
        zip_bytes = _batch_export_zip(data, [0, 1], data["metadata.masses"])
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
        self.assertEqual(len(names), 2)
        # All files should be HTML
        for name in names:
            self.assertTrue(name.endswith(".html"), msg=f"Not HTML: {name}")

    def test_empty_mass_indices_produces_empty_zip(self):
        import zipfile
        data = self._make_data()
        zip_bytes = _batch_export_zip(data, [], data["metadata.masses"])
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            self.assertEqual(len(zf.namelist()), 0)


class TestStreamlitConfigExists(unittest.TestCase):
    """Verify the air-gapped Streamlit config file is in place."""

    def test_config_toml_exists(self):
        config_path = Path(__file__).resolve().parent.parent / "dashboard" / ".streamlit" / "config.toml"
        self.assertTrue(config_path.exists(), f"Missing: {config_path}")

    def test_config_disables_telemetry(self):
        config_path = Path(__file__).resolve().parent.parent / "dashboard" / ".streamlit" / "config.toml"
        text = config_path.read_text()
        self.assertIn("gatherUsageStats = false", text)

    def test_config_binds_to_loopback(self):
        config_path = Path(__file__).resolve().parent.parent / "dashboard" / ".streamlit" / "config.toml"
        text = config_path.read_text()
        self.assertIn('address = "127.0.0.1"', text)

    def test_config_headless(self):
        config_path = Path(__file__).resolve().parent.parent / "dashboard" / ".streamlit" / "config.toml"
        text = config_path.read_text()
        self.assertIn("headless = true", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
