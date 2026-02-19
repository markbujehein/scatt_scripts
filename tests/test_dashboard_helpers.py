"""Tests for the Nanoscience Explorer dashboard helper functions.

These tests verify the pure-Python (non-Streamlit) utility functions in
``dashboard/result_viewer.py`` without launching a browser or Streamlit
server.  They depend only on NumPy and the project's StreamManager.

Run with::

    python -m pytest tests/test_dashboard_helpers.py -v
"""

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
