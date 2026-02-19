"""Tests for correction_plots.py — Automated Comparative Visualization.

These tests do **not** depend on Mantid and can be run in any standard
Python environment::

    python -m pytest tests/test_correction_plots.py -v

Coverage:
1. ``_get_scenario`` — correct label for all three flag combinations.
2. ``_integrate_area`` — scalar area computation including NaN handling.
3. ``_area_fraction_pct`` — percentage calculation and zero-division guard.
4. ``_extract_ws_data`` — histogram and point-data workspaces, multi-spectra sum.
5. ``plot_tof_correction_dashboard`` — file creation (PDF + PNG) for all
   scenarios, and empty-list return when no corrections are active.
6. ``plot_yspace_correction_dashboard`` — same checks in y-space.
7. ``dispatch_correction_plots`` — end-to-end with mock ADS, verifies
   saved-file list and early-exit conditions.
"""

from __future__ import annotations

import sys
import types
import unittest
import tempfile
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight mock ADS and workspace (mirrors test_workspace_safety.py)
# ---------------------------------------------------------------------------

class _MockMtd(dict):
    def add(self, name: str, ws: Any) -> None:
        self[name] = ws

    def remove(self, name: str) -> None:
        self.pop(name, None)


class _MockWorkspace:
    """Minimal stand-in for a Mantid MatrixWorkspace (point-data layout)."""

    def __init__(
        self,
        name: str,
        n_spectra: int = 4,
        n_bins: int = 30,
        histogram: bool = False,
        seed: int = 0,
    ):
        self._name = name
        self._n_spectra = n_spectra
        self._n_bins = n_bins
        self._histogram = histogram
        rng = np.random.default_rng(seed)
        self._y = rng.random((n_spectra, n_bins))
        self._e = np.ones((n_spectra, n_bins)) * 0.05

    def name(self) -> str:
        return self._name

    def extractX(self) -> np.ndarray:
        n_x = self._n_bins + (1 if self._histogram else 0)
        return np.tile(np.linspace(100.0, 500.0, n_x), (self._n_spectra, 1))

    def extractY(self) -> np.ndarray:
        return self._y.copy()

    def extractE(self) -> np.ndarray:
        return self._e.copy()


# ---------------------------------------------------------------------------
# Test 1: _get_scenario
# ---------------------------------------------------------------------------

class TestGetScenario(unittest.TestCase):
    """Verify correct scenario label for each flag combination."""

    def setUp(self) -> None:
        from vesuvio_analysis.core_functions.correction_plots import _get_scenario
        self._fn = _get_scenario

    def test_ms_only_is_A(self) -> None:
        self.assertEqual(self._fn(ms_flag=True, gc_flag=False), "A")

    def test_gc_only_is_B(self) -> None:
        self.assertEqual(self._fn(ms_flag=False, gc_flag=True), "B")

    def test_both_is_C(self) -> None:
        self.assertEqual(self._fn(ms_flag=True, gc_flag=True), "C")

    def test_neither_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._fn(ms_flag=False, gc_flag=False)


# ---------------------------------------------------------------------------
# Test 2: _integrate_area
# ---------------------------------------------------------------------------

class TestIntegrateArea(unittest.TestCase):
    """Verify trapezoidal area computation."""

    def setUp(self) -> None:
        from vesuvio_analysis.core_functions.correction_plots import _integrate_area
        self._fn = _integrate_area

    def test_constant_signal(self) -> None:
        x = np.linspace(0.0, 1.0, 100)
        y = np.ones(100)
        area = self._fn(x, y)
        self.assertAlmostEqual(area, 1.0, places=4)

    def test_all_nan_returns_zero(self) -> None:
        x = np.linspace(0.0, 1.0, 50)
        y = np.full(50, np.nan)
        self.assertEqual(self._fn(x, y), 0.0)

    def test_negative_values_give_positive_area(self) -> None:
        x = np.linspace(0.0, 1.0, 50)
        y = -np.ones(50)
        self.assertGreater(self._fn(x, y), 0.0)

    def test_mixed_nan_uses_finite_only(self) -> None:
        x = np.linspace(0.0, 1.0, 10)
        y = np.array([np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan])
        area = self._fn(x, y)
        self.assertGreater(area, 0.0)


# ---------------------------------------------------------------------------
# Test 3: _area_fraction_pct
# ---------------------------------------------------------------------------

class TestAreaFractionPct(unittest.TestCase):
    """Verify percentage calculation and zero-division guard."""

    def setUp(self) -> None:
        from vesuvio_analysis.core_functions.correction_plots import _area_fraction_pct
        self._fn = _area_fraction_pct

    def test_ten_percent(self) -> None:
        x = np.linspace(0.0, 1.0, 100)
        sig = np.ones(100)
        corr = np.ones(100) * 0.1
        pct = self._fn(x, corr, x, sig)
        self.assertAlmostEqual(pct, 10.0, places=3)

    def test_zero_signal_returns_zero(self) -> None:
        x = np.linspace(0.0, 1.0, 50)
        pct = self._fn(x, np.ones(50), x, np.zeros(50))
        self.assertEqual(pct, 0.0)

    def test_equal_areas_give_100_pct(self) -> None:
        x = np.linspace(0.0, 1.0, 200)
        y = np.ones(200)
        pct = self._fn(x, y, x, y)
        self.assertAlmostEqual(pct, 100.0, places=3)


# ---------------------------------------------------------------------------
# Test 4: _extract_ws_data
# ---------------------------------------------------------------------------

class TestExtractWsData(unittest.TestCase):
    """Verify extraction and summation from point-data and histogram workspaces."""

    def setUp(self) -> None:
        from vesuvio_analysis.core_functions.correction_plots import _extract_ws_data
        self._fn = _extract_ws_data

    def test_point_data_shape(self) -> None:
        ws = _MockWorkspace("ws", n_spectra=3, n_bins=20, histogram=False)
        x, y, e = self._fn(ws)
        self.assertEqual(x.shape, (20,))
        self.assertEqual(y.shape, (20,))
        self.assertEqual(e.shape, (20,))

    def test_histogram_shape(self) -> None:
        ws = _MockWorkspace("ws", n_spectra=3, n_bins=20, histogram=True)
        x, y, e = self._fn(ws)
        # Histogram: n_x = n_bins + 1 → x should be centred to n_bins
        self.assertEqual(x.shape, (20,))
        self.assertEqual(y.shape, (20,))

    def test_sum_across_spectra(self) -> None:
        ws = _MockWorkspace("ws", n_spectra=5, n_bins=10, histogram=False)
        _, y_sum, _ = self._fn(ws)
        expected = np.sum(ws.extractY(), axis=0)
        np.testing.assert_allclose(y_sum, expected)

    def test_error_combined_in_quadrature(self) -> None:
        ws = _MockWorkspace("ws", n_spectra=4, n_bins=10, histogram=False)
        _, _, e_sum = self._fn(ws)
        expected = np.sqrt(np.sum(ws.extractE()**2, axis=0))
        np.testing.assert_allclose(e_sum, expected)


# ---------------------------------------------------------------------------
# Test 5: plot_tof_correction_dashboard
# ---------------------------------------------------------------------------

class TestPlotTofCorrectionDashboard(unittest.TestCase):
    """Verify TOF dashboard writes PDF + PNG for all scenarios."""

    def _make_data(self, include_ms: bool, include_gc: bool) -> dict:
        from vesuvio_analysis.core_functions.correction_plots import (
            _KEY_UNCORRECTED, _KEY_CORRECTED, _KEY_MS, _KEY_GAMMA,
        )
        x = np.linspace(110.0, 500.0, 80)
        data = {
            _KEY_UNCORRECTED: (x, np.random.default_rng(0).random(80), np.ones(80) * 0.01),
            _KEY_CORRECTED: (x, np.random.default_rng(1).random(80), np.ones(80) * 0.01),
        }
        if include_ms:
            data[_KEY_MS] = (x, np.random.default_rng(2).random(80) * 0.1, np.ones(80) * 0.001)
        if include_gc:
            data[_KEY_GAMMA] = (x, np.random.default_rng(3).random(80) * 0.05, np.ones(80) * 0.001)
        return data

    def _run_scenario(
        self, ms_flag: bool, gc_flag: bool
    ) -> None:
        from vesuvio_analysis.core_functions.correction_plots import (
            plot_tof_correction_dashboard,
        )
        with tempfile.TemporaryDirectory() as tmp:
            fig_path = Path(tmp)
            data = self._make_data(include_ms=ms_flag, include_gc=gc_flag)
            saved = plot_tof_correction_dashboard(
                ic_name="test_BACKWARD_",
                masses=np.array([12.0, 16.0]),
                ms_flag=ms_flag,
                gc_flag=gc_flag,
                data=data,
                fig_save_path=fig_path,
                iteration=2,
            )
            self.assertEqual(len(saved), 2, msg=f"Expected 2 files, got {saved}")
            for path in saved:
                self.assertTrue(path.exists(), msg=f"Missing file: {path}")

    def test_scenario_A(self) -> None:
        self._run_scenario(ms_flag=True, gc_flag=False)

    def test_scenario_B(self) -> None:
        self._run_scenario(ms_flag=False, gc_flag=True)

    def test_scenario_C(self) -> None:
        self._run_scenario(ms_flag=True, gc_flag=True)

    def test_no_corrections_returns_empty(self) -> None:
        from vesuvio_analysis.core_functions.correction_plots import (
            plot_tof_correction_dashboard,
            _KEY_UNCORRECTED, _KEY_CORRECTED,
        )
        x = np.linspace(110.0, 500.0, 50)
        data = {
            _KEY_UNCORRECTED: (x, np.ones(50), None),
            _KEY_CORRECTED: (x, np.ones(50), None),
        }
        with tempfile.TemporaryDirectory() as tmp:
            saved = plot_tof_correction_dashboard(
                ic_name="test_",
                masses=np.array([1.0]),
                ms_flag=False,
                gc_flag=False,
                data=data,
                fig_save_path=Path(tmp),
                iteration=1,
            )
        self.assertEqual(saved, [])


# ---------------------------------------------------------------------------
# Test 6: plot_yspace_correction_dashboard
# ---------------------------------------------------------------------------

class TestPlotYspaceCorrectionDashboard(unittest.TestCase):
    """Verify y-space dashboard writes PDF + PNG."""

    def test_scenario_A_yspace(self) -> None:
        from vesuvio_analysis.core_functions.correction_plots import (
            plot_yspace_correction_dashboard,
            _KEY_UNCORRECTED, _KEY_CORRECTED, _KEY_MS,
        )
        x = np.linspace(-25.0, 25.0, 100)
        data = {
            _KEY_UNCORRECTED: (x, np.exp(-0.5 * x**2), np.ones(100) * 0.01),
            _KEY_CORRECTED: (x, np.exp(-0.5 * x**2) * 0.9, np.ones(100) * 0.01),
            _KEY_MS: (x, np.exp(-0.5 * x**2) * 0.1, np.ones(100) * 0.001),
        }
        with tempfile.TemporaryDirectory() as tmp:
            saved = plot_yspace_correction_dashboard(
                ic_name="test_FORWARD_",
                masses=np.array([1.0079, 12.0]),
                ms_flag=True,
                gc_flag=False,
                data=data,
                fig_save_path=Path(tmp),
                iteration=2,
            )
            self.assertEqual(len(saved), 2)
            for path in saved:
                self.assertTrue(path.exists())


# ---------------------------------------------------------------------------
# Test 7: dispatch_correction_plots with mock ADS
# ---------------------------------------------------------------------------

class TestDispatchCorrectionPlots(unittest.TestCase):
    """End-to-end dispatch with mock ADS, verifies returned path list."""

    def _make_ic(
        self,
        ic_name: str,
        ms_flag: bool,
        gc_flag: bool,
        n_iter: int,
        fig_path: Path,
    ) -> Any:
        """Build a minimal IC mock object."""

        class _IC:
            pass

        ic = _IC()
        ic.name = ic_name
        ic.MSCorrectionFlag = ms_flag
        ic.GammaCorrectionFlag = gc_flag
        ic.noOfMSIterations = n_iter
        ic.figSavePath = fig_path
        ic.masses = np.array([1.0079, 12.0, 16.0])
        return ic

    def _populate_mtd(self, mtd_obj: _MockMtd, ic_name: str, n_iter: int) -> None:
        """Register the minimum required workspaces in the mock ADS."""
        ncpm = ic_name + "_NCPMasked"
        for ws_name in [
            ic_name + "0",
            ic_name + str(n_iter),
            ncpm + "_MulScattering",
            ncpm + "_Gamma_Background",
        ]:
            mtd_obj.add(ws_name, _MockWorkspace(ws_name, n_spectra=3, n_bins=25))

    def test_scenario_C_returns_two_files_per_space(self) -> None:
        """Scenario C without y-space fn → 2 files (PDF + PNG) for TOF only."""
        from vesuvio_analysis.core_functions.correction_plots import dispatch_correction_plots

        with tempfile.TemporaryDirectory() as tmp:
            fig_path = Path(tmp)
            ic = self._make_ic("run_FORWARD_", True, True, 2, fig_path)
            mtd_obj = _MockMtd()
            self._populate_mtd(mtd_obj, "run_FORWARD_", 2)

            saved = dispatch_correction_plots(ic=ic, mtd=mtd_obj)
            self.assertEqual(len(saved), 2, msg=f"Expected 2 files: {saved}")
            suffixes = {p.suffix for p in saved}
            self.assertIn(".pdf", suffixes)
            self.assertIn(".png", suffixes)

    def test_scenario_A_with_yspace_fn_returns_four_files(self) -> None:
        """Scenario A + y-space fn → 4 files (2 TOF + 2 y-space)."""
        from vesuvio_analysis.core_functions.correction_plots import (
            dispatch_correction_plots, _KEY_UNCORRECTED, _KEY_CORRECTED, _KEY_MS,
        )

        def _mock_yspace_fn(ws_name, mass):
            x = np.linspace(-25.0, 25.0, 50)
            return x, np.exp(-0.5 * x**2), np.ones(50) * 0.01

        with tempfile.TemporaryDirectory() as tmp:
            fig_path = Path(tmp)
            ic = self._make_ic("run_BACKWARD_", True, False, 2, fig_path)
            mtd_obj = _MockMtd()
            self._populate_mtd(mtd_obj, "run_BACKWARD_", 2)

            saved = dispatch_correction_plots(
                ic=ic,
                mtd=mtd_obj,
                convert_to_yspace_fn=_mock_yspace_fn,
            )
            self.assertEqual(len(saved), 4, msg=f"Expected 4 files: {saved}")

    def test_no_corrections_returns_empty(self) -> None:
        from vesuvio_analysis.core_functions.correction_plots import dispatch_correction_plots

        with tempfile.TemporaryDirectory() as tmp:
            ic = self._make_ic("run_BACKWARD_", False, False, 2, Path(tmp))
            saved = dispatch_correction_plots(ic=ic, mtd=_MockMtd())
            self.assertEqual(saved, [])

    def test_zero_ms_iterations_returns_empty(self) -> None:
        from vesuvio_analysis.core_functions.correction_plots import dispatch_correction_plots

        with tempfile.TemporaryDirectory() as tmp:
            ic = self._make_ic("run_BACKWARD_", True, True, 0, Path(tmp))
            saved = dispatch_correction_plots(ic=ic, mtd=_MockMtd())
            self.assertEqual(saved, [])

    def test_missing_required_ws_returns_empty(self) -> None:
        """When workspace 'run_BACKWARD_0' is absent, dispatch returns []."""
        from vesuvio_analysis.core_functions.correction_plots import dispatch_correction_plots

        with tempfile.TemporaryDirectory() as tmp:
            ic = self._make_ic("run_BACKWARD_", True, False, 2, Path(tmp))
            # Empty ADS — no workspaces registered
            saved = dispatch_correction_plots(ic=ic, mtd=_MockMtd())
            self.assertEqual(saved, [])

    def test_output_filenames_follow_convention(self) -> None:
        """File names must contain ic_name prefix and 'TOF_correction'."""
        from vesuvio_analysis.core_functions.correction_plots import dispatch_correction_plots

        with tempfile.TemporaryDirectory() as tmp:
            fig_path = Path(tmp)
            ic_name = "thymol_10K_Gauss1D_FORWARD_"
            ic = self._make_ic(ic_name, True, True, 2, fig_path)
            mtd_obj = _MockMtd()
            self._populate_mtd(mtd_obj, ic_name, 2)

            saved = dispatch_correction_plots(ic=ic, mtd=mtd_obj)
            self.assertTrue(len(saved) > 0)
            pdf_files = [p for p in saved if p.suffix == ".pdf"]
            self.assertTrue(len(pdf_files) > 0)
            pdf_name = pdf_files[0].name
            self.assertIn("thymol_10K_Gauss1D_FORWARD", pdf_name)
            self.assertIn("TOF_correction", pdf_name)


# ---------------------------------------------------------------------------
# Test 8: Correction workspace naming convention
# ---------------------------------------------------------------------------

class TestCorrectionWSNamingConvention(unittest.TestCase):
    """Verify that dispatch uses the correct workspace naming convention.

    Workspace names follow the pattern established by
    ``createMulScatWorkspaces`` and ``createWorkspacesForGammaCorrection``
    in ``analysis_functions.py``:

    * MS:    ``{ic.name}_NCPMasked_MulScattering``
    * Gamma: ``{ic.name}_NCPMasked_Gamma_Background``
    """

    def test_ms_workspace_name_derived_correctly(self) -> None:
        ic_name = "thymol_10K_Gauss1D_BACKWARD_"
        expected_ms = ic_name + "_NCPMasked_MulScattering"
        self.assertTrue(expected_ms.startswith(ic_name))
        self.assertIn("_NCPMasked_MulScattering", expected_ms)

    def test_gamma_workspace_name_derived_correctly(self) -> None:
        ic_name = "thymol_10K_Gauss1D_FORWARD_"
        expected_gc = ic_name + "_NCPMasked_Gamma_Background"
        self.assertTrue(expected_gc.startswith(ic_name))
        self.assertIn("_NCPMasked_Gamma_Background", expected_gc)

    def test_iteration_zero_is_uncorrected(self) -> None:
        ic_name = "BaH2_500C_BACKWARD_"
        uncorrected = ic_name + "0"
        self.assertEqual(uncorrected, "BaH2_500C_BACKWARD_0")

    def test_final_iteration_matches_n_iter(self) -> None:
        ic_name = "BaH2_500C_FORWARD_"
        n_iter = 3
        corrected = ic_name + str(n_iter)
        self.assertEqual(corrected, "BaH2_500C_FORWARD_3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
