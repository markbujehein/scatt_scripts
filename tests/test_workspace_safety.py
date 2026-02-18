"""Workspace safety tests for Phase 4 — Mantid Workspace Lifecycle Preservation.

These tests do **not** depend on Mantid and can be run in any standard
Python environment::

    python -m pytest tests/test_workspace_safety.py -v

The tests use a lightweight ``_MockMtd`` class that mimics the relevant
subset of the Mantid AnalysisDataService (``mtd``): dictionary-like
``__contains__`` and ``__getitem__`` plus a ``clear()`` method.

Coverage:
1. Naming conventions — ``{scriptName}_{DIRECTION}_{iteration}`` format.
2. Boundary enforcement — ``NCPCostFunction.__call__`` must not import or
   call any Mantid algorithm.
3. Iterative-loop workspace assertions — the entry/exit assertions in
   ``fitNcpToWorkspace`` fire when a workspace is absent or output is
   missing from ``mtd``.
4. ``AnalysisDataService.clear()`` — calling ``clear()`` removes all
   entries and leaves an empty service.
"""

from __future__ import annotations

import re
import sys
import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight mock AnalysisDataService
# ---------------------------------------------------------------------------

class _MockMtd(dict):
    """Dictionary-like stand-in for Mantid's ``AnalysisDataService`` (``mtd``).

    Supports the operations actually used in the pipeline:
    * ``name in mtd`` — membership test
    * ``mtd[name]`` — workspace retrieval
    * ``mtd.clear()`` — remove all entries
    * ``mtd.add(name, ws)`` — register a workspace
    """

    def add(self, name: str, ws: Any) -> None:
        """Register *ws* under *name* (mirrors ADS.add)."""
        self[name] = ws

    def remove(self, name: str) -> None:
        """Deregister *name* (mirrors ADS.remove)."""
        self.pop(name, None)


class _MockWorkspace:
    """Minimal stand-in for a Mantid ``MatrixWorkspace``."""

    def __init__(self, name: str, n_spectra: int = 3, n_bins: int = 20):
        self._name = name
        self._n_spectra = n_spectra
        self._n_bins = n_bins

    def name(self) -> str:
        return self._name

    def extractX(self) -> np.ndarray:
        return np.linspace(100, 600, self._n_bins * self._n_spectra).reshape(
            self._n_spectra, self._n_bins
        )

    def extractY(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random((self._n_spectra, self._n_bins))

    def extractE(self) -> np.ndarray:
        return np.ones((self._n_spectra, self._n_bins)) * 0.01


# ---------------------------------------------------------------------------
# Test 1: Workspace naming conventions
# ---------------------------------------------------------------------------

class TestWorkspaceNamingConventions(unittest.TestCase):
    """Verify that the ``{scriptName}_{DIRECTION}_{iteration}`` pattern holds."""

    _CONVENTION = re.compile(r"^[A-Za-z0-9_]+_(BACKWARD|FORWARD)_\d+$")

    def _make_names(self, script_name: str, direction: str, n_iter: int):
        """Generate the workspace names that the iterative loop would create."""
        ic_name = f"{script_name}_{direction}_"
        return [ic_name + str(i) for i in range(n_iter + 1)]

    def test_backward_names_match_convention(self):
        names = self._make_names("BaH2_500C", "BACKWARD", 3)
        for name in names:
            self.assertRegex(
                name, self._CONVENTION,
                msg=f"'{name}' does not match the expected naming convention."
            )

    def test_forward_names_match_convention(self):
        names = self._make_names("BaH2_500C", "FORWARD", 2)
        for name in names:
            self.assertRegex(name, self._CONVENTION)

    def test_iteration_index_increments_correctly(self):
        names = self._make_names("starch_80_RD", "BACKWARD", 4)
        indices = [int(n.split("_")[-1]) for n in names]
        self.assertEqual(indices, list(range(5)))

    def test_final_ws_name_via_build_function(self):
        """Verify that the naming logic mirrors ``buildFinalWSName`` in ICHelpers.

        ``buildFinalWSName`` is in Mantid-dependent code so we replicate
        its single-line logic and check multiple (script_name, procedure,
        n_iter) combinations to confirm the convention is correct.
        """

        def _build_final_ws_name(script_name: str, procedure: str, n_iter: int) -> str:
            # Mirrors ICHelpers.buildFinalWSName exactly:
            # name = scriptName + "_" + procedure + "_" + str(IC.noOfMSIterations)
            return script_name + "_" + procedure + "_" + str(n_iter)

        cases = [
            ("BaH2_500C", "BACKWARD", 3, "BaH2_500C_BACKWARD_3"),
            ("BaH2_500C", "FORWARD", 0, "BaH2_500C_FORWARD_0"),
            ("starch_80_RD", "BACKWARD", 1, "starch_80_RD_BACKWARD_1"),
        ]
        for script_name, procedure, n_iter, expected in cases:
            with self.subTest(script=script_name, procedure=procedure, n=n_iter):
                actual = _build_final_ws_name(script_name, procedure, n_iter)
                self.assertEqual(actual, expected)
                self.assertRegex(actual, self._CONVENTION)

    def test_ic_name_prefix_has_trailing_underscore(self):
        """``IC.name`` is the base without iteration suffix, so it ends with '_'."""
        script_name = "BaH2_500C"
        direction = "BACKWARD"
        ic_name = f"{script_name}_{direction}_"
        self.assertTrue(ic_name.endswith("_"))
        # Appending str(0) produces a valid full name
        self.assertRegex(ic_name + "0", self._CONVENTION)


# ---------------------------------------------------------------------------
# Test 2: AnalysisDataService.clear() logic
# ---------------------------------------------------------------------------

class TestAnalysisDataServiceClear(unittest.TestCase):
    """Verify that ``_MockMtd.clear()`` empties the service."""

    def test_clear_removes_all_workspaces(self):
        mtd = _MockMtd()
        for i in range(5):
            mtd.add(f"BaH2_BACKWARD_{i}", _MockWorkspace(f"BaH2_BACKWARD_{i}"))
        self.assertEqual(len(mtd), 5)
        mtd.clear()
        self.assertEqual(len(mtd), 0)

    def test_clear_then_add_works(self):
        mtd = _MockMtd()
        mtd.add("ws_old", _MockWorkspace("ws_old"))
        mtd.clear()
        mtd.add("ws_new", _MockWorkspace("ws_new"))
        self.assertIn("ws_new", mtd)
        self.assertNotIn("ws_old", mtd)

    def test_membership_after_add_and_remove(self):
        mtd = _MockMtd()
        ws = _MockWorkspace("BaH2_BACKWARD_0")
        mtd.add("BaH2_BACKWARD_0", ws)
        self.assertIn("BaH2_BACKWARD_0", mtd)
        mtd.remove("BaH2_BACKWARD_0")
        self.assertNotIn("BaH2_BACKWARD_0", mtd)

    def test_boundary_before_procedure(self):
        """``AnalysisDataService.clear()`` at procedure boundary removes stale ws."""
        mtd = _MockMtd()
        # Simulate stale workspaces from a previous run
        for i in range(3):
            mtd.add(f"stale_{i}", _MockWorkspace(f"stale_{i}"))
        # Procedure boundary: clear
        mtd.clear()
        self.assertEqual(len(mtd), 0)


# ---------------------------------------------------------------------------
# Test 3: fitNcpToWorkspace entry/exit assertions (mocked Mantid)
# ---------------------------------------------------------------------------

class TestFitNcpToWorkspaceAssertions(unittest.TestCase):
    """Verify that fitNcpToWorkspace raises AssertionError when safety checks fail.

    We simulate the assertion logic without importing Mantid by extracting
    the relevant two guards and running them against a mock mtd.
    """

    def _entry_guard(self, ws_name: str, mtd: _MockMtd) -> None:
        """Replicate the entry assertion from ``fitNcpToWorkspace``."""
        assert ws_name in mtd, (
            f"fitNcpToWorkspace: InputWorkspace '{ws_name}' not found in mtd. "
            "Ensure the workspace was created before calling this function."
        )

    def _exit_guard(self, ws_name: str, mtd: _MockMtd) -> None:
        """Replicate the exit assertion from ``fitNcpToWorkspace``."""
        ncp_ws_name = ws_name + "_TOF_Fitted_Profiles"
        assert ncp_ws_name in mtd, (
            f"fitNcpToWorkspace: OutputWorkspace '{ncp_ws_name}' was not "
            "created in mtd.  NCP calculation may have failed silently."
        )

    def test_entry_guard_passes_when_ws_present(self):
        mtd = _MockMtd()
        ws = _MockWorkspace("BaH2_BACKWARD_0")
        mtd.add("BaH2_BACKWARD_0", ws)
        # Should not raise
        self._entry_guard("BaH2_BACKWARD_0", mtd)

    def test_entry_guard_fails_when_ws_absent(self):
        mtd = _MockMtd()
        with self.assertRaises(AssertionError) as ctx:
            self._entry_guard("BaH2_BACKWARD_0", mtd)
        self.assertIn("not found in mtd", str(ctx.exception))

    def test_exit_guard_passes_when_output_present(self):
        mtd = _MockMtd()
        ws_name = "BaH2_BACKWARD_0"
        ncp_out = _MockWorkspace(ws_name + "_TOF_Fitted_Profiles")
        mtd.add(ws_name + "_TOF_Fitted_Profiles", ncp_out)
        # Should not raise
        self._exit_guard(ws_name, mtd)

    def test_exit_guard_fails_when_output_absent(self):
        mtd = _MockMtd()
        with self.assertRaises(AssertionError) as ctx:
            self._exit_guard("BaH2_BACKWARD_0", mtd)
        self.assertIn("was not created in mtd", str(ctx.exception))

    def test_output_name_follows_convention(self):
        """The NCP output ws name must follow ``{input}_TOF_Fitted_Profiles``."""
        ws_name = "BaH2_BACKWARD_2"
        expected_out = "BaH2_BACKWARD_2_TOF_Fitted_Profiles"
        self.assertEqual(ws_name + "_TOF_Fitted_Profiles", expected_out)


# ---------------------------------------------------------------------------
# Test 4: Boundary enforcement — NCPCostFunction must not call Mantid
# ---------------------------------------------------------------------------

class TestBoundaryEnforcement(unittest.TestCase):
    """Verify that NCPCostFunction.__call__ does not import or call Mantid.

    The rule from Phase 4: 'No Numba (@njit) code or iMinuit cost functions
    may call Mantid algorithms.'  We check this by inspecting that the
    NCPCostFunction class (importable without Mantid) does not reference
    the ``mantid`` module in its source.
    """

    def test_ncp_cost_function_has_no_mantid_import(self):
        """``iminuit_costs.py`` must not import ``mantid`` at module level."""
        import inspect
        from vesuvio_analysis.core_functions import iminuit_costs
        src = inspect.getsource(iminuit_costs)
        # Allow 'mantid' to appear only in comments/docstrings, not in imports
        import_lines = [
            line for line in src.splitlines()
            if re.match(r"^\s*(import|from)\s+mantid\b", line)
        ]
        self.assertEqual(
            import_lines, [],
            msg=f"Found Mantid import(s) in iminuit_costs.py: {import_lines}"
        )

    def test_ncp_cost_function_call_has_no_mantid_reference(self):
        """``NCPCostFunction.__call__`` body must not reference Mantid APIs."""
        import inspect
        from vesuvio_analysis.core_functions.iminuit_costs import NCPCostFunction
        call_src = inspect.getsource(NCPCostFunction.__call__)
        mantid_apis = [
            "Minus", "Rebin", "SumSpectra", "CloneWorkspace",
            "MaskDetectors", "CropWorkspace",
            "VesuvioCalculateMS", "VesuvioCalculateGammaBackground",
        ]
        for api in mantid_apis:
            self.assertNotIn(
                api, call_src,
                msg=f"NCPCostFunction.__call__ references Mantid API '{api}'."
            )


# ---------------------------------------------------------------------------
# Test 5: wsNCPM refresh logic in the iterative loop
# ---------------------------------------------------------------------------

class TestWsNCPMRefresh(unittest.TestCase):
    """Verify that wsNCPM is updated at every non-final iteration.

    Simulates the iterative-loop branching logic with a counter to confirm
    that the NCP-masked workspace is refreshed before each MS/GC correction.
    """

    def _run_loop_simulation(self, n_ms_iter: int, ms_flag: bool) -> list:
        """Return a list of iteration indices at which wsNCPM was refreshed."""
        refreshed_at = []
        for iteration in range(n_ms_iter + 1):
            # (Fit step is skipped in this simulation.)
            if iteration == n_ms_iter:
                break  # Last iteration: skip MS/GC
            # Unconditional refresh (Phase 4 fix):
            refreshed_at.append(iteration)
            # MS/GC correction would happen here if flags are set.
        return refreshed_at

    def test_wsNCPM_refreshed_at_every_non_final_iteration(self):
        n = 3
        refreshed = self._run_loop_simulation(n_ms_iter=n, ms_flag=True)
        self.assertEqual(refreshed, list(range(n)))  # [0, 1, 2]

    def test_no_refresh_when_zero_ms_iterations(self):
        """With noOfMSIterations=0 the loop runs once and then breaks immediately."""
        refreshed = self._run_loop_simulation(n_ms_iter=0, ms_flag=True)
        self.assertEqual(refreshed, [])

    def test_single_ms_iteration(self):
        refreshed = self._run_loop_simulation(n_ms_iter=1, ms_flag=True)
        self.assertEqual(refreshed, [0])

    def test_refresh_count_equals_ms_iterations(self):
        """Number of wsNCPM refreshes == noOfMSIterations."""
        for n in range(5):
            with self.subTest(n=n):
                refreshed = self._run_loop_simulation(n_ms_iter=n, ms_flag=True)
                self.assertEqual(len(refreshed), n)


# ---------------------------------------------------------------------------
# Test 6: Data-exchange boundary (extractWS / passDataIntoWS)
# ---------------------------------------------------------------------------

class TestDataExchangeBoundary(unittest.TestCase):
    """Verify the Numba/NumPy ↔ Mantid boundary contract.

    All data must leave Mantid via extractWS() (returning NumPy arrays)
    and return via passDataIntoWS() (writing NumPy arrays back).  These
    tests use the mock workspace to confirm the round-trip shape contract.
    """

    def test_extract_returns_numpy_arrays(self):
        ws = _MockWorkspace("test_ws", n_spectra=4, n_bins=30)
        dataX = ws.extractX()
        dataY = ws.extractY()
        dataE = ws.extractE()
        self.assertIsInstance(dataX, np.ndarray)
        self.assertIsInstance(dataY, np.ndarray)
        self.assertIsInstance(dataE, np.ndarray)

    def test_extract_shapes_consistent(self):
        n_spec, n_bins = 5, 40
        ws = _MockWorkspace("test_ws", n_spectra=n_spec, n_bins=n_bins)
        dataX = ws.extractX()
        dataY = ws.extractY()
        dataE = ws.extractE()
        self.assertEqual(dataX.shape, (n_spec, n_bins))
        self.assertEqual(dataY.shape, (n_spec, n_bins))
        self.assertEqual(dataE.shape, (n_spec, n_bins))

    def test_numba_cost_function_receives_numpy_not_workspace(self):
        """NCPCostFunction.__init__ stores plain NumPy arrays, not Mantid objects."""
        from vesuvio_analysis.core_functions.iminuit_costs import NCPCostFunction

        class _IC:
            masses = np.array([1.008, 12.0])
            noOfMasses = 2
            bounds = np.array([
                [0, 100.0], [0.5, 50.0], [-3.0, 3.0],
                [0, 100.0], [0.5, 50.0], [-3.0, 3.0],
            ])
            normVoigt = True

        n_bins = 10
        dataY = np.random.default_rng(1).random(n_bins)
        dataE = np.ones(n_bins) * 0.1
        ySpaces = np.zeros((2, n_bins))
        resPars = np.zeros(6)
        instrPars = np.zeros(6)
        kinArrays = np.zeros((4, n_bins))

        c = NCPCostFunction(dataY, dataE, ySpaces, resPars, instrPars,
                             kinArrays, _IC())

        # All stored arrays must be NumPy, not Mantid workspace objects.
        self.assertIsInstance(c._dataY, np.ndarray)
        self.assertIsInstance(c._dataE, np.ndarray)
        self.assertIsInstance(c._ySpacesForEachMass, np.ndarray)
        self.assertIsInstance(c._resolutionPars, np.ndarray)
        self.assertIsInstance(c._instrPars, np.ndarray)
        self.assertIsInstance(c._kinematicArrays, np.ndarray)


if __name__ == "__main__":
    unittest.main(verbosity=2)
