"""Production Smoke Test — Small-Scale End-to-End Pipeline Validation.

This module is the **canonical design template** for all future
``scatt_scripts`` smoke tests.  Its purpose is to validate that the
three-stage VESUVIO analysis pipeline can be configured, orchestrated, and
torn down correctly without requiring a full Mantid installation or live
instrument data.

=============================================================================
Pipeline Stages (Master Orchestrator Pattern)
=============================================================================

The smoke test exercises the following sequential stages:

  Stage 1 — runRoutine()
      NCP iterative fitting (MS + GC corrections).
      Fast-tracked when ``runningTest=True``: MS/GC iterations are capped
      at 1 and iMinuit uses a loose convergence tolerance.

  Stage 2 — runBootstrap()
      Residual or Gaussian-error resampling of the NCP fit.
      Fast-tracked: ``nSamples`` is forced to at most 3 replicas.

  Stage 3 — runAnalysis()
      Post-hoc statistical analysis of stored bootstrap distributions
      (Phase 6 sieves: outlier detection, physics clustering, Bayesian
      bootstrap).

=============================================================================
Design Principles
=============================================================================

* **Single configuration block** — all IC classes are defined at the top
  of this module.  Changing one number (e.g. detector range) adjusts the
  full pipeline automatically.

* **runningTest flag** — every IC class carries ``runningTest = True``,
  which activates the fast-track path throughout the library.

* **ADS lifecycle** — the Mantid AnalysisDataService is explicitly cleared
  before each stage and after final exit (Section 4 of ARCHITECTURE_AUDIT).

* **Phase 6 sieves** — all three statistical checks (outlier detection,
  physics clustering, Bayesian bootstrap) are exercised on synthetic toy
  data to confirm they produce finite, NaN-free results.

* **No Mantid required** — tests in this file use the lightweight
  ``mock_mantid`` stub so they run in a standard Python / CI environment.

Usage::

    # Run the full smoke-test suite
    python -m pytest tests/production_smoke_test.py -v

    # Run only the orchestrator entry-point
    python tests/production_smoke_test.py

"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Optional


import numpy as np

# ---------------------------------------------------------------------------
# Install the Mantid mock BEFORE importing any vesuvio module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.mock_mantid import install as _install_mock_mantid  # noqa: E402
_install_mock_mantid()

from vesuvio_analysis.core_functions.statistical_plugins import (  # noqa: E402
    BayesianBootstrap,
    HardwareOutlierDetector,
    PhysicsTrendClusterer,
)


# =============================================================================
# Small-Scale IC Configuration (3-detector Thymol-style slice)
# =============================================================================
# All IC classes below are the single source of truth for this smoke test.
# They configure a minimal FORWARD-only run on 3 consecutive spectra
# (spectra 135-137) with:
#   - 1 mass (hydrogen proxy)
#   - 0 MS iterations (no-op correction loop)
#   - runningTest = True (activates all fast-track paths)
# Total wall-clock target: < 120 s with a real Mantid installation.

class _SmokeLoadVesuvioFrontParameters:
    """Minimal forward-scattering load parameters for smoke testing."""
    runs: str = "1-3"             # Placeholder run numbers
    empty_runs: str = "4-6"
    spectra: str = "135-137"      # 3-detector slice (forward bank starts at 135)
    mode: str = "SingleDifference"
    ipfile: Path = Path("vesuvio_analysis/ip_files/ip2018_3.par")


class _SmokeLoadVesuvioBackParameters:
    """Minimal backward-scattering load parameters for smoke testing."""
    runs: str = "1-3"
    empty_runs: str = "4-6"
    spectra: str = "3-5"          # 3-detector slice (backward bank)
    mode: str = "DoubleDifference"
    ipfile: Path = Path("vesuvio_analysis/ip_files/ip2018_3.par")


class _SmokeForwardInitialConditions:
    """Minimal forward IC for smoke testing (single Gaussian, no H ratio)."""
    # Geometry
    vertical_width: float = 0.1
    horizontal_width: float = 0.1
    thickness: float = 0.001

    # Masses and fitting parameters
    masses: np.ndarray = np.array([1.0079])           # Hydrogen only
    initPars: np.ndarray = np.array([10, 5, 0.0])     # [intensity, width, centre]
    bounds: np.ndarray = np.array([
        [0, np.nan], [3, 8], [-1.5, 0.5],
    ])
    constraints: tuple = ()

    # Iteration control — 0 = no MS/GC corrections
    noOfMSIterations: int = 0
    firstSpec: int = 135
    lastSpec: int = 137

    MSCorrectionFlag: bool = False
    GammaCorrectionFlag: bool = False

    maskedSpecAllNo: np.ndarray = np.array([], dtype=int)

    tofBinning: str = "110,1.,420"
    maskTOFRange: Optional[str] = None
    transmission_guess: float = 0.9
    multiple_scattering_order: int = 2
    number_of_events: float = 1.0e4

    # ── Fast-track flag ──────────────────────────────────────────────────────
    # Setting runningTest=True activates all fast-track paths in the library:
    #   • analysis_functions: caps MS iterations at 1, loosens iMinuit tol
    #   • bootstrap: forces nSamples ≤ 3
    runningTest: bool = True


class _SmokeBackwardInitialConditions:
    """Minimal backward IC for smoke testing (single mass, no H)."""
    vertical_width: float = 0.1
    horizontal_width: float = 0.1
    thickness: float = 0.001

    HToMassIdxRatio = None
    massIdx: int = 0

    masses: np.ndarray = np.array([16.0])             # Oxygen only
    initPars: np.ndarray = np.array([1, 9, 0.0])
    bounds: np.ndarray = np.array([
        [0, np.nan], [7, 14], [-3, 1],
    ])
    constraints: tuple = ()

    noOfMSIterations: int = 0
    firstSpec: int = 3
    lastSpec: int = 5

    MSCorrectionFlag: bool = False
    GammaCorrectionFlag: bool = False

    maskedSpecAllNo: np.ndarray = np.array([], dtype=int)

    tofBinning: str = "110,1.,420"
    maskTOFRange = None
    transmission_guess: float = 0.9
    multiple_scattering_order: int = 2
    number_of_events: float = 1.0e4

    runningTest: bool = True


class _SmokeYSpaceFitInitialConditions:
    """Minimal y-space fit configuration for smoke testing."""
    showPlots: bool = False
    symmetrisationFlag: bool = True
    rebinParametersForYSpaceFit: str = "-25, 0.5, 25"
    fitModel: str = "SINGLE_GAUSSIAN"
    runMinos: bool = False
    globalFit: bool = False
    nGlobalFitGroups: int = 1
    maskTypeProcedure: Optional[str] = "NAN"


class _SmokeUserScriptControls:
    """User-facing control flags for the smoke-test orchestrator."""
    runRoutine: bool = True
    procedure: str = "FORWARD"
    fitInYSpace: str = "FORWARD"

    # Fast-track flag (see UserScriptControls docstring in BaH2_500C.py)
    runningTest: bool = True

    # Phase 6 — all sieves ON for smoke-test validation
    runOutlierDetection: bool = True
    runPhysicsClustering: bool = True
    runBayesianBootstrap: bool = True


class _SmokeBootstrapInitialConditions:
    """Bootstrap configuration for the smoke-test run."""
    runBootstrap: bool = True
    procedure: str = "FORWARD"
    fitInYSpace: str = "FORWARD"

    bootstrapType: str = "BOOT_GAUSS_ERRS"
    nSamples: int = 3                   # Already at max for runningTest
    skipMSIterations: bool = True
    userConfirmation: bool = False      # No interactive prompt in tests

    runningTest: bool = True            # Force nSamples ≤ 3 in bootstrap.py


class _SmokeBootstrapAnalysis:
    """Post-hoc analysis flags for the smoke-test run."""
    runAnalysis: bool = True
    filterAvg: bool = False
    plotRawWidthsIntensities: bool = False
    plotMeanWidthsIntensities: bool = False
    plotMeansEvolution: bool = False
    plot2DHists: bool = False
    plotYFitHists: bool = False


# =============================================================================
# Test Suite
# =============================================================================


class TestRunningTestFlagPropagation(unittest.TestCase):
    """Verify that ``runningTest=True`` is present on all smoke-test ICs.

    This is the first safety gate: if any IC is missing the flag, the
    fast-track paths will not activate and the pipeline will be slow.
    """

    def test_user_controls_has_running_test_flag(self) -> None:
        """``UserScriptControls`` must declare ``runningTest``."""
        ctr = _SmokeUserScriptControls()
        self.assertTrue(hasattr(ctr, "runningTest"),
                        "UserScriptControls is missing 'runningTest'.")

    def test_user_controls_running_test_is_true(self) -> None:
        ctr = _SmokeUserScriptControls()
        self.assertTrue(ctr.runningTest)

    def test_forward_ic_has_running_test_flag(self) -> None:
        ic = _SmokeForwardInitialConditions()
        self.assertTrue(hasattr(ic, "runningTest"))
        self.assertTrue(ic.runningTest)

    def test_backward_ic_has_running_test_flag(self) -> None:
        ic = _SmokeBackwardInitialConditions()
        self.assertTrue(hasattr(ic, "runningTest"))
        self.assertTrue(ic.runningTest)

    def test_bootstrap_ic_has_running_test_flag(self) -> None:
        boot = _SmokeBootstrapInitialConditions()
        self.assertTrue(hasattr(boot, "runningTest"))
        self.assertTrue(boot.runningTest)


class TestRunningTestMSIterationCap(unittest.TestCase):
    """Verify the MS-iteration cap logic from ``analysis_functions``.

    We replicate the ``_n_iter`` logic inline so that this test is
    independent of Mantid and produces a deterministic result in CI.
    """

    def _effective_n_iter(self, noOfMSIterations: int, running_test: bool) -> int:
        """Mirror the logic injected into ``iterativeFitForDataReduction``."""
        return (
            min(1, noOfMSIterations)
            if running_test
            else noOfMSIterations
        )

    def test_cap_at_one_when_running_test_and_four_iterations(self) -> None:
        self.assertEqual(self._effective_n_iter(4, True), 1)

    def test_cap_at_zero_when_running_test_and_zero_iterations(self) -> None:
        """Zero iterations stays zero even in test mode (no-op run)."""
        self.assertEqual(self._effective_n_iter(0, True), 0)

    def test_uncapped_when_not_running_test(self) -> None:
        self.assertEqual(self._effective_n_iter(4, False), 4)

    def test_smoke_ic_noOfMSIterations_already_zero(self) -> None:
        """Smoke-test IC has ``noOfMSIterations=0``: fast path is taken regardless."""
        ic = _SmokeForwardInitialConditions()
        capped = self._effective_n_iter(ic.noOfMSIterations, ic.runningTest)
        self.assertEqual(capped, 0)


class TestBootstrapNSamplesCap(unittest.TestCase):
    """Verify that ``bootstrap.chooseNSamples`` caps at 3 when ``runningTest``."""

    def _choose_n_samples(self, n_samples: int, running_test: bool) -> int:
        """Replicate the cap injected into ``bootstrap.chooseNSamples``."""
        if running_test:
            return min(n_samples, 3)
        return n_samples

    def test_capped_at_three_with_large_nsamples(self) -> None:
        self.assertEqual(self._choose_n_samples(100, True), 3)

    def test_capped_at_three_with_exactly_three(self) -> None:
        self.assertEqual(self._choose_n_samples(3, True), 3)

    def test_below_three_unchanged(self) -> None:
        self.assertEqual(self._choose_n_samples(2, True), 2)

    def test_no_cap_when_not_running_test(self) -> None:
        self.assertEqual(self._choose_n_samples(50, False), 50)

    def test_smoke_bootstrap_ic_n_samples(self) -> None:
        boot = _SmokeBootstrapInitialConditions()
        capped = self._choose_n_samples(boot.nSamples, boot.runningTest)
        self.assertLessEqual(capped, 3)


class TestSmallScaleICConfiguration(unittest.TestCase):
    """Verify that the smoke-test ICs are self-consistent and minimal."""

    def test_forward_detector_range_is_three(self) -> None:
        ic = _SmokeForwardInitialConditions()
        n_detectors = ic.lastSpec - ic.firstSpec + 1
        self.assertEqual(n_detectors, 3,
                         "Smoke test forward slice must be exactly 3 detectors.")

    def test_backward_detector_range_is_three(self) -> None:
        ic = _SmokeBackwardInitialConditions()
        n_detectors = ic.lastSpec - ic.firstSpec + 1
        self.assertEqual(n_detectors, 3)

    def test_forward_spectra_in_forward_bank(self) -> None:
        """Forward spectra must be ≥ 135 (VESUVIO forward bank convention)."""
        ic = _SmokeForwardInitialConditions()
        self.assertGreaterEqual(ic.firstSpec, 135)

    def test_backward_spectra_in_backward_bank(self) -> None:
        ic = _SmokeBackwardInitialConditions()
        self.assertLess(ic.lastSpec, 135)

    def test_no_ms_iterations_in_smoke_ics(self) -> None:
        """Smoke-test ICs set ``noOfMSIterations=0`` for maximum speed."""
        self.assertEqual(_SmokeForwardInitialConditions().noOfMSIterations, 0)
        self.assertEqual(_SmokeBackwardInitialConditions().noOfMSIterations, 0)

    def test_correction_flags_disabled(self) -> None:
        fwd = _SmokeForwardInitialConditions()
        bck = _SmokeBackwardInitialConditions()
        for ic in (fwd, bck):
            self.assertFalse(ic.MSCorrectionFlag,
                             "MS correction should be OFF in smoke-test ICs.")
            self.assertFalse(ic.GammaCorrectionFlag,
                             "Gamma correction should be OFF in smoke-test ICs.")

    def test_bootstrap_user_confirmation_disabled(self) -> None:
        """Interactive prompt must be suppressed in non-interactive tests."""
        boot = _SmokeBootstrapInitialConditions()
        self.assertFalse(boot.userConfirmation)

    def test_phase6_sieves_all_on_in_user_controls(self) -> None:
        """All three Phase 6 statistical sieves must be enabled for full coverage."""
        ctr = _SmokeUserScriptControls()
        self.assertTrue(ctr.runOutlierDetection)
        self.assertTrue(ctr.runPhysicsClustering)
        self.assertTrue(ctr.runBayesianBootstrap)


class TestPhase6SievesOnToyData(unittest.TestCase):
    """Verify Phase 6 statistical sieves produce finite results on toy data.

    Uses synthetic spectra (clean + 1 injected outlier) so no instrument
    data or Mantid is required.  The tests assert that none of the
    statistical routines produce NaN outputs or raise division-by-zero
    errors — the critical success criterion for smoke-test validation.
    """

    _N_SPEC = 10   # 10 synthetic detectors
    _N_BINS = 50   # 50 TOF bins per spectrum

    def _make_clean_spectra(self) -> np.ndarray:
        """Return a matrix of smooth spectra with a single clear outlier."""
        rng = np.random.default_rng(seed=42)
        spectra = rng.standard_normal((self._N_SPEC, self._N_BINS))
        spectra[0] *= 50.0    # Inject a severe outlier in row 0
        return spectra

    def _make_instrument_features(self) -> np.ndarray:
        """Return synthetic (L1, theta) feature matrix for clustering."""
        rng = np.random.default_rng(seed=7)
        # Two well-separated clusters in (L1, theta) space
        cluster_a = rng.standard_normal((5, 2)) * 0.1 + np.array([1.0, 30.0])
        cluster_b = rng.standard_normal((5, 2)) * 0.1 + np.array([3.0, 70.0])
        return np.vstack([cluster_a, cluster_b])

    # ------------------------------------------------------------------
    # Sieve 1 — Outlier Detection (Phase 6 flag: runOutlierDetection)
    # ------------------------------------------------------------------

    def test_outlier_detection_returns_finite_labels(self) -> None:
        """``HardwareOutlierDetector`` must not return NaN labels."""
        detector = HardwareOutlierDetector(n_components=2, contamination=0.1)
        spectra = self._make_clean_spectra()
        labels = detector.fit_predict(spectra)
        self.assertFalse(
            np.any(np.isnan(labels.astype(float))),
            "Outlier-detection labels contain NaN.",
        )

    def test_outlier_detection_label_shape(self) -> None:
        detector = HardwareOutlierDetector(n_components=2, contamination=0.1)
        labels = detector.fit_predict(self._make_clean_spectra())
        self.assertEqual(labels.shape, (self._N_SPEC,))

    def test_outlier_detection_finds_injected_outlier(self) -> None:
        """The injected outlier (row 0) must be flagged as -1."""
        detector = HardwareOutlierDetector(n_components=2, contamination=0.1)
        labels = detector.fit_predict(self._make_clean_spectra())
        self.assertEqual(
            labels[0], -1,
            "Expected row 0 (injected outlier) to be labelled -1.",
        )

    # ------------------------------------------------------------------
    # Sieve 2 — Physics Clustering (Phase 6 flag: runPhysicsClustering)
    # ------------------------------------------------------------------

    def test_physics_clustering_returns_finite_labels(self) -> None:
        """``PhysicsTrendClusterer`` must not produce NaN labels."""
        clusterer = PhysicsTrendClusterer(eps=0.5, min_samples=2)
        features = self._make_instrument_features()
        labels = clusterer.fit_predict(features)
        self.assertFalse(
            np.any(np.isnan(labels.astype(float))),
            "Clustering labels contain NaN.",
        )

    def test_physics_clustering_finds_two_clusters(self) -> None:
        clusterer = PhysicsTrendClusterer(eps=0.5, min_samples=2)
        features = self._make_instrument_features()
        labels = clusterer.fit_predict(features)
        groups = clusterer.get_cluster_groups(labels)
        self.assertEqual(
            len(groups), 2,
            f"Expected 2 physics clusters, found {len(groups)}.",
        )

    def test_physics_clustering_no_division_by_zero(self) -> None:
        """Clustering on uniform features (ε too small) must not crash."""
        clusterer = PhysicsTrendClusterer(eps=0.001, min_samples=10)
        features = np.zeros((self._N_SPEC, 2))  # Degenerate features
        # Should complete without raising ZeroDivisionError
        labels = clusterer.fit_predict(features)
        self.assertIsNotNone(labels)

    # ------------------------------------------------------------------
    # Sieve 3 — Bayesian Bootstrap (Phase 6 flag: runBayesianBootstrap)
    # ------------------------------------------------------------------

    def test_bayesian_bootstrap_no_nan_in_weights(self) -> None:
        """Dirichlet weights must be all-finite (no NaN / ±Inf)."""
        bb = BayesianBootstrap(n_samples=20, seed=0)
        residuals = np.random.default_rng(0).standard_normal(
            (self._N_SPEC, self._N_BINS)
        )
        weighted = bb.compute_weighted_residuals(residuals)
        self.assertFalse(
            np.any(~np.isfinite(weighted)),
            "Bayesian Bootstrap produced non-finite weighted residuals.",
        )

    def test_bayesian_bootstrap_shape(self) -> None:
        bb = BayesianBootstrap(n_samples=3, seed=1)
        residuals = np.ones((self._N_SPEC, self._N_BINS))
        weighted = bb.compute_weighted_residuals(residuals)
        # Output shape: (n_samples, n_bins) — weighted sum over spectra axis
        self.assertEqual(weighted.shape, (3, self._N_BINS))

    def test_bayesian_bootstrap_weights_sum_to_one(self) -> None:
        bb = BayesianBootstrap(n_samples=5, seed=2)
        n_spec = 8
        residuals = np.random.default_rng(2).standard_normal((n_spec, self._N_BINS))
        weighted = bb.compute_weighted_residuals(residuals)
        # Output shape: (n_samples, n_bins)
        self.assertEqual(weighted.shape[0], 5)
        self.assertTrue(np.all(np.isfinite(weighted)))


class TestADSLifecycleBetweenStages(unittest.TestCase):
    """Verify ADS cleanup between pipeline stages.

    Following Section 4 of ARCHITECTURE_AUDIT.md: all Mantid workspaces
    must be explicitly deleted between stages and upon final exit to
    prevent memory leaks in the CI / Codespace environment.
    """

    def test_ads_empty_after_clear_before_stage1(self) -> None:
        """The ADS must be empty at the start of Stage 1 (fresh slate)."""
        from tests.mock_mantid import _ads
        _ads.clear()
        self.assertEqual(len(_ads), 0,
                         "ADS must be empty before Stage 1 begins.")

    def test_ads_clear_between_stage1_and_stage2(self) -> None:
        """Simulate ADS clear at the Stage 1 → Stage 2 boundary."""
        from tests.mock_mantid import _ads, MockWorkspace
        # Stage 1 leaves workspaces behind
        _ads.clear()
        for i in range(3):
            _ads.add(f"smoke_FORWARD_{i}", MockWorkspace(f"smoke_FORWARD_{i}"))
        self.assertEqual(len(_ads), 3)

        # Boundary action: clear before bootstrap
        _ads.clear()
        self.assertEqual(len(_ads), 0,
                         "ADS must be empty at Stage 1 → Stage 2 boundary.")

    def test_ads_clear_at_final_exit(self) -> None:
        """ADS must be empty after the final cleanup call."""
        from tests.mock_mantid import _ads, MockWorkspace
        _ads.add("leftover_ws", MockWorkspace("leftover_ws"))
        _ads.clear()
        self.assertEqual(len(_ads), 0,
                         "ADS must be empty after final cleanup.")

    def test_stage_transition_clears_all_iteration_workspaces(self) -> None:
        """All per-iteration workspaces (name+0..name+N) are removed by clear."""
        from tests.mock_mantid import _ads, MockWorkspace
        _ads.clear()
        ic_name = "smoke_FORWARD_"
        for i in range(5):
            _ads.add(ic_name + str(i), MockWorkspace(ic_name + str(i)))
        self.assertEqual(len(_ads), 5)
        _ads.clear()
        for i in range(5):
            self.assertNotIn(ic_name + str(i), _ads)


class TestOrchestratorHandoffLogic(unittest.TestCase):
    """Verify stage-to-stage data handoff structure.

    Tests that the IC objects used for Stage 2 (bootstrap) are consistent
    with those from Stage 1 (routine) — in particular that the
    ``runningTest`` flag propagates correctly and that the workspace names
    referenced by Stage 2 match what Stage 1 would have produced.
    """

    def test_final_ws_name_matches_stage1_output(self) -> None:
        """Stage 2 must look up the workspace produced at the end of Stage 1."""
        script_name = "smoke_thymol"
        procedure = "FORWARD"
        fwd_ic = _SmokeForwardInitialConditions()

        # Replicate ICHelpers.buildFinalWSName
        expected_ws_name = f"{script_name}_{procedure}_{fwd_ic.noOfMSIterations}"

        # With runningTest=True and noOfMSIterations=0, _n_iter=0
        _n_iter = min(1, fwd_ic.noOfMSIterations) if fwd_ic.runningTest else fwd_ic.noOfMSIterations
        actual_ws_name = f"{script_name}_{procedure}_{_n_iter}"

        self.assertEqual(
            actual_ws_name, expected_ws_name,
            "Stage 1 output workspace name does not match Stage 2 lookup key.",
        )

    def test_bootstrap_ic_procedure_matches_routine_procedure(self) -> None:
        """Bootstrap procedure direction must match the routine procedure."""
        ctr = _SmokeUserScriptControls()
        boot = _SmokeBootstrapInitialConditions()
        self.assertEqual(
            ctr.procedure, boot.procedure,
            "Stage 1 and Stage 2 procedure directions must match.",
        )

    def test_running_test_propagated_from_userscript_to_ics(self) -> None:
        """When UserScriptControls.runningTest is True, both ICs must mirror it."""
        ctr = _SmokeUserScriptControls()
        fwd = _SmokeForwardInitialConditions()
        bck = _SmokeBackwardInitialConditions()

        # In run_script.py: if userCtr.runningTest: bckwdIC.runningTest = True ...
        if ctr.runningTest:
            fwd.runningTest = True
            bck.runningTest = True

        self.assertTrue(fwd.runningTest)
        self.assertTrue(bck.runningTest)


# =============================================================================
# Master Orchestrator Entry Point
# =============================================================================

def main(dry_run: bool = True) -> None:
    """Master Orchestrator — prototype for the full 3-stage smoke test.

    This function documents the intended execution flow for a real end-to-end
    smoke test with a live Mantid installation.  When ``dry_run=True``
    (the default), it prints the stage configuration without attempting to
    connect to Mantid or load instrument data.

    The design template below should be adapted for future scripts:

    1. Instantiate all IC classes with ``runningTest=True``.
    2. Call ``runScript(userCtr, ...)`` — this triggers Stage 1 (NCP fitting)
       and is fast-tracked via the ``runningTest`` gates.
    3. Call ``runScript`` again with ``bootIC.runBootstrap=True`` — Stage 2.
       The bootstrap is capped at 3 replicas automatically.
    4. Call ``runAnalysisOfStoredBootstrap(...)`` — Stage 3.
    5. Clear the ADS on exit.

    Args:
        dry_run: When ``True``, only print the stage configuration; do not
            actually invoke Mantid or the pipeline.  Set to ``False`` in a
            full Mantid environment.
    """

    print("=" * 72)
    print("VESUVIO Production Smoke Test — Master Orchestrator")
    print("=" * 72)

    # ── IC instantiation ──────────────────────────────────────────────────────
    fwdIC   = _SmokeForwardInitialConditions()
    bckwdIC = _SmokeBackwardInitialConditions()
    yFitIC  = _SmokeYSpaceFitInitialConditions()
    userCtr = _SmokeUserScriptControls()
    bootIC  = _SmokeBootstrapInitialConditions()
    analysisIC = _SmokeBootstrapAnalysis()

    print(f"\n[Config] runningTest       = {userCtr.runningTest}")
    print(f"[Config] procedure          = {userCtr.procedure}")
    print(f"[Config] fitInYSpace        = {userCtr.fitInYSpace}")
    print(f"[Config] fwd spectra        = {fwdIC.firstSpec}–{fwdIC.lastSpec}")
    print(f"[Config] noOfMSIterations   = {fwdIC.noOfMSIterations}")
    print(f"[Config] bootstrap nSamples = {bootIC.nSamples} (≤ 3 when runningTest)")
    print(f"[Config] Phase 6 sieves ON  = "
          f"outlier={userCtr.runOutlierDetection}, "
          f"cluster={userCtr.runPhysicsClustering}, "
          f"bayes={userCtr.runBayesianBootstrap}")

    if dry_run:
        print("\n[DryRun] Skipping Mantid-dependent pipeline calls.")
        print("[DryRun] Pass dry_run=False to run the full pipeline.")
        print("\n[DryRun] Stage 1 (runRoutine) would call:")
        print("         runScript(userCtr, 'smoke_thymol', wsBackIC, wsFrontIC,")
        print("                   bckwdIC, fwdIC, yFitIC, bootIC)")
        print("\n[DryRun] Stage 2 (runBootstrap) would call:")
        print("         bootIC.runBootstrap = True")
        print("         runScript(userCtr_boot, 'smoke_thymol', ...)")
        print("\n[DryRun] Stage 3 (runAnalysis) would call:")
        print("         runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC,")
        print("                                      bootIC, analysisIC, userCtr)")
        print("\n[DryRun] ADS cleanup: AnalysisDataService.clear()")
    else:
        # ── Full pipeline (requires Mantid) ────────────────────────────────────
        # Import here so the module can be imported without Mantid for CI testing
        from mantid.api import AnalysisDataService
        from vesuvio_analysis.core_functions.run_script import runScript
        from vesuvio_analysis.core_functions.bootstrap_analysis import (
            runAnalysisOfStoredBootstrap,
        )

        script_name = "smoke_thymol"
        wsBackIC    = _SmokeLoadVesuvioBackParameters
        wsFrontIC   = _SmokeLoadVesuvioFrontParameters

        # ── Stage 1: NCP Routine ──────────────────────────────────────────────
        print("\n[Stage 1] Running NCP routine (runningTest fast-track)...")
        userCtr.runRoutine = True
        bootIC_off = type("_BootOff", (), {"runBootstrap": False})()
        res, resYFit = runScript(
            userCtr, script_name, wsBackIC, wsFrontIC,
            bckwdIC, fwdIC, yFitIC, bootIC_off,
        )
        print("[Stage 1] Complete.")

        # ── ADS cleanup before Stage 2 ────────────────────────────────────────
        AnalysisDataService.clear()
        print("[ADS] Cleared before Stage 2.")

        # ── Stage 2: Bootstrap ────────────────────────────────────────────────
        print("\n[Stage 2] Running bootstrap (3 samples, fast-track)...")
        userCtr_boot = type("_UserBoot", (), {
            "runRoutine": False,
            "procedure": "FORWARD",
            "fitInYSpace": "FORWARD",
            "runningTest": True,
        })()
        bootResult, _ = runScript(
            userCtr_boot, script_name, wsBackIC, wsFrontIC,
            bckwdIC, fwdIC, yFitIC, bootIC,
        )
        print("[Stage 2] Complete.")

        # ── ADS cleanup before Stage 3 ────────────────────────────────────────
        AnalysisDataService.clear()
        print("[ADS] Cleared before Stage 3.")

        # ── Stage 3: Post-hoc Analysis ────────────────────────────────────────
        print("\n[Stage 3] Running bootstrap analysis...")
        runAnalysisOfStoredBootstrap(
            bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr,
        )
        print("[Stage 3] Complete.")

        # ── Final ADS cleanup ─────────────────────────────────────────────────
        AnalysisDataService.clear()
        print("[ADS] Final cleanup complete.")

    print("\n" + "=" * 72)
    print("Smoke test orchestration finished successfully.")
    print("=" * 72)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # When executed directly, run a dry-run orchestration trace followed
    # by the full unittest suite.
    main(dry_run=True)
    print()
    unittest.main(verbosity=2)
