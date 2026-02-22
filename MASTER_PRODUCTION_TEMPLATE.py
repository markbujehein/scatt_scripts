"""VESUVIO Analysis — Master Production Template.

This file is the canonical starting point for every new VESUVIO experiment
analysis.  All bank settings, naming conventions, and physics constants are
transferred from the Thymol 10 K single-Gaussian reference experiment
(``thymol_10K_Gauss1D.py``), with the structural orchestration pattern
inherited from ``tests/production_smoke_test.py``.

=============================================================================
Usage
=============================================================================

Run without human intervention::

    pixi run python MASTER_PRODUCTION_TEMPLATE.py

The script is fully non-interactive.  All configuration is encoded in the
IC classes below.  No ``input()`` prompts are present anywhere in the
pipeline when the script is used as-is.

=============================================================================
Design Principles
=============================================================================

* **Non-interactive** — zero ``input()`` calls.  ``userConfirmation = False``
  on ``BootstrapInitialConditions`` disables the bootstrap runtime estimator
  prompt.  The H-ratio estimation procedure runs automatically when needed.

* **Explicit nSamples on BackwardInitialConditions** — required by
  ``_autoSelectPreliminaryIterations()`` if the preliminary H-ratio
  estimation is triggered (i.e. when ``HToMassIdxRatio`` is ``None`` or
  ``1.0``).

* **Single configuration block** — all physics constants, detector ranges,
  and control flags live in the IC classes at the top of this file.  Change
  one number and the full pipeline adjusts automatically.

* **ADS lifecycle** — Mantid workspaces are cleared explicitly before each
  stage following Section 4 of ``ARCHITECTURE_AUDIT.md``.

* **StreamManager-ready** — optional ``stream_manager`` parameter is
  forwarded to ``iterativeFitForDataReduction`` and
  ``fitInYSpaceProcedure`` for L0–L3 data capture.

=============================================================================
Adapting for a new experiment
=============================================================================

1. Copy this file and rename it (e.g. ``my_sample_300K.py``).
2. Update the run numbers in ``LoadVesuvioBackParameters`` and
   ``LoadVesuvioFrontParameters``.
3. Adjust masses, ``initPars``, ``bounds``, and ``noOfMSIterations`` in
   ``BackwardInitialConditions`` and ``ForwardInitialConditions``.
4. Set ``HToMassIdxRatio`` on ``BackwardInitialConditions``:
   * Known value  → enter it directly (e.g. ``20.1``).
   * Unknown / H absent → set to ``None`` or ``1.0`` and the pipeline will
     estimate it automatically using ``nSamples`` preliminary iterations.
5. Set ``nSamples`` on ``BootstrapInitialConditions`` to the desired number
   of bootstrap replicas.
6. Run: ``pixi run python my_sample_300K.py``
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from vesuvio_analysis.core_functions.bootstrap_analysis import (
    runAnalysisOfStoredBootstrap,
)
from vesuvio_analysis.core_functions.run_script import runScript

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
scriptName = Path(__file__).stem   # e.g. "MASTER_PRODUCTION_TEMPLATE"
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


# =============================================================================
# Stage 0 — Data Loading Parameters
# =============================================================================

class LoadVesuvioBackParameters:
    """Backward-scattering Vesuvio data loading parameters (Thymol 10 K).

    Attributes:
        runs: ISIS run numbers for the backward-scattering measurement.
        empty_runs: ISIS run numbers for the empty-can background subtraction.
        spectra: Detector spectra range covering the backward bank (3–134).
        mode: Vesuvio difference mode for the backward bank.
        ipfile: Instrument parameter file used by LoadVesuvio (Mantid).
    """

    runs = "50888-50900"
    empty_runs = "51382-51415"
    spectra = "3-134"
    mode = "DoubleDifference"
    ipfile = ipFilesPath / "IP_2024_10_31_E1_variable_unmasked.par"


class LoadVesuvioFrontParameters:
    """Forward-scattering Vesuvio data loading parameters (Thymol 10 K).

    Attributes:
        runs: ISIS run numbers for the forward-scattering measurement.
        empty_runs: ISIS run numbers for the empty-can background subtraction.
        spectra: Detector spectra range covering the forward bank (135–182).
        mode: Vesuvio difference mode for the forward bank.
        ipfile: Instrument parameter file used by LoadVesuvio (Mantid).
    """

    runs = "50888-50900"
    empty_runs = "51382-51415"
    spectra = "135-182"
    mode = "SingleDifference"
    ipfile = ipFilesPath / "IP_2024_10_31_E1_variable_unmasked.par"


# =============================================================================
# Stage 1 — NCP Fitting: Shared Geometry
# =============================================================================

class GeneralInitialConditions:
    """Sample slab geometry shared by backward and forward analyses.

    Attributes:
        vertical_width: Vertical extent of the sample slab (metres).
        horizontal_width: Horizontal extent of the sample slab (metres).
        thickness: Slab thickness along the beam direction (metres).
    """

    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001


# =============================================================================
# Stage 1 — NCP Fitting: Backward Bank (C10H14O, spectra 3–134)
# =============================================================================

class BackwardInitialConditions(GeneralInitialConditions):
    """Backward-scattering NCP configuration for Thymol at 10 K.

    Inherits slab geometry from ``GeneralInitialConditions``.

    The sample is Thymol (C10H14O).  In the backward bank the three fitted
    masses are Carbon (12 u), Oxygen (16 u), and the aluminium can (27 u).

    Hydrogen is resolved in the forward bank.  ``HToMassIdxRatio`` encodes
    the H peak intensity relative to the Oxygen peak.  When set to ``None``
    or ``1.0``, the pipeline runs an automatic preliminary estimation using
    ``nSamples`` iterations before the main fit.

    Attributes:
        HToMassIdxRatio: H / reference-mass intensity ratio.  Known value
            for Thymol 10 K is 20.1.  Set to ``None`` or ``1.0`` to trigger
            automatic estimation (requires ``nSamples`` to be set).
        massIdx: Index of the reference mass (Oxygen, index 1 in this bank).
        nSamples: Number of preliminary iterations for H-ratio estimation.
            Only used when automatic estimation is triggered.  Must not be
            ``None`` in that case (raises ``ValueError`` otherwise).
        masses: Nuclear masses in a.m.u., shape ``(n_masses,)``.
        initPars: Flat array ``[intensity, width, centre] * n_masses``.
        bounds: Optimiser bounds per parameter, shape ``(3*n_masses, 2)``.
            Use ``np.nan`` for an unbounded limit.
        noOfMSIterations: MS correction iterations (0 = disabled).
        firstSpec / lastSpec: Inclusive detector range for the backward bank.
        maskedSpecAllNo: Absolute spectrum numbers excluded from the fit.
        MSCorrectionFlag: Enable multiple-scattering correction.
        GammaCorrectionFlag: Enable gamma-background correction.
        tofBinning: Rebin string ``'start,step,end'`` in µs.
        transmission_guess: Estimated neutron transmission of the sample.
        multiple_scattering_order: Max MS order for VesuvioCalculateMS.
        number_of_events: Monte Carlo events for the MS correction.
    """

    subEmptyFromRaw = True
    scaleEmpty = 1
    scaleRaw = 1

    # C10H14O: H/O ratio ≈ 20.1 (14 H atoms, σ_H ≈ 82 barn vs σ_O ≈ 4.2 barn)
    HToMassIdxRatio = 20.1   # Set to None or 1.0 to trigger auto-estimation
    massIdx = 1              # Reference mass index: Oxygen (index 1 in bckwd masses)

    # Preliminary H-ratio estimation iterations.
    # Required (must not be None) when HToMassIdxRatio is None or 1.0.
    nSamples: int = 4

    masses = np.array([12, 16, 27])   # Carbon, Oxygen, Aluminium

    initPars = np.array([
        # Intensities, NCP widths, NCP centres
        1,    4.9,  0.0,   # Carbon
        1,    4.9,  0.0,   # Oxygen
        1,    9.27, 0.0,   # Aluminium
    ])
    bounds = np.array([
        [0, np.nan], [1.573, 12],  [-10, 10],   # Carbon
        [0, np.nan], [1.816, 12],  [-10, 10],   # Oxygen
        [0, np.nan], [9.27, 9.27], [-10, 10],   # Aluminium
    ])
    constraints = []

    noOfMSIterations = 2
    firstSpec = 3
    lastSpec = 134

    maskedSpecAllNo = np.array([18, 34, 45, 52, 62])

    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    tofBinning = "110,1.,500"
    maskTOFRange = None

    transmission_guess = 0.6
    multiple_scattering_order, number_of_events = 2, 1.0e5

    runHistData = False


# =============================================================================
# Stage 1 — NCP Fitting: Forward Bank (C10H14O, spectra 135–182)
# =============================================================================

class ForwardInitialConditions(GeneralInitialConditions):
    """Forward-scattering NCP configuration for Thymol at 10 K.

    Inherits slab geometry from ``GeneralInitialConditions``.

    The four fitted masses in the forward bank are:
        index 0 → Hydrogen  (1.0079 u) — primary J(y) interest
        index 1 → Carbon    (12 u)
        index 2 → Oxygen    (16 u)
        index 3 → Aluminium (27 u)

    Attributes:
        masses: Nuclear masses in a.m.u., shape ``(n_masses,)``.
        initPars: Flat array ``[intensity, width, centre] * n_masses``.
        bounds: Optimiser bounds per parameter, shape ``(3*n_masses, 2)``.
        noOfMSIterations: MS correction iterations.
        firstSpec / lastSpec: Inclusive detector range for the forward bank.
        maskedSpecAllNo: Absolute spectrum numbers excluded from the fit.
        MSCorrectionFlag: Enable multiple-scattering correction.
        GammaCorrectionFlag: Enable gamma-background correction.
        tofBinning: Rebin string ``'start,step,end'`` in µs.
        transmission_guess: Estimated neutron transmission of the sample.
        multiple_scattering_order: Max MS order for VesuvioCalculateMS.
        number_of_events: Monte Carlo events for the MS correction.
    """

    subEmptyFromRaw = False
    scaleEmpty = 1
    scaleRaw = 1

    masses = np.array([1.0079, 12, 16, 27])   # H, C, O, Al

    initPars = np.array([
        # Intensities, NCP widths, NCP centres
        1,    5,    0.0,   # Hydrogen
        1,    4.9,  0.0,   # Carbon
        1,    4.9,  0.0,   # Oxygen
        1,    9.27, 0.0,   # Aluminium
    ])
    bounds = np.array([
        [0, np.nan], [4,     6     ], [-1,  1  ],   # Hydrogen
        [0, np.nan], [1.573, 12    ], [-10, 10 ],   # Carbon
        [0, np.nan], [1.816, 12    ], [-10, 10 ],   # Oxygen
        [0, np.nan], [9.27,  9.27  ], [-10, 10 ],   # Aluminium
    ])
    constraints = []

    noOfMSIterations = 2
    firstSpec = 135
    lastSpec = 182

    maskedSpecAllNo = np.array([173, 174, 178])

    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    tofBinning = "110,1,430"
    maskTOFRange = None

    transmission_guess = 0.87
    multiple_scattering_order, number_of_events = 2, 1.0e5

    runHistData = True


# =============================================================================
# Stage 1 — Y-Space Fit (hydrogen J(y) analysis)
# =============================================================================

class YSpaceFitInitialConditions:
    """Y-space fitting configuration for the Thymol 10 K hydrogen J(y) analysis.

    Controls the fit of J(y) after the NCP pipeline has isolated the
    hydrogen recoil peak and converted to y-space via
    ``J(y) = (M / ℏq) * (E - E_recoil)``.

    Attributes:
        showPlots: Generate diagnostic plots after the fit.
        symmetrisationFlag: Symmetrise J(y) about y = 0 before fitting.
        rebinParametersForYSpaceFit: Rebin string in Å⁻¹ (must be symmetric
            about zero).
        fitModel: Model function for J(y).  Options: ``'SINGLE_GAUSSIAN'``,
            ``'GC_C4'``, ``'GC_C6'``, ``'GC_C4_C6'``, ``'DOUBLE_WELL'``,
            ``'ANSIO_GAUSSIAN'``.
        runMinos: Run MINOS for asymmetric confidence intervals (Minuit).
        globalFit: Simultaneous global fit across detector groups.
        nGlobalFitGroups: Number of detector groups (or ``'ALL'``).
        maskTypeProcedure: Masked-bin handling.  Options: ``'NCP'``,
            ``'NAN'``, ``None``.
    """

    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"
    fitModel = "SINGLE_GAUSSIAN"
    runMinos = True
    globalFit = True
    nGlobalFitGroups = 3
    maskTypeProcedure = "NAN"


# =============================================================================
# Top-Level Execution Controls
# =============================================================================

class UserScriptControls:
    """Top-level flags controlling which pipeline stages are active.

    Attributes:
        runRoutine: Execute the full NCP + y-space fitting pipeline.
        procedure: Scattering direction(s) for the NCP routine.
            Options: ``None``, ``'BACKWARD'``, ``'FORWARD'``, ``'JOINT'``.
        fitInYSpace: Which final workspace is passed to the y-space fit.
            Options: ``None``, ``'BACKWARD'``, ``'FORWARD'``, ``'JOINT'``.
        runningTest: Fast-track flag; set to ``True`` only for CI smoke tests.
        runOutlierDetection: PCA hardware-outlier detection (Phase 6).
        runPhysicsClustering: DBSCAN physics-trend clustering (Phase 6).
        runBayesianBootstrap: Bayesian Bootstrap resampling (Phase 6).
        verbose: Print pipeline headers, footers, and optimizer agreement.
    """

    runRoutine = True

    procedure = "JOINT"      # Options: None, "BACKWARD", "FORWARD", "JOINT"
    fitInYSpace = "FORWARD"  # Options: None, "BACKWARD", "FORWARD", "JOINT"

    runningTest: bool = False

    runOutlierDetection: bool = False
    runPhysicsClustering: bool = False
    runBayesianBootstrap: bool = False

    verbose: bool = True


# =============================================================================
# Stage 2 — Bootstrap / Jackknife Resampling
# =============================================================================

class BootstrapInitialConditions:
    """Bootstrap / jackknife resampling configuration.

    Attributes:
        runBootstrap: Redirect execution to the bootstrap procedure.
            Mutually exclusive with ``UserScriptControls.runRoutine``.
        procedure: Scattering direction for each bootstrap replica.
        fitInYSpace: Workspace fitted in y-space per replica.
        bootstrapType: Resampling strategy.
            Options: ``'JACKKNIFE'``, ``'BOOT_RESIDUALS'``,
            ``'BOOT_GAUSS_ERRS'``.
        nSamples: Number of bootstrap replicas.  Must not be ``None``
            (raises ``ValueError`` if missing).
        skipMSIterations: Each replica skips MS / GC corrections.
        userConfirmation: Prompt before starting the bootstrap loop.
            Set to ``False`` for non-interactive production runs.
        runningTest: Suppress side-effects (file I/O, plots) in tests.
    """

    runBootstrap = False

    procedure = "BACKWARD"
    fitInYSpace = None

    bootstrapType = "BOOT_RESIDUALS"
    nSamples = 650          # Must not be None — validated by bootstrap.chooseNSamples()
    skipMSIterations = False
    userConfirmation = False  # Non-interactive: no runtime-estimator prompt
    runningTest = False


# =============================================================================
# Stage 3 — Post-Hoc Bootstrap Analysis
# =============================================================================

class BootstrapAnalysis:
    """Post-hoc analysis configuration for stored bootstrap results.

    Run independently after ``runScript()`` returns by calling
    ``runAnalysisOfStoredBootstrap()``.

    Attributes:
        runAnalysis: Execute analysis of previously stored bootstrap files.
        filterAvg: Apply the original outlier-rejection filter to widths
            and intensities before computing averages.
        plotRawWidthsIntensities: Plot all per-replica distributions.
        plotMeanWidthsIntensities: Plot mean widths and intensities with
            bootstrap uncertainties.
        plotMeansEvolution: Plot evolution of mean values vs sample index.
        plot2DHists: Plot 2-D joint histograms of widths and intensities.
        plotYFitHists: Plot histograms of y-space fit parameters.
    """

    runAnalysis = False

    filterAvg = True

    plotRawWidthsIntensities = True
    plotMeanWidthsIntensities = True
    plotMeansEvolution = True
    plot2DHists = True
    plotYFitHists = True


# =============================================================================
# Entry Point — Non-Interactive Master Orchestrator
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()

    wsBackIC  = LoadVesuvioBackParameters
    wsFrontIC = LoadVesuvioFrontParameters
    bckwdIC   = BackwardInitialConditions
    fwdIC     = ForwardInitialConditions
    yFitIC    = YSpaceFitInitialConditions
    bootIC    = BootstrapInitialConditions
    userCtr   = UserScriptControls

    # Stage 1 + optional Stage 2 (controlled by UserScriptControls /
    # BootstrapInitialConditions flags above).
    runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

    # Stage 3 — post-hoc analysis of stored bootstrap results.
    analysisIC = BootstrapAnalysis
    runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr)

    end_time = time.time()
    _elapsed = end_time - start_time
    _m, _s = divmod(_elapsed, 60)
    print(f"\nTotal Running Time: {_elapsed:.2f} seconds ({int(_m)}m {int(_s)}s)")
