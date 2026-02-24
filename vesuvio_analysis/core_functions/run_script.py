from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mantid.api import mtd
from mantid.simpleapi import (
    ConvertToYSpace,
    DeleteWorkspaces,
    Rebin,
    SumSpectra,
)


def build_metadata_map(
    ic: Any,
) -> Dict[int, Dict[str, Any]]:
    """Build a metadata dictionary mapping array indices to physical detector info.

    Uses the instrument parameter file (columns: spec, det, angle, T0, L0, L1)
    to create a lookup from zero-based array index to physical identifiers.

    Args:
        ic: Completed initial-conditions object with ``InstrParsPath``,
            ``firstSpec``, and ``lastSpec``.

    Returns:
        ``{array_index: {'spec_no': int, 'angle': float, 'detector_id': int}}``
    """
    from vesuvio_analysis.core_functions.analysis_functions import (
        loadInstrParsFileIntoArray,
    )

    try:
        instrPars = loadInstrParsFileIntoArray(
            ic.InstrParsPath, ic.firstSpec, ic.lastSpec,
        )
    except Exception as exc:
        warnings.warn(f"build_metadata_map: failed to load instrument parameters: {exc}")
        return {}

    metadata: Dict[int, Dict[str, Any]] = {}
    for i in range(len(instrPars)):
        metadata[i] = {
            "spec_no": int(instrPars[i, 0]),
            "angle": float(instrPars[i, 2]),
            "detector_id": int(instrPars[i, 1]),
        }
    return metadata

from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.core_functions.correction_plots import dispatch_correction_plots
from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.ICHelpers import (
    buildFinalWSName,
    completeBootIC,
    completeICFromInputs,
    completeYFitIC,
)
from vesuvio_analysis.core_functions.log_manager import RunLogger
from vesuvio_analysis.core_functions.procedures import (
    createTableWSHRatios,
    isHPresent,
    runIndependentIterativeProcedure,
    runJointBackAndForwardProcedure,
    runPreProcToEstHRatio,
)

logger = logging.getLogger(__name__)

_SEP_DOUBLE = "=" * 60
_SEP_SINGLE = "-" * 60


def runScript(
    userCtr: Any,
    scriptName: str,
    wsBackIC: Any,
    wsFrontIC: Any,
    bckwdIC: Any,
    fwdIC: Any,
    yFitIC: Any,
    bootIC: Any,
) -> Optional[Tuple[Any, Any]]:
    """Dispatch the VESUVIO analysis pipeline according to user flags.

    This is the top-level orchestrator called from every submission script.
    It completes the initial-condition objects, validates inputs, and
    branches into one of three paths:

    * **Bootstrap** (``bootIC.runBootstrap``): delegates to
      ``bootstrap.runBootstrap`` and returns immediately.
    * **Routine** (``userCtr.runRoutine``): runs the iterative NCP
      fitting procedure (``BACKWARD``, ``FORWARD``, or ``JOINT``),
      followed by fitting in y-space via
      ``fit_in_yspace.fitInYSpaceProcedure``.
    * **No-op**: if neither flag is set, nothing is executed.

    ``runRoutine`` and ``runBootstrap`` are mutually exclusive; an
    ``AssertionError`` is raised if both are ``True``.

    Expects the Mantid AnalysisDataService (``mtd``) to be available.
    When ``runRoutine`` is ``True`` and the final workspace already
    exists in ``mtd``, the NCP fitting is skipped and only the
    y-space fit is performed.

    Args:
        userCtr: ``UserScriptControls`` class with ``runRoutine``,
            ``procedure``, and ``fitInYSpace`` flags.
        scriptName: Base name of the submission script (without
            ``.py``), used to construct workspace and file names.
        wsBackIC: ``LoadVesuvioBackParameters`` class with run numbers,
            spectra, mode, and ip-file for backward scattering.
        wsFrontIC: ``LoadVesuvioFrontParameters`` class with the same
            fields for forward scattering.
        bckwdIC: ``BackwardInitialConditions`` class with masses,
            fit parameters, bounds, and correction flags.
        fwdIC: ``ForwardInitialConditions`` class with masses,
            fit parameters, bounds, and correction flags.
        yFitIC: ``YSpaceFitInitialConditions`` class controlling
            rebinning, symmetrisation, model selection, and Minos.
        bootIC: ``BootstrapInitialConditions`` class controlling
            the resampling procedure.

    Returns:
        A 2-tuple ``(res, resYFit)`` where *res* is the result object
        from the iterative NCP fit (or ``None``) and *resYFit* is the
        ``ResultsYFitObject`` from the y-space fit (or ``None``).
        When bootstrap is active, returns ``(bootResult, None)``.

    Raises:
        AssertionError: If both ``runRoutine`` and ``runBootstrap`` are
            ``True``, or if input flags contain invalid values.
    """

    # --- Mantid log suppression: silence repetitive Notice messages ---
    try:
        from mantid.kernel import ConfigService
        ConfigService.setLogLevel(3)  # 3 = Warning; keeps errors visible
    except Exception:
        pass

    _verbose = getattr(userCtr, "verbose", True)

    # Set extra attributes from user attributes
    completeICFromInputs(fwdIC, scriptName, wsFrontIC)
    completeICFromInputs(bckwdIC, scriptName, wsBackIC)

    # Propagate the fast-track smoke-test flag to all IC objects so that
    # analysis_functions and bootstrap logic can cap iterations, limit samples,
    # and adjust tolerances appropriately.
    # Must be done before completeBootIC so directory/file naming uses the correct mode.
    if getattr(userCtr, "runningTest", False):
        bckwdIC.runningTest = True
        fwdIC.runningTest = True
        bootIC.runningTest = True

    completeBootIC(bootIC, bckwdIC, fwdIC, yFitIC)
    completeYFitIC(yFitIC, scriptName)

    checkInputs(userCtr)
    checkInputs(bootIC)
    if userCtr.runRoutine and bootIC.runBootstrap:
        raise ValueError("Main routine and bootstrap both set to run!")

    # --- Configuration-drift guard: bootstrap ↔ main procedure alignment ---
    if bootIC.runBootstrap:
        # Auto-default: if the user did not explicitly set the bootstrap
        # procedure, inherit it from the primary fit so the bootstrap always
        # operates on the same data domain.
        if getattr(bootIC, "procedure", None) is None:
            bootIC.procedure = userCtr.procedure

        _boot_proc = getattr(bootIC, "procedure", None)
        _main_proc = getattr(userCtr, "procedure", None)
        _single_bank_modes = ("FORWARD", "BACKWARD")
        if _boot_proc == "JOINT" and _main_proc in _single_bank_modes:
            raise ValueError(
                f"Bootstrap procedure 'JOINT' is incompatible with main "
                f"procedure '{_main_proc}'.  A JOINT bootstrap requires both "
                "detector banks to have been fitted (including the "
                "HToMassIdxRatio estimation for the backward bank), but the "
                f"primary fit only ran the '{_main_proc}' bank.\n"
                "Fix: set BootstrapInitialConditions.procedure = "
                f'"{_main_proc}" to match the primary fit, '
                'or set UserScriptControls.procedure = "JOINT" '
                "to run both banks during the primary fit."
            )

    # --- Logging setup ---
    direction = getattr(userCtr, "procedure", None) or "NONE"
    _log_output_dir = getattr(
        bckwdIC, "resultsSavePath",
        getattr(fwdIC, "resultsSavePath", None),
    )
    if _log_output_dir is not None:
        _log_output_dir = _log_output_dir.parent
    else:
        import tempfile
        _log_output_dir = tempfile.mkdtemp()
    run_logger = RunLogger(scriptName, direction, _log_output_dir)
    run_logger.log_environment()
    run_logger.log_ic("UserScriptControls", userCtr)
    run_logger.log_ic("LoadVesuvioBackParameters", wsBackIC)
    run_logger.log_ic("LoadVesuvioFrontParameters", wsFrontIC)
    run_logger.log_ic("BackwardInitialConditions", bckwdIC)
    run_logger.log_ic("ForwardInitialConditions", fwdIC)
    run_logger.log_ic("YSpaceFitInitialConditions", yFitIC)
    run_logger.log_ic("BootstrapInitialConditions", bootIC)
    run_logger.log_flags(
        MSCorrectionFlag_bckwd=getattr(bckwdIC, "MSCorrectionFlag", None),
        GammaCorrectionFlag_bckwd=getattr(bckwdIC, "GammaCorrectionFlag", None),
        MSCorrectionFlag_fwd=getattr(fwdIC, "MSCorrectionFlag", None),
        GammaCorrectionFlag_fwd=getattr(fwdIC, "GammaCorrectionFlag", None),
        runRoutine=getattr(userCtr, "runRoutine", None),
        runBootstrap=getattr(bootIC, "runBootstrap", None),
    )

    def runProcedure():
        proc = userCtr.procedure  # Shorthad to make it easier to read

        if proc == None:
            return

        ranPreliminary = False
        if (proc == "BACKWARD") | (proc == "JOINT"):
            h_present = isHPresent(fwdIC.masses)
            if h_present:
                # In JOINT mode the ratio is consumed from backward IC when
                # projecting backward intensities into forward initial conditions.
                # If user supplied it on forward IC, use that value as seed.
                if (bckwdIC.HToMassIdxRatio is None) and (fwdIC.HToMassIdxRatio is not None):
                    bckwdIC.HToMassIdxRatio = fwdIC.HToMassIdxRatio

                if bckwdIC.HToMassIdxRatio is None:
                    HRatios, massIdxs = runPreProcToEstHRatio(
                        bckwdIC, fwdIC, nIter=fwdIC.preliminaryNoOfIterations
                    )
                    ranPreliminary = True

                assert bckwdIC.HToMassIdxRatio is not None, (
                    "HToMassIdxRatio is required for JOINT/BACKWARD when H is present."
                )

        if proc == "BACKWARD":
            res = runIndependentIterativeProcedure(bckwdIC)
        if proc == "FORWARD":
            res = runIndependentIterativeProcedure(fwdIC)
        if proc == "JOINT":
            res = runJointBackAndForwardProcedure(bckwdIC, fwdIC)

        # If preliminary procedure ran, make TableWS with H ratios values
        if ranPreliminary:
            createTableWSHRatios(HRatios, massIdxs)
        return res

    # Names of workspaces to be fitted in y space (bank-aware)
    wsNames: List[str] = []
    ICs: List[Any] = []
    wsModes: List[str] = []
    for mode, IC in zip(["BACKWARD", "FORWARD"], [bckwdIC, fwdIC]):
        if (userCtr.fitInYSpace == mode) | (userCtr.fitInYSpace == "JOINT"):
            wsNames.append(buildFinalWSName(scriptName, mode, IC))
            ICs.append(IC)
            wsModes.append(mode)

    # If bootstrap is not None, run bootstrap procedure and finish
    if bootIC.runBootstrap:
        assert (
            (bootIC.procedure == "FORWARD")
            | (bootIC.procedure == "BACKWARD")
            | (bootIC.procedure == "JOINT")
        ), "Invalid Bootstrap procedure."
        if _verbose:
            print(f"\n{_SEP_DOUBLE}")
            print(f"  Bootstrap Procedure — {bootIC.procedure}")
            print(f"{_SEP_DOUBLE}\n")
        _t0 = time.time()
        run_logger.write()

        # Bayesian Bootstrap: fast Dirichlet-weighted resampling on NCP
        # residuals.  Runs the parent NCP fit first, then generates
        # weighted residual profiles — no iterative re-fitting.
        if getattr(bootIC, "bootstrapType", "") == "BOOT_BAYESIAN":
            boot_result = _runBayesianBootstrapProcedure(
                bckwdIC, fwdIC, bootIC, yFitIC,
            )
        else:
            boot_result = runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC)

        if _verbose:
            _elapsed = time.time() - _t0
            _m, _s = divmod(_elapsed, 60)
            print(f"\n{_SEP_SINGLE}")
            print(f"  Bootstrap complete — {_elapsed:.2f}s ({int(_m)}m {int(_s)}s)")
            print(f"{_SEP_SINGLE}\n")
        return boot_result, None

    # Default workflow for procedure + fit in y space
    if userCtr.runRoutine:
        if _verbose:
            print(f"\n{_SEP_DOUBLE}")
            print(f"  VESUVIO Routine — {direction}")
            print(f"{_SEP_DOUBLE}\n")
        _t0 = time.time()

        # Check if final ws are loaded:
        wsInMtd = [ws in mtd for ws in wsNames]  # Bool list
        if (len(wsInMtd) > 0) and all(wsInMtd):  # When wsName is empty list, loop doesn't run
            for wsName, IC, mode in zip(wsNames, ICs, wsModes):
                # bank-aware validation before performing the fit
                checkInputs(userCtr, bank=mode)
                if wsName not in mtd:
                    logging.warning(
                        f"Workspace '{wsName}' not found in Mantid Analysis Data Service. "
                        f"Skipping Y-Space fit for {mode} bank."
                    )
                    continue
                # temporarily align fitInYSpace for diagnostics/logging
                prev_fit = userCtr.fitInYSpace
                userCtr.fitInYSpace = mode
                if hasattr(yFitIC, "detectorQualityWeightsByWorkspace"):
                    yFitIC.detectorQualityWeights = yFitIC.detectorQualityWeightsByWorkspace.get(wsName)
                resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
                userCtr.fitInYSpace = prev_fit
            run_logger.write()
            if _verbose:
                _elapsed = time.time() - _t0
                _m, _s = divmod(_elapsed, 60)
                print(f"\n{_SEP_SINGLE}")
                print(f"  Analysis complete — {_elapsed:.2f}s ({int(_m)}m {int(_s)}s)")
                print(f"{_SEP_SINGLE}\n")
            return None, resYFit  # To match return below.

        # Skip interactive workspace-clear prompt when running a smoke test
        if not getattr(userCtr, "runningTest", False):
            checkUserClearWS()  # Check if user is OK with cleaning all workspaces

        res = None
        resYFit = None
        try:
            run_logger.log_timestamp("ncp_start")
            res = runProcedure()
            run_logger.log_timestamp("ncp_end")
            run_logger.log_final_results(res[1] if res is not None and len(res) >= 2 else None)

            # --- Correction Dashboard Plots ---
            _dispatchCorrectionPlots(userCtr, bckwdIC, fwdIC)

            # --- Phase 6a: Pre-fit statistical analysis ---
            # Outlier detection and DBSCAN clustering run BEFORE the
            # y-space global fit so their results can configure
            # nGlobalFitGroups and exclude broken detectors.
            run_logger.log_timestamp("phase6_prefit_start")
            _runPreFitStatistics(
                userCtr, res, bckwdIC, fwdIC, yFitIC,
            )
            run_logger.log_timestamp("phase6_prefit_end")

            run_logger.log_timestamp("yspace_start")
            for wsName, IC, mode in zip(wsNames, ICs, wsModes):
                checkInputs(userCtr, bank=mode)
                if wsName not in mtd:
                    logging.warning(
                        f"Workspace '{wsName}' not found in Mantid Analysis Data Service. "
                        f"Skipping Y-Space fit for {mode} bank."
                    )
                    continue
                prev_fit = userCtr.fitInYSpace
                userCtr.fitInYSpace = mode
                if hasattr(yFitIC, "detectorQualityWeightsByWorkspace"):
                    yFitIC.detectorQualityWeights = yFitIC.detectorQualityWeightsByWorkspace.get(wsName)
                resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
                userCtr.fitInYSpace = prev_fit
            run_logger.log_timestamp("yspace_end")

        except Exception as exc:
            run_logger.log_error(exc)
            run_logger.write()
            raise

        if _verbose:
            _elapsed = time.time() - _t0
            _m, _s = divmod(_elapsed, 60)
            print(f"\n{_SEP_SINGLE}")
            print(f"  Analysis complete — {_elapsed:.2f}s ({int(_m)}m {int(_s)}s)")
            print(f"{_SEP_SINGLE}\n")

        run_logger.write()
        return res, resYFit  # Return results used only in tests


def checkUserClearWS() -> None:
    """Prompt the user before clearing all loaded Mantid workspaces.

    If any workspaces are present in the AnalysisDataService, the user
    is asked to confirm.  A ``KeyboardInterrupt`` is raised if the user
    declines.

    Raises:
        KeyboardInterrupt: If the user does not confirm the action.
    """

    if len(mtd) != 0:
        userInput = input(
            "This action will clean all current workspaces to start anew. Proceed? (y/n): "
        )
        if (userInput == "y") | (userInput == "Y"):
            pass
        else:
            raise KeyboardInterrupt("Run of procedure canceled.")
    return


def checkInputs(crtIC: Any, bank: str | None = None) -> None:
    """Validate procedure and fitInYSpace flags on a control class.

    This function is used in two contexts:

    * **Global validation** (called with ``bank=None``).  Only the flag
      values themselves are checked; no cross‑comparison is performed.
      This is appropriate for ``UserScriptControls`` objects where the
      bank being processed is not yet known.
    * **Bank-aware validation** (``bank`` set to "BACKWARD" or
      "FORWARD").  In non‑joint execution the pipeline may run the
      two banks sequentially, so we must not raise an error simply because
      ``procedure`` and ``fitInYSpace`` refer to different banks.  The
      check below only complains if the *same* bank appears in both
      fields and the names disagree, or if a flag is outright invalid
      for the given bank.

    Args:
        crtIC: A ``UserScriptControls`` or ``BootstrapInitialConditions``
            class whose ``procedure`` and ``fitInYSpace`` attributes are
            validated.
        bank: Optional bank name used for bank‑specific checks.  If
            ``None`` the routine performs only value validity checking.

    Raises:
        ValueError: If any flag value is invalid, or if both flags are
            set to the *same* bank but disagree, or if a flag is
            incompatible with the supplied ``bank`` argument.
    """

    try:
        if not crtIC.runRoutine:
            return
    except AttributeError:
        if not crtIC.runBootstrap:
            return

    # --- basic legality check ---
    for flag in [crtIC.procedure, crtIC.fitInYSpace]:
        if not (
            (flag == "BACKWARD")
            | (flag == "FORWARD")
            | (flag == "JOINT")
            | (flag is None)
        ):
            raise ValueError(
                f"Invalid option '{flag}' not recognized. "
                "Must be 'BACKWARD', 'FORWARD', 'JOINT', or None."
            )

    if bank is not None:
        # Bank‑specific consistency: the flag must either be unset, JOINT,
        # or match the bank being processed.
        for name, flag in ("procedure", crtIC.procedure), ("fitInYSpace", crtIC.fitInYSpace):
            if flag not in (None, "JOINT", bank):
                raise ValueError(
                    f"{name}='{flag}' is incompatible with bank '{bank}'."
                )
        # if both flags mention this same bank but disagree, that's wrong
        if (crtIC.procedure == bank) and (crtIC.fitInYSpace == bank) and (
            crtIC.procedure != crtIC.fitInYSpace
        ):
            raise ValueError(
                f"procedure and fitInYSpace both refer to '{bank}' but differ: "
                f"{crtIC.procedure} vs {crtIC.fitInYSpace}."
            )
    # else: when bank is None we skip any cross comparison


def _convertToYSpaceSummed(
    ws_name: str, mass: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a named TOF workspace to y-space and return summed 1-D arrays.

    Calls Mantid ``ConvertToYSpace`` followed by ``SumSpectra`` to
    produce a single averaged spectrum, then extracts NumPy arrays.
    The temporary y-space workspaces are deleted from the ADS after
    extraction so they do not accumulate.

    Args:
        ws_name: Name of the TOF workspace in the ADS.
        mass: Atomic mass (a.m.u.) for the y-scaling
            $J(y) = M / (\\hbar q) \\cdot (E - E_{\\text{recoil}})$.

    Returns:
        ``(x, y, err)`` — 1-D NumPy arrays (summed across spectra)
        in y-space units ($\\AA^{-1}$).
    """
    tmp_name = ws_name + "_JoY_tmp"
    ws_joy = ConvertToYSpace(ws_name, Mass=mass, OutputWorkspace=tmp_name)

    # ConvertToYSpace produces detector-specific (ragged) bin edges because the
    # y-space transformation depends on each detector's angle and flight path.
    # SumSpectra requires common bin boundaries for all histograms, so we must
    # rebin to a common grid first.  The grid is inferred from the workspace
    # itself to avoid hard-coding a y-space range, ensuring the fix is valid
    # regardless of the runHistData setting.
    x_all = ws_joy.extractX()
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    step = float(np.median(np.abs(np.diff(x_all, axis=1))))
    if step <= 0.0:
        raise ValueError(
            f"_convertToYSpaceSummed: inferred y-space bin width is "
            f"{step:.6g} for workspace '{ws_name}'. "
            "ConvertToYSpace may have produced a degenerate workspace."
        )
    rebin_params = f"{x_min:.6f},{step:.6f},{x_max:.6f}"
    ws_joy = Rebin(
        InputWorkspace=ws_joy,
        Params=rebin_params,
        FullBinsOnly=True,
        OutputWorkspace=tmp_name,
    )
    ws_sum = SumSpectra(ws_joy, OutputWorkspace=tmp_name + "_Sum")

    raw_x = ws_sum.extractX()
    raw_y = ws_sum.extractY()
    raw_e = ws_sum.extractE()

    x = raw_x[0]
    y = raw_y[0]
    e = raw_e[0]
    if len(x) == len(y) + 1:
        x = 0.5 * (x[:-1] + x[1:])

    DeleteWorkspaces([tmp_name, tmp_name + "_Sum"])
    return x, y, e


def _dispatchCorrectionPlots(
    userCtr: Any, bckwdIC: Any, fwdIC: Any
) -> None:
    """Dispatch correction dashboard plots for all active ICs.

    Called from ``runScript`` immediately after ``runProcedure()``
    completes and only when ``userCtr.runRoutine`` is ``True``.
    Iterates over the ICs that were actually fitted (selected by
    ``userCtr.procedure``) and calls ``dispatch_correction_plots``
    for each one whose correction flags are active.

    The y-space conversion is provided via ``_convertToYSpaceSummed``
    so that every correction term is shown in both TOF and y-space.

    Args:
        userCtr: ``UserScriptControls`` with a ``procedure`` attribute.
        bckwdIC: Completed backward initial-conditions object.
        fwdIC: Completed forward initial-conditions object.
    """
    proc = getattr(userCtr, "procedure", None)
    if proc is None:
        return

    ics: List[Any] = []
    if proc in ("BACKWARD", "JOINT"):
        ics.append(bckwdIC)
    if proc in ("FORWARD", "JOINT"):
        ics.append(fwdIC)

    for ic in ics:
        ms_flag = bool(getattr(ic, "MSCorrectionFlag", False))
        gc_flag = bool(getattr(ic, "GammaCorrectionFlag", False))
        if not (ms_flag or gc_flag):
            continue
        try:
            dispatch_correction_plots(
                ic=ic,
                mtd=mtd,
                convert_to_yspace_fn=_convertToYSpaceSummed,
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"_dispatchCorrectionPlots: plotting failed for "
                f"'{getattr(ic, 'name', '?')}': {exc}"
            )


def _runPreFitStatistics(
    userCtr: Any,
    res: Any,
    bckwdIC: Any,
    fwdIC: Any,
    yFitIC: Any,
) -> Dict[str, Any]:
    """Phase 6a: Diagnostic-only outlier detection and DBSCAN clustering.

    PHILOSOPHY — DIAGNOSTIC ASSISTANCE, NOT AUTOMATED REMOVAL:
        Automated outlier masking is **deprecated**.  Real samples (e.g.
        Thymol) can be anisotropic; automated removal risks discarding
        genuine physical signals.  This function now operates as an
        *Anisotropy & Health Monitor*: suspicious detectors are flagged
        for manual review via formatted diagnostic tables.

        The **only** spectra excluded from the fit are those explicitly
        listed by the user in ``maskedSpecAllNo``.  If the summary-feature
        outlier pipeline detects an outlier whose Spectrum ID
        is **not** in ``maskedSpecAllNo``, it is flagged as
        "Unrecognized Outlier — Manual Review Required".

    Runs **before** the y-space global fit so that DBSCAN results can
    dynamically configure ``yFitIC.nGlobalFitGroups``.

    Diagnostic outputs:
        1. Formatted terminal table with Index, Spectrum ID, Scattering
           Angle, Fisher Score, and Recommendation.
        2. Physical interpretation hints (angular trends, cluster count).
        3. Unified Phase 6 Diagnostic Summary PDF:
           - Panel 1: Feature-space scatter (color-coded by Fisher Score).
           - Panel 2: Fisher 1D histogram of discriminant scores.
           - Panel 3: Residuals vs Scattering Angle (anisotropy check).
           - Panel 4: ROC curve (detector trustworthiness).

    Args:
        userCtr: ``UserScriptControls`` with detection/clustering flags.
        res: Result tuple from the iterative NCP fit (may be ``None``).
        bckwdIC: Completed backward initial-conditions object.
        fwdIC: Completed forward initial-conditions object.
        yFitIC: ``YSpaceFitInitialConditions`` — ``nGlobalFitGroups``
            may be mutated by DBSCAN.

    Returns:
        A dict with diagnostic info (outlier indices, cluster labels,
        etc.) or an empty dict if nothing ran.
    """
    from vesuvio_analysis.core_functions.statistical_plugins import (
        apply_detector_intensity_calibration,
        build_detector_feature_matrix,
        build_fidelity_labels,
        centroid_distance_scores,
        compute_anisotropy_residuals,
        detector_quality_weights,
        detector_relative_difference_metrics,
        fisher_lda_with_roc,
        format_diagnostic_table,
        generate_physical_interpretation_hint,
        HardwareOutlierDetector,
        PhysicsTrendClusterer,
        plot_cluster_ltheta,
        plot_detector_calibration_distribution,
        plot_fisher_distribution,
        plot_fisher_roc,
        plot_outlier_scatter,
        plot_phase6_diagnostic_dashboard,
        plot_residuals_vs_angle,
        plot_feature_annotated,
        plot_feature_lda_overlay,
    )
    from vesuvio_analysis.core_functions.analysis_functions import (
        loadInstrParsFileIntoArray,
        plot_optimizer_divergence_spectra,
    )

    diagnostics: Dict[str, Any] = {}

    any_enabled = (
        getattr(userCtr, "runOutlierDetection", False)
        or getattr(userCtr, "runPhysicsClustering", False)
    )
    if not any_enabled or res is None:
        return diagnostics

    # ---- Extract resultsObject(s) and their corresponding ICs ----
    if len(res) == 3:
        results_and_ics = [(res[1], bckwdIC), (res[2], fwdIC)]
    else:
        proc = userCtr.procedure
        ic = bckwdIC if proc == "BACKWARD" else fwdIC
        results_and_ics = [(res[1], ic)]

    for results, ic in results_and_ics:
        # --- Extract NCP profiles for diagnostic analysis ---
        spectra = results.all_fit_workspaces[-1]
        ncp_total = results.all_tot_ncp[-1]

        # Align shapes when runHistData=True
        if spectra.shape[1] == ncp_total.shape[1] + 1:
            spectra = spectra[:, :-1].copy()

        n_total = spectra.shape[0]
        outlier_mask: Optional[np.ndarray] = None
        cluster_labels_full: Optional[np.ndarray] = None

        # Resolve the Mantid workspace name
        ws_name = ic.name + str(ic.noOfMSIterations)
        print(
            f"\n{_SEP_DOUBLE}\n"
            f"  [Phase 6] Anisotropy & Health Monitor — {ws_name}\n"
            f"{_SEP_DOUBLE}"
        )

        # --- Build MetadataMap: array_index → {spec_no, angle, detector_id} ---
        metadata_map = build_metadata_map(ic)
        diagnostics["metadata_map"] = metadata_map

        # --- Resolve user-defined masks ---
        masked_spec_all_no = np.asarray(
            getattr(ic, "maskedSpecAllNo", []), dtype=np.intp,
        )

        # Load instrument parameters (needed for θ, L, spec_no mapping)
        instrPars = loadInstrParsFileIntoArray(
            ic.InstrParsPath, ic.firstSpec, ic.lastSpec,
        )
        theta_deg = instrPars[:, 2]
        all_spec_nos = instrPars[:, 0]

        # ---- Outlier Detection (Summary-Feature + EllipticEnvelope) — DIAGNOSTIC ONLY ----
        if getattr(userCtr, "runOutlierDetection", False):
            n_outlier_passes = max(1, int(getattr(userCtr, "outlierPasses", 1)))

            all_flagged_indices: List[int] = []
            all_flagged_spec_ids: List[int] = []
            current_spectra = spectra.copy()
            active_indices = np.arange(n_total)

            for pass_no in range(1, n_outlier_passes + 1):
                if current_spectra.shape[0] < 4:
                    print(
                        f"[Phase 6] Pass {pass_no}: Too few spectra "
                        f"remaining ({current_spectra.shape[0]}) — stopping."
                    )
                    break

                detector = HardwareOutlierDetector(
                    contamination=0.1,
                )
                labels = detector.fit_predict(current_spectra)
                outlier_local = np.where(labels == -1)[0]
                n_outliers_pass = len(outlier_local)

                if n_outliers_pass == 0:
                    print(
                        f"[Phase 6] Pass {pass_no}: No anomalies detected — "
                        f"converged after {pass_no} pass(es)."
                    )
                    break

                outlier_original = active_indices[outlier_local]
                flagged_spec_ids_pass = []
                for oi in outlier_original:
                    meta = metadata_map.get(int(oi), {})
                    spec_id = meta.get("spec_no", oi)
                    flagged_spec_ids_pass.append(spec_id)
                    all_flagged_spec_ids.append(spec_id)
                all_flagged_indices.extend(outlier_original.tolist())

                pct_remaining = 100.0 * (n_total - len(all_flagged_indices)) / n_total
                print(
                    f"[Phase 6] Pass {pass_no}: Flagged Spectrum IDs "
                    f"{flagged_spec_ids_pass}  "
                    f"({n_outliers_pass} anomaly(ies) this pass, "
                    f"{len(all_flagged_indices)}/{n_total} total flagged, "
                    f"{pct_remaining:.1f}% clean)"
                )

                if pass_no < n_outlier_passes:
                    keep_mask = np.ones(len(active_indices), dtype=bool)
                    keep_mask[outlier_local] = False
                    active_indices = active_indices[keep_mask]
                    current_spectra = current_spectra[keep_mask]

            # Final summary
            outlier_idx = np.array(sorted(set(all_flagged_indices)), dtype=np.intp)
            n_outliers = len(outlier_idx)
            pct_remaining = 100.0 * (n_total - n_outliers) / n_total

            diagnostics["outlier_passes"] = min(pass_no, n_outlier_passes)

            # Reconstruct labels for the full original array
            labels = np.zeros(n_total, dtype=np.intp)
            labels[outlier_idx] = -1

            diagnostics["outlier_indices"] = outlier_idx
            diagnostics["outlier_labels"] = labels
            outlier_mask = labels == -1

            # ---- Calibration pass (AppStat Week 5 style) ----
            calib = detector_relative_difference_metrics(spectra, ncp_total)
            det_bias = calib["bias"]
            det_rms = calib["rms"]
            diagnostics["detector_bias"] = det_bias
            diagnostics["detector_rms"] = det_rms

            fig_dir = getattr(ic, "figSavePath", None)
            if fig_dir is not None:
                try:
                    plot_detector_calibration_distribution(
                        det_bias, det_rms,
                        save_path=fig_dir / f"{ws_name}_calibration_distribution.pdf",
                    )
                except Exception as exc:
                    warnings.warn(f"Phase 6 calibration distribution plot failed: {exc}")

            median_rms = float(np.nanmedian(det_rms))
            rms_gate = float(getattr(userCtr, "calibrationRMSGate", 0.02))
            apply_calibration = median_rms > rms_gate

            bias_sigma = float(getattr(userCtr, "calibrationBiasSigma", 2.0))
            bias_threshold = bias_sigma * float(np.nanstd(det_bias))
            biased_mask = np.abs(det_bias) > bias_threshold

            if apply_calibration and np.any(biased_mask) and (ws_name in mtd):
                ws_calib = mtd[ws_name]
                y_corr, e_corr, corr_factor = apply_detector_intensity_calibration(
                    ws_calib.extractY(),
                    ws_calib.extractE(),
                    det_bias,
                    biased_mask,
                )
                for row_idx in range(ws_calib.getNumberHistograms()):
                    ws_calib.dataY(row_idx)[:] = y_corr[row_idx, :]
                    ws_calib.dataE(row_idx)[:] = e_corr[row_idx, :]
                spectra = y_corr[:, :spectra.shape[1]]
                diagnostics["calibration_factors"] = corr_factor
                logger.info(
                    "Phase 6 calibration pass: corrected %d/%d detectors "
                    "(median RMS %.4g > %.4g).",
                    int(np.sum(biased_mask)), int(len(biased_mask)),
                    median_rms, rms_gate,
                )
            else:
                logger.info(
                    "Phase 6 calibration pass skipped "
                    "(median RMS %.4g, gate %.4g, biased=%d).",
                    median_rms, rms_gate, int(np.sum(biased_mask)),
                )

            # ---- Anisotropy Detection: residuals vs scattering angle ----
            anisotropy = compute_anisotropy_residuals(
                spectra, ncp_total, theta_deg,
            )
            diagnostics["anisotropy"] = anisotropy

            # ---- Fisher discriminant (LDA) + ROC from convergence labels ----
            lda_result = None
            fidelity_labels = None
            fisher_scores = None
            feature_names = ["Total Counts", "RMS Shape", "Width σ", "Angle θ"]

            if ws_name + "_Optimizer_Diagnostics" in mtd:
                opt_table = mtd[ws_name + "_Optimizer_Diagnostics"]

                n_det = len(spectra)
                spec_nos_tab = np.array(opt_table.column("Spec Idx"), dtype=float)
                chi2_rel_tab = np.array(opt_table.column("Chi2 Rel Diff"), dtype=float)
                par_rel_tab = np.array(opt_table.column("Max Par Rel Diff"), dtype=float)
                migrad_valid_tab = np.array(opt_table.column("Migrad Valid"), dtype=float) > 0.5

                agreement_full = np.full(n_det, np.inf)
                migrad_valid_full = np.zeros(n_det, dtype=bool)
                for k, sno in enumerate(spec_nos_tab):
                    row = np.where(np.isclose(all_spec_nos, sno))[0]
                    if len(row) == 1:
                        agreement_full[row[0]] = max(
                            float(chi2_rel_tab[k]), float(par_rel_tab[k]),
                        )
                        migrad_valid_full[row[0]] = bool(migrad_valid_tab[k])

                fidelity_labels = build_fidelity_labels(
                    agreement_full, migrad_valid_full,
                    hi_fidelity_thr=0.01, poor_fidelity_thr=0.05,
                )
                diagnostics["fidelity_labels"] = fidelity_labels

                width_proxy = np.zeros(len(spectra), dtype=float)
                fit_table_name = ws_name + "_Best_Fit_NCP_Parameters"
                if fit_table_name in mtd and "Width 0" in mtd[fit_table_name].keys():
                    width_proxy = np.array(
                        mtd[fit_table_name].column("Width 0"), dtype=float,
                    )

                features = build_detector_feature_matrix(
                    spectra, theta_deg, width_proxy,
                )

                lda_result = fisher_lda_with_roc(features, fidelity_labels)
                if lda_result is not None:
                    fisher_scores = np.array(lda_result["scores"], dtype=float)
                    diagnostics["fisher_scores"] = fisher_scores
                else:
                    # No Poor-Fidelity class (e.g. Forward bank, all Hi-Fidelity).
                    # Fall back to centroid-distance proxy so the dashboard
                    # receives a meaningful per-detector score instead of N/A.
                    fisher_scores = centroid_distance_scores(
                        detector.summary_features_,
                    )
                    diagnostics["fisher_scores"] = fisher_scores
                    logger.info(
                        "[Phase 6] Fisher/LDA unavailable for %s "
                        "(no Poor-Fidelity class) — centroid-distance proxy "
                        "scores computed, n=%d.",
                        ws_name, len(fisher_scores),
                    )

                    if fig_dir is not None:
                        try:
                            plot_fisher_roc(
                                lda_result["fpr"], lda_result["tpr"],
                                float(lda_result["auc"]),
                                save_path=fig_dir / f"{ws_name}_fisher_roc.pdf",
                            )
                        except Exception as exc:
                            warnings.warn(f"Phase 6 Fisher ROC plot failed: {exc}")

                        try:
                            plot_fisher_distribution(
                                fisher_scores, fidelity_labels,
                                feature_names, lda_result["model"],
                                float(lda_result["auc"]),
                                save_path=fig_dir / f"{ws_name}_fisher_distribution.pdf",
                            )
                        except Exception as exc:
                            warnings.warn(f"Phase 6 Fisher distribution plot failed: {exc}")

            # ---- Detector quality weights ----
            p_fail = None if lda_result is None else np.array(
                lda_result["p_fail"], dtype=float,
            )
            det_weights = detector_quality_weights(det_rms, p_fail)
            diagnostics["detector_weights"] = det_weights

            if not hasattr(yFitIC, "detectorQualityWeightsByWorkspace"):
                yFitIC.detectorQualityWeightsByWorkspace = {}
            yFitIC.detectorQualityWeightsByWorkspace[ws_name] = det_weights

            # ---- DIAGNOSTIC TABLE — Transparency-first output ----
            print(
                format_diagnostic_table(
                    outlier_idx, metadata_map,
                    masked_spec_all_no, fisher_scores,
                )
            )
            logger.info(
                "Phase 6 diagnostic: %d/%d detectors flagged at Spec IDs %s "
                "(%.1f%% clean)",
                n_outliers, n_total,
                sorted(set(all_flagged_spec_ids)), pct_remaining,
            )

            # ---- DEPRECATED: Automated masking ----
            # Automated MaskDetectors calls have been removed. Only
            # spectra in maskedSpecAllNo are excluded (handled earlier
            # via ICHelpers.completeICFromInputs → maskedDetectorIdx).
            if getattr(userCtr, "removeOutliers", False):
                warnings.warn(
                    "removeOutliers=True is DEPRECATED and has no effect. "
                    "Outlier detection now operates as a diagnostic-only "
                    "Anisotropy & Health Monitor. To exclude spectra, add "
                    "their Spectrum IDs to maskedSpecAllNo in the IC class.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            # ---- Optimizer divergence plots (Transparency First metadata) ----
            try:
                _fs_map: Dict[float, float] = (
                    {
                        float(sno): float(fs)
                        for sno, fs in zip(all_spec_nos, fisher_scores)
                    }
                    if fisher_scores is not None
                    else {}
                )
                plot_optimizer_divergence_spectra(
                    ic,
                    debug_mode=getattr(userCtr, "debug_mode", False),
                    fisher_scores_map=_fs_map,
                )
            except Exception as _opt_exc:
                warnings.warn(f"Optimizer divergence plots failed: {_opt_exc}")

            # ---- Diagnostic plots ----
            features_before = detector.summary_features_.copy()
            labels_before = labels.copy()

            if fig_dir is not None:
                _outlier_summary = {
                    "n_total": n_total,
                    "n_outliers": n_outliers,
                }
                try:
                    # Annotated feature-space scatter with Spectrum IDs, cluster angles,
                    # and Spearman ρ as a Physics Metric inset.
                    _sp_r_val = anisotropy.get("spearman_r", float("nan")) if isinstance(anisotropy, dict) else float("nan")
                    _sp_p_val = anisotropy.get("spearman_p", float("nan")) if isinstance(anisotropy, dict) else float("nan")
                    plot_feature_annotated(
                        features_before, labels_before,
                        metadata_map,
                        cluster_labels=cluster_labels_full,
                        spearman_r=float(_sp_r_val),
                        spearman_p=float(_sp_p_val),
                        save_path=fig_dir / f"{ws_name}_diagnostic_features.pdf",
                    )
                except Exception as exc:
                    warnings.warn(f"Phase 6 annotated feature plot failed: {exc}")

                try:
                    if (lda_result is not None) and (p_fail is not None):
                        plot_feature_lda_overlay(
                            features_before, p_fail,
                            labels=fidelity_labels,
                            save_path=fig_dir / f"{ws_name}_feature_outliers.pdf",
                        )
                    else:
                        plot_outlier_scatter(
                            features_before, labels_before,
                            save_path=fig_dir / f"{ws_name}_feature_outliers.pdf",
                            summary_stats=_outlier_summary,
                        )
                except Exception as exc:
                    warnings.warn(f"Phase 6 outlier plot failed: {exc}")

                # Residuals vs angle plot
                try:
                    plot_residuals_vs_angle(
                        anisotropy, metadata_map,
                        outlier_indices=outlier_idx,
                        save_path=fig_dir / f"{ws_name}_residuals_vs_angle.pdf",
                    )
                except Exception as exc:
                    warnings.warn(f"Phase 6 residuals vs angle plot failed: {exc}")

        # ---- Physics Clustering (DBSCAN) — pre-Global Fit ----
        if getattr(userCtr, "runPhysicsClustering", False):
            L_total = instrPars[:, 4] + instrPars[:, 5]
            theta = instrPars[:, 2]
            features = np.column_stack([L_total, theta])

            if outlier_mask is not None:
                features_clean = features[~outlier_mask]
            else:
                features_clean = features

            _n_det_clust = len(features_clean)
            _min_samp = max(3, int(0.05 * _n_det_clust))
            logger.info(
                "[Phase 6] DBSCAN bank-aware min_samples=%d (n_det=%d, 5%% rule)",
                _min_samp, _n_det_clust,
            )
            clusterer = PhysicsTrendClusterer(eps=0.5, min_samples=_min_samp)
            cluster_labels = clusterer.fit_predict(features_clean)
            groups = clusterer.get_cluster_groups(cluster_labels)
            n_noise = int(np.sum(cluster_labels == -1))
            n_clusters = len(groups)
            cluster_labels_full = cluster_labels

            print(
                f"[Phase 6] DBSCAN clustering: {n_clusters} cluster(s) "
                f"found, {n_noise} noise point(s) excluded"
            )
            logger.info(
                "Phase 6 DBSCAN: %d clusters, %d noise points",
                n_clusters, n_noise,
            )

            diagnostics["cluster_labels"] = cluster_labels
            diagnostics["cluster_groups"] = groups
            diagnostics["n_noise"] = n_noise

            if n_clusters > 0:
                old_n = yFitIC.nGlobalFitGroups
                yFitIC.nGlobalFitGroups = n_clusters
                print(
                    f"[Phase 6] nGlobalFitGroups updated: "
                    f"{old_n} → {n_clusters} (from DBSCAN)"
                )
                logger.info(
                    "nGlobalFitGroups updated: %s → %d (DBSCAN)",
                    old_n, n_clusters,
                )
            else:
                print(
                    "[Phase 6] DBSCAN found 0 clusters; "
                    "nGlobalFitGroups left unchanged."
                )

            fig_dir = getattr(ic, "figSavePath", None)
            if fig_dir is not None:
                try:
                    plot_cluster_ltheta(
                        features_clean, cluster_labels,
                        save_path=fig_dir / f"{ws_name}_cluster_ltheta.pdf",
                    )
                except Exception as exc:
                    warnings.warn(f"Phase 6 cluster plot failed: {exc}")

        # ---- "Show Your Work" Physical Interpretation Hints ----
        _n_clusters = diagnostics.get("n_noise", 0)
        if "cluster_labels" in diagnostics:
            _n_clusters = len(set(diagnostics["cluster_labels"]) - {-1})
        _sp_r = diagnostics.get("anisotropy", {}).get("spearman_r", float("nan"))
        _sp_p = diagnostics.get("anisotropy", {}).get("spearman_p", float("nan"))
        _n_out = len(diagnostics.get("outlier_indices", []))
        print(
            generate_physical_interpretation_hint(
                _n_clusters, _sp_r, _sp_p, _n_out, n_total,
            )
        )

        # ---- Unified Phase 6 Diagnostic Dashboard PDF ----
        fig_dir = getattr(ic, "figSavePath", None)
        if fig_dir is not None and getattr(userCtr, "runOutlierDetection", False):
            try:
                _aniso = diagnostics.get("anisotropy", {
                    "theta": theta_deg,
                    "mean_residual": np.zeros_like(theta_deg),
                    "std_residual": np.zeros_like(theta_deg),
                    "spearman_r": float("nan"),
                    "spearman_p": float("nan"),
                })
                _emb = diagnostics.get("outlier_labels", np.zeros(n_total, dtype=np.intp))
                _features = getattr(detector, "summary_features_", np.zeros((n_total, 4)))
                plot_phase6_diagnostic_dashboard(
                    summary_features=_features,
                    labels=_emb,
                    fisher_scores=fisher_scores,
                    fidelity_labels=fidelity_labels,
                    theta_deg=theta_deg,
                    anisotropy=_aniso,
                    roc_data=lda_result,
                    metadata_map=metadata_map,
                    ws_name=ws_name,
                    save_path=fig_dir / f"{ws_name}_Phase6_Diagnostic_Summary.pdf",
                )
            except Exception as exc:
                warnings.warn(f"Phase 6 dashboard generation failed: {exc}")
                logger.error(
                    "[Phase 6] Dashboard shape debug — "
                    "summary_features: %s, labels: %s, "
                    "fisher_scores: %s, fidelity_labels: %s, theta_deg: %s",
                    getattr(_features, "shape", type(_features)),
                    getattr(_emb, "shape", type(_emb)),
                    getattr(fisher_scores, "shape", None),
                    getattr(fidelity_labels, "shape", None),
                    getattr(theta_deg, "shape", type(theta_deg)),
                )

    return diagnostics


def _runBayesianBootstrapProcedure(
    bckwdIC: Any,
    fwdIC: Any,
    bootIC: Any,
    yFitIC: Any,
) -> Dict[str, Any]:
    """Run the Bayesian Bootstrap as a standalone bootstrap procedure.

    This is invoked when ``bootIC.bootstrapType == "BOOT_BAYESIAN"``
    and ``bootIC.runBootstrap == True``.  It first runs the parent NCP
    fit (via the same path as the standard bootstrap's
    ``runOriginalBeforeBootstrap``), then applies Dirichlet-weighted
    resampling to the NCP residuals.

    Unlike the frequentist bootstrap methods (JACKKNIFE,
    BOOT_RESIDUALS, BOOT_GAUSS_ERRS), this does **not** re-fit each
    replica — it produces weighted residual profiles in a single
    matrix multiplication.

    Args:
        bckwdIC: Completed backward initial-conditions object.
        fwdIC: Completed forward initial-conditions object.
        bootIC: ``BootstrapInitialConditions`` with ``nSamples``,
            ``procedure``, and ``bootstrapType == "BOOT_BAYESIAN"``.
        yFitIC: ``YSpaceFitInitialConditions`` for y-space fitting.

    Returns:
        Dict with keys ``weighted_residuals``, ``boot_mean``,
        ``boot_std``, and ``n_samples``.
    """
    from vesuvio_analysis.core_functions.statistical_plugins import (
        BayesianBootstrap,
        plot_bootstrap_convergence,
    )
    from vesuvio_analysis.core_functions.procedures import (
        runIndependentIterativeProcedure,
        runJointBackAndForwardProcedure,
    )
    from vesuvio_analysis.core_functions.fit_in_yspace import (
        fitInYSpaceProcedure,
    )
    from vesuvio_analysis.core_functions.ICHelpers import buildFinalWSName

    print("[Phase 6] Running Bayesian Bootstrap (BOOT_BAYESIAN) ...")

    # Run the parent NCP fit to obtain residuals
    proc = bootIC.procedure
    if proc == "BACKWARD":
        res = runIndependentIterativeProcedure(bckwdIC)
    elif proc == "FORWARD":
        res = runIndependentIterativeProcedure(fwdIC)
    elif proc == "JOINT":
        # Pre-flight: the joint procedure calls runJoint which consumes
        # bckwdIC.HToMassIdxRatio when H is present.  If the primary fit
        # ran only a single bank this ratio was never estimated, so we must
        # catch the misconfiguration here rather than letting it crash inside
        # Mantid with an opaque AssertionError.
        if isHPresent(fwdIC.masses) and getattr(bckwdIC, "HToMassIdxRatio", None) is None:
            raise ValueError(
                "Cannot run a JOINT Bayesian Bootstrap: "
                "bckwdIC.HToMassIdxRatio is not set.  "
                "The backward-bank H/mass intensity ratio must be estimated "
                "by a prior JOINT or BACKWARD primary fit before the bootstrap "
                "can propagate backward constraints into the forward fit.  "
                "Set BootstrapInitialConditions.procedure = 'FORWARD' to "
                "bootstrap only the forward bank, or run the full JOINT "
                "primary procedure first."
            )
        # Verify both bank final workspaces are present in mtd.
        bckwd_ws = bckwdIC.name + str(bckwdIC.noOfMSIterations)
        fwd_ws = fwdIC.name + str(fwdIC.noOfMSIterations)
        missing = [ws for ws in (bckwd_ws, fwd_ws) if ws not in mtd]
        if missing:
            raise ValueError(
                f"Cannot run JOINT Bayesian Bootstrap: workspace(s) "
                f"{missing} not found in the Mantid Analysis Data Service.  "
                "Ensure the full JOINT primary fit has completed successfully "
                "before running a JOINT bootstrap."
            )
        res = runJointBackAndForwardProcedure(bckwdIC, fwdIC)
    else:
        raise ValueError(f"Invalid bootstrap procedure: {proc}")

    # Extract NCP results
    if len(res) == 3:
        results_list = [(res[1], bckwdIC), (res[2], fwdIC)]
    else:
        ic = bckwdIC if proc == "BACKWARD" else fwdIC
        results_list = [(res[1], ic)]

    all_results: Dict[str, Any] = {}
    n_samples = getattr(bootIC, "nSamples", 1000)

    for results, ic in results_list:
        spectra = results.all_fit_workspaces[-1]
        ncp_total = results.all_tot_ncp[-1]

        if spectra.shape[1] == ncp_total.shape[1] + 1:
            spectra = spectra[:, :-1].copy()

        residuals = spectra - ncp_total
        bootstrap = BayesianBootstrap(n_samples=n_samples, seed=42)

        # Pre-sanitize residuals (mirrors BayesianBootstrap.compute_weighted_residuals)
        bad_rows = ~np.any(np.isfinite(residuals), axis=1)
        n_valid = int(np.sum(~bad_rows))
        clean_res = residuals.copy()
        clean_res[bad_rows, :] = 0.0
        np.nan_to_num(clean_res, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        print(
            f"[Bootstrap] Valid spectra: {n_valid}/{residuals.shape[0]}  "
            f"Residual range: [{np.nanmin(residuals):.4g}, {np.nanmax(residuals):.4g}]"
        )

        # Per-10th-replica diagnostic log: generate weights explicitly so we
        # can inspect individual replicas without re-doing the full multiply.
        weights_all = bootstrap.generate_weights(residuals.shape[0])   # (n_samples, n_spectra)
        if bad_rows.any():
            weights_all[:, bad_rows] = 0.0
            _rs = weights_all.sum(axis=1, keepdims=True)
            weights_all = weights_all / np.where(_rs > 0.0, _rs, 1.0)

        weighted = weights_all @ clean_res   # (n_samples, n_bins)

        for _i in range(0, n_samples, max(1, n_samples // 10)):
            _rep = weighted[_i]
            _mean_r = float(np.mean(_rep))
            _valid_bins = int(np.sum(np.isfinite(_rep)))
            print(
                f"[Bootstrap] Replica {_i:>4d}: "
                f"mean residual = {_mean_r:+.4g}, "
                f"valid bins = {_valid_bins}/{len(_rep)}"
            )

        boot_mean = float(np.mean(weighted))
        boot_std = float(np.std(weighted))

        direction = getattr(ic, "modeRunning", "UNKNOWN")
        print(
            f"[Phase 6] Bayesian Bootstrap ({direction}): "
            f"{weighted.shape[0]} replicas, shape {weighted.shape}"
        )
        print(
            f"[Phase 6] Bootstrap summary ({direction}): "
            f"mean = {boot_mean:.4f}, std = {boot_std:.4f}"
        )

        # --- Bootstrap convergence diagnostic plot ---
        fig_dir = getattr(ic, "figSavePath", None)
        if fig_dir is not None:
            try:
                plot_bootstrap_convergence(
                    weighted,
                    save_path=fig_dir / f"stats_bootstrap_convergence_{direction}.pdf",
                )
            except Exception as exc:
                warnings.warn(f"Phase 6 bootstrap convergence plot failed: {exc}")

        all_results[direction] = {
            "weighted_residuals": weighted,
            "boot_mean": boot_mean,
            "boot_std": boot_std,
            "n_samples": n_samples,
        }

    return all_results
