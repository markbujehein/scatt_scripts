from typing import Any, List, Optional, Tuple

import numpy as np
from mantid.api import mtd
from mantid.simpleapi import ConvertToYSpace, DeleteWorkspaces, SumSpectra

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
    assert not (userCtr.runRoutine & bootIC.runBootstrap), (
        "Main routine and bootstrap both set to run!"
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
    logger = RunLogger(scriptName, direction, _log_output_dir)
    logger.log_environment()
    logger.log_ic("UserScriptControls", userCtr)
    logger.log_ic("LoadVesuvioBackParameters", wsBackIC)
    logger.log_ic("LoadVesuvioFrontParameters", wsFrontIC)
    logger.log_ic("BackwardInitialConditions", bckwdIC)
    logger.log_ic("ForwardInitialConditions", fwdIC)
    logger.log_ic("YSpaceFitInitialConditions", yFitIC)
    logger.log_ic("BootstrapInitialConditions", bootIC)
    logger.log_flags(
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
            if isHPresent(fwdIC.masses) & (bckwdIC.HToMassIdxRatio == None):
                HRatios, massIdxs = runPreProcToEstHRatio(
                    bckwdIC, fwdIC
                )  # Sets H ratio to bckwdIC automatically
                ranPreliminary = True
            assert isHPresent(fwdIC.masses) != (bckwdIC.HToMassIdxRatio == None), (
                "When H is not present, HToMassIdxRatio has to be set to None"
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

    # Names of workspaces to be fitted in y space
    wsNames = []
    ICs = []
    for mode, IC in zip(["BACKWARD", "FORWARD"], [bckwdIC, fwdIC]):
        if (userCtr.fitInYSpace == mode) | (userCtr.fitInYSpace == "JOINT"):
            wsNames.append(buildFinalWSName(scriptName, mode, IC))
            ICs.append(IC)

    # If bootstrap is not None, run bootstrap procedure and finish
    if bootIC.runBootstrap:
        assert (
            (bootIC.procedure == "FORWARD")
            | (bootIC.procedure == "BACKWARD")
            | (bootIC.procedure == "JOINT")
        ), "Invalid Bootstrap procedure."
        logger.write()
        return runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC), None

    # Default workflow for procedure + fit in y space
    if userCtr.runRoutine:
        # Check if final ws are loaded:
        wsInMtd = [ws in mtd for ws in wsNames]  # Bool list
        if (len(wsInMtd) > 0) and all(
            wsInMtd
        ):  # When wsName is empty list, loop doesn't run
            for wsName, IC in zip(wsNames, ICs):
                resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
            logger.write()
            return None, resYFit  # To match return below.

        # Skip interactive workspace-clear prompt when running a smoke test
        if not getattr(userCtr, "runningTest", False):
            checkUserClearWS()  # Check if user is OK with cleaning all workspaces

        res = None
        resYFit = None
        try:
            logger.log_timestamp("ncp_start")
            res = runProcedure()
            logger.log_timestamp("ncp_end")
            logger.log_final_results(res[1] if res is not None and len(res) >= 2 else None)

            # --- Correction Dashboard Plots ---
            _dispatchCorrectionPlots(userCtr, bckwdIC, fwdIC)

            logger.log_timestamp("yspace_start")
            for wsName, IC in zip(wsNames, ICs):
                resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
            logger.log_timestamp("yspace_end")

        except Exception as exc:
            logger.log_error(exc)
            logger.write()
            raise

        # --- Phase 6: Statistical Analysis (post-fit) ---
        _runStatisticalAnalysis(userCtr, res, bckwdIC, fwdIC)

        logger.write()
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


def checkInputs(crtIC: Any) -> None:
    """Validate procedure and fitInYSpace flags on a control class.

    Checks that ``crtIC.procedure`` and ``crtIC.fitInYSpace`` are among
    the accepted values (``"BACKWARD"``, ``"FORWARD"``, ``"JOINT"``, or
    ``None``) and that they are consistent with each other.  Silently
    returns when the corresponding run flag is ``False``.

    Args:
        crtIC: A ``UserScriptControls`` or ``BootstrapInitialConditions``
            class whose ``procedure`` and ``fitInYSpace`` attributes are
            validated.

    Raises:
        AssertionError: If any flag value is invalid or if ``procedure``
            and ``fitInYSpace`` are inconsistent.
    """

    try:
        if ~crtIC.runRoutine:
            return
    except AttributeError:
        if ~crtIC.runBootstrap:
            return

    for flag in [crtIC.procedure, crtIC.fitInYSpace]:
        assert (
            (flag == "BACKWARD")
            | (flag == "FORWARD")
            | (flag == "JOINT")
            | (flag == None)
        ), "Option not recognized."

    if (crtIC.procedure != "JOINT") & (crtIC.fitInYSpace != None):
        assert crtIC.procedure == crtIC.fitInYSpace


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
            import warnings
            warnings.warn(
                f"_dispatchCorrectionPlots: plotting failed for "
                f"'{getattr(ic, 'name', '?')}': {exc}"
            )


def _runStatisticalAnalysis(
    userCtr: Any, res: Any, bckwdIC: Any, fwdIC: Any,
) -> None:
    """Runs Phase 6 statistical analysis steps when their flags are set.

    Called after the main NCP fitting and y-space fitting have completed.
    Each step is gated by its own boolean flag on ``userCtr`` and runs
    only when the flag is ``True``.

    The pipeline extracts per-spectrum fitted NCP profiles from ``res``
    (the last iteration) and uses them for outlier detection and
    bootstrap resampling.  Instrument parameters (L, theta) are loaded
    from the IC objects for physics-trend clustering.

    Args:
        userCtr: ``UserScriptControls`` class with
            ``runOutlierDetection``, ``runPhysicsClustering``, and
            ``runBayesianBootstrap`` flags.
        res: Result tuple from the iterative NCP fit (may be ``None``).
            For BACKWARD/FORWARD: ``(wsFinal, resultsObject)``.
            For JOINT: ``(wsFinal, bckwdResults, fwdResults)``.
        bckwdIC: Completed backward initial-conditions object.
        fwdIC: Completed forward initial-conditions object.
    """
    from vesuvio_analysis.core_functions.statistical_plugins import (
        BayesianBootstrap,
        HardwareOutlierDetector,
        PhysicsTrendClusterer,
    )
    from vesuvio_analysis.core_functions.analysis_functions import (
        loadInstrParsFileIntoArray,
    )

    any_enabled = (
        getattr(userCtr, "runOutlierDetection", False)
        or getattr(userCtr, "runPhysicsClustering", False)
        or getattr(userCtr, "runBayesianBootstrap", False)
    )
    if not any_enabled or res is None:
        return

    # Extract the resultsObject(s) and their corresponding IC from the
    # procedure return value.
    # BACKWARD/FORWARD: (wsFinal, resultsObject)
    # JOINT: (wsFinal, bckwdResults, fwdResults)
    if len(res) == 3:
        results_and_ics = [(res[1], bckwdIC), (res[2], fwdIC)]
    else:
        proc = userCtr.procedure
        ic = bckwdIC if proc == "BACKWARD" else fwdIC
        results_and_ics = [(res[1], ic)]

    for results, ic in results_and_ics:
        # Last-iteration spectra: shape (n_spectra, n_bins)
        spectra = results.all_fit_workspaces[-1]
        ncp_total = results.all_tot_ncp[-1]

        # When runHistData=True the fitted workspace stores N histogram bins
        # while the NCP profile is computed on N-1 point-data bins.  Trim
        # the trailing column of spectra so both arrays are in point-data
        # representation before any arithmetic or outlier analysis.
        if spectra.shape[1] == ncp_total.shape[1] + 1:
            spectra = spectra[:, :-1].copy()

        if getattr(userCtr, "runOutlierDetection", False):
            detector = HardwareOutlierDetector(
                n_components=5, contamination=0.1,
            )
            labels = detector.fit_predict(spectra)
            n_outliers = int(np.sum(labels == -1))
            outlier_idx = np.where(labels == -1)[0]
            print(
                f"[Phase 6] Outlier detection: {n_outliers} outlier(s) "
                f"found at indices {outlier_idx.tolist()}"
            )

        if getattr(userCtr, "runPhysicsClustering", False):
            instrPars = loadInstrParsFileIntoArray(
                ic.InstrParsPath, ic.firstSpec, ic.lastSpec,
            )
            # instrPars columns: [spec, det, angle, T0, L0, L1]
            L1 = instrPars[:, 5]
            theta = instrPars[:, 2]
            features = np.column_stack([L1, theta])
            clusterer = PhysicsTrendClusterer(eps=0.5, min_samples=3)
            labels = clusterer.fit_predict(features)
            groups = clusterer.get_cluster_groups(labels)
            n_noise = int(np.sum(labels == -1))
            print(
                f"[Phase 6] Physics clustering: {len(groups)} cluster(s) "
                f"found, {n_noise} noise point(s) excluded"
            )

        if getattr(userCtr, "runBayesianBootstrap", False):
            residuals = spectra - ncp_total
            bootstrap = BayesianBootstrap(n_samples=1000, seed=42)
            weighted = bootstrap.compute_weighted_residuals(residuals)
            print(
                f"[Phase 6] Bayesian bootstrap: generated "
                f"{weighted.shape[0]} weighted residual profiles, "
                f"shape {weighted.shape}"
            )
