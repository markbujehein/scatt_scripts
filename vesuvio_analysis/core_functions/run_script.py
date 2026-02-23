from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import warnings

import numpy as np
from mantid.api import mtd
from mantid.simpleapi import (
    ConvertToYSpace,
    DeleteWorkspaces,
    MaskDetectors,
    Rebin,
    SumSpectra,
)

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
    """Phase 6a: two-pass outlier masking and DBSCAN clustering (pre-fit).

    Runs **before** the y-space global fit so that its results can
    dynamically configure ``yFitIC.nGlobalFitGroups`` and exclude
    broken detectors from subsequent fitting.

    Two-pass outlier removal workflow:
        **Pass 1** — Extract NCP profiles from the final-iteration
        workspace.  Run UMAP + ``EllipticEnvelope`` outlier detection
        on the raw NCP array.  Map the Python array indices of the
        anomalous spectra to Mantid workspace indices and mask them
        via ``MaskDetectors(Workspace=ws, WorkspaceIndexList=[…])``.

        **Pass 2** — Re-extract the (now clean) NCP profiles from the
        masked workspace.  Counts and errors for masked spectra are
        strictly zeroed by Mantid, so the downstream
        ``takeOutMaskedSpectra`` in ``runGlobalFit`` will
        automatically exclude them.

    DBSCAN grouping:
        When ``userCtr.runPhysicsClustering`` is ``True``, DBSCAN
        clusters the cleaned (outlier-free) NCP profiles in
        (L, θ) space.  The number of valid (non-noise) clusters
        dynamically overwrites ``yFitIC.nGlobalFitGroups``.  Noise
        points (label ``-1``) are excluded from all groups.

    Args:
        userCtr: ``UserScriptControls`` with detection/clustering flags
            and UMAP hyperparameters.
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
        HardwareOutlierDetector,
        PhysicsTrendClusterer,
        plot_cluster_ltheta,
        plot_fisher_discriminant,
        plot_outlier_before_after,
        plot_outlier_scatter,
    )
    from vesuvio_analysis.core_functions.analysis_functions import (
        loadInstrParsFileIntoArray,
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
        # --- Extract NCP profiles ---
        spectra = results.all_fit_workspaces[-1]
        ncp_total = results.all_tot_ncp[-1]

        # Align shapes when runHistData=True
        if spectra.shape[1] == ncp_total.shape[1] + 1:
            spectra = spectra[:, :-1].copy()

        n_total = spectra.shape[0]
        outlier_mask: Optional[np.ndarray] = None

        # Resolve the Mantid workspace name: ic.name = "{script}_{DIR}_"
        # Final iteration workspace = ic.name + str(ic.noOfMSIterations)
        ws_name = ic.name + str(ic.noOfMSIterations)
        print(
            f"[Phase 6] Entering pre-fit statistics for workspace: {ws_name}"
        )

        # ---- Build MetadataMap: {array_index: {spec_no, angle, detector_id}} ----
        # Populated via Mantid API where available; falls back to NaN/−1 on error.
        metadata_map: Dict[int, Dict[str, Any]] = {}
        if ws_name in mtd:
            _ws = mtd[ws_name]
            try:
                _sample_pos = _ws.getInstrument().getSample().getPos()
                _source_pos = _ws.getInstrument().getSource().getPos()
                _beam_dir = _sample_pos - _source_pos
            except Exception:
                _sample_pos = None
                _beam_dir = None
            for _i in range(n_total):
                try:
                    spec_no = int(_ws.getSpectrum(_i).getSpectrumNo())
                except Exception:
                    spec_no = _i
                try:
                    det = _ws.getDetector(_i)
                    det_id = int(det.getID())
                except Exception:
                    det = None
                    det_id = -1
                try:
                    if det is not None and _sample_pos is not None and _beam_dir is not None:
                        angle = float(
                            np.degrees(det.getTwoTheta(_sample_pos, _beam_dir))
                        )
                    else:
                        angle = float("nan")
                except Exception:
                    angle = float("nan")
                metadata_map[_i] = {
                    "spec_no": spec_no,
                    "angle": angle,
                    "detector_id": det_id,
                }
        else:
            # Workspace not in ADS; build map from array indices only
            for _i in range(n_total):
                metadata_map[_i] = {
                    "spec_no": _i,
                    "angle": float("nan"),
                    "detector_id": -1,
                }

        diagnostics["metadata_map"] = metadata_map

        # ---- Outlier Detection (UMAP + EllipticEnvelope) ----
        if getattr(userCtr, "runOutlierDetection", False):
            # UMAP hyperparameters from user config
            n_neighbors = getattr(userCtr, "umapNNeighbors", 15)
            min_dist = getattr(userCtr, "umapMinDist", 0.1)
            n_components = getattr(userCtr, "umapNComponents", 2)
            n_passes = max(1, int(getattr(userCtr, "outlierPasses", 1)))

            # Cumulative set of masked array indices across all passes
            cumulative_outlier_indices: List[int] = []
            # Working copy of the spectra array (rows masked between passes)
            spectra_working = spectra.copy()
            # Maps rows in spectra_working → original array indices
            active_indices = list(range(n_total))

            # Store Pass-1 embedding for before/after visualisation
            embedding_before: Optional[np.ndarray] = None
            labels_before: Optional[np.ndarray] = None

            for pass_num in range(1, n_passes + 1):
                if spectra_working.shape[0] < 4:
                    print(
                        f"[Phase 6] Pass {pass_num}: fewer than 4 spectra "
                        f"remaining — stopping iterative removal."
                    )
                    break

                detector = HardwareOutlierDetector(
                    n_components=n_components,
                    n_neighbors=max(2, min(n_neighbors, spectra_working.shape[0] - 1)),
                    min_dist=min_dist,
                    contamination=0.1,
                )
                pass_labels = detector.fit_predict(spectra_working)
                pass_outlier_local = np.where(pass_labels == -1)[0]

                # Map local row indices → original array indices
                pass_outlier_orig = [active_indices[j]
                                     for j in pass_outlier_local]

                # Resolve physical Spectrum IDs via MetadataMap
                spec_ids_removed = [
                    metadata_map[orig_i]["spec_no"]
                    for orig_i in pass_outlier_orig
                ]
                angles_removed = [
                    metadata_map[orig_i]["angle"]
                    for orig_i in pass_outlier_orig
                ]

                print(
                    f"[Phase 6] Pass {pass_num}: "
                    f"Removing Spectrum IDs {spec_ids_removed} "
                    f"(array indices {pass_outlier_orig}, "
                    f"angles {[f'{a:.2f}' if np.isfinite(a) else 'N/A' for a in angles_removed]}°)"
                )

                if pass_num == 1:
                    embedding_before = detector.embedding_coords_.copy()
                    labels_before = pass_labels.copy()

                cumulative_outlier_indices.extend(pass_outlier_orig)

                if not pass_outlier_orig:
                    print(
                        f"[Phase 6] Pass {pass_num}: no outliers found — "
                        f"stopping iterative removal."
                    )
                    break

                # Remove outliers from working arrays for next pass
                keep_mask = np.ones(spectra_working.shape[0], dtype=bool)
                keep_mask[pass_outlier_local] = False
                spectra_working = spectra_working[keep_mask]
                active_indices = [active_indices[j]
                                  for j in range(len(active_indices))
                                  if keep_mask[j]]

            # Build full-length labels array from cumulative removals
            outlier_idx = np.array(sorted(set(cumulative_outlier_indices)),
                                   dtype=np.intp)
            n_outliers = len(outlier_idx)
            pct_remaining = 100.0 * (n_total - n_outliers) / n_total

            # Build full-length label vector (−1 = outlier, 0 = inlier)
            labels = np.zeros(n_total, dtype=np.intp)
            labels[outlier_idx] = -1

            if embedding_before is None:
                # No passes ran (e.g. < 4 spectra from the start)
                embedding_before = np.zeros((n_total, 2))
                labels_before = labels.copy()

            print(
                f"[Phase 6] Total Spectra Removed: {n_outliers} / "
                f"Original Population: {n_total} "
                f"({pct_remaining:.1f}% remaining)"
            )

            # generate masking summary plot (spectrum index vs counts)
            fig_dir = getattr(ic, "figSavePath", None)
            if n_outliers > 0 and fig_dir is not None:
                try:
                    total_counts = np.sum(spectra, axis=1)
                    fig, ax = plt.subplots()
                    idxs = np.arange(n_total)
                    ax.plot(idxs, total_counts, '.', color='blue', label='counts')
                    ax.plot(idxs[outlier_idx], total_counts[outlier_idx], 'ro', label='masked')
                    ax.set_xlabel('Spectrum Index')
                    ax.set_ylabel('Total Counts')
                    ax.set_title(f"Masking summary — {ws_name}")
                    ax.legend()
                    fig.savefig(fig_dir / f"{ws_name}_masking_summary.pdf")
                    plt.close(fig)
                except Exception as exc:  # pragma: no cover
                    warnings.warn(f"Masking summary plot failed: {exc}")
            logger.info(
                "Phase 6 outlier detection: %d/%d outliers at %s (%.1f%% remaining)",
                n_outliers, n_total, outlier_idx.tolist(), pct_remaining,
            )

            diagnostics["outlier_indices"] = outlier_idx
            diagnostics["outlier_labels"] = labels

            # --- Masking: Map array indices → Mantid WorkspaceIndices ---
            if getattr(userCtr, "removeOutliers", False) and n_outliers > 0:
                ws_indices_to_mask = outlier_idx.tolist()

                if ws_name in mtd:
                    MaskDetectors(
                        Workspace=ws_name,
                        WorkspaceIndexList=ws_indices_to_mask,
                    )
                    masked_spec_ids = [metadata_map[i]["spec_no"]
                                       for i in ws_indices_to_mask]
                    print(
                        f"[Phase 6] Masked {n_outliers} outlier spectra "
                        f"in workspace '{ws_name}' — "
                        f"Spectrum IDs: {masked_spec_ids} "
                        f"(WorkspaceIndexList: {ws_indices_to_mask})"
                    )
                    logger.info(
                        "Phase 6 masked %d spectra in '%s': %s",
                        n_outliers, ws_name, ws_indices_to_mask,
                    )

                    # Re-extract clean NCP from masked workspace
                    ws_masked = mtd[ws_name]
                    spectra = ws_masked.extractY()
                    if spectra.shape[1] == ncp_total.shape[1] + 1:
                        spectra = spectra[:, :-1].copy()
                    print(
                        f"[Phase 6] Entering Pass 2 for Workspace: {ws_name}"
                    )
                    print(
                        f"[Phase 6] Pass 2: Confirming mask on "
                        f"{len(outlier_idx)} spectra for {ws_name}."
                    )

                    # Re-embed clean spectra for before/after plot
                    clean_mask = labels_before == 0
                    spectra_clean = spectra[clean_mask]
                    if spectra_clean.shape[0] >= 4:
                        detector_pass2 = HardwareOutlierDetector(
                            n_components=n_components,
                            n_neighbors=min(n_neighbors, spectra_clean.shape[0] - 1),
                            min_dist=min_dist,
                            contamination=0.1,
                        )
                        labels_after = detector_pass2.fit_predict(spectra_clean)
                        embedding_after = detector_pass2.embedding_coords_
                    else:
                        embedding_after = embedding_before[clean_mask]
                        labels_after = np.zeros(clean_mask.sum(), dtype=np.intp)
                else:
                    print(
                        f"[Phase 6] Workspace '{ws_name}' not in ADS. "
                        f"Outlier indices recorded but not masked."
                    )
                    embedding_after = embedding_before
                    labels_after = labels_before

                outlier_mask = labels == -1
            else:
                embedding_after = embedding_before
                labels_after = labels_before

            # ---- Determine DBSCAN cluster count for summary (if available) ----
            _n_clusters_for_summary = diagnostics.get("n_clusters_summary")

            fig_dir = getattr(ic, "figSavePath", None)
            if fig_dir is not None:
                summary_info: Dict[str, Any] = {
                    "n_total": n_total,
                    "n_outliers": n_outliers,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                }
                if _n_clusters_for_summary is not None:
                    summary_info["n_clusters"] = _n_clusters_for_summary
                try:
                    # Before/after side-by-side subplot
                    plot_outlier_before_after(
                        embedding_before, labels_before,
                        embedding_after, labels_after,
                        save_path=fig_dir / f"{ws_name}_umap_before_after.pdf",
                    )
                    # Standalone before-only scatter with summary table
                    plot_outlier_scatter(
                        embedding_before, labels_before,
                        save_path=fig_dir / f"{ws_name}_umap_outliers.pdf",
                        summary_info=summary_info,
                    )
                except Exception as exc:
                    warnings.warn(f"Phase 6 outlier plot failed: {exc}")

                # Fisher Discriminant Visualization
                try:
                    total_counts = np.sum(spectra, axis=1)
                    spectral_width = np.std(spectra, axis=1)
                    angles_arr = np.array(
                        [metadata_map[i]["angle"] for i in range(n_total)]
                    )
                    fisher_features = np.column_stack(
                        [angles_arr, total_counts, spectral_width]
                    )
                    # Replace NaN angles with column mean; fall back to 0 if all NaN
                    col_mean = np.nanmean(fisher_features[:, 0])
                    if not np.isfinite(col_mean):
                        col_mean = 0.0
                    nan_mask = ~np.isfinite(fisher_features[:, 0])
                    fisher_features[nan_mask, 0] = col_mean
                    plot_fisher_discriminant(
                        fisher_features, labels,
                        feature_names=["Angle", "Counts", "Width"],
                        save_path=fig_dir / f"{ws_name}_fisher_discriminant_distribution.pdf",
                    )
                except Exception as exc:
                    warnings.warn(f"Phase 6 Fisher discriminant plot failed: {exc}")

        # ---- Physics Clustering (DBSCAN) — pre-Global Fit ----
        if getattr(userCtr, "runPhysicsClustering", False):
            instrPars = loadInstrParsFileIntoArray(
                ic.InstrParsPath, ic.firstSpec, ic.lastSpec,
            )
            L_total = instrPars[:, 4] + instrPars[:, 5]
            theta = instrPars[:, 2]
            features = np.column_stack([L_total, theta])

            # If outliers were removed, exclude them from clustering
            if outlier_mask is not None:
                features_clean = features[~outlier_mask]
            else:
                features_clean = features

            clusterer = PhysicsTrendClusterer(eps=0.5, min_samples=3)
            cluster_labels = clusterer.fit_predict(features_clean)
            groups = clusterer.get_cluster_groups(cluster_labels)
            n_noise = int(np.sum(cluster_labels == -1))
            n_clusters = len(groups)

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
            # Make n_clusters available to the outlier scatter summary
            diagnostics["n_clusters_summary"] = n_clusters

            # --- Dynamic overwrite of nGlobalFitGroups ---
            # This executes *before* the global fit so that
            # groupDetectors() uses the DBSCAN count instead of the
            # legacy k-means default.
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
        weighted = bootstrap.compute_weighted_residuals(residuals)
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
