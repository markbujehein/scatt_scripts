from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging
import os

if TYPE_CHECKING:
    from .stream_manager import StreamManager

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit
from mantid.simpleapi import *
from scipy import optimize

from vesuvio_analysis.core_functions.fit_in_yspace import passDataIntoWS, replaceZerosWithNCP, switchFirstTwoAxis
from vesuvio_analysis.core_functions.iminuit_costs import NCPCostFunction
from vesuvio_analysis.core_functions.plot_style import set_thesis_style, figure_factory, set_print_options, COLORBLIND_PALETTE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numba acceleration toggle — set to False to revert to pure-NumPy paths
# ---------------------------------------------------------------------------
USE_NUMBA = True

try:
    from .numba_routines import calculateNcpSpec_numba as _calculateNcpSpec_numba
except ImportError:
    USE_NUMBA = False

# Format print output of arrays
set_print_options()

# ---------------------------------------------------------------------------
# Optimizer agreement check accumulator — cleared before each workspace fit
# and populated per-spectrum inside fitNcpToSingleSpec.
# ---------------------------------------------------------------------------
_optimizer_check_log: list = []

# Fit comparison data for visualization: stores (specNo, scipy_chi2, iminuit_chi2, scipy_pars, iminuit_pars)
_fit_comparison_log: list = []

_SEP_DOUBLE = "=" * 60
_SEP_SINGLE = "-" * 60

# iMinuit–Scipy agreement threshold for optimizer cross-validation
from vesuvio_analysis.core_functions.constants import _AGREEMENT_THRESHOLD


def _print_optimizer_agreement_summary() -> None:
    """Print an aggregated iMinuit–Scipy Agreement Check summary.

    Reads from the module-level ``_optimizer_check_log`` list which is
    populated per-spectrum inside ``fitNcpToSingleSpec``.  Should be
    called immediately after ``fitNcpToArray`` completes.
    """
    if not _optimizer_check_log:
        return
    max_diff = max(max(c, p) for c, p in _optimizer_check_log)
    n_fail = sum(
        1 for c, p in _optimizer_check_log
        if c > _AGREEMENT_THRESHOLD or p > _AGREEMENT_THRESHOLD
    )
    n_total = len(_optimizer_check_log)
    gate_pass = n_fail == 0
    status = "PASS" if gate_pass else "FAIL"
    print(f"\n{_SEP_SINGLE}")
    print(
        f"  iMinuit\u2013SciPy Agreement Check: {status} "
        f"(Max Difference: {max_diff * 100:.2f}% | "
        f"Tolerance: {_AGREEMENT_THRESHOLD * 100:.2f}%)"
    )
    print(f"  Spectra checked: {n_total}, Failed: {n_fail}")

    # Fail-safe: identify which physical parameters are causing
    # systematic divergence across multiple spectra.
    if not gate_pass and _fit_comparison_log:
        _print_per_parameter_divergence_summary()

    print(_SEP_SINGLE)


def _print_per_parameter_divergence_summary() -> None:
    """Print a breakdown of which physical parameters diverge most often.

    Analyses ``_fit_comparison_log`` entries that contain per-parameter
    relative differences, counts how often each named parameter exceeds
    the 1% threshold, and reports the top offenders.  This helps
    diagnose whether the local-minimum trap is caused by a specific
    physical quantity (e.g. $\\sigma$ vs $A$).
    """
    from collections import Counter
    param_fail_counts: Counter = Counter()
    param_worst_diffs: Dict[str, float] = {}

    for entry in _fit_comparison_log:
        par_rel = entry.get("par_rel_diff")
        par_names = entry.get("par_names")
        if par_rel is None or par_names is None:
            continue
        for k, (name, diff) in enumerate(zip(par_names, par_rel)):
            if diff > _AGREEMENT_THRESHOLD:
                param_fail_counts[name] += 1
                if name not in param_worst_diffs or diff > param_worst_diffs[name]:
                    param_worst_diffs[name] = diff

    if param_fail_counts:
        print("  Per-parameter divergence summary (top offenders):")
        for name, count in param_fail_counts.most_common(6):
            worst = param_worst_diffs[name]
            print(f"    {name}: failed in {count} spectra (worst: {worst * 100:.1f}%)")
    else:
        print("  (No per-parameter breakdown available.)")


def _plot_optimizer_comparison(ic: Any) -> None:
    """Plot iMinuit vs scipy chi-squared and parameter differences.

    Creates a diagnostic figure showing:
    1. Chi-squared comparison across spectra
    2. Parameter-wise relative differences for worst-case spectra

    Args:
        ic: Initial-conditions object with figSavePath.
    """
    if not _fit_comparison_log or ic.runningSampleWS:
        return

    set_thesis_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    spec_nos = [d['specNo'] for d in _fit_comparison_log]
    scipy_chi2s = [d['scipy_chi2'] for d in _fit_comparison_log]
    iminuit_chi2s = [d['iminuit_chi2'] for d in _fit_comparison_log]
    
    # Plot 1: Chi-squared comparison
    ax = axes[0]
    ax.scatter(spec_nos, scipy_chi2s, label='SciPy', alpha=0.6, s=30)
    ax.scatter(spec_nos, iminuit_chi2s, label='iMinuit', alpha=0.6, s=30)
    ax.set_xlabel('Spectrum Number')
    ax.set_ylabel('χ²')
    ax.set_title('SciPy vs iMinuit: Chi-squared')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Relative chi-squared difference
    ax = axes[1]
    chi2_rel_diffs = [
        abs(s - m) / max(abs(s), 1e-10) if s != 0 else 0
        for s, m in zip(scipy_chi2s, iminuit_chi2s)
    ]
    colors = ['red' if d > _AGREEMENT_THRESHOLD else 'green' for d in chi2_rel_diffs]
    ax.bar(range(len(chi2_rel_diffs)), [d*100 for d in chi2_rel_diffs], color=colors, alpha=0.7)
    ax.axhline(_AGREEMENT_THRESHOLD * 100, color='black', linestyle='--', label='2.5% Threshold')
    ax.set_xlabel('Spectrum Index')
    ax.set_ylabel('Relative Difference (%)')
    ax.set_title('Chi-squared Relative Difference (SciPy vs iMinuit)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    try:
        fileName = f"{ic.name}_Optimizer_Comparison.pdf"
        savePath = ic.figSavePath / fileName
        plt.savefig(savePath, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {fileName} to {ic.figSavePath}")
    except Exception as e:
        logger.warning(f"Failed to save optimizer comparison plot: {e}")
    
    plt.close(fig)


def iterativeFitForDataReduction(
    ic: Any,
    stream_manager: Optional["StreamManager"] = None,
) -> Tuple[Any, "resultsObject"]:
    """Run the iterative NCP fitting and optional MS/GC correction loop.

    This is the core data-reduction routine.  It loads and crops the
    raw/empty workspaces, then iterates ``noOfMSIterations + 1`` times:

    1. Fit the Neutron Compton Profile (NCP) to every spectrum via
       ``fitNcpToWorkspace``.
    2. Compute mean widths and intensity ratios (with sigma-clipping).
    3. Unless on the final iteration, apply multiple-scattering and/or
       gamma-background corrections and subtract them from the
       workspace for the next iteration.

    Expects the Mantid AnalysisDataService to be available.  At
    completion, the corrected workspace for the last iteration and all
    intermediate workspaces (``ic.name + "0"`` … ``ic.name + str(N)``)
    are present in ``mtd``.

    Args:
        ic: A completed ``BackwardInitialConditions`` or
            ``ForwardInitialConditions`` object (after
            ``completeICFromInputs``).
        stream_manager: Optional :class:`StreamManager` for capturing
            intermediate data streams.  When ``None`` (default),
            no additional persistence is performed.

    Returns:
        A 2-tuple ``(wsFinal, fittingResults)`` where *wsFinal* is the
        Mantid workspace from the last iteration and *fittingResults* is
        a ``resultsObject`` containing all per-iteration arrays.
    """
    from .stream_manager import DataLevel

    createTableInitialParameters(ic)

    # When VESUVIO_RUNNING_TEST=1 (active inside Codespaces and the fast-track
    # CI gate) cap MS correction iterations to 0 so each run completes quickly.
    # This mirrors the runningTest fast-track flag described in PR #35 and
    # allows high-frequency code validation without a full production run.
    if os.environ.get("VESUVIO_RUNNING_TEST", "0") == "1":
        prev_iterations = getattr(ic, "noOfMSIterations", None)
        ic.noOfMSIterations = 0
        logger.info(
            "VESUVIO_RUNNING_TEST=1: overriding ic.noOfMSIterations from %s to 0",
            prev_iterations,
        )

    initialWs = loadRawAndEmptyWsFromUserPath(ic)  # Do this before alternative bootstrap to extract name()   

    if ic.runningSampleWS:
        initialWs = RenameWorkspace(InputWorkspace=ic.sampleWS, OutputWorkspace=initialWs.name())

    cropedWs = cropAndMaskWorkspace(ic, initialWs)
    wsToBeFitted = CloneWorkspace(InputWorkspace=cropedWs, OutputWorkspace=cropedWs.name()+"0")

    # L0 — capture raw counts before any corrections
    if stream_manager is not None:
        dataX_raw, dataY_raw, dataE_raw = extractWS(wsToBeFitted)
        stream_manager.capture("dataX", dataX_raw, DataLevel.RAW, domain="tof")
        stream_manager.capture("dataY", dataY_raw, DataLevel.RAW, domain="tof")
        stream_manager.capture("dataE", dataE_raw, DataLevel.RAW, domain="tof")
        stream_manager.set_metadata("masses", np.array(ic.masses))
        stream_manager.set_metadata("n_iterations", ic.noOfMSIterations + 1)

    # When running a smoke test, cap MS/GC iterations at 1 for speed.
    # Override ic.noOfMSIterations so all downstream naming/lookups are consistent.
    if getattr(ic, "runningTest", False):
        ic.noOfMSIterations = min(1, ic.noOfMSIterations)

    for iteration in range(ic.noOfMSIterations + 1):
        # Workspace from previous iteration
        wsToBeFitted = mtd[ic.name+str(iteration)]

        ncpTotal = fitNcpToWorkspace(ic, wsToBeFitted)
        
        mWidths, stdWidths, mIntRatios, stdIntRatios = extractMeans(wsToBeFitted.name(), ic)
        createMeansAndStdTableWS(wsToBeFitted.name(), ic, mWidths, stdWidths, mIntRatios, stdIntRatios)

        # L3 — capture NCP total for each iteration
        if stream_manager is not None:
            stream_manager.capture(
                "ncp_total", ncpTotal, DataLevel.FINAL_PHYSICS,
                domain="tof", iteration=iteration,
            )

        # When last iteration, skip MS and GC
        if iteration == ic.noOfMSIterations: break 

        # Replace zero columns (bins) with ncp total fit
        # If ws has no zero column, then remains unchanged.
        # Refreshed at every iteration so MS/GC use the latest NCP estimate.
        wsNCPM = replaceZerosWithNCP(mtd[ic.name], ncpTotal)

        CloneWorkspace(InputWorkspace=ic.name, OutputWorkspace="tmpNameWs")

        if ic.MSCorrectionFlag:
            wsMS = createWorkspacesForMSCorrection(ic, mWidths, mIntRatios, wsNCPM)
            # L1 — capture MS correction profile
            if stream_manager is not None:
                stream_manager.capture(
                    "ms", wsMS.extractY(),
                    DataLevel.CORRECTION_COMPONENTS,
                    domain="tof", iteration=iteration,
                )
            Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=wsMS, OutputWorkspace="tmpNameWs")

        if ic.GammaCorrectionFlag:  
            wsGC = createWorkspacesForGammaCorrection(ic, mWidths, mIntRatios, wsNCPM)
            # L1 — capture Gamma correction profile
            if stream_manager is not None:
                stream_manager.capture(
                    "gamma", wsGC.extractY(),
                    DataLevel.CORRECTION_COMPONENTS,
                    domain="tof", iteration=iteration,
                )
            Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=wsGC, OutputWorkspace="tmpNameWs")

        remaskValues(ic.name, "tmpNameWs")    # Masks cols in the same place as in ic.name
        RenameWorkspace(InputWorkspace="tmpNameWs", OutputWorkspace=ic.name+str(iteration+1))

        # L2 — capture corrected signal after MS/GC subtraction
        if stream_manager is not None:
            wsCorrected = mtd[ic.name+str(iteration+1)]
            stream_manager.capture(
                "corrected", wsCorrected.extractY(),
                DataLevel.INTERMEDIATE_CORRECTED,
                domain="tof", iteration=iteration,
            )

    # L3 — capture final corrected signal
    wsFinal = mtd[ic.name+str(ic.noOfMSIterations)]
    if stream_manager is not None:
        stream_manager.capture(
            "corrected_final", wsFinal.extractY(),
            DataLevel.FINAL_PHYSICS, domain="tof",
        )

    fittingResults = resultsObject(ic)
    fittingResults.save()
    return wsFinal, fittingResults


def remaskValues(wsName: str, wsToMaskName: str) -> None:
    """Re-apply column masks after an MS or gamma-background correction.

    Copies the zero-column pattern from the pre-correction workspace
    onto the post-correction workspace so that masked bins remain
    masked.

    Expects workspaces ``wsName`` and ``wsToMaskName`` to be present
    in the AnalysisDataService.

    Args:
        wsName: Name of the workspace **before** correction, used as the
            masking reference.
        wsToMaskName: Name of the workspace **after** correction, whose
            dataY columns are zeroed where the reference has zeros.
    """
    ws = mtd[wsName]
    dataX, dataY, dataE = extractWS(ws)
    mask = np.all(dataY==0, axis=0)

    wsM = mtd[wsToMaskName]
    dataXM, dataYM, dataEM = extractWS(wsM)
    dataYM[:, mask] = 0
    if np.all(dataE==0): dataEM = np.zeros(dataEM.shape)
    
    passDataIntoWS(dataXM, dataYM, dataEM, wsM)
    return


def createTableInitialParameters(ic: Any) -> None:
    """Create a Mantid TableWorkspace listing the starting fit parameters.

    Writes a table named ``ic.name + "_Initial_Parameters"`` into the
    AnalysisDataService, with columns for mass, initial widths/bounds,
    initial intensities/bounds, and initial centres/bounds.

    Args:
        ic: Completed initial-conditions object whose ``masses``,
            ``initPars``, and ``bounds`` arrays are read.
    """

    print("\nRUNNING ", ic.modeRunning, " SCATTERING.\n")
    if ic.modeRunning == "BACKWARD":
        print(f"\nH ratio to mass with idx={ic.massIdx}: {ic.HToMassIdxRatio}\n")

    meansTableWS = CreateEmptyTableWorkspace(OutputWorkspace=ic.name+"_Initial_Parameters")
    meansTableWS.addColumn(type='float', name="Mass")
    meansTableWS.addColumn(type='float', name="Initial Widths")
    meansTableWS.addColumn(type='str', name="Bounds Widths")
    meansTableWS.addColumn(type='float', name="Initial Intensities")
    meansTableWS.addColumn(type='str', name="Bounds Intensities")
    meansTableWS.addColumn(type='float', name="Initial Centers")
    meansTableWS.addColumn(type='str', name="Bounds Centers")

    print("\nCreated Table with Initial Parameters:")
    for m, iw, bw, ii, bi, inc, bc in zip(ic.masses.astype(float), ic.initPars[1::3], ic.bounds[1::3], ic.initPars[0::3], ic.bounds[0::3], ic.initPars[2::3], ic.bounds[2::3]):
        meansTableWS.addRow([m, iw, str(bw), ii, str(bi), inc, str(bc)])
        print("\nMass: ", m)
        print(f"{'Initial Intensity:':>20s} {ii:<8.4f} Bounds: {bi}")
        print(f"{'Initial Width:':>20s} {iw:<8.4f} Bounds: {bw}")
        print(f"{'Initial Center:':>20s} {inc:<8.4f} Bounds: {bc}")
    print("\n")    


def loadRawAndEmptyWsFromUserPath(ic: Any) -> Any:
    """Load cached raw and empty Nexus files, rebin, scale, and subtract.

    Loads the raw workspace from ``ic.userWsRawPath``, rebins to
    ``ic.tofBinning``, and scales by ``ic.scaleRaw``.  If
    ``ic.subEmptyFromRaw`` is ``True``, the empty workspace is loaded,
    rebinned, scaled, and subtracted.  Sum spectra workspaces are
    created for diagnostic plots.

    Expects the Nexus files to already exist (cached by
    ``ICHelpers.saveWSFromLoadVesuvio``).

    Args:
        ic: Completed initial-conditions object with ``userWsRawPath``,
            ``userWsEmptyPath``, ``tofBinning``, ``scaleRaw``,
            ``scaleEmpty``, and ``subEmptyFromRaw``.

    Returns:
        The Mantid workspace ready for cropping, named
        ``ic.name + "uncroped_unmasked"``.
    """

    print('\nLoading local workspaces ...\n')
    Load(Filename=str(ic.userWsRawPath), OutputWorkspace=ic.name+"raw")
    Rebin(InputWorkspace=ic.name+'raw', Params=ic.tofBinning,
          OutputWorkspace=ic.name+'raw')

    assert (type(ic.scaleRaw)==float) | (type(ic.scaleRaw)==int), "Scaling factor of raw ws needs to be float or int."
    Scale(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw', Factor=str(ic.scaleRaw))
 
    SumSpectra(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw'+'_sum')
    wsToBeFitted = CloneWorkspace(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+"uncroped_unmasked")

    # if ic.mode=="DoubleDifference":
    if ic.subEmptyFromRaw:
        Load(Filename=str(ic.userWsEmptyPath), OutputWorkspace=ic.name+"empty")
        Rebin(InputWorkspace=ic.name+'empty', Params=ic.tofBinning,
            OutputWorkspace=ic.name+'empty')

        assert (type(ic.scaleEmpty)==float) | (type(ic.scaleEmpty)==int), "Scaling factor of empty ws needs to be float or int"
        Scale(InputWorkspace=ic.name+'empty', OutputWorkspace=ic.name+'empty', Factor=str(ic.scaleEmpty))

        SumSpectra(InputWorkspace=ic.name+'empty', OutputWorkspace=ic.name+'empty'+'_sum')
        
        wsToBeFitted = Minus(LHSWorkspace=ic.name+'raw', RHSWorkspace=ic.name+'empty',
                            OutputWorkspace=ic.name+"uncroped_unmasked")
    return wsToBeFitted


def cropAndMaskWorkspace(ic: Any, ws: Any) -> Any:
    """Crop the workspace to the requested spectrum range and apply masks.

    Uses Mantid ``CropWorkspace`` and ``MaskDetectors``.  Any resonance
    peak TOF range specified by ``ic.maskTOFRange`` is zeroed out via
    ``maskBinsWithZeros``.

    Args:
        ic: Completed initial-conditions object with ``firstSpec``,
            ``lastSpec``, ``maskedDetectorIdx``, and ``maskTOFRange``.
        ws: Input Mantid workspace (uncropped/unmasked).

    Returns:
        The cropped and masked Mantid workspace, named after ``ic.name``.
    """
    # Read initial Spectrum number
    wsFirstSpec = ws.getSpectrumNumbers()[0]
    assert ic.firstSpec >= wsFirstSpec, "Can't crop workspace, firstSpec < first spectrum in workspace."
    
    initialIdx = ic.firstSpec - wsFirstSpec
    lastIdx = ic.lastSpec - wsFirstSpec
    
    newWsName = ws.name().split("uncroped")[0]  # Retrieve original name
    wsCrop = CropWorkspace(
        InputWorkspace=ws, 
        StartWorkspaceIndex=initialIdx, EndWorkspaceIndex=lastIdx, 
        OutputWorkspace=newWsName
        )

    maskBinsWithZeros(wsCrop, ic)    # Used to mask resonance peaks

    MaskDetectors(Workspace=wsCrop, WorkspaceIndexList=ic.maskedDetectorIdx)
    return wsCrop


def maskBinsWithZeros(ws: Any, IC: Any) -> None:
    """Zero out dataY bins inside a specified TOF range.

    Sets dataY to zero for all spectra in the given TOF window,
    leaving dataE unchanged (errors are needed downstream).  Used to
    mask resonance peaks.  Does nothing if ``IC.maskTOFRange`` is
    ``None``.

    Args:
        ws: Mantid workspace whose dataY is modified in-place.
        IC: Initial-conditions object with ``maskTOFRange`` (a string
            ``"start, end"`` or ``None``).
    """

    if IC.maskTOFRange==None:     # Masked TOF bins not found, skip
        return

    dataX, dataY, dataE = extractWS(ws)
    start, end = [int(s) for s in IC.maskTOFRange.split(",")]
    assert start <= end, "Start value for masking needs to be smaller or equal than end."
    mask = (dataX >= start) & (dataX <= end)    # TOF region to mask

    dataY[mask] = 0     

    passDataIntoWS(dataX, dataY, dataE, ws)
    return 


def fitNcpToWorkspace(IC: Any, ws: Any) -> np.ndarray:
    """Fit the Neutron Compton Profile to every spectrum in a workspace.

    Prepares kinematic and resolution arrays, then fits each spectrum
    independently via ``fitNcpToSingleSpec`` (scipy SLSQP + parallel
    iMinuit cross-validation).  The best-fit parameters are stored in a
    TableWorkspace and the synthetic NCP profiles are written back as
    separate workspaces.

    **Workspace lifecycle:**

    * ``ws`` (``InputWorkspace``) — read-only; data is extracted via
      :func:`extractWS` and the object is not modified in-place.
    * ``ws.name() + "_Best_Fit_NCP_Parameters"`` — **created** as a new
      TableWorkspace in the AnalysisDataService.
    * ``ws.name() + "_TOF_Fitted_Profiles"`` — **created** as a new
      MatrixWorkspace containing the synthetic total NCP.
    * ``ws.name() + "_TOF_Fitted_Profile_<i>"`` — **created**, one per
      mass *i*.
    * ``ws.name() + "_Sum"`` — **created** via ``SumSpectra``.

    Args:
        IC: Completed initial-conditions object with fit parameters,
            masses, bounds, constraints, and path information.
        ws: Mantid workspace containing the TOF data to fit, shape
            ``(n_spectra, n_bins)``.  Must already be registered in the
            ``mtd`` AnalysisDataService.

    Returns:
        ``ncpTotal``, the summed NCP over all masses for each spectrum,
        shape ``(n_spectra, n_bins)``.

    Raises:
        AssertionError: If ``ws`` is not present in ``mtd`` on entry, or
            if the output NCP workspace is missing from ``mtd`` on exit.
    """
    # --- Entry guard: workspace must be in the AnalysisDataService ---
    assert ws.name() in mtd, (
        f"fitNcpToWorkspace: InputWorkspace '{ws.name()}' not found in mtd. "
        "Ensure the workspace was created before calling this function."
    )

    dataX, dataY, dataE = extractWS(ws)
    if IC.runHistData:     # Converts point data from workspaces to histogram data
        dataY, dataX, dataE = histToPointData(dataY, dataX, dataE)      


    resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(IC, dataX)
    
    print("\nFitting NCP:\n")

    _optimizer_check_log.clear()
    _fit_comparison_log.clear()
    arrFitPars = fitNcpToArray(IC, dataY, dataE, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)
    _print_optimizer_agreement_summary()
    _plot_optimizer_comparison(IC)
    createTableWSForFitPars(ws.name(), IC.noOfMasses, arrFitPars)
    arrBestFitPars = arrFitPars[:, 1:-2]
    ncpForEachMass, ncpTotal = calculateNcpArr(IC, arrBestFitPars, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)
    ncpSumWSs = createNcpWorkspaces(ncpForEachMass, ncpTotal, ws, IC)

    # --- Exit guard: output NCP workspace must now exist in mtd ---
    _ncp_ws_name = ws.name() + "_TOF_Fitted_Profiles"
    assert _ncp_ws_name in mtd, (
        f"fitNcpToWorkspace: OutputWorkspace '{_ncp_ws_name}' was not "
        "created in mtd.  NCP calculation may have failed silently."
    )

    wsDataSum = SumSpectra(InputWorkspace=ws, OutputWorkspace=ws.name()+"_Sum")
    plotSumNCPFits(wsDataSum, *ncpSumWSs, IC)
    plotIndividualNCPFits(ncpSumWSs[1], wsDataSum, IC)
    return ncpTotal


def extractWS(ws: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract dataX, dataY, and dataE arrays from a Mantid workspace.

    Args:
        ws: A Mantid MatrixWorkspace.

    Returns:
        A 3-tuple ``(dataX, dataY, dataE)`` of NumPy arrays, each of
        shape ``(n_spectra, n_bins)`` (or ``n_bins + 1`` for histogram
        dataX).
    """
    return ws.extractX(), ws.extractY(), ws.extractE()


def histToPointData(
    dataY: np.ndarray, dataX: np.ndarray, dataE: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert histogram data to point data by centring bins.

    Removes the last column and shifts dataX to bin centres.  Used only
    for regression comparison with the original (histogram-based) results.

    Args:
        dataY: Histogram counts, shape ``(n_spectra, n_bins)``.
        dataX: Bin edges, shape ``(n_spectra, n_bins)`` or ``n_bins + 1``.
        dataE: Errors, same shape as *dataY*.

    Returns:
        A 3-tuple ``(dataYp, dataXp, dataEp)`` with the last column
        removed and dataX shifted to bin centres.
    """

    histWidths = dataX[:, 1:] - dataX[:, :-1]
    assert np.min(histWidths) == np.max(histWidths), "Histogram widhts need to be the same length"
    
    dataYp = dataY[:, :-1]
    dataEp = dataE[:, :-1] 
    dataXp = dataX[:, :-1] + histWidths[0, 0]/2 
    return dataYp, dataXp, dataEp


def prepareFitArgs(
    ic: Any, dataX: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build all arrays required by the NCP fitting loop.

    Loads instrument parameters, computes resolution parameters,
    kinematics (v0, E0, deltaE, deltaQ), and y-spaces for each mass.
    All returned arrays are reshaped so that the first axis is the
    spectrum index.

    Args:
        ic: Completed initial-conditions object with ``InstrParsPath``,
            ``firstSpec``, ``lastSpec``, and ``masses``.
        dataX: TOF bin centres, shape ``(n_spectra, n_bins)``.

    Returns:
        A 4-tuple ``(resolutionPars, instrPars, kinematicArrays,
        ySpacesForEachMass)`` where each array has its leading axis
        indexed per spectrum.
    """

    instrPars = loadInstrParsFileIntoArray(ic.InstrParsPath, ic.firstSpec, ic.lastSpec)       
    resolutionPars = loadResolutionPars(instrPars)                                   

    v0, E0, delta_E, delta_Q = calculateKinematicsArrays(dataX, instrPars)   
    kinematicArrays = np.array([v0, E0, delta_E, delta_Q])
    ySpacesForEachMass = convertDataXToYSpacesForEachMass(dataX, ic.masses, delta_Q, delta_E)        
    
    kinematicArrays = reshapeArrayPerSpectrum(kinematicArrays)
    ySpacesForEachMass = reshapeArrayPerSpectrum(ySpacesForEachMass)
    return resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass


def loadInstrParsFileIntoArray(
    InstrParsPath: str, firstSpec: int, lastSpec: int
) -> np.ndarray:
    """Load the instrument parameter file and select the requested spectra.

    The file is expected to have a header row followed by numeric data
    with columns ``[spec, det, angle, T0, L0, L1]``.

    Args:
        InstrParsPath: Path to the ``.par`` instrument parameter file.
        firstSpec: First spectrum number to include.
        lastSpec: Last spectrum number to include.

    Returns:
        Instrument parameters array, shape ``(n_selected_spectra, 6)``.
    """

    data = np.loadtxt(InstrParsPath, dtype=str)[1:].astype(float)

    spectra = data[:, 0]
    select_rows = np.where((spectra >= firstSpec) & (spectra <= lastSpec))
    instrPars = data[select_rows]
    return instrPars


def loadResolutionPars(instrPars: np.ndarray) -> np.ndarray:
    """Build the detector resolution parameter matrix from instrument data.

    Assigns Gaussian and Lorentzian energy-resolution widths, TOF jitter,
    angular uncertainty, and flight-path uncertainties based on whether
    each spectrum is backward (< 135) or forward (>= 135).

    Args:
        instrPars: Instrument parameters, shape ``(n_spectra, 6)``.
            Column 0 is the spectrum number.

    Returns:
        Resolution parameters, shape ``(n_spectra, 6)`` with columns
        ``[dE1, dTOF, dTheta, dL0, dL1, dE1_lorz]``.
    """
    spectrums = instrPars[:, 0] 
    L = len(spectrums)
    # For spec no below 135, back scattering detectors, mode is double difference
    # For spec no 135 or above, front scattering detectors, mode is single difference
    dE1 = np.where(spectrums < 135, 88.7, 73)       #meV, STD
    dE1_lorz = np.where(spectrums < 135, 40.3, 24)  #meV, HFHM
    dTOF = np.repeat(0.37, L)      #us
    dTheta = np.repeat(0.016, L)   #rad
    dL0 = np.repeat(0.021, L)      #meters
    dL1 = np.repeat(0.023, L)      #meters
    
    resolutionPars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose() 
    return resolutionPars 


def calculateKinematicsArrays(
    dataX: np.ndarray, instrPars: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute neutron kinematics from TOF data and instrument geometry.

    Calculates initial velocity *v0*, initial energy *E0*, energy
    transfer *deltaE*, and momentum transfer *deltaQ* for every
    (spectrum, bin) pair.

    Physics: ``v0 = vf * L0 / (vf * t_us - L1)``,
    ``E0 = (v0 / en_to_vel)^2``, ``deltaE = E0 - Ef``,
    ``deltaQ = sqrt(2 mN (E0 + Ef - 2 sqrt(E0 Ef) cos(theta)) / hbar^2)``.

    Args:
        dataX: TOF values in microseconds, shape ``(n_spectra, n_bins)``.
        instrPars: Instrument parameters, shape ``(n_spectra, 6)`` with
            columns ``[det, plick, angle, T0, L0, L1]``.

    Returns:
        A 4-tuple ``(v0, E0, delta_E, delta_Q)`` each of shape
        ``(n_spectra, n_bins)``.
    """

    mN, Ef, en_to_vel, vf, hbar = loadConstants()    
    det, plick, angle, T0, L0, L1 = np.hsplit(instrPars, 6)     #each is of len(dataX)
    t_us = dataX - T0                                           #T0 is electronic delay due to instruments
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  np.square( v0 / en_to_vel )            #en_to_vel is a factor used to easily change velocity to energy and vice-versa
    
    delta_E = E0 - Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt(E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q              #shape(no of spectrums, no of bins)


def reshapeArrayPerSpectrum(A: np.ndarray) -> np.ndarray:
    """Transpose the first two axes so the leading axis is per-spectrum.

    Used to rearrange arrays computed per-quantity into arrays indexed by
    spectrum for the per-spectrum fitting loop.

    Args:
        A: Array with shape ``(n_quantities, n_spectra, n_bins)``.

    Returns:
        Array with shape ``(n_spectra, n_quantities, n_bins)``.
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def convertDataXToYSpacesForEachMass(
    dataX: np.ndarray,
    masses: np.ndarray,
    delta_Q: np.ndarray,
    delta_E: np.ndarray,
) -> np.ndarray:
    """Convert TOF data to y-space for each atomic mass via the y-scaling relation.

    Applies the y-scaling equation:
    ``y = M / (hbar^2 * deltaQ) * (deltaE - E_recoil)``
    where ``E_recoil = (hbar * deltaQ)^2 / (2 * M)``.

    Args:
        dataX: TOF values, shape ``(n_spectra, n_bins)`` (unused in
            computation but broadcast for alignment).
        masses: Atomic masses in a.m.u., shape ``(n_masses,)``.
        delta_Q: Momentum transfer, shape ``(n_spectra, n_bins)``.
        delta_E: Energy transfer, shape ``(n_spectra, n_bins)``.

    Returns:
        y-spaces, shape ``(n_masses, n_spectra, n_bins)``.
    """
    
    # Prepare arrays to broadcast
    dataX = dataX[np.newaxis, :, :]
    delta_Q = delta_Q[np.newaxis, :, :]
    delta_E = delta_E[np.newaxis, :, :]  

    mN, Ef, en_to_vel, vf, hbar = loadConstants()
    masses = masses.reshape(masses.size, 1, 1)

    energyRecoil = np.square( hbar * delta_Q ) / 2. / masses              
    ySpacesForEachMass = masses / hbar**2 /delta_Q * (delta_E - energyRecoil)    #y-scaling  
    return ySpacesForEachMass


def fitNcpToArray(
    ic: Any,
    dataY: np.ndarray,
    dataE: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
    kinematicArrays: np.ndarray,
    ySpacesForEachMass: np.ndarray,
) -> np.ndarray:
    """Fit the NCP model to every spectrum in a 2-D data array.

    Iterates over spectra and calls ``fitNcpToSingleSpec`` for each.
    The result array has one row per spectrum with columns:
    ``[specNo, *fitPars, normChi2, nIter]``.

    Args:
        ic: Completed initial-conditions object.
        dataY: Observed counts, shape ``(n_spectra, n_bins)``.
        dataE: Errors on counts, same shape as *dataY*.
        resolutionPars: Per-spectrum resolution, shape
            ``(n_spectra, 6)``.
        instrPars: Per-spectrum instrument parameters, shape
            ``(n_spectra, 6)``.
        kinematicArrays: Per-spectrum ``[v0, E0, deltaE, deltaQ]``,
            shape ``(n_spectra, 4, n_bins)``.
        ySpacesForEachMass: Per-spectrum y-spaces, shape
            ``(n_spectra, n_masses, n_bins)``.

    Returns:
        Best-fit parameter array, shape
        ``(n_spectra, 3 * n_masses + 3)``.
    """

    arrFitPars = np.zeros((len(dataY), len(ic.initPars)+3))
    for i in range(len(dataY)):

        specFitPars = fitNcpToSingleSpec(
            dataY[i],
            dataE[i],
            ySpacesForEachMass[i],
            resolutionPars[i],
            instrPars[i],
            kinematicArrays[i],
            ic
            ) 

        arrFitPars[i] = specFitPars

        if np.all(specFitPars==0):
            print("Skipped spectra.")
        else:
            print(f"Fitted spectra {int(specFitPars[0]):3}")
    
    assert ~np.all(arrFitPars==0), "Either Fits are all zero or assignment of fitting not working"
    return arrFitPars


def createTableWSForFitPars(
    wsName: str, noOfMasses: int, arrFitPars: np.ndarray
) -> None:
    """Store NCP fit parameters in a Mantid TableWorkspace.

    Creates a table named ``wsName + "_Best_Fit_NCP_Parameters"`` with
    columns for spectrum index, per-mass intensity/width/centre, normalised
    chi-squared, and number of optimiser iterations.

    Args:
        wsName: Base name of the workspace that was fitted.
        noOfMasses: Number of atomic masses in the fit.
        arrFitPars: Parameter array from ``fitNcpToArray``, shape
            ``(n_spectra, 3 * n_masses + 3)``.
    """

    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsName+"_Best_Fit_NCP_Parameters")
    tableWS.setTitle("SCIPY Fit")
    tableWS.addColumn(type='float', name="Spec Idx")
    for i in range(int(noOfMasses)):
        tableWS.addColumn(type='float', name=f"Intensity {i}")
        tableWS.addColumn(type='float', name=f"Width {i}")
        tableWS.addColumn(type='float', name=f"Center {i}")
    tableWS.addColumn(type='float', name="Norm Chi2")
    tableWS.addColumn(type='float', name="No Iter")

    for row in arrFitPars:    # Pass array onto table ws
        tableWS.addRow(row)
    return 


def calculateNcpArr(
    ic: Any,
    arrBestFitPars: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
    kinematicArrays: np.ndarray,
    ySpacesForEachMass: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recalculate NCP arrays from the best-fit parameter matrix.

    Iterates over spectra and calls ``calculateNcpRow`` for each.

    Args:
        ic: Completed initial-conditions object.
        arrBestFitPars: Best-fit intensities/widths/centres, shape
            ``(n_spectra, 3 * n_masses)``.
        resolutionPars: Per-spectrum resolution, shape
            ``(n_spectra, 6)``.
        instrPars: Per-spectrum instrument parameters, shape
            ``(n_spectra, 6)``.
        kinematicArrays: Per-spectrum kinematics, shape
            ``(n_spectra, 4, n_bins)``.
        ySpacesForEachMass: Per-spectrum y-spaces, shape
            ``(n_spectra, n_masses, n_bins)``.

    Returns:
        A 2-tuple ``(allNcpForEachMass, allNcpTotal)`` where the first
        has shape ``(n_spectra, n_masses, n_bins)`` and the second
        ``(n_spectra, n_bins)``.
    """

    allNcpForEachMass = []
    for i in range(len(arrBestFitPars)):

        ncpForEachMass = calculateNcpRow(
            arrBestFitPars[i],
            ySpacesForEachMass[i], 
            resolutionPars[i], 
            instrPars[i], 
            kinematicArrays[i],
            ic
            )
            
        allNcpForEachMass.append(ncpForEachMass)

    allNcpForEachMass = np.array(allNcpForEachMass)
    allNcpTotal = np.sum(allNcpForEachMass, axis=1)        
    return allNcpForEachMass, allNcpTotal


def calculateNcpRow(
    initPars: np.ndarray,
    ySpacesForEachMass: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
    kinematicArrays: np.ndarray,
    ic: Any,
) -> np.ndarray:
    """Compute the NCP for each mass for a single spectrum.

    Delegates to ``calculateNcpSpec``.  Returns zeros when the input
    parameters are all zero (masked spectrum).

    Args:
        initPars: Flat fit parameters for this spectrum, shape
            ``(3 * n_masses,)``.
        ySpacesForEachMass: y-spaces for this spectrum, shape
            ``(n_masses, n_bins)``.
        resolutionPars: Resolution parameters for this spectrum, shape
            ``(6,)``.
        instrPars: Instrument parameters for this spectrum, shape
            ``(6,)``.
        kinematicArrays: Kinematics for this spectrum, shape
            ``(4, n_bins)``.
        ic: Completed initial-conditions object.

    Returns:
        NCP per mass, shape ``(n_masses, n_bins)``.
    """

    if np.all(initPars==0):  
        return np.zeros(ySpacesForEachMass.shape) 
    
    ncpForEachMass, ncpTotal = calculateNcpSpec(ic, initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)        
    return ncpForEachMass


def createNcpWorkspaces(
    ncpForEachMass: np.ndarray,
    ncpTotal: np.ndarray,
    ws: Any,
    ic: Any,
) -> Tuple[Any, List[Any]]:
    """Store NCP arrays as named Mantid workspaces.

    Creates a total-NCP workspace and one workspace per mass.  Both the
    full-spectra and summed-spectra variants are written to the
    AnalysisDataService.

    Args:
        ncpForEachMass: NCP per mass, shape
            ``(n_spectra, n_masses, n_bins)``.
        ncpTotal: Total NCP, shape ``(n_spectra, n_bins)``.
        ws: The data workspace (used for dataX and naming).
        ic: Completed initial-conditions object with
            ``maskedDetectorIdx``.

    Returns:
        A 2-tuple ``(wsTotNCPSum, wsMNCPSum)`` — the summed-spectra
        workspace for the total NCP and a list of summed-spectra
        workspaces, one per mass.
    """

    # Need to rearrage array of yspaces into seperate arrays for each mass
    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)

    # Use ws dataX to match with histogram data
    dataX = ws.extractX()[:, :ncpTotal.shape[1]]  # Make dataX match ncp shape automatically
    assert ncpTotal.shape == dataX.shape, "DataX and DataY in ws need to be the same shape."

    ncpTotWS = createWS(dataX, ncpTotal, np.zeros(dataX.shape), ws.name()+"_TOF_Fitted_Profiles")
    MaskDetectors(Workspace=ncpTotWS, WorkspaceIndexList=ic.maskedDetectorIdx)
    wsTotNCPSum = SumSpectra(InputWorkspace=ncpTotWS, OutputWorkspace=ncpTotWS.name()+"_Sum" )

    # Individual ncp workspaces
    wsMNCPSum = []
    for i, ncp_m in enumerate(ncpForEachMass):
        ncpMWS = createWS(dataX, ncp_m, np.zeros(dataX.shape), ws.name()+"_TOF_Fitted_Profile_"+str(i))
        MaskDetectors(Workspace=ncpMWS, WorkspaceIndexList=ic.maskedDetectorIdx)
        wsNCPSum = SumSpectra(InputWorkspace=ncpMWS, OutputWorkspace=ncpMWS.name()+"_Sum" )
        wsMNCPSum.append(wsNCPSum)
        
    return wsTotNCPSum, wsMNCPSum


def createWS(
    dataX: np.ndarray, dataY: np.ndarray, dataE: np.ndarray, wsName: str
) -> Any:
    """Create a Mantid MatrixWorkspace from NumPy arrays.

    Args:
        dataX: X-axis values, shape ``(n_spectra, n_bins)``.
        dataY: Y-axis values, same shape.
        dataE: Error values, same shape.
        wsName: Output workspace name in the AnalysisDataService.

    Returns:
        The created Mantid workspace.
    """

    ws = CreateWorkspace(
        DataX=dataX.flatten(),
        DataY=dataY.flatten(),
        DataE=dataE.flatten(),
        Nspec=len(dataY),
        OutputWorkspace=wsName
    )
    return ws


def plotSumNCPFits(wsDataSum: Any, wsTotNCPSum: Any, wsMNCPSum: List[Any], IC: Any) -> None:
    """Save a PDF plot comparing the summed data to the fitted NCP profiles.

    Skipped when running a bootstrap sample (``IC.runningSampleWS`` is
    ``True``).

    Args:
        wsDataSum: Summed-spectra data workspace.
        wsTotNCPSum: Summed-spectra total NCP workspace.
        wsMNCPSum: List of summed-spectra NCP workspaces, one per mass.
        IC: Completed initial-conditions object with ``masses``,
            ``runningSampleWS``, and ``figSavePath``.
    """

    if IC.runningSampleWS:   # Skip saving figure if running bootstrap
        return         

    set_thesis_style()
    fig, ax = figure_factory(subplot_kw={"projection": "mantid"})
    ax.errorbar(wsDataSum, "k.", label="Spectra")

    ax.plot(wsTotNCPSum, "r-", label="Total NCP")
    for m, wsNcp in zip(IC.masses, wsMNCPSum):
        ax.plot(wsNcp, label=f"NCP m={m}")
    
    ax.set_xlabel("TOF")
    ax.set_ylabel("Counts")
    ax.set_title("Sum of NCP fits")
    ax.legend()

    fileName = wsDataSum.name()+"_NCP_Fits.pdf"
    savePath = IC.figSavePath / fileName
    plt.savefig(savePath, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {fileName} to {IC.figSavePath}")
    plt.close(fig)
    return


def plotIndividualNCPFits(
    wsMNCPSum: List[Any], wsDataSum: Any, IC: Any
) -> None:
    """Save individual per-mass NCP diagnostic plots as PDFs.

    Generates one PDF per atomic species (Mass0, Mass1, …) showing the
    summed data alongside the mass-specific NCP component.  Skipped
    when running a bootstrap sample (``IC.runningSampleWS`` is
    ``True``).

    Args:
        wsMNCPSum: List of summed-spectra NCP workspaces, one per mass
            (from :func:`createNcpWorkspaces`).
        wsDataSum: Summed-spectra data workspace.
        IC: Completed initial-conditions object with ``masses``,
            ``runningSampleWS``, and ``figSavePath``.
    """
    if IC.runningSampleWS:
        return

    set_thesis_style()
    for mass_idx, (m, wsNcp) in enumerate(zip(IC.masses, wsMNCPSum)):
        fig, ax = figure_factory(subplot_kw={"projection": "mantid"})
        ax.errorbar(wsDataSum, "k.", label="Spectra")
        ax.plot(wsNcp, color=COLORBLIND_PALETTE[mass_idx % len(COLORBLIND_PALETTE)],
                label=f"NCP m={m:.4g}")
        ax.set_xlabel("TOF")
        ax.set_ylabel("Counts")
        ax.set_title(f"NCP Fit — Mass{mass_idx} (m = {m:.4g})")
        ax.legend()

        fileName = f"{wsDataSum.name()}_Mass{mass_idx}_NCP.pdf"
        savePath = IC.figSavePath / fileName
        plt.savefig(savePath, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {fileName} to {IC.figSavePath}")
        plt.close(fig)


def extractMeans(
    wsName: str, IC: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract mean widths and intensity ratios from the fit-parameter table.

    Reads the ``_Best_Fit_NCP_Parameters`` TableWorkspace, applies
    sigma-clipping via ``calculateMeansAndStds``, and returns the
    filtered means and standard deviations.

    Expects workspace ``wsName + "_Best_Fit_NCP_Parameters"`` in ``mtd``.

    Args:
        wsName: Base name of the fitted workspace.
        IC: Completed initial-conditions object with ``noOfMasses``.

    Returns:
        A 4-tuple ``(meanWidths, stdWidths, meanIntensityRatios,
        stdIntensityRatios)`` each of shape ``(n_masses,)``.
    """

    fitParsTable = mtd[wsName+"_Best_Fit_NCP_Parameters"]
    widths = np.zeros((IC.noOfMasses, fitParsTable.rowCount()))
    intensities = np.zeros(widths.shape)
    for i in range(IC.noOfMasses):
        widths[i] = fitParsTable.column(f"Width {i}")
        intensities[i] = fitParsTable.column(f"Intensity {i}")

    meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios = calculateMeansAndStds(widths, intensities, IC)

    assert len(widths) == IC.noOfMasses, "Widths and intensities must be in shape (noOfMasses, noOfSpec)"
    return meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios


def createMeansAndStdTableWS(
    wsName: str,
    IC: Any,
    meanWidths: np.ndarray,
    stdWidths: np.ndarray,
    meanIntensityRatios: np.ndarray,
    stdIntensityRatios: np.ndarray,
) -> None:
    """Create a Mantid TableWorkspace with per-mass mean widths and intensities.

    Writes a table named ``wsName + "_Mean_Widths_And_Intensities"``
    to the AnalysisDataService.

    Args:
        wsName: Base name of the fitted workspace (used for naming).
        IC: Completed initial-conditions object with ``masses``.
        meanWidths: Mean NCP widths, shape ``(n_masses,)``.
        stdWidths: Standard deviation of widths, shape ``(n_masses,)``.
        meanIntensityRatios: Mean intensity ratios, shape
            ``(n_masses,)``.
        stdIntensityRatios: Standard deviation of intensities, shape
            ``(n_masses,)``.
    """

    meansTableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsName+"_Mean_Widths_And_Intensities")
    meansTableWS.addColumn(type='float', name="Mass")
    meansTableWS.addColumn(type='float', name="Mean Widths")
    meansTableWS.addColumn(type='float', name="Std Widths")
    meansTableWS.addColumn(type='float', name="Mean Intensities")
    meansTableWS.addColumn(type='float', name="Std Intensities")

    print("\nCreated Table with means and std:")
    print("\nMass    Mean \u00B1 Std Widths    Mean \u00B1 Std Intensities\n")
    for m, mw, stdw, mi, stdi in zip(IC.masses.astype(float), meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios):
        meansTableWS.addRow([m, mw, stdw, mi, stdi])
        print(f"{m:5.2f}  {mw:10.5f} \u00B1 {stdw:7.5f}  {mi:10.5f} \u00B1 {stdi:7.5f}")
    print("\n")
    return 


def calculateMeansAndStds(
    widthsIn: np.ndarray, intensitiesIn: np.ndarray, IC: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute filtered means and standard deviations of widths and intensities.

    Applies ``filterWidthsAndIntensities`` (sigma-clipping) before
    computing ``nanmean`` and ``nanstd``.

    Args:
        widthsIn: Raw widths, shape ``(n_masses, n_spectra)``.
        intensitiesIn: Raw intensities, shape ``(n_masses, n_spectra)``.
        IC: Completed initial-conditions object.

    Returns:
        A 4-tuple ``(meanWidths, stdWidths, meanIntensityRatios,
        stdIntensityRatios)`` each of shape ``(n_masses,)``.
    """

    betterWidths, betterIntensities = filterWidthsAndIntensities(widthsIn, intensitiesIn, IC)
    
    meanWidths = np.nanmean(betterWidths, axis=1)  
    stdWidths = np.nanstd(betterWidths, axis=1)

    meanIntensityRatios = np.nanmean(betterIntensities, axis=1)
    stdIntensityRatios = np.nanstd(betterIntensities, axis=1)

    return meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios


def filterWidthsAndIntensities(
    widthsIn: np.ndarray, intensitiesIn: np.ndarray, IC: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply sigma-clipping to reject outlier widths and normalise intensities.

    For each mass, any spectrum whose width deviates from the mean by
    more than one standard deviation is replaced with ``NaN``.  The
    surviving intensities are then normalised so that they sum to one
    across masses for each spectrum.

    Args:
        widthsIn: Raw widths, shape ``(n_masses, n_spectra)``.
        intensitiesIn: Raw intensities, same shape.
        IC: Completed initial-conditions object.  Uses
            ``runningPreliminary`` and ``noOfMSIterations`` for
            edge-case handling.

    Returns:
        A 2-tuple ``(betterWidths, betterIntensities)`` with outliers
        replaced by ``NaN``.
    """

    widths = widthsIn.copy()      # Copy to avoid accidental changes in arrays
    intensities = intensitiesIn.copy()

    zeroSpecs = np.all(widths==0, axis=0)   # Catches all failed fits, not just masked spectra
    widths[:, zeroSpecs] = np.nan
    intensities[:, zeroSpecs] = np.nan

    meanWidths = np.nanmean(widths, axis=1)[:, np.newaxis]  

    widthDeviation = np.abs(widths - meanWidths)
    stdWidths = np.nanstd(widths, axis=1)[:, np.newaxis]  

    # Put nan in places where width deviation is bigger than std
    filterMask = widthDeviation > stdWidths
    betterWidths = np.where(filterMask, np.nan, widths)
    
    maskedIntensities = np.where(filterMask, np.nan, intensities)
    betterIntensities = maskedIntensities / np.sum(maskedIntensities, axis=0)   # Not nansum()      
    
    # When trying to estimate HToMassIdxRatio and normalization fails, skip normalization
    if np.all(np.isnan(betterIntensities)) & IC.runningPreliminary:
        assert IC.noOfMSIterations == 0, "Calculation of mean intensities failed, cannot proceed with MS correction. Try to run again with noOfMSIterations=0."
        betterIntensities = maskedIntensities 
    else:
        pass
  
    assert np.all(meanWidths!=np.nan), "At least one mean of widths is nan!"
    assert np.sum(filterMask) >= 1, "No widths survive filtering condition"
    assert not(np.all(np.isnan(betterWidths))), "All filtered widths are nan"
    assert not(np.all(np.isnan(betterIntensities))), "All filtered intensities are nan"
    if len(betterWidths)>1:
        assert np.nanmax(betterWidths) != np.nanmin(betterWidths), f"All fitered widths have the same value: {np.nanmin(betterWidths)}"
        assert np.nanmax(betterIntensities) != np.nanmin(betterIntensities), f"All fitered widths have the same value: {np.nanmin(betterIntensities)}"
   
    return betterWidths, betterIntensities


def fitNcpToSingleSpec(
    dataY: np.ndarray,
    dataE: np.ndarray,
    ySpacesForEachMass: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
    kinematicArrays: np.ndarray,
    ic: Any,
) -> np.ndarray:
    """Fit the NCP model to a single spectrum using SciPy SLSQP and iMinuit.

    Minimises ``errorFunction`` with ``scipy.optimize.minimize``
    (method ``'SLSQP'``), then runs a parallel iMinuit fit
    (MIGRAD + Hesse, optionally Minos) for cross-validation and
    rigorous error estimation.

    **Boundary synchronisation:** Both optimisers receive identical
    parameter bounds derived from ``ic.bounds``.  ``np.nan`` entries
    are mapped to ``None`` (unbounded) for iMinuit and ``±np.inf``
    for SciPy, ensuring the same feasible region.

    **iMinuit step sizes:** Initial errors (step sizes) are set to
    physically reasonable fractions of the initial parameter guesses
    (10% of ``|initPar|`` or 0.1 for near-zero parameters).  This
    prevents MIGRAD from taking oversized steps that traverse
    different cost-function basins.

    **Cost function alignment:** Both optimisers evaluate the
    identical chi-squared definition:
    ``χ² = Σ (NCP_model - data)² / σ²``.  iMinuit's ``errordef``
    is explicitly set to ``Minuit.LEAST_SQUARES`` (= 1.0).

    **Fail-safe:** When the 1% tolerance is exceeded, the divergent
    parameters are identified by physical name (``I_m`` = intensity,
    ``σ_m`` = width, ``C_m`` = centre for mass *m*) and their
    individual values are logged to diagnose local-minimum traps.

    Args:
        dataY: Observed counts for one spectrum, shape ``(n_bins,)``.
        dataE: Errors for one spectrum, shape ``(n_bins,)``.
        ySpacesForEachMass: y-spaces for this spectrum, shape
            ``(n_masses, n_bins)``.
        resolutionPars: Resolution parameters, shape ``(6,)``.
        instrPars: Instrument parameters, shape ``(6,)``.
        kinematicArrays: Kinematics ``[v0, E0, deltaE, deltaQ]``,
            shape ``(4, n_bins)``.
        ic: Completed initial-conditions object with ``initPars``,
            ``bounds``, ``constraints``, and ``masses``.

    Returns:
        Array of shape ``(3 * n_masses + 3,)`` containing
        ``[specNo, *fitPars, normChi2, nIter]``, or all zeros if
        the spectrum was masked.  The primary result comes from
        SciPy; iMinuit results are logged for comparison.
    """

    if np.all(dataY == 0):
        return np.zeros(len(ic.initPars) + 3)

    # --- Strict boundary synchronisation ---
    # Normalise ic.bounds: replace np.nan with np.inf/-np.inf for SciPy
    scipy_bounds = []
    for lo, hi in ic.bounds:
        lo_val = -np.inf if np.isnan(lo) else float(lo)
        hi_val = np.inf if np.isnan(hi) else float(hi)
        scipy_bounds.append((lo_val, hi_val))

    # --- Boundary buffer: nudge initial guesses off exact boundaries ---
    # SLSQP can stall when starting exactly on a constraint boundary.
    # Shift inward by a small fraction of the feasible range (or an
    # absolute floor for semi-infinite bounds).
    _BOUNDARY_BUFFER_FRAC = 1e-4
    _BOUNDARY_BUFFER_ABS = 1e-4
    init_pars = ic.initPars.copy()
    for k, (lo, hi) in enumerate(scipy_bounds):
        if np.isfinite(hi - lo):
            span = hi - lo
            nudge = max(abs(span) * _BOUNDARY_BUFFER_FRAC, _BOUNDARY_BUFFER_ABS)
        else:
            # Infinite span: fallback nudge relative to initial guess
            nudge = abs(init_pars[k]) * _BOUNDARY_BUFFER_FRAC
        if np.isfinite(lo) and init_pars[k] <= lo + nudge:
            init_pars[k] = lo + nudge
        if np.isfinite(hi) and init_pars[k] >= hi - nudge:
            init_pars[k] = hi - nudge

    # --- SciPy SLSQP fit (primary) ---
    result = optimize.minimize(
        errorFunction,
        init_pars,
        args=(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic),
        method='SLSQP',
        bounds=scipy_bounds,
        constraints=ic.constraints,
    )

    fitPars = result["x"]

    noDegreesOfFreedom = len(dataY) - len(fitPars)
    specFitPars = np.append(instrPars[0], fitPars)

    # --- Build physical parameter names for diagnostics ---
    # Parameters are ordered [I0, W0, C0, I1, W1, C1, ...] for n_masses.
    n_masses = len(ic.masses)
    _param_labels = ("I", "σ", "C")  # intensity, width, centre
    par_names = [
        f"{_param_labels[j % 3]}_{j // 3}" for j in range(len(ic.initPars))
    ]

    # --- iMinuit MIGRAD + Hesse fit (parallel cross-validation) ---
    try:
        cost_fn = NCPCostFunction(
            dataY, dataE, ySpacesForEachMass,
            resolutionPars, instrPars, kinematicArrays, ic,
        )
        m = Minuit(cost_fn, *init_pars)

        # Synchronise bounds: apply identical limits to iMinuit.
        # NCPCostFunction._parameters already sets these via
        # _build_parameters_dict, but we enforce them explicitly here
        # to guarantee strict equivalence with the SciPy bounds.
        for k, (lo, hi) in enumerate(scipy_bounds):
            lo_m = None if lo == -np.inf else lo
            hi_m = None if hi == np.inf else hi
            m.limits[k] = (lo_m, hi_m)

        # Set initial step sizes (errors) to physically reasonable
        # fractions of the initial guesses.  MIGRAD uses these as the
        # starting step size for gradient estimation.  Without this,
        # MIGRAD may take steps that are too large (especially for
        # centres near zero) and converge to a different local minimum.
        _STEP_FRACTION = 0.1  # 10% of |initPar|
        _STEP_FLOOR = 0.1     # Minimum step for near-zero parameters
        for k, p0 in enumerate(init_pars):
            step = max(abs(p0) * _STEP_FRACTION, _STEP_FLOOR)
            m.errors[k] = step

        if getattr(ic, "runningTest", False):
            m.tol = 1.0  # Loose EDM tolerance for fast smoke-test convergence

        m.migrad()
        m.hesse()

        # --- Migrad convergence and bound-hit diagnostics ---
        if not m.valid:
            logger.warning(
                "OptimizerCheck Spec %.0f: MIGRAD did NOT converge "
                "(edm=%.4g, is_above_max_edm=%s).",
                instrPars[0], m.fmin.edm, m.fmin.is_above_max_edm,
            )
        if m.fmin.has_parameters_at_limit:
            at_limit = [
                par_names[k] for k in range(len(init_pars))
                if m.params[k].is_at_lower_limit or m.params[k].is_at_upper_limit
            ]
            logger.warning(
                "OptimizerCheck Spec %.0f: parameters at limits: %s",
                instrPars[0], at_limit,
            )
        # Log SciPy exit status
        scipy_status = result.get("status", -1)
        scipy_msg = result.get("message", "")
        if scipy_status != 0:
            logger.warning(
                "OptimizerCheck Spec %.0f: SciPy SLSQP exit=%d (%s)",
                instrPars[0], scipy_status, scipy_msg,
            )

        runMinos = getattr(ic, "runMinos", False)
        if runMinos:
            m.minos()

        # --- iMinuit–SciPy Numerical Agreement Check ---
        scipy_chi2 = result["fun"]
        iminuit_chi2 = m.fval

        # χ² comparison
        if scipy_chi2 > 0:
            chi2_rel_diff = abs(scipy_chi2 - iminuit_chi2) / scipy_chi2
            if chi2_rel_diff > _AGREEMENT_THRESHOLD:
                logger.warning(
                    "OptimizerCheck Spec %.0f: χ² disagreement — "
                    "SciPy=%.6g vs iMinuit=%.6g (%.2f%%)",
                    instrPars[0], scipy_chi2, iminuit_chi2,
                    chi2_rel_diff * 100,
                )
        else:
            chi2_rel_diff = 0.0

        # Parameter-vector comparison (hybrid metric)
        iminuit_pars = np.array(m.values)
        scipy_pars = fitPars

        _PARAM_SCALE_THRESHOLD = 0.01
        _PARAM_ABS_TOLERANCE = 1e-4

        par_abs_diff = np.abs(scipy_pars - iminuit_pars)
        par_rel_diff = np.zeros_like(scipy_pars)

        large_param_mask = np.abs(scipy_pars) > _PARAM_SCALE_THRESHOLD
        par_rel_diff[large_param_mask] = (
            par_abs_diff[large_param_mask] / np.abs(scipy_pars[large_param_mask])
        )

        small_param_mask = ~large_param_mask
        par_rel_diff[small_param_mask] = (
            par_abs_diff[small_param_mask] / _PARAM_ABS_TOLERANCE
        )

        max_par_diff = float(np.max(par_rel_diff))

        # --- Fail-safe: per-parameter divergence diagnostics ---
        if max_par_diff > _AGREEMENT_THRESHOLD:
            # Log ALL divergent parameters with their physical names
            divergent_params = np.where(par_rel_diff > _AGREEMENT_THRESHOLD)[0]
            for pidx in divergent_params:
                pname = par_names[pidx] if pidx < len(par_names) else f"p{pidx}"
                logger.warning(
                    "OptimizerCheck Spec %.0f: %s diverged — "
                    "SciPy=%.6g, iMinuit=%.6g "
                    "(abs: %.4g, rel: %.2f%%)",
                    instrPars[0], pname,
                    scipy_pars[pidx], iminuit_pars[pidx],
                    par_abs_diff[pidx], par_rel_diff[pidx] * 100,
                )

        # Accumulate for the end-of-workspace summary
        _optimizer_check_log.append((chi2_rel_diff, max_par_diff))

        # Store data for fit comparison visualization
        _fit_comparison_log.append({
            'specNo': instrPars[0],
            'scipy_chi2': scipy_chi2,
            'iminuit_chi2': iminuit_chi2,
            'scipy_pars': scipy_pars.copy(),
            'iminuit_pars': iminuit_pars.copy(),
            'par_names': par_names,
            'par_rel_diff': par_rel_diff.copy(),
        })
    except Exception:
        logger.debug(
            "iMinuit fit failed for spec %.0f, SciPy result used.",
            instrPars[0], exc_info=True,
        )

    return np.append(specFitPars, [result["fun"] / noDegreesOfFreedom, result["nit"]])


def errorFunction(
    pars: np.ndarray,
    dataY: np.ndarray,
    dataE: np.ndarray,
    ySpacesForEachMass: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
    kinematicArrays: np.ndarray,
    ic: Any,
) -> float:
    """Compute the chi-squared cost for the NCP model on a single spectrum.

    Called by ``scipy.optimize.minimize`` at every iteration.  Masked
    bins (dataY == 0) are excluded.  When errors are all zero the
    un-weighted sum of squared residuals is returned.

    **Numba candidate** — the inner call to ``calculateNcpSpec`` and
    the chi-squared arithmetic are pure NumPy.

    Args:
        pars: Current fit parameters, shape ``(3 * n_masses,)``.
        dataY: Observed counts, shape ``(n_bins,)``.
        dataE: Errors, shape ``(n_bins,)``.
        ySpacesForEachMass: y-spaces, shape ``(n_masses, n_bins)``.
        resolutionPars: Resolution parameters, shape ``(6,)``.
        instrPars: Instrument parameters, shape ``(6,)``.
        kinematicArrays: Kinematics, shape ``(4, n_bins)``.
        ic: Completed initial-conditions object.

    Returns:
        Scalar chi-squared value (weighted if errors are present).
    """

    ncpForEachMass, ncpTotal = calculateNcpSpec(ic, pars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)

    # Ignore masked bins (dataY==0) AND zero-error bins (dataE==0).
    # MaskDetectors zeros both dataY and dataE; resonance masking
    # (maskBinsWithZeros) zeros dataY only.  Individual dataE==0 bins
    # from single-count edges or Mantid masking must also be excluded
    # to prevent division-by-zero in the chi-squared denominator.
    validMask = (dataY != 0) & (dataE != 0)
    ncpTotal = ncpTotal[validMask]
    dataYf = dataY[validMask]
    dataEf = dataE[validMask]

    if len(dataYf) == 0 or np.all(dataEf == 0):
        return np.sum((ncpTotal - dataYf)**2)

    return np.sum((ncpTotal - dataYf)**2 / dataEf**2)


def calculateNcpSpec(
    ic: Any,
    pars: np.ndarray,
    ySpacesForEachMass: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
    kinematicArrays: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesise the Neutron Compton Profile C(t) for one spectrum.

    Constructs J(y) for each mass as a pseudo-Voigt (Gaussian
    resolution ⊕ Lorentzian resolution), adds the Final-State Effects
    (FSE) term via a numerical third derivative, and converts to
    TOF-space counts:
    ``NCP_m = I_m * (J(y) + FSE) * E0 * E0^{-0.92} * M / deltaQ``

    When ``USE_NUMBA`` is ``True`` (default), the IC object is
    *unrolled*: ``ic.masses`` (1-D array) and ``ic.normVoigt`` (bool)
    are extracted as plain scalars/arrays and forwarded to the
    ``@njit``-compiled ``calculateNcpSpec_numba`` in
    ``numba_routines.py``.  This avoids passing any Python object into
    the Numba nopython boundary.

    Args:
        ic: Completed initial-conditions object (``normVoigt`` flag
            and ``masses`` array are used).
        pars: Fit parameters ``[I0, W0, C0, I1, W1, C1, …]``, shape
            ``(3 * n_masses,)``.
        ySpacesForEachMass: y-spaces, shape ``(n_masses, n_bins)``.
        resolutionPars: Resolution parameters for this spectrum,
            shape ``(6,)``.
        instrPars: Instrument parameters for this spectrum,
            shape ``(6,)``.
        kinematicArrays: ``[v0, E0, deltaE, deltaQ]`` for this
            spectrum, shape ``(4, n_bins)``.

    Returns:
        A 2-tuple ``(ncpForEachMass, ncpTotal)`` where the first has
        shape ``(n_masses, n_bins)`` and the second ``(n_bins,)``.
    """

    # --- Numba-accelerated path (IC unrolling) ---
    if USE_NUMBA:
        # Flatten IC object attributes into plain scalars/arrays:
        #   ic.masses  → 1-D float64 array  (n_masses,)
        #   ic.normVoigt → Python bool
        masses_1d = np.asarray(ic.masses, dtype=np.float64)
        normVoigt = bool(ic.normVoigt)
        return _calculateNcpSpec_numba(
            masses_1d, pars, ySpacesForEachMass,
            resolutionPars, instrPars, kinematicArrays,
            normVoigt,
        )

    # --- Legacy NumPy path ---
    masses, intensities, widths, centers = prepareArraysFromPars(ic, pars) 
    v0, E0, deltaE, deltaQ = kinematicArrays
    
    gaussRes, lorzRes = caculateResolutionForEachMass(
        masses, ySpacesForEachMass, centers, resolutionPars, instrPars, kinematicArrays
        )
    totalGaussWidth = np.sqrt(widths**2 + gaussRes**2)                 
    
    JOfY = pseudoVoigt(ySpacesForEachMass - centers, totalGaussWidth, lorzRes, ic)  
    
    FSE =  - numericalThirdDerivative(ySpacesForEachMass, JOfY) * widths**4 / deltaQ * 0.72 
    
    ncpForEachMass = intensities * (JOfY + FSE) * E0 * E0**(-0.92) * masses / deltaQ   
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    return ncpForEachMass, ncpTotal


def prepareArraysFromPars(
    ic: Any, initPars: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack flat fit parameters into per-mass column vectors.

    Splits the interleaved ``[I, W, C, I, W, C, …]`` array into
    separate arrays and reshapes to column vectors for broadcasting.

    Args:
        ic: Initial-conditions object (``masses`` is used).
        initPars: Flat fit parameters, shape ``(3 * n_masses,)``.

    Returns:
        A 4-tuple ``(masses, intensities, widths, centers)`` each of
        shape ``(n_masses, 1)``.
    """

    masses = ic.masses[:, np.newaxis]    
    intensities = initPars[::3].reshape(masses.shape)
    widths = initPars[1::3].reshape(masses.shape)
    centers = initPars[2::3].reshape(masses.shape)  
    return masses, intensities, widths, centers 


def caculateResolutionForEachMass(
    masses: np.ndarray,
    ySpacesForEachMass: np.ndarray,
    centers: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
    kinematicArrays: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian and Lorentzian resolution widths for each mass.

    Evaluates kinematics at the y-space centres of each NCP peak and
    propagates the instrument uncertainties into Gaussian and
    Lorentzian resolution widths in inverse-Ångström space.

    Args:
        masses: Column vector of masses, shape ``(n_masses, 1)``.
        ySpacesForEachMass: y-spaces, shape ``(n_masses, n_bins)``.
        centers: NCP centres, shape ``(n_masses, 1)``.
        resolutionPars: Resolution parameters, shape ``(6,)``.
        instrPars: Instrument parameters, shape ``(6,)``.
        kinematicArrays: Kinematics, shape ``(4, n_bins)``.

    Returns:
        A 2-tuple ``(gaussianResWidth, lorentzianResWidth)`` each of
        shape ``(n_masses, 1)``.
    """
    
    v0, E0, delta_E, delta_Q = kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays)
    
    gaussianResWidth = calcGaussianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    lorentzianResWidth = calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    return gaussianResWidth, lorentzianResWidth


def kinematicsAtYCenters(
    ySpacesForEachMass: np.ndarray,
    centers: np.ndarray,
    kinematicArrays: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate kinematics at the y-space bin closest to each NCP centre.

    Selects the bin nearest to each mass's centre in y-space and
    returns the corresponding v0, E0, deltaE, deltaQ values.

    **Numba candidate** — pure NumPy index operations.

    Args:
        ySpacesForEachMass: y-spaces, shape ``(n_masses, n_bins)``.
        centers: NCP peak centres, shape ``(n_masses, 1)``.
        kinematicArrays: ``[v0, E0, deltaE, deltaQ]``, shape
            ``(4, n_bins)``.

    Returns:
        A 4-tuple ``(v0, E0, deltaE, deltaQ)`` each of shape
        ``(n_masses, 1)``.
    """

    shapeOfArrays = centers.shape
    proximityToYCenters = np.abs(ySpacesForEachMass - centers)
    yClosestToCenters = proximityToYCenters.min(axis=1).reshape(shapeOfArrays)
    yCentersMask = proximityToYCenters == yClosestToCenters

    v0, E0, deltaE, deltaQ = kinematicArrays

    # Expand arrays to match shape of yCentersMask
    v0 = v0 * np.ones(shapeOfArrays)
    E0 = E0 * np.ones(shapeOfArrays)
    deltaE = deltaE * np.ones(shapeOfArrays)
    deltaQ = deltaQ * np.ones(shapeOfArrays)

    v0 = v0[yCentersMask].reshape(shapeOfArrays)
    E0 = E0[yCentersMask].reshape(shapeOfArrays)
    deltaE = deltaE[yCentersMask].reshape(shapeOfArrays)
    deltaQ = deltaQ[yCentersMask].reshape(shapeOfArrays)
    return v0, E0, deltaE, deltaQ


def calcGaussianResolution(
    masses: np.ndarray,
    v0: np.ndarray,
    E0: np.ndarray,
    delta_E: np.ndarray,
    delta_Q: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
) -> np.ndarray:
    """Compute the Gaussian component of the TOF resolution in y-space.

    Propagates energy and momentum uncertainties (dE1, dTOF, dL0, dL1,
    dTheta) into a total Gaussian resolution width in Å⁻¹.  This is
    the dominant computational bottleneck in the fitting loop.

    **Numba candidate** — pure NumPy arithmetic on pre-extracted arrays.

    Args:
        masses: Atomic masses, shape ``(n_masses, 1)``.
        v0: Initial neutron velocity, shape ``(n_masses, 1)``.
        E0: Initial neutron energy (meV), shape ``(n_masses, 1)``.
        delta_E: Energy transfer (meV), shape ``(n_masses, 1)``.
        delta_Q: Momentum transfer (Å⁻¹), shape ``(n_masses, 1)``.
        resolutionPars: ``[dE1, dTOF, dTheta, dL0, dL1, dE1_lorz]``.
        instrPars: ``[det, plick, angle, T0, L0, L1]``.

    Returns:
        Gaussian resolution width in Å⁻¹, shape ``(n_masses, 1)``.
    """
    # Currently the function that takes the most time in the fitting
    assert masses.shape == (masses.size, 1), f"masses.shape: {masses.shape}. The shape of the masses array needs to be a collumn!"

    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resolutionPars
    mN, Ef, en_to_vel, vf, hbar = loadConstants()

    angle = angle * np.pi/180

    dWdE1 = 1. + (E0 / Ef)**1.5 * (L1 / L0)
    dWdTOF = 2. * E0 * v0 / L0
    dWdL1 = 2. * E0**1.5 / Ef**0.5 / L0
    dWdL0 = 2. * E0 / L0

    dW2 = dWdE1**2*dE1**2 + dWdTOF**2*dTOF**2 + dWdL1**2*dL1**2 + dWdL0**2*dL0**2
    # conversion from meV^2 to A^-2, dydW = (M/q)^2
    dW2 *= (masses / hbar**2 / delta_Q)**2

    dQdE1 = 1. - (E0 / Ef)**1.5 * L1/L0 - np.cos(angle) * ((E0 / Ef)**0.5 - L1/L0 * E0/Ef)
    dQdTOF = 2.*E0 * v0/L0
    dQdL1 = 2.*E0**1.5 / L0 / Ef**0.5
    dQdL0 = 2.*E0 / L0
    dQdTheta = 2. * np.sqrt(E0 * Ef) * np.sin(angle)

    dQ2 = dQdE1**2*dE1**2 + (dQdTOF**2*dTOF**2 + dQdL1**2*dL1**2 + dQdL0 **
                             2*dL0**2)*np.abs(Ef/E0*np.cos(angle)-1) + dQdTheta**2*dTheta**2
    dQ2 *= (mN / hbar**2 / delta_Q)**2

    # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2
    gaussianResWidth = np.sqrt(dW2 + dQ2)
    return gaussianResWidth


def calcLorentzianResolution(
    masses: np.ndarray,
    v0: np.ndarray,
    E0: np.ndarray,
    delta_E: np.ndarray,
    delta_Q: np.ndarray,
    resolutionPars: np.ndarray,
    instrPars: np.ndarray,
) -> np.ndarray:
    """Compute the Lorentzian component of the TOF resolution in y-space.

    Propagates the Lorentzian energy width ``dE1_lorz`` through the
    partial derivatives of energy and momentum transfer.

    **Numba candidate** — pure NumPy arithmetic.

    Args:
        masses: Atomic masses, shape ``(n_masses, 1)``.
        v0: Initial neutron velocity, shape ``(n_masses, 1)``.
        E0: Initial neutron energy (meV), shape ``(n_masses, 1)``.
        delta_E: Energy transfer (meV), shape ``(n_masses, 1)``.
        delta_Q: Momentum transfer (Å⁻¹), shape ``(n_masses, 1)``.
        resolutionPars: ``[dE1, dTOF, dTheta, dL0, dL1, dE1_lorz]``.
        instrPars: ``[det, plick, angle, T0, L0, L1]``.

    Returns:
        Lorentzian resolution HWHM in Å⁻¹, shape ``(n_masses, 1)``.
    """

    assert masses.shape == (masses.size, 1), "The shape of the masses array needs to be a collumn!"
        
    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resolutionPars
    mN, Ef, en_to_vel, vf, hbar = loadConstants()

    angle = angle * np.pi / 180

    dWdE1_lor = (1. + (E0/Ef)**1.5 * (L1/L0))**2
    # conversion from meV^2 to A^-2
    dWdE1_lor *= (masses / hbar**2 / delta_Q)**2

    dQdE1_lor = (1. - (E0/Ef)**1.5 * L1/L0 - np.cos(angle)
                 * ((E0/Ef)**0.5 + L1/L0 * E0/Ef))**2
    dQdE1_lor *= (mN / hbar**2 / delta_Q)**2

    lorentzianResWidth = np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz   # in A-1
    return lorentzianResWidth


def loadConstants() -> Tuple[float, float, float, float, float]:
    """Return fundamental physical constants used in VESUVIO kinematics.

    Constants are specific to the VESUVIO instrument at ISIS:
    * ``mN`` — neutron mass in atomic mass units.
    * ``Ef`` — final neutron energy selected by the gold analyser foil (meV).
    * ``en_to_vel`` — factor converting sqrt(energy) to velocity (m/μs).
    * ``vf`` — final neutron velocity (m/μs).
    * ``hbar`` — reduced Planck constant in Å⁻¹·a.m.u.·m/μs units.

    Returns:
        A 5-tuple ``(mN, Ef, en_to_vel, vf, hbar)``.
    """
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf = np.sqrt(Ef) * en_to_vel  # m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar


def pseudoVoigt(
    x: np.ndarray, sigma: np.ndarray, gamma: np.ndarray, IC: Any
) -> np.ndarray:
    """Approximate pseudo-Voigt profile (Thompson–Cox–Hastings).

    Approximates the convolution of a Gaussian (std *sigma*) and a
    Lorentzian (HWHM *gamma*) as a linear combination weighted by the
    mixing parameter *eta*.  Optionally normalised by trapezoidal
    integration when ``IC.normVoigt`` is ``True``.

    **Numba candidate** — pure NumPy arithmetic; ``np.trapz`` would
    need replacing with a manual trapezoidal rule under ``@njit``.

    Args:
        x: Abscissa values, shape ``(n_masses, n_bins)``.
        sigma: Gaussian standard deviation, shape ``(n_masses, 1)``.
        gamma: Lorentzian HWHM, shape ``(n_masses, 1)``.
        IC: Initial-conditions object with ``normVoigt`` flag.

    Returns:
        Pseudo-Voigt profile, shape ``(n_masses, n_bins)``.
    """
    fg, fl = 2.*sigma*np.sqrt(2.*np.log(2.)), 2.*gamma
    f = 0.5346 * fl + np.sqrt(0.2166*fl**2 + fg**2)
    eta = 1.36603 * fl/f - 0.47719 * (fl/f)**2 + 0.11116 * (fl/f)**3
    sigma_v, gamma_v = f/(2.*np.sqrt(2.*np.log(2.))), f / 2.
    pseudo_voigt = eta * lorentizian(x, gamma_v) + (1.-eta) * gaussian(x, sigma_v)
    
    norm = np.abs(np.trapz(pseudo_voigt, x, axis=1))[:, np.newaxis] if IC.normVoigt else 1
    return pseudo_voigt / norm


def gaussian(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Normalised Gaussian centred at zero.

    Args:
        x: Abscissa values.
        sigma: Standard deviation (same shape or broadcastable).

    Returns:
        Gaussian values, same shape as *x*.
    """
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian


def lorentizian(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Normalised Lorentzian centred at zero.

    Args:
        x: Abscissa values.
        gamma: Half-width at half-maximum (same shape or broadcastable).

    Returns:
        Lorentzian values, same shape as *x*.
    """
    lorentzian = gamma/np.pi / (x**2 + gamma**2)
    return lorentzian


def numericalThirdDerivative(x: np.ndarray, fun: np.ndarray) -> np.ndarray:
    """Compute the third derivative of *fun* using a 13-point stencil.

    Uses a symmetric finite-difference stencil of radius 6 bins.
    The result is zero-padded on the left and right to preserve the
    input shape.  Used to compute the Final-State Effects (FSE) term.

    **Numba candidate** — pure slicing and arithmetic.

    Args:
        x: Abscissa values, shape ``(n_masses, n_bins)``.
        fun: Function values, same shape.

    Returns:
        Approximate third derivative, same shape as *fun*.
    """

    k6 = (- fun[:, 12:] + fun[:, :-12]) * 1
    k5 = (+ fun[:, 11:-1] - fun[:, 1:-11]) * 24
    k4 = (- fun[:, 10:-2] + fun[:, 2:-10]) * 192
    k3 = (+ fun[:,  9:-3] - fun[:, 3:-9]) * 488
    k2 = (+ fun[:,  8:-4] - fun[:, 4:-8]) * 387
    k1 = (- fun[:,  7:-5] + fun[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3

    derivative = np.zeros(fun.shape)
    derivative[:, 6:-6] = dev
    # Padded with zeros left and right to return array with same shape
    return derivative


def createWorkspacesForMSCorrection(
    ic: Any,
    meanWidths: np.ndarray,
    meanIntensityRatios: np.ndarray,
    wsNCPM: Any,
) -> Any:
    """Run the Mantid multiple-scattering correction.

    Constructs the slab sample geometry, builds the sample-property
    list (optionally adding H for backward scattering), and calls
    ``VesuvioThickness`` and ``VesuvioCalculateMS``.  The resulting
    ``_MulScattering`` workspace is returned for subtraction.

    Expects ``wsNCPM`` to be present in the AnalysisDataService.

    Args:
        ic: Completed initial-conditions object with geometry,
            ``modeRunning``, ``HToMassIdxRatio``, ``masses``, and
            MS settings.
        meanWidths: Mean NCP widths, shape ``(n_masses,)``.
        meanIntensityRatios: Mean intensity ratios, shape
            ``(n_masses,)``.
        wsNCPM: Mantid workspace with NCP-masked data.

    Returns:
        The ``_MulScattering`` workspace to be subtracted from the
        data.
    """

    createSlabGeometry(ic, wsNCPM)    # Sample properties for MS correction 

    sampleProperties = calcMSCorrectionSampleProperties(ic, meanWidths, meanIntensityRatios)
    print("\nThe sample properties for Multiple Scattering correction are:\n\n", 
            sampleProperties, "\n")
    
    return createMulScatWorkspaces(ic, wsNCPM, sampleProperties)


def createSlabGeometry(ic: Any, wsNCPM: Any) -> None:
    """Attach a cuboid sample shape to a workspace for MS correction.

    Builds an XML string describing a slab of dimensions
    ``vertical_width × horizontal_width × thickness`` and calls
    ``CreateSampleShape``.

    Args:
        ic: Initial-conditions object with ``vertical_width``,
            ``horizontal_width``, and ``thickness`` (metres).
        wsNCPM: Mantid workspace to which the shape is attached.
    """

    half_height, half_width, half_thick = 0.5*ic.vertical_width, 0.5*ic.horizontal_width, 0.5*ic.thickness
    xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"

    CreateSampleShape(wsNCPM, xml_str)


def calcMSCorrectionSampleProperties(
    ic: Any, meanWidths: np.ndarray, meanIntensityRatios: np.ndarray
) -> List[float]:
    """Build the flat sample-property list for ``VesuvioCalculateMS``.

    The list is interleaved as ``[mass0, intensity0, width0, mass1, …]``.
    When running backward scattering with H present, hydrogen is
    appended using ``HToMassIdxRatio``.

    Args:
        ic: Completed initial-conditions object.
        meanWidths: Mean NCP widths, shape ``(n_masses,)``.
        meanIntensityRatios: Mean intensity ratios, shape
            ``(n_masses,)``.

    Returns:
        A flat list of floats suitable for ``VesuvioCalculateMS``.
    """

    masses = ic.masses.flatten()

    # If Backsscattering mode and H is present in the sample, add H to MS properties
    if (ic.modeRunning == "BACKWARD"):
        if (ic.HToMassIdxRatio != None):  # If H is present, ratio is a number
            masses = np.append(masses, 1.0079)
            meanWidths = np.append(meanWidths, 5.0)

            HIntensity = ic.HToMassIdxRatio * meanIntensityRatios[ic.massIdx]
            meanIntensityRatios = np.append(meanIntensityRatios, HIntensity)
            meanIntensityRatios /= np.sum(meanIntensityRatios)

    MSProperties = np.zeros(3*len(masses))
    MSProperties[::3] = masses
    MSProperties[1::3] = meanIntensityRatios
    MSProperties[2::3] = meanWidths
    sampleProperties = list(MSProperties)   

    return sampleProperties


def createMulScatWorkspaces(
    ic: Any, ws: Any, sampleProperties: List[float]
) -> Any:
    """Execute ``VesuvioCalculateMS`` and normalise the correction workspaces.

    Creates ``_TotScattering`` and ``_MulScattering`` workspaces,
    normalises them to the data, and renames them with the parent
    workspace name prefix.

    Args:
        ic: Completed initial-conditions object with
            ``transmission_guess``, ``multiple_scattering_order``, and
            ``number_of_events``.
        ws: The data workspace (used for naming and normalisation).
        sampleProperties: Flat interleaved list from
            ``calcMSCorrectionSampleProperties``.

    Returns:
        The ``_MulScattering`` workspace.
    """

    print("\nEvaluating the Multiple Scattering Correction...\n")
    # selects only the masses, every 3 numbers
    MS_masses = sampleProperties[::3]
    # same as above, but starts at first intensities
    MS_amplitudes = sampleProperties[1::3]

    dens, trans = VesuvioThickness(
        Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=ic.transmission_guess, Thickness=0.1
        )

    _TotScattering, _MulScattering = VesuvioCalculateMS(
        ws, 
        NoOfMasses=len(MS_masses), 
        SampleDensity=dens.cell(9, 1),
        AtomicProperties=sampleProperties, 
        BeamRadius=2.5,
        NumScatters=ic.multiple_scattering_order,
        NumEventsPerRun=int(ic.number_of_events)
        )

    data_normalisation = Integration(ws)
    simulation_normalisation = Integration("_TotScattering")
    for workspace in ("_MulScattering", "_TotScattering"):
        Divide(LHSWorkspace=workspace, RHSWorkspace=simulation_normalisation, OutputWorkspace=workspace)
        Multiply(LHSWorkspace=workspace, RHSWorkspace=data_normalisation, OutputWorkspace=workspace)
        RenameWorkspace(InputWorkspace=workspace, OutputWorkspace=ws.name()+workspace)
        SumSpectra(ws.name()+workspace, OutputWorkspace=ws.name()+workspace+"_Sum")
        
    DeleteWorkspaces(
        [data_normalisation, simulation_normalisation, trans, dens]
        )
    # The only remaining workspaces are the _MulScattering and _TotScattering
    return mtd[ws.name()+"_MulScattering"]


def createWorkspacesForGammaCorrection(
    ic: Any, meanWidths: np.ndarray, meanIntensityRatios: np.ndarray, wsNCPM: Any
) -> Any:
    """Run the Mantid gamma-background correction and return the result.

    Calls ``VesuvioCalculateGammaBackground`` with Gaussian Compton
    profiles built from the fitted mean widths and intensities.  The
    resulting background is scaled by 0.9 before being returned for
    subtraction.

    Args:
        ic: Completed initial-conditions object with ``masses``.
        meanWidths: Mean NCP widths, shape ``(n_masses,)``.
        meanIntensityRatios: Mean intensity ratios, shape
            ``(n_masses,)``.
        wsNCPM: Mantid workspace with NCP-masked data.

    Returns:
        The ``_Gamma_Background`` workspace to be subtracted.
    """

    inputWS = wsNCPM.name()

    # I do not know why, but setting these instrument parameters is required
    SetInstrumentParameter(inputWS, ParameterName='hwhm_lorentz', 
                            ParameterType='Number', Value='24.0')
    SetInstrumentParameter(inputWS, ParameterName='sigma_gauss', 
                            ParameterType='Number', Value='73.0')

    profiles = calcGammaCorrectionProfiles(ic.masses, meanWidths, meanIntensityRatios)

    background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=inputWS, ComptonFunction=profiles)
    
    RenameWorkspace(InputWorkspace= background, OutputWorkspace = inputWS+"_Gamma_Background")
    Scale(InputWorkspace = inputWS+"_Gamma_Background", OutputWorkspace = inputWS+"_Gamma_Background", 
        Factor=0.9, Operation="Multiply")
    DeleteWorkspace(corrected)
    return mtd[inputWS+"_Gamma_Background"]


def calcGammaCorrectionProfiles(
    masses: np.ndarray, meanWidths: np.ndarray, meanIntensityRatios: np.ndarray
) -> str:
    """Build the Mantid ``ComptonFunction`` string for gamma correction.

    Concatenates ``GaussianComptonProfile`` entries for each mass,
    separated by semicolons.

    Args:
        masses: Atomic masses, shape ``(n_masses,)``.
        meanWidths: Mean NCP widths, shape ``(n_masses,)``.
        meanIntensityRatios: Mean intensity ratios, shape
            ``(n_masses,)``.

    Returns:
        A semicolon-delimited profile string for
        ``VesuvioCalculateGammaBackground``.
    """

    masses = masses.flatten()
    profiles = ""
    for mass, width, intensity in zip(masses, meanWidths, meanIntensityRatios):
        profiles += "name=GaussianComptonProfile,Mass="   \
                    + str(mass) + ",Width=" + str(width)  \
                    + ",Intensity=" + str(intensity) + ';'
    print("\n The sample properties for Gamma Correction are:\n",
            profiles)
    return profiles


class resultsObject:
    """Collector for per-iteration NCP fitting results.

    Gathers all fit workspaces, best-fit parameters, NCP arrays, and
    mean widths/intensities from the AnalysisDataService into NumPy
    arrays that can be saved as ``.npz`` files for regression testing.

    Expects workspaces named ``ic.name + "0"``, ``ic.name + "1"``, …
    (and their associated ``_TOF_Fitted_Profiles``,
    ``_Best_Fit_NCP_Parameters``, ``_Mean_Widths_And_Intensities``
    tables) to be present in ``mtd``.

    Attributes:
        all_fit_workspaces: DataY from each iteration's fitted
            workspace, shape ``(n_iter, n_spectra, n_bins)``.
        all_spec_best_par_chi_nit: Best-fit parameter tables, shape
            ``(n_iter, n_spectra, 3*n_masses + 3)``.
        all_tot_ncp: Total NCP per iteration, shape
            ``(n_iter, n_spectra, n_bins)``.
        all_ncp_for_each_mass: Per-mass NCP per iteration, shape
            ``(n_iter, n_spectra, n_masses, n_bins)``.
        all_mean_widths: Mean widths per iteration, shape
            ``(n_iter, n_masses)``.
        all_mean_intensities: Mean intensities per iteration, shape
            ``(n_iter, n_masses)``.
        all_std_widths: Std widths per iteration, shape
            ``(n_iter, n_masses)``.
        all_std_intensities: Std intensities per iteration, shape
            ``(n_iter, n_masses)``.
    """

    def __init__(self, ic: Any) -> None:

        allIterNcp = []
        allFitWs = []
        allTotNcp = []
        allBestPar = []
        allMeanWidhts = []
        allMeanIntensities = []
        allStdWidths = []
        allStdIntensities = []
        j=0
        while True:
            try:
                wsIterName = ic.name+str(j)

                # Extract ws that were fitted
                ws = mtd[wsIterName]
                allFitWs.append(ws.extractY())

                # Extract total ncp
                totNcpWs = mtd[wsIterName+"_TOF_Fitted_Profiles"]
                allTotNcp.append(totNcpWs.extractY())

                # Extract best fit parameters
                fitParTable = mtd[wsIterName+"_Best_Fit_NCP_Parameters"]
                bestFitPars = []
                for key in fitParTable.keys():
                    bestFitPars.append(fitParTable.column(key))
                allBestPar.append(np.array(bestFitPars).T)
                
                # Extract individual ncp 
                allNCP = []
                i = 0
                while True:   # By default, looks for all ncp ws until it breaks
                    try:
                        ncpWsToAppend = mtd[wsIterName+"_TOF_Fitted_Profile_"+str(i)]
                        allNCP.append(ncpWsToAppend.extractY())
                        i += 1
                    except KeyError:
                        break
                allNCP = switchFirstTwoAxis(np.array(allNCP))
                allIterNcp.append(allNCP)
                
                # Extract Mean and Std Widths, Intensities
                meansTable = mtd[wsIterName + "_Mean_Widths_And_Intensities"]
                allMeanWidhts.append(meansTable.column("Mean Widths"))
                allStdWidths.append(meansTable.column("Std Widths"))
                allMeanIntensities.append(meansTable.column("Mean Intensities"))
                allStdIntensities.append(meansTable.column("Std Intensities"))  
                
                j+=1
            except KeyError:
                break

        self.all_fit_workspaces = np.array(allFitWs)
        self.all_spec_best_par_chi_nit = np.array(allBestPar)
        self.all_tot_ncp = np.array(allTotNcp)
        self.all_ncp_for_each_mass = np.array(allIterNcp)

        self.all_mean_widths = np.array(allMeanWidhts)
        self.all_mean_intensities = np.array(allMeanIntensities)
        self.all_std_widths = np.array(allStdWidths)
        self.all_std_intensities = np.array(allStdIntensities)

        # Pass all attributes of ic into attributes to be used whithin this object
        self.maskedDetectorIdx = ic.maskedDetectorIdx
        self.masses = ic.masses
        self.noOfMasses = ic.noOfMasses
        self.resultsSavePath = ic.resultsSavePath


    def save(self) -> None:
        """Save all result arrays to an ``.npz`` file at ``self.resultsSavePath``."""

        # TODO: Take out nans next time when running original results
        # Because original results were recently saved with nans, mask spectra with nans
        self.all_spec_best_par_chi_nit[:, self.maskedDetectorIdx, :] = np.nan
        self.all_ncp_for_each_mass[:, self.maskedDetectorIdx, :, :] = np.nan
        self.all_tot_ncp[:, self.maskedDetectorIdx, :] = np.nan

        savePath = self.resultsSavePath
        np.savez(savePath,
                 all_fit_workspaces=self.all_fit_workspaces,
                 all_spec_best_par_chi_nit=self.all_spec_best_par_chi_nit,
                 all_mean_widths=self.all_mean_widths,
                 all_mean_intensities=self.all_mean_intensities,
                 all_std_widths=self.all_std_widths,
                 all_std_intensities=self.all_std_intensities,
                 all_tot_ncp=self.all_tot_ncp,
                 all_ncp_for_each_mass=self.all_ncp_for_each_mass)

           
