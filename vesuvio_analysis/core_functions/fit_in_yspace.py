from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .stream_manager import StreamManager

from dataclasses import replace
import matplotlib.pyplot as plt
import numpy as np
from mantid.simpleapi import *
from scipy import optimize
from scipy import  signal
from pathlib import Path
from iminuit import Minuit, cost, util
from iminuit.util import describe
from vesuvio_analysis.core_functions.iminuit_costs import GlobalNCPCostFunction
from vesuvio_analysis.core_functions.plot_style import set_thesis_style, figure_factory
import jacobi
import time

repoPath = Path(__file__).absolute().parent  # Path to the repository


def fitInYSpaceProcedure(
    yFitIC: Any,
    IC: Any,
    wsTOF: Any,
    stream_manager: Optional["StreamManager"] = None,
) -> "ResultsYFitObject":
    """Perform the full y-space fitting procedure on a corrected TOF workspace.

    Orchestrates the conversion from TOF to y-space, optional
    symmetrisation, iMinuit fitting, Mantid Fit validation, and
    optional global fit.  All intermediate workspaces are stored in the
    AnalysisDataService.

    Steps:
        1. Extract NCP profiles from named workspaces.
        2. Compute Mantid resolution for the first mass.
        3. Subtract all masses except the first to isolate mass₀.
        4. Convert to y-space, rebin, normalise, and weight-average.
        5. Optionally symmetrise about y = 0.
        6. Fit with iMinuit (``fitProfileMinuit``).
        7. Fit with Mantid Fit (``fitProfileMantidFit``).
        8. Optionally run a global fit across detector groups.

    Expects workspace ``wsTOF.name()`` and its associated
    ``_TOF_Fitted_Profile_*`` workspaces to be present in ``mtd``.

    Args:
        yFitIC: ``YSpaceFitInitialConditions`` controlling rebinning,
            symmetrisation, model, Minos, and global fit.
        IC: Completed ``BackwardInitialConditions`` or
            ``ForwardInitialConditions`` object.
        wsTOF: Mantid workspace containing the fully-corrected TOF
            data from the last MS/GC iteration.
        stream_manager: Optional :class:`StreamManager` for capturing
            y-space data streams.  When ``None`` (default),
            no additional persistence is performed.

    Returns:
        A ``ResultsYFitObject`` containing y-space fit results and
        saved to ``.npz``.
    """
    from .stream_manager import DataLevel

    ncpForEachMass = extractNCPFromWorkspaces(wsTOF, IC)
    wsResSum, wsRes = calculateMantidResolutionFirstMass(IC, yFitIC, wsTOF)

    wsTOFMass0 = subtractAllMassesExceptFirst(IC, wsTOF, ncpForEachMass)
    
    wsJoY, wsJoYAvg = ySpaceReduction(wsTOFMass0, IC.masses[0], yFitIC, ncpForEachMass[:, 0, :])
    
    if yFitIC.symmetrisationFlag:
        wsJoYAvg = symmetrizeWs(wsJoYAvg)

    fitProfileMinuit(yFitIC, wsJoYAvg, wsResSum)
    fitProfileMantidFit(yFitIC, wsJoYAvg, wsResSum)
    
    printYSpaceFitResults(wsJoYAvg.name())

    yfitResults = ResultsYFitObject(IC, yFitIC, wsTOF.name(), wsJoYAvg.name())
    yfitResults.save()

    # L3 — capture y-space fit results
    if stream_manager is not None:
        stream_manager.capture(
            "joy_avg", yfitResults.YSpaceSymSumDataY,
            DataLevel.FINAL_PHYSICS, domain="y",
        )
        stream_manager.capture(
            "joy_avg_err", yfitResults.YSpaceSymSumDataE,
            DataLevel.FINAL_PHYSICS, domain="y",
        )
        stream_manager.capture(
            "resolution", yfitResults.resolution,
            DataLevel.FINAL_PHYSICS, domain="y",
        )
        stream_manager.capture(
            "fit_popt", yfitResults.popt,
            DataLevel.FINAL_PHYSICS, domain="y",
        )
        stream_manager.capture(
            "fit_perr", yfitResults.perr,
            DataLevel.FINAL_PHYSICS, domain="y",
        )
        stream_manager.set_metadata("fit_model", str(yFitIC.fitModel))

    if yFitIC.globalFit:
        runGlobalFit(wsJoY, wsRes, IC, yFitIC) 
        
    return yfitResults


def extractNCPFromWorkspaces(wsFinal: Any, ic: Any) -> np.ndarray:
    """Extract per-mass NCP arrays from named Mantid workspaces.

    Reads workspaces ``wsFinal.name() + "_TOF_Fitted_Profile_0"``
    through ``…_Profile_{n-1}`` and stacks them into a single array
    organised per spectrum.

    Expects all per-mass NCP workspaces to be present in ``mtd``.

    Args:
        wsFinal: The final corrected TOF workspace.
        ic: Completed initial-conditions object with ``noOfMasses``.

    Returns:
        NCP array organised by spectrum, shape
        ``(n_spectra, n_masses, n_bins)``.
    """

    ws0_y = mtd[wsFinal.name() + "_TOF_Fitted_Profile_0"].extractY()
    n_spectra, n_bins = ws0_y.shape
    ncpForEachMass = np.empty((ic.noOfMasses, n_spectra, n_bins))
    ncpForEachMass[0] = ws0_y
    for i in range(1, ic.noOfMasses):
        ncpForEachMass[i] = mtd[wsFinal.name() + "_TOF_Fitted_Profile_" + str(i)].extractY()

    # Ensure shape of ncp matches data
    shape = ncpForEachMass.shape
    assert shape[0] == ic.noOfMasses
    assert shape[1] == wsFinal.getNumberHistograms()
    # Final dimension can be missing last col or not
    assert ((shape[2]==wsFinal.blocksize()) | (shape[2]==wsFinal.blocksize()-1))

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)  # Organizes ncp by spectra
    print(f"\nExtracted NCP profiles from workspaces.\n")
    return ncpForEachMass


def calculateMantidResolutionFirstMass(
    IC: Any, yFitIC: Any, ws: Any
) -> Tuple[Any, Any]:
    """Compute the Mantid VesuvioResolution for the first mass.

    Calls ``VesuvioResolution`` per spectrum, rebins to the y-space
    grid, sums across spectra, and normalises.  The resolution
    workspace is used for convolution in the y-space fit.

    Args:
        IC: Completed initial-conditions object with ``masses`` and
            ``maskedDetectorIdx``.
        yFitIC: Y-space fit configuration with
            ``rebinParametersForYSpaceFit``.
        ws: The corrected TOF workspace.

    Returns:
        A 2-tuple ``(wsResSum, wsRes)`` where *wsResSum* is the
        normalised summed-spectra resolution and *wsRes* is the
        per-spectrum resolution workspace.
    """

    mass = IC.masses[0]

    resName = ws.name()+"_Resolution"
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
        Rebin(InputWorkspace="tmp", Params=yFitIC.rebinParametersForYSpaceFit, OutputWorkspace="tmp")

        if index == 0:   # Ensures that workspace has desired units
            RenameWorkspace("tmp",  resName)
        else:
            AppendSpectra(resName, "tmp", OutputWorkspace=resName)
   
    MaskDetectors(resName, WorkspaceIndexList=IC.maskedDetectorIdx)
    wsResSum = SumSpectra(InputWorkspace=resName, OutputWorkspace=resName+"_Sum")
 
    normalise_workspace(wsResSum)
    DeleteWorkspace("tmp")
    return wsResSum, mtd[resName]


def subtractAllMassesExceptFirst(
    ic: Any, ws: Any, ncpForEachMass: np.ndarray
) -> Any:
    """Subtract all NCP profiles except the first mass from the TOF data.

    Isolates the contribution of mass₀ (typically hydrogen) by
    summing and subtracting the NCP of all heavier masses.  Masked
    bins are preserved.

    Args:
        ic: Completed initial-conditions object with
            ``maskedDetectorIdx``.
        ws: The corrected TOF workspace.
        ncpForEachMass: NCP per spectrum and mass, shape
            ``(n_spectra, n_masses, n_bins)``.

    Returns:
        A Mantid workspace named ``ws.name() + "_Mass0"`` containing
        only the mass₀ signal.
    """

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMassExceptFirst = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotalExceptFirst = np.sum(ncpForEachMassExceptFirst, axis=0)

    dataX, dataY, dataE = extractWS(ws)

    # Adjust for last column missing or not
    dataY[:, :ncpTotalExceptFirst.shape[1]] -= ncpTotalExceptFirst

    # Ignore any masked bins (columns) from initial ws
    mask = np.all(ws.extractY()==0, axis=0)
    dataY[:, mask] = 0

    wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_Mass0")
    passDataIntoWS(dataX, dataY, dataE, wsSubMass)
    MaskDetectors(Workspace=wsSubMass, WorkspaceIndexList=ic.maskedDetectorIdx)  
    SumSpectra(InputWorkspace=wsSubMass.name(), OutputWorkspace=wsSubMass.name()+"_Sum")
    return wsSubMass


def switchFirstTwoAxis(A: np.ndarray) -> np.ndarray:
    """Transpose the first two axes of a 3-D array.

    Args:
        A: Array with shape ``(a, b, c)``.

    Returns:
        Array with shape ``(b, a, c)``.
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def ySpaceReduction(
    wsTOF: Any, mass0: float, yFitIC: Any, ncp: np.ndarray
) -> Tuple[Any, Any]:
    """Convert TOF data to y-space and produce a weighted-average spectrum.

    Handles two masking strategies:

    * **NAN**: Accumulates data points per y-space bin and performs a
      weighted average that naturally handles missing data.
    * **NCP**: Fills masked bins with the fitted NCP before standard
      rebinning and normalisation.

    If no masked columns are present, standard rebinning is used.

    Args:
        wsTOF: Mantid workspace with mass₀ TOF data.
        mass0: Mass of the first atom in a.m.u.
        yFitIC: Y-space fit configuration with
            ``rebinParametersForYSpaceFit`` and ``maskTypeProcedure``.
        ncp: NCP for mass₀ only, shape ``(n_spectra, n_bins)``.

    Returns:
        A 2-tuple ``(wsJoYN, wsJoYAvg)`` — the normalised per-spectrum
        J(y) workspace and its weighted average.
    """
    
    rebinPars = yFitIC.rebinParametersForYSpaceFit
    
    if np.any(np.all(wsTOF.extractY()==0, axis=0)):  # Masked columns present
        
        if yFitIC.maskTypeProcedure=="NAN":
            # Build special workspace to store accumulated points
            wsJoY = convertToYSpace(wsTOF, mass0)
            xp = buildXRangeFromRebinPars(yFitIC)
            wsJoYB = dataXBining(wsJoY, xp)      # Unusual ws with several dataY points per each dataX point

            # Need normalisation values from NCP masked workspace
            wsTOFNCP = replaceZerosWithNCP(wsTOF, ncp)
            wsJoYNCP = convertToYSpace(wsTOFNCP, mass0)
            wsJoYNCPN, wsJoYInt = rebinAndNorm(wsJoYNCP, rebinPars)

            # Normalize spectra of specieal workspace
            wsJoYN = Divide(wsJoYB, wsJoYInt, OutputWorkspace=wsJoYB.name()+"_Normalised")
            wsJoYAvg = weightedAvgXBins(wsJoYN, xp)
            return wsJoYN, wsJoYAvg
        
        elif yFitIC.maskTypeProcedure=="NCP":
            wsTOF = replaceZerosWithNCP(wsTOF, ncp)

        else:
            raise ValueError("""
            Masked TOF bins were found but no valid procedure in y-space fit was selected.
            Options: 'NAN', 'NCP'
            """)

    wsJoY = convertToYSpace(wsTOF, mass0)
    wsJoYN, wsJoYI = rebinAndNorm(wsJoY, rebinPars)
    wsJoYAvg = weightedAvgCols(wsJoYN)
    return wsJoYN, wsJoYAvg


def convertToYSpace(wsTOF: Any, mass0: float) -> Any:
    """Convert a TOF workspace to y-space via Mantid ``ConvertToYSpace``.

    Args:
        wsTOF: Input TOF workspace.
        mass0: Atomic mass for the conversion (a.m.u.).

    Returns:
        The y-space Mantid workspace.
    """

    wsJoY = ConvertToYSpace(wsTOF, Mass=mass0, OutputWorkspace=wsTOF.name()+"_JoY")
    return wsJoY


def rebinAndNorm(wsJoY: Any, rebinPars: str) -> Tuple[Any, Any]:
    """Rebin a y-space workspace and normalise by the integrated intensity.

    Args:
        wsJoY: Input y-space workspace.
        rebinPars: Mantid rebin parameter string ``"start, step, end"``.

    Returns:
        A 2-tuple ``(wsJoYNorm, wsJoYInt)`` — the normalised workspace
        and the integration workspace used for normalisation.
    """

    wsJoYR = Rebin(InputWorkspace=wsJoY, Params=rebinPars, FullBinsOnly=True, OutputWorkspace=wsJoY.name()+"_Rebinned")
    wsJoYInt = Integration(wsJoYR, OutputWorkspace=wsJoYR.name()+"_Integrated")
    wsJoYNorm = Divide(wsJoYR, wsJoYInt, OutputWorkspace=wsJoYR.name()+"_Normalised")
    return wsJoYNorm, wsJoYInt


def replaceZerosWithNCP(ws: Any, ncp: np.ndarray) -> Any:
    """Replace zero-masked columns in dataY with the NCP prediction.

    Creates a clone named ``ws.name() + "_NCPMasked"`` with the
    replacement applied.  Errors are not modified.

    Args:
        ws: Mantid workspace with masked columns (dataY == 0).
        ncp: NCP values to fill in, shape ``(n_spectra, n_bins)``.

    Returns:
        A cloned workspace with zero columns replaced by NCP values.
    """
    dataX, dataY, dataE = extractWS(ws)
    mask = np.all(dataY==0, axis=0)    # Masked Cols 

    dataY[:, mask] = ncp[:, mask[:ncp.shape[1]]]   # mask of ncp adjusted for last col present or not

    wsMasked = CloneWorkspace(ws, OutputWorkspace=ws.name()+"_NCPMasked")
    passDataIntoWS(dataX, dataY, dataE, wsMasked)
    SumSpectra(wsMasked, OutputWorkspace=wsMasked.name()+"_Sum")
    return wsMasked  


def buildXRangeFromRebinPars(yFitIC: Any) -> np.ndarray:
    """Build an array of bin centres from the rebin parameter string.

    Args:
        yFitIC: Y-space fit configuration with
            ``rebinParametersForYSpaceFit``.

    Returns:
        Array of bin centres, shape ``(n_bins,)``.
    """
    # Range used in case mask is set to NAN
    first, step, last = [float(s) for s in yFitIC.rebinParametersForYSpaceFit.split(",")]
    xp = np.arange(first, last, step) + step/2   # Correction to match Mantid range
    return xp


def dataXBining(ws: Any, xp: np.ndarray) -> Any:
    """Re-bin dataX to the closest centres in *xp*, allowing multiple dataY per bin.

    Modifies the dataX values of a workspace so that each point is
    snapped to the nearest bin centre from *xp*.  Points outside the
    range are set to ``NaN``.  The output workspace may have several
    dataY values per dataX point.

    Args:
        ws: Input Mantid workspace.
        xp: Array of target bin centres, shape ``(n_bins,)``.

    Returns:
        A cloned workspace with modified dataX.
    """

    assert np.min(xp[:-1]-xp[1:]) == np.max(xp[:-1]-xp[1:]), "Bin widths need to be the same."
    step = xp[1] - xp[0]   # Calculate step from first two numbers
    # Form bins with xp being the centers
    bins = np.append(xp, [xp[-1]+step]) - step/2

    dataX, dataY, dataE = extractWS(ws)
    # Loop below changes only the values of DataX
    for i, x in enumerate(dataX):

        # Select only valid range xr
        mask = (x<np.min(bins)) | (x>np.max(bins))
        xr = x[~mask]

        idxs = np.digitize(xr, bins)
        newXR = np.array([xp[idx] for idx in idxs-1])  # Bin idx 1 refers to first bin ie idx 0 of centers

        # Pad invalid values with nans
        newX = x
        newX[mask] = np.nan    # Cannot use 0 as to not be confused with a dataX value
        newX[~mask] = newXR
        dataX[i] = newX       # Update DataX

    # Mask DataE values in same places as DataY values 
    dataE[dataY==0] = 0

    wsXBins = CloneWorkspace(ws, OutputWorkspace=ws.name()+"_XBinned")
    wsXBins = passDataIntoWS(dataX, dataY, dataE, wsXBins)
    return wsXBins


def weightedAvgXBins(wsXBins: Any, xp: np.ndarray) -> Any:
    """Weighted average over grouped dataY points sharing the same bin centre.

    Args:
        wsXBins: Workspace from ``dataXBining`` with multiple dataY
            per bin centre.
        xp: Target bin centres, shape ``(n_bins,)``.

    Returns:
        A 1-spectrum workspace with the weighted average.
    """
    dataX, dataY, dataE = extractWS(wsXBins)

    meansY, meansE = weightedAvgXBinsArr(dataX, dataY, dataE, xp)

    wsYSpaceAvg = CreateWorkspace(DataX=xp, DataY=meansY, DataE=meansE, NSpec=1, OutputWorkspace=wsXBins.name()+"_WeightedAvg")
    return wsYSpaceAvg


def weightedAvgXBinsArr(
    dataX: np.ndarray, dataY: np.ndarray, dataE: np.ndarray, xp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted average over arrays where several dataY correspond to a single dataX.

    Groups all dataY (and dataE) values that share the same bin centre
    from *xp* and computes the inverse-variance weighted mean.  Zero
    and NaN values are ignored.

    Args:
        dataX: X values (snapped to centres), shape
            ``(n_spectra, n_pts)``.
        dataY: Y values, same shape.
        dataE: Error values, same shape.
        xp: Target bin centres, shape ``(n_bins,)``.

    Returns:
        A 2-tuple ``(meansY, meansE)`` each of shape ``(n_bins,)``.
    """
    meansY = np.zeros(len(xp))
    meansE = np.zeros(len(xp))

    for i in range(len(xp)):
        # Perform weighted average over all dataY and dataE values with the same xp[i]
        # Change shape to column to match weighted average function
        pointMask = dataX==xp[i]
        allY = dataY[pointMask][:, np.newaxis]
        allE = dataE[pointMask][:, np.newaxis]

        # If no points were found for a given abcissae
        if (np.sum(pointMask)==0):   mY, mE = 0, 0  # Mask with zeros
        
        # If one point was found, set to that point
        elif (np.sum(pointMask)==1):  mY, mE = allY.flatten(), allE.flatten()

        # Weighted avg over all spectra and several points per spectra
        else:
            # Case of bootstrap replica with no errors
            if np.all(dataE==0):      
                mY = avgArr(allY)
                mE = 0

            # Default for most cases
            else:
                mY, mE = weightedAvgArr(allY, allE)    # Outputs masked values as zeros
        
        # DataY and DataE should never reach NaN, but safeguard in case they do
        if (mE==np.nan) | (mY==np.nan):  
            mY, mE = 0, 0

        meansY[i] = mY
        meansE[i] = mE
    
    return meansY, meansE


def weightedAvgCols(wsYSpace: Any) -> Any:
    """Compute the inverse-variance weighted average across spectra.

    Args:
        wsYSpace: Per-spectrum y-space workspace.

    Returns:
        A 1-spectrum workspace with the weighted average.
    """
    dataX, dataY, dataE = extractWS(wsYSpace)
    if np.all(dataE==0):      # Bootstrap case where errors are not used
        meanY = avgArr(dataY)
        meanE = np.zeros(meanY.shape)
    else:
        meanY, meanE = weightedAvgArr(dataY, dataE)
    wsYSpaceAvg = CreateWorkspace(DataX=dataX[0, :], DataY=meanY, DataE=meanE, NSpec=1, OutputWorkspace=wsYSpace.name()+"_WeightedAvg")
    return wsYSpaceAvg


def avgArr(dataYO: np.ndarray) -> np.ndarray:
    """Simple mean over rows (spectra) ignoring zeros.

    Zeros are treated as masked and replaced with ``NaN`` before
    averaging.

    Args:
        dataYO: 2-D array, shape ``(n_spectra, n_bins)``.

    Returns:
        Row-averaged array, shape ``(n_bins,)``.
    """

    assert len(dataYO)>1, "Averaging needs more than one element."

    dataY = dataYO.copy()
    dataY[dataY==0] = np.nan
    meanY = np.nanmean(dataY, axis=0)
    meanY[meanY==np.nan] = 0

    assert np.all(np.all(dataYO==0, axis=0)==(meanY==0)), "Columns of zeros should give zero."
    return meanY


def weightedAvgArr(
    dataYO: np.ndarray, dataEO: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-variance weighted average over rows (spectra).

    Zeros and NaNs are ignored.  The output masks columns that are
    entirely zero.

    Args:
        dataYO: Y values, shape ``(n_spectra, n_bins)``.
        dataEO: Error values, same shape.

    Returns:
        A 2-tuple ``(meanY, meanE)`` each of shape ``(n_bins,)``.
    """

    # Run some tests
    assert dataYO.shape==dataEO.shape, "Y and E arrays should have same shape for weighted average."
    assert np.all((dataYO==0)==(dataEO==0)), f"Masked zeros should match in DataY and DataE: {np.argwhere((dataYO==0)!=(dataEO==0))}"
    assert np.all(np.isnan(dataYO)==np.isnan(dataEO)), "Masked nans should match in DataY and DataE."
    assert len(dataYO) > 1, "Weighted average needs more than one element to be performed."

    dataY = dataYO.copy()  # Copy arrays not to change original data
    dataE = dataEO.copy()

    # Ignore invalid data by changing zeros to nans
    # If data is already masked with nans, it remains unaltered
    zerosMask = (dataY==0)
    dataY[zerosMask] = np.nan  
    dataE[zerosMask] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        invVar = 1 / np.square(dataE)
        sumInvVar = np.nansum(invVar, axis=0)
        meanY = np.nansum(dataY * invVar, axis=0) / sumInvVar
        meanE = np.sqrt(1 / sumInvVar)

    # Change invalid data back to original masking format with zeros
    nanInfMask = np.isinf(meanE) | np.isnan(meanE) | np.isnan(meanY)
    meanY[nanInfMask] = 0
    meanE[nanInfMask] = 0

    # Test that columns of zeros are left unchanged
    assert np.all((meanY==0)==(meanE==0)), "Weighted avg output should have masks in the same DataY and DataE."
    assert np.all((np.all(dataYO==0, axis=0) | np.all(np.isnan(dataYO), axis=0)) == (meanY==0)), "Masked cols should be ignored."
    
    return meanY, meanE


def normalise_workspace(ws_name: Any) -> None:
    """Normalise a workspace by its integrated intensity (in-place).

    Args:
        ws_name: Workspace name or handle to normalise.
    """
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def extractWS(ws: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract dataX, dataY, and dataE arrays from a Mantid workspace.

    Args:
        ws: A Mantid MatrixWorkspace.

    Returns:
        A 3-tuple ``(dataX, dataY, dataE)``.
    """
    return ws.extractX(), ws.extractY(), ws.extractE()


def passDataIntoWS(
    dataX: np.ndarray, dataY: np.ndarray, dataE: np.ndarray, ws: Any
) -> Any:
    """Write NumPy arrays back into a Mantid workspace in-place.

    Modifies all spectra of *ws* to hold the values from the input
    arrays.

    Args:
        dataX: X-axis values, shape ``(n_spectra, n_bins)``.
        dataY: Y-axis values, same shape.
        dataE: Error values, same shape.
        ws: The Mantid workspace to modify.

    Returns:
        The modified workspace *ws*.
    """
    for i in range(ws.getNumberHistograms()):
        ws.dataX(i)[:] = dataX[i, :]
        ws.dataY(i)[:] = dataY[i, :]
        ws.dataE(i)[:] = dataE[i, :]
    return ws


def symmetrizeWs(avgYSpace: Any) -> Any:
    """Symmetrise a y-space workspace about y = 0.

    For weighted data, inverse-variance symmetrisation is used; for
    data without errors (bootstrap), a simple mean of opposing points
    is computed.  Requires symmetric rebinning.

    Args:
        avgYSpace: 1-spectrum y-space workspace (weighted average).

    Returns:
        A cloned, symmetrised workspace named
        ``avgYSpace.name() + "_Symmetrised"``.
    """

    dataX, dataY, dataE = extractWS(avgYSpace)
    
    if np.all(dataE==0):
        dataYS = symArr(dataY)
        dataES = np.zeros(dataYS.shape)
    else:
        dataYS, dataES = weightedSymArr(dataY, dataE)

    wsSym = CloneWorkspace(avgYSpace, OutputWorkspace=avgYSpace.name()+"_Symmetrised")
    wsSym = passDataIntoWS(dataX, dataYS, dataES, wsSym)
    return wsSym


def symArr(dataYO: np.ndarray) -> np.ndarray:
    """Average each y-point with its mirror image about y = 0.

    Zeros are treated as absent and the mirror value is kept.

    Args:
        dataYO: 2-D array, shape ``(1, n_bins)`` (or general 2-D).

    Returns:
        Symmetrised array, same shape.
    """

    assert len(dataYO.shape) == 2, "Symmetrization is written for 2D arrays."
    dataY = dataYO.copy()  # Copy arrays not to risk changing original data
    coMask = dataY==0
    dataY[coMask] = np.nan

    yFlip = np.flip(dataY, axis=1)

    dataYS = np.nanmean(np.stack((dataY, yFlip)), axis=0)  # Normal avg between two numbers, cut-offs get ignored

    dataYS[dataYS==np.nan] = 0
    np.testing.assert_array_equal(dataYS, np.flip(dataYS, axis=1)), f"Symmetrisation failed in {np.argwhere(dataYS!=np.flip(dataYS))}"
    np.testing.assert_allclose(dataYS[coMask], np.flip(dataYO, axis=1)[coMask])
    return dataYS


def weightedSymArr(
    dataYO: np.ndarray, dataEO: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse-variance weighted symmetrisation of y-space data.

    Combines each point with its mirror image using inverse-variance
    weighting.  Cut-off values (zero-error points) are left
    unchanged by setting their error to infinity before the
    calculation.

    Args:
        dataYO: Y values, shape ``(1, n_bins)`` (or general 2-D).
        dataEO: Error values, same shape.

    Returns:
        A 2-tuple ``(dataYS, dataES)`` of symmetrised arrays.
    """
    assert len(dataYO.shape) == 2, "Symmetrization is written for 2D arrays."
    assert np.all((dataYO==0)==(dataEO==0)), "Masked values should have zeros on both dataY and dataE."
    
    dataY = dataYO.copy()  # Copy arrays not to risk changing original data
    dataE = dataEO.copy()

    cutOffMask = dataY==0
    # Change values of yerr to leave cut-offs unchanged during symmetrisation
    dataE[cutOffMask] = np.full(np.sum(cutOffMask), np.inf)

    yFlip = np.flip(dataY, axis=1)
    eFlip = np.flip(dataE, axis=1)

    # Inverse variance weighting
    dataYS = (dataY/dataE**2 + yFlip/eFlip**2) / (1/dataE**2 + 1/eFlip**2)
    dataES = 1 / np.sqrt(1/dataE**2 + 1/eFlip**2)

    # Deal with effects from previously changing dataE=np.inf
    nanInfMask = (dataES==np.inf) | (dataES==np.nan) | (dataYS==np.nan)
    dataYS[nanInfMask] = 0
    dataES[nanInfMask] = 0

    # Test that arrays are symmetrised
    np.testing.assert_array_equal(dataYS, np.flip(dataYS, axis=1)), f"Symmetrisation failed in {np.argwhere(dataYS!=np.flip(dataYS))}"
    np.testing.assert_array_equal(dataES, np.flip(dataES, axis=1)), f"Symmetrisation failed in {np.argwhere(dataES!=np.flip(dataES))}"

    # Test that cut-offs were not included in the symmetrisation
    np.testing.assert_allclose(dataYS[cutOffMask], np.flip(dataYO, axis=1)[cutOffMask])
    np.testing.assert_allclose(dataES[cutOffMask], np.flip(dataEO, axis=1)[cutOffMask])

    return dataYS, dataES


def fitProfileMinuit(yFitIC: Any, wsYSpaceSym: Any, wsRes: Any) -> None:
    """Fit the y-space profile using iMinuit.

    Builds a convolved model (J(y) model ⊗ resolution), creates the
    appropriate cost function (``cost.LeastSquares`` with errors or
    ``MyLeastSquares`` without), and minimises using Simplex → Migrad
    (for Gaussian) or Simplex → constrained Scipy (for non-Gaussian
    models).  Hessian errors are always computed.  Optionally runs
    Minos for asymmetric errors.

    Expects ``wsYSpaceSym`` and ``wsRes`` to be present in ``mtd``.

    Args:
        yFitIC: Y-space fit configuration with ``fitModel``,
            ``runMinos``, and ``showPlots``.
        wsYSpaceSym: The (optionally symmetrised) weighted-average
            J(y) workspace (1 spectrum).
        wsRes: The summed resolution workspace (1 spectrum).
    """

    dataX, dataY, dataE = extractFirstSpectra(wsYSpaceSym)
    resX, resY, resE = extractFirstSpectra(wsRes)
    assert np.all(dataX==resX), "Resolution should operate on the same range as DataX"

    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitModel)

    xDelta, resDense = oddPointsRes(resX, resY)
    def convolvedModel(x, y0, *pars):
        return y0 + signal.convolve(model(x, *pars), resDense, mode="same") * xDelta

    signature = describe(model)[:]      # Build signature of convolved function
    annotations = describe(model, annotations=True)
    signature[1:1] = ["y0"]     # Add intercept as first fitting parameter after range 'x'

    # Propagate model annotations for limits; y0 is unbounded (None)
    convolvedModel._parameters = {"y0": None}
    for name in signature[2:]:  # skip 'x' and 'y0'
        convolvedModel._parameters[name] = annotations.get(name)
    defaultPars["y0"] = 0    # Add initialization of parameter to dictionary

    # Fit only valid values, ignore cut-offs 
    dataXNZ, dataYNZ, dataENZ = selectNonZeros(dataX, dataY, dataE)

    # Fit with Minuit
    if np.all(dataE==0):   # Choose fitting without weights
        costFun = MyLeastSquares(dataXNZ, dataYNZ, convolvedModel)
    else:
        costFun = cost.LeastSquares(dataXNZ, dataYNZ, dataENZ, convolvedModel)
    # cost.LeastSquares (and MyLeastSquares) derive parameter names from
    # describe(convolvedModel). When _parameters is present on the model,
    # describe returns names from that mapping and the wrappers skip the first
    # entry as the independent variable, which can drop 'y0'. Propagate the
    # model _parameters explicitly so Minuit sees all fitting parameters.
    costFun._parameters = convolvedModel._parameters

    m = Minuit(costFun, **defaultPars)

    m.limits["A"] = (0, None)
    if yFitIC.fitModel=="DOUBLE_WELL":
        m.limits["d"] = (0, None)
        m.limits["R"] = (0, None)

    if yFitIC.fitModel=="SINGLE_GAUSSIAN":
        m.simplex()
        m.migrad()

        def constrFunc()->None:  # No constraint function for gaussian profile
            return
    else:
        def constrFunc(*pars):   # Constrain physical model before convolution
            return model(dataXNZ, *pars[1:])   # First parameter is intercept, not part of model()
        
        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

    # Explicit calculation of Hessian after the fit
    m.hesse()

    # Weighted Chi2
    chi2 = m.fval / (len(dataXNZ)-m.nfit)

    # Best fit and confidence band
    # Calculated for the whole range of dataX, including where zero
    dataYFit, dataYCov = jacobi.propagate(lambda pars: convolvedModel(dataX, *pars), m.values, m.covariance)
    dataYSigma = np.sqrt(np.diag(dataYCov))
    dataYSigma *= chi2        # Weight the confidence band
    Residuals = dataY - dataYFit

    # Create workspace to store best fit curve and errors on the fit
    wsMinFit = createFitResultsWorkspace(wsYSpaceSym, dataX, dataY, dataE, dataYFit, dataYSigma, Residuals)
    saveMinuitPlot(yFitIC, wsMinFit, m)

    # Calculate correlation matrix
    corrMatrix = m.covariance.correlation()
    corrMatrix *= 100

    # Create correlation tableWorkspace
    createCorrelationTableWorkspace(wsYSpaceSym, m.parameters, corrMatrix)

    # Run Minos
    fitCols = runMinos(m, yFitIC, constrFunc, wsYSpaceSym.name())

    # Create workspace with final fitting parameters and their errors
    createFitParametersTableWorkspace(wsYSpaceSym, *fitCols, chi2)
    return 


def extractFirstSpectra(ws: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the first spectrum from a workspace.

    Args:
        ws: Mantid workspace.

    Returns:
        A 3-tuple ``(dataX, dataY, dataE)`` each of shape
        ``(n_bins,)``.
    """
    dataY = ws.extractY()[0]
    dataX = ws.extractX()[0]
    dataE = ws.extractE()[0]
    return dataX, dataY, dataE


def selectModelAndPars(
    modelFlag: str,
) -> Tuple[Callable, Dict[str, float], List[str]]:
    """Return the J(y) model function, default parameters, and shared-parameter names.

    The model is a plain Python callable with signature
    ``model(x, *pars)``.  The *sharedPars* list identifies parameters
    that are shared across detector groups in the global fit; they must
    appear last in the function signature.

    Supported models:
        * ``"SINGLE_GAUSSIAN"``
        * ``"GC_C4"``, ``"GC_C6"``, ``"GC_C4_C6"`` (Gram–Charlier)
        * ``"DOUBLE_WELL"``
        * ``"ANSIO_GAUSSIAN"``
        * ``"MULTIVARIATE_GAUSSIAN"``

    Args:
        modelFlag: String identifier selecting the model.

    Returns:
        A 3-tuple ``(model, defaultPars, sharedPars)`` where *model*
        is the callable, *defaultPars* is a dict of starting values,
        and *sharedPars* is a list of parameter name strings.

    Raises:
        ValueError: If *modelFlag* is not recognised.
    """

    if modelFlag == "SINGLE_GAUSSIAN":
        def model(x, A, x0, sigma):
            return  A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

        defaultPars = {"A":1, "x0":0, "sigma":5}
        sharedPars = ["sigma"]    # Used only in Global fit

    elif (modelFlag=="GC_C4_C6"):
        def model(x, A, x0, sigma1, c4, c6):
            return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*np.pi*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
        
        defaultPars = {"A":1, "x0":0, "sigma1":6, "c4":0, "c6":0} 
        sharedPars = ["sigma1", "c4", "c6"]     # Used only in Global fit

    elif modelFlag=="GC_C4":
        def model(x, A, x0, sigma1, c4):
            return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*np.pi*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12))
        
        defaultPars = {"A":1, "x0":0, "sigma1":6, "c4":0} 
        sharedPars = ["sigma1", "c4"]     # Used only in Global fit   
    
    elif modelFlag=="GC_C6":
        def model(x, A, x0, sigma1, c6):
            return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*np.pi*sigma1**2)) \
                    *(1 + +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
        
        
        defaultPars = {"A":1, "x0":0, "sigma1":6, "c6":0} 
        sharedPars = ["sigma1", "c6"]     # Used only in Global fit   

    elif modelFlag=="DOUBLE_WELL":
        def model(x, A, d, R, sig1, sig2):
            h = 2.04
            theta = np.linspace(0, np.pi, 300)[:, np.newaxis]   # 300 points seem like a good estimate for ~10 examples
            y = x[np.newaxis, :]

            sigTH = np.sqrt( sig1**2*np.cos(theta)**2 + sig2**2*np.sin(theta)**2 )
            alpha = 2*( d*sig2*sig1*np.sin(theta) / sigTH )**2
            beta = ( 2*sig1**2*d*np.cos(theta) / sigTH**2 ) * y
            denom = 2.506628 * sigTH * (1 + R**2 + 2*R*np.exp(-2*d**2*sig1**2))
            jp = np.exp( -y**2/(2*sigTH**2)) * (1 + R**2 + 2*R*np.exp(-alpha)*np.cos(beta)) / denom
            jp *= np.sin(theta)

            JBest = np.trapz(jp, x=theta, axis=0)
            JBest /= np.abs(np.trapz(JBest, x=y))
            JBest *= A
            return JBest

        defaultPars = {"A":1, "d":1, "R":1, "sig1":3, "sig2":5}  # TODO: Starting parameters and bounds?
        sharedPars = ["d", "R", "sig1", "sig2"]      # Only varying parameter is amplitude A     

    elif modelFlag=="ANSIO_GAUSSIAN":
        # Ansiotropic case
        def model(x, A, sig1, sig2):
            h = 2.04
            theta = np.linspace(0, np.pi, 300)[:, np.newaxis]
            y = x[np.newaxis, :]

            sigTH = np.sqrt( sig1**2*np.cos(theta)**2 + sig2**2*np.sin(theta)**2 )
            jp = np.exp( -y**2/(2*sigTH**2)) / (2.506628*sigTH)
            jp *= np.sin(theta)

            JBest = np.trapz(jp, x=theta, axis=0)
            JBest /= np.abs(np.trapz(JBest, x=y))
            JBest *= A
            return JBest

        defaultPars = {"A":1, "sig1":3, "sig2":5}
        sharedPars = ["sig1", "sig2"]           

    elif modelFlag=="MULTIVARIATE_GAUSSIAN":
        def model(x, A, sig_x, sig_y, sig_z):

            y = x[:, np.newaxis, np.newaxis]
            n_steps = 50       # Low number of integration steps because otherwise too slow
            theta = np.linspace(0, np.pi / 2, n_steps)[np.newaxis, :, np.newaxis]
            phi = np.linspace(0, np.pi / 2, n_steps)[np.newaxis, np.newaxis, :]


            S2_inv = np.sin(theta)**2 * np.cos(phi)**2 / sig_x**2     \
                   + np.sin(theta)**2 * np.sin(phi)**2 / sig_y**2   \
                   + np.cos(theta)**2 / sig_z**2

            J = np.sin(theta) / S2_inv * np.exp(- y**2 / 2 * S2_inv)

            J = np.trapz(J, x=phi, axis=2)[:, :, np.newaxis]    # Keep shape
            J = np.trapz(J, x=theta, axis=1)

            J *= A * 2 / np.pi * 1 / np.sqrt(2 * np.pi) * 1 / (sig_x * sig_y * sig_z)    # Normalisation 
            J = J.squeeze()
            return J

        defaultPars = {"A": 1, "sig_x": 5, "sig_y": 5, "sig_z": 5}
        sharedPars = ["sig_x", "sig_y", "sig_z"]

    else:
        raise ValueError("Fitting Model not recognized, available options: 'SINGLE_GAUSSIAN', 'GC_C4_C6', 'GC_C4'")
    
    print("\nShared Parameters: ", [key for key in sharedPars])
    print("\nUnshared Parameters: ", [key for key in defaultPars if key not in sharedPars])
    
    assert all(isinstance(item, str) for item in sharedPars), "Parameters in list must be strings."
    assert describe(model)[-len(sharedPars):]==sharedPars, "Function signature needs to have shared parameters at the end: model(*unsharedPars, *sharedPars)"
    
    return model, defaultPars, sharedPars


def selectNonZeros(
    dataX: np.ndarray, dataY: np.ndarray, dataE: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select only non-zero data points (mask removal).

    Uses zeros in *dataY* as the mask indicator because *dataE* can be
    all zeros in certain bootstrap types.

    Args:
        dataX: X values, shape ``(n_bins,)``.
        dataY: Y values, same shape.
        dataE: Error values, same shape.

    Returns:
        A 3-tuple of the non-zero subsets.
    """
    zeroMask = dataY==0  

    dataXNZ = dataX[~zeroMask]
    dataYNZ = dataY[~zeroMask]
    dataENZ = dataE[~zeroMask]   
    return dataXNZ, dataYNZ, dataENZ 


class MyLeastSquares:
    """Unweighted least-squares cost function compatible with iMinuit.

    Used when ``dataE`` is all zeros (e.g. bootstrap replicas without
    error propagation).  Implements the modern iMinuit v2.x class-based
    cost-function pattern with ``errordef``, ``__call__``,
    ``_parameters``, and ``ndata``.

    Attributes:
        errordef: Set to ``Minuit.LEAST_SQUARES`` so Minuit computes
            parameter errors correctly.
        model: The model callable ``model(x, *par)``.
        x: Abscissa values.
        y: Observed values.
        _parameters: Dict mapping parameter names to limits (or
            ``None`` for unbounded).  Used by ``Minuit`` for automatic
            signature detection, replacing the deprecated ``func_code``.
    """

    errordef = Minuit.LEAST_SQUARES # for Minuit to compute errors correctly

    def __init__(self, x: np.ndarray, y: np.ndarray, model: Callable) -> None:
        """Initialise the cost function.

        Args:
            x: Abscissa values, shape ``(n_points,)``.
            y: Observed values, same shape.
            model: Callable ``model(x, *par)`` returning predicted y.
        """

        self.model = model  # model predicts y for given x
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        # Use annotations=True to propagate any type-annotation limits
        # from the model signature, following iminuit best practice
        # (see: scikit-hep.org/iminuit/notebooks/generic_least_squares.html)
        pars = describe(model, annotations=True)
        model_args = iter(pars)
        next(model_args)  # skip the first argument (independent variable x)
        self._parameters = {k: pars[k] for k in model_args}

    def __call__(self, *par: float) -> float:
        """Evaluate the unweighted sum of squared residuals.

        Args:
            *par: Model parameters.

        Returns:
            Scalar cost value.
        """

        ym = self.model(self.x, *par)
        return np.sum((self.y - ym) ** 2)

    @property
    def ndata(self) -> int:
        """Number of non-masked (non-zero) data points."""
        return len(self.x)


def createFitResultsWorkspace(
    wsYSpaceSym: Any,
    dataX: np.ndarray,
    dataY: np.ndarray,
    dataE: np.ndarray,
    dataYFit: np.ndarray,
    dataYSigma: np.ndarray,
    Residuals: np.ndarray,
) -> Any:
    """Create a 3-spectrum Mantid workspace with data, fit, and residuals.

    Mimics the output format of the Mantid ``Fit`` algorithm.

    Args:
        wsYSpaceSym: The fitted workspace (used for naming).
        dataX: X values, shape ``(n_bins,)``.
        dataY: Observed Y values.
        dataE: Observed errors.
        dataYFit: Best-fit Y values.
        dataYSigma: Confidence-band width.
        Residuals: ``dataY - dataYFit``.

    Returns:
        A 3-spectrum workspace named
        ``wsYSpaceSym.name() + "_Fitted_Minuit"``.
    """

    wsMinFit = CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)), 
                    DataY=np.concatenate((dataY, dataYFit, Residuals)), 
                    DataE=np.concatenate((dataE, dataYSigma, np.zeros(len(dataE)))),
                    NSpec=3,
                    OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit")
    return wsMinFit


def saveMinuitPlot(yFitIC: Any, wsMinuitFit: Any, mObj: Minuit) -> None:
    """Save a PDF plot of the Minuit fit result.

    Args:
        yFitIC: Y-space fit configuration with ``figSavePath``.
        wsMinuitFit: The 3-spectrum fit-result workspace.
        mObj: The ``Minuit`` object (used for the legend).
    """

    leg = ""
    for p, v, e in zip(mObj.parameters, mObj.values, mObj.errors):
        leg += f"${p}={v:.2f} \pm {e:.2f}$\n"

    set_thesis_style()
    fig, ax = figure_factory(subplot_kw={"projection": "mantid"})
    ax.errorbar(wsMinuitFit, "k.", wkspIndex=0, label="Weighted Avg")
    ax.errorbar(wsMinuitFit, "r-", wkspIndex=1, label=leg)
    ax.set_xlabel("YSpace")
    ax.set_ylabel("Counts")
    ax.set_title("Minuit Fit")
    ax.legend()

    fileName = wsMinuitFit.name()+".pdf"
    savePath = yFitIC.figSavePath / fileName
    plt.savefig(savePath, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return


def createCorrelationTableWorkspace(
    wsYSpaceSym: Any, parameters: List[str], corrMatrix: np.ndarray
) -> None:
    """Create a Mantid TableWorkspace with the normalised correlation matrix.

    Args:
        wsYSpaceSym: The fitted workspace (used for naming).
        parameters: List of parameter names.
        corrMatrix: Correlation matrix (%), shape
            ``(n_pars, n_pars)``.
    """

    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_NormalizedCovarianceMatrix")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str',name="Name")
    for p in parameters:
        tableWS.addColumn(type='float',name=p)
    for p, arr in zip(parameters, corrMatrix):
        tableWS.addRow([p] + list(arr))
 

def runMinos(
    mObj: Minuit,
    yFitIC: Any,
    constrFunc: Callable,
    wsName: str,
) -> Tuple[List[str], List[float], List[float], List[np.ndarray], List[np.ndarray]]:
    """Run Minos asymmetric error analysis and return formatted columns.

    If ``yFitIC.runMinos`` is ``False``, returns zero-filled Minos
    error columns.  For ``SINGLE_GAUSSIAN``, uses iMinuit's automatic
    ``minos()``; for non-Gaussian models, uses a manual
    profile-likelihood scan (``runAndPlotManualMinos``).

    Args:
        mObj: The ``Minuit`` object after the main fit.
        yFitIC: Y-space fit configuration with ``runMinos``,
            ``fitModel``, and ``showPlots``.
        constrFunc: Positivity constraint callable (or no-op for
            Gaussian).
        wsName: Base workspace name (used for plot titles).

    Returns:
        A 5-tuple ``(parameters, values, errors, minosAutoErr,
        minosManErr)`` suitable for
        ``createFitParametersTableWorkspace``.
    """

    # Extract info from fit before running any MINOS
    parameters = list(mObj.parameters)
    values = list(mObj.values)
    errors = list(mObj.errors)

    # If minos is set not to run, ouput columns with zeros on minos errors
    if not(yFitIC.runMinos):
        minosAutoErr = list(np.zeros((len(parameters), 2)))
        minosManErr = list(np.zeros((len(parameters), 2)))
        return parameters, values, errors, minosAutoErr, minosManErr
    
    bestFitVals = {}
    bestFitErrs = {}
    for p, v, e in zip(mObj.parameters, mObj.values, mObj.errors):
        bestFitVals[p] = v
        bestFitErrs[p] = e

    if (yFitIC.fitModel=="SINGLE_GAUSSIAN"):   # Case with no positivity constraint, can use automatic minos()
        mObj.minos()
        me = mObj.merrors

        # Build minos errors lists in suitable format
        minosAutoErr = []
        for p in parameters:
            minosAutoErr.append([me[p].lower, me[p].upper])
        minosManErr = list(np.zeros(np.array(minosAutoErr).shape))

        if yFitIC.showPlots:
            plotAutoMinos(mObj, wsName)

    else:   # Case with positivity constraint on function, use manual implementation
        merrors, fig = runAndPlotManualMinos(mObj, constrFunc, bestFitVals, bestFitErrs, yFitIC.showPlots)     # Changes values of minuit obj m, do not use m below this point
        
        # Same as above, but the other way around
        minosManErr = []
        for p in parameters:
            minosManErr.append(merrors[p])
        minosAutoErr = list(np.zeros(np.array(minosManErr).shape))

        if yFitIC.showPlots:
            fig.canvas.setWindowTitle(wsName+"_Manual_Implementation_MINOS")
            fig.show()

    return    parameters, values, errors, minosAutoErr, minosManErr


def runAndPlotManualMinos(
    minuitObj: Minuit,
    constrFunc: Callable,
    bestFitVals: Dict[str, float],
    bestFitErrs: Dict[str, float],
    showPlots: bool,
) -> Tuple[Dict[str, np.ndarray], Any]:
    """Brute-force Minos profile-likelihood scan with plots.

    For each parameter, fixes the parameter, re-optimises, and records
    the cost-function value across a grid.  Minos errors are extracted
    from the ΔF = 1 intersection.

    Args:
        minuitObj: The ``Minuit`` object.
        constrFunc: Positivity constraint callable.
        bestFitVals: Best-fit parameter values.
        bestFitErrs: Hessian errors at the best fit.
        showPlots: Whether to display the profile plots.

    Returns:
        A 2-tuple ``(merrors, fig)`` where *merrors* is a dict mapping
        parameter names to ``[lower, upper]`` error arrays and *fig*
        is the Matplotlib figure.
    """
    # Reason for two distinct operations inside the same function is that its easier
    # to build the minos plots for each parameter as they are being calculated.
    print("\nRunning Minos ... \n")

    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})  #subplot_kw={'projection':'mantid'}
    # fig.canvas.setWindowTitle("Plot of Manual Implementation MINOS")

    merrors = {}
    for p, ax in zip(minuitObj.parameters, axs.flat):
        lerr, uerr = runMinosForPar(minuitObj, constrFunc, p, 2, ax, bestFitVals, bestFitErrs, showPlots)
        merrors[p] = np.array([lerr, uerr])

    # if showPlots:
    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
        # fig.show()
    return merrors, fig


def runMinosForPar(
    minuitObj: Minuit,
    constrFunc: Callable,
    var: str,
    bound: int,
    ax: Any,
    bestFitVals: Dict[str, float],
    bestFitErrs: Dict[str, float],
    showPlots: bool,
) -> Tuple[float, float]:
    """Run the profile-likelihood scan for a single parameter.

    Scans a range of ``bound`` × ``σ`` around the best-fit value,
    using both constrained Scipy and unconstrained Migrad.

    Args:
        minuitObj: The ``Minuit`` object.
        constrFunc: Positivity constraint callable.
        var: Name of the parameter to profile.
        bound: Number of standard deviations for the scan range.
        ax: Matplotlib axis for plotting.
        bestFitVals: Best-fit parameter values.
        bestFitErrs: Hessian errors.
        showPlots: Whether to draw on *ax*.

    Returns:
        A 2-tuple ``(lerr, uerr)`` of lower and upper Minos errors.
    """

    resetMinuit(minuitObj, bestFitVals, bestFitErrs)
    # Run Fitting procedures again to be on the safe side and reset to minimum
    minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
    minuitObj.hesse()

    # Extract parameters from minimum
    varVal = minuitObj.values[var]
    varErr = minuitObj.errors[var]
    # Store fval of best fit
    fValsMin = minuitObj.fval      # Used to calculate error bands at the end

    varSpace = buildVarRange(bound, varVal, varErr) 
    
    # Split variable space into right and left side
    lhsVarSpace, rhsVarSpace = np.split(varSpace, 2)
    lhsVarSpace = np.flip(lhsVarSpace)   # Flip to start at minimum

    for minimizer in ("Scipy", "Migrad"):
        resetMinuit(minuitObj, bestFitVals, bestFitErrs)
        rhsMinos = runMinosOnRange(minuitObj, var, rhsVarSpace, minimizer, constrFunc)
        
        resetMinuit(minuitObj, bestFitVals, bestFitErrs)
        lhsMinos = runMinosOnRange(minuitObj, var, lhsVarSpace, minimizer, constrFunc)

        wholeMinos = np.concatenate((np.flip(lhsMinos), rhsMinos), axis=None)   # Flip left hand side again

        if minimizer == "Scipy":   # Calculate minos errors from constrained scipy
            lerr, uerr = errsFromMinosCurve(varSpace, varVal, wholeMinos, fValsMin, dChi2=1)
            ax.plot(varSpace, wholeMinos, label="fVals Constr Scipy")

        elif minimizer == "Migrad":   # Plot migrad as well to see the difference between constrained and unconstrained
            plotProfile(ax, var, varSpace, wholeMinos, lerr, uerr, fValsMin, varVal, varErr)
        else:
            raise ValueError("Minimizer not recognized.")

    resetMinuit(minuitObj, bestFitVals, bestFitErrs)
    return lerr, uerr


def resetMinuit(
    minuitObj: Minuit,
    bestFitVals: Dict[str, float],
    bestFitErrs: Dict[str, float],
) -> None:
    """Reset Minuit parameters to the best-fit values and errors.

    Args:
        minuitObj: The ``Minuit`` object.
        bestFitVals: Dict mapping parameter names to best-fit values.
        bestFitErrs: Dict mapping parameter names to Hessian errors.
    """
    for p in bestFitVals:
        minuitObj.values[p] = bestFitVals[p]
        minuitObj.errors[p] = bestFitErrs[p]
    return


def buildVarRange(bound: int, varVal: float, varErr: float) -> np.ndarray:
    """Build a quadratically-spaced scan range for the Minos profile.

    The grid is denser near the minimum and spans ±``bound`` standard
    deviations.

    Args:
        bound: Number of standard deviations to span.
        varVal: Best-fit value of the parameter.
        varErr: Hessian error on the parameter.

    Returns:
        Array of scan points, shape ``(30,)`` (even number required).
    """
    # Create variable space more dense near the minima using a quadratic density
    limit = (bound*varErr)**(1/2)     # Square root is corrected below
    varSpace = np.linspace(-limit, limit, 30)
    varSpace = varSpace**2 * np.sign(varSpace) + varVal
    assert len(varSpace)%2 == 0, "Number of points in Minos range needs to be even"
    return varSpace


def runMinosOnRange(
    minuitObj: Minuit,
    var: str,
    varRange: np.ndarray,
    minimizer: str,
    constrFunc: Callable,
) -> np.ndarray:
    """Evaluate the cost function with one parameter fixed across a range.

    Fixes *var*, sets it to each value in *varRange*, re-optimises the
    remaining parameters, and records the minimum cost at each point.

    Args:
        minuitObj: The ``Minuit`` object.
        var: Name of the parameter to fix.
        varRange: Scan values, shape ``(n_points,)``.
        minimizer: ``"Scipy"`` (constrained) or ``"Migrad"``.
        constrFunc: Positivity constraint callable.

    Returns:
        Array of cost-function values, shape ``(n_points,)``.
    """

    result = np.zeros(varRange.size)
    minuitObj.fixed[var] = True

    # Unconstrained fit over side range
    for i, value in enumerate(varRange):

        minuitObj.values[var] = value      # Fix variable

        if minimizer == "Migrad":
            minuitObj.migrad()                       # Fit 
        elif minimizer == "Scipy":
            minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

        result[i] = minuitObj.fval          # Store minimum

    minuitObj.fixed[var] = False 
    return result                    


def errsFromMinosCurve(
    varSpace: np.ndarray,
    varVal: float,
    fValsScipy: np.ndarray,
    fValsMin: float,
    dChi2: float = 1,
) -> Tuple[float, float]:
    """Extract Minos errors from a profile-likelihood curve.

    Interpolates the profile to a dense grid and finds the two
    intersections with the ``ΔF = dChi2`` line.

    Args:
        varSpace: Scan-point values, shape ``(n_points,)``.
        varVal: Best-fit value.
        fValsScipy: Cost-function values at each scan point.
        fValsMin: Cost-function value at the minimum.
        dChi2: Delta chi-squared threshold (default 1 for 1σ).

    Returns:
        A 2-tuple ``(lerr, uerr)`` — lower and upper errors relative
        to *varVal*.  Both are zero if intersections are not found.
    """
    # Use intenpolation to create dense array of fmin values 
    varSpaceDense = np.linspace(np.min(varSpace), np.max(varSpace), 100000)
    fValsScipyDense = np.interp(varSpaceDense, varSpace, fValsScipy)
    # Calculate points of intersection with line delta fmin val = 1
    idxErr = np.argwhere(np.diff(np.sign(fValsScipyDense - fValsMin - 1)))
    
    if idxErr.size != 2:    # Intersections not found, do not plot error range
        lerr, uerr = 0., 0.   
    else:
        lerr, uerr = varSpaceDense[idxErr].flatten() - varVal

        if lerr*uerr >= 0:     # Case where we get either two positive or two negative errors, ill defined profile
            lerr, uerr = 0, 0
 
    return lerr, uerr


def plotAutoMinos(minuitObj: Minuit, wsName: str) -> None:
    """Plot the automatic Minos profiles from iMinuit.

    Args:
        minuitObj: The ``Minuit`` object (after ``minos()``).
        wsName: Base workspace name for the figure title.
    """
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})
    # fig.canvas.setWindowTitle(wsName+"_Plot_Automatic_MINOS")
    fig.canvas.setWindowTitle(wsName+"_Plot_Automatic_MINOS")
 
    for p, ax in zip(minuitObj.parameters, axs.flat):
        loc, fvals, status = minuitObj.mnprofile(p, bound=2)

        minfval = minuitObj.fval
        minp = minuitObj.values[p]
        hessp = minuitObj.errors[p]
        lerr = minuitObj.merrors[p].lower
        uerr = minuitObj.merrors[p].upper
        plotProfile(ax, p, loc, fvals, lerr, uerr, minfval, minp, hessp)

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    fig.show()   


def plotProfile(
    ax: Any,
    var: str,
    varSpace: np.ndarray,
    fValsMigrad: np.ndarray,
    lerr: float,
    uerr: float,
    fValsMin: float,
    varVal: float,
    varErr: float,
) -> None:
    """Plot a single parameter's likelihood profile on an axis.

    Draws the cost-function curve, the Minos error band (red), and
    the Hessian error band (green).

    Args:
        ax: Matplotlib axis.
        var: Parameter name (used in title).
        varSpace: Scan-point values.
        fValsMigrad: Cost-function values (Migrad).
        lerr: Lower Minos error.
        uerr: Upper Minos error.
        fValsMin: Cost-function minimum.
        varVal: Best-fit value.
        varErr: Hessian error.
    """

    ax.set_title(var+f" = {varVal:.3f} {lerr:.3f} {uerr:+.3f}")

    ax.plot(varSpace, fValsMigrad, label="fVals Migrad")

    ax.axvspan(lerr+varVal, uerr+varVal, alpha=0.2, color="red", label="Minos error")
    ax.axvspan(varVal-varErr, varVal+varErr, alpha=0.2, color="green", label="Hessian Std error")
    
    ax.axvline(varVal, 0.03, 0.97, color="k", ls="--")
    ax.axhline(fValsMin+1, 0.03, 0.97, color="k")
    ax.axhline(fValsMin, 0.03, 0.97, color="k")


def createFitParametersTableWorkspace(
    wsYSpaceSym: Any,
    parameters: List[str],
    values: List[float],
    errors: List[float],
    minosAutoErr: List[np.ndarray],
    minosManualErr: List[np.ndarray],
    chi2: float,
) -> None:
    """Create a Mantid TableWorkspace with all fit parameters and errors.

    Includes Hessian errors and both automatic and manual Minos errors.

    Args:
        wsYSpaceSym: The fitted workspace (used for naming).
        parameters: Parameter names.
        values: Best-fit values.
        errors: Hessian errors.
        minosAutoErr: Automatic Minos ``[lower, upper]`` per parameter.
        minosManualErr: Manual Minos ``[lower, upper]`` per parameter.
        chi2: Normalised chi-squared value.
    """
    # Create Parameters workspace
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_Parameters")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str', name="Name")
    tableWS.addColumn(type='float', name="Value")
    tableWS.addColumn(type='float', name="Error")
    tableWS.addColumn(type='float', name="Auto Minos Error-")
    tableWS.addColumn(type='float', name="Auto Minos Error+")
    tableWS.addColumn(type='float', name="Manual Minos Error-")
    tableWS.addColumn(type='float', name="Manual Minos Error+")

    for p, v, e, mae, mme in zip(parameters, values, errors, minosAutoErr, minosManualErr):
        tableWS.addRow([p, v, e, mae[0], mae[1], mme[0], mme[1]])

    tableWS.addRow(["Cost function", chi2, 0, 0, 0, 0, 0])
    return


def oddPointsRes(
    x: np.ndarray, res: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Resample the resolution to an odd-number grid for symmetric convolution.

    Ensures a single peak at the centre by forcing an odd number of
    grid points.

    Args:
        x: Abscissa values (must be symmetric about zero), shape
            ``(n_bins,)``.
        res: Resolution values, same shape.

    Returns:
        A 2-tuple ``(xDelta, resDense)`` — the grid spacing and the
        resampled resolution array.
    """

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if res.size % 2 == 0:
        dens = res.size+1  # If even change to odd
    else:
        dens = res.size    # If odd, keep being odd

    xDense = np.linspace(np.min(x), np.max(x), dens)    # Make gridd with odd number of points - peak at center
    xDelta = xDense[1] - xDense[0]

    resDense = np.interp(xDense, x, res)

    return xDelta, resDense


def fitProfileMantidFit(yFitIC: Any, wsYSpaceSym: Any, wsRes: Any) -> None:
    """Fit the y-space profile using Mantid Fit for cross-validation.

    Runs both Levenberg–Marquardt and Simplex minimisers on a convolved
    ``UserFunction``.  Skipped for ``DOUBLE_WELL``,
    ``ANSIO_GAUSSIAN``, and ``MULTIVARIATE_GAUSSIAN`` models.

    Args:
        yFitIC: Y-space fit configuration with ``fitModel``.
        wsYSpaceSym: The (symmetrised) weighted-average J(y) workspace.
        wsRes: The summed resolution workspace.
    """

    print('\nFitting on the sum of spectra in the West domain ...\n')     
    for minimizer in ['Levenberg-Marquardt','Simplex']:
        
        if yFitIC.fitModel=="SINGLE_GAUSSIAN":
            function=f"""composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0;
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()"""

        elif yFitIC.fitModel=="GC_C4_C6":
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12)+c6/384*(64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120)),
            y0=0, A=1,x0=0,sigma1=4.0,c4=0.0,c6=0.0,ties=(),constraints=(0<c4,0<c6)
            """
        elif yFitIC.fitModel=="GC_C4":
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12)),
            y0=0, A=1,x0=0,sigma1=4.0,c4=0.0,ties=()
            """
        elif yFitIC.fitModel=="GC_C6":
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c6/384*(64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120)),
            y0=0, A=1,x0=0,sigma1=4.0,c6=0.0,ties=()
            """
        elif (yFitIC.fitModel=="DOUBLE_WELL") | (yFitIC.fitModel=="ANSIO_GAUSSIAN") | (yFitIC.fitModel=="MULTIVARIATE_GAUSSIAN"):
            return
        else: raise ValueError("fitmodel not recognized.")

        outputName = wsYSpaceSym.name()+"_Fitted_"+minimizer
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = outputName)

        Fit(
            Function=function, 
            InputWorkspace=outputName,
            Output=outputName,
            Minimizer=minimizer
            )
        # Fit produces output workspaces with results
    return 


def printYSpaceFitResults(wsJoYName: str) -> None:
    """Print a summary of y-space fit results from all available optimisers.

    Reads ``_Fitted_Levenberg-Marquardt_Parameters``,
    ``_Fitted_Simplex_Parameters``, and
    ``_Fitted_Minuit_Parameters`` TableWorkspaces from ``mtd``.

    Args:
        wsJoYName: Base name of the weighted-average J(y) workspace.
    """
    print("\nFit in Y Space results:")
    foundWS = []
    try:
        wsFitLM = mtd[wsJoYName + "_Fitted_Levenberg-Marquardt_Parameters"]
        foundWS.append(wsFitLM)
    except KeyError: pass
    try:
        wsFitSimplex = mtd[wsJoYName + "_Fitted_Simplex_Parameters"]
        foundWS.append(wsFitSimplex)
    except KeyError: pass
    try:
        wsFitMinuit = mtd[wsJoYName + "_Fitted_Minuit_Parameters"]
        foundWS.append(wsFitMinuit)
    except KeyError: pass

    for tableWS in foundWS:
        print("\n"+" ".join(tableWS.getName().split("_")[-3:])+":")
        # print("    ".join(tableWS.keys()))
        for key in tableWS.keys():
            if key=="Name":
                print(f"{key:>20s}:  "+"  ".join([f"{elem:7.8s}" for elem in tableWS.column(key)]))
            else:
                print(f"{key:>20s}: "+"  ".join([f"{elem:7.4f}" for elem in tableWS.column(key)]))
    print("\n")


class ResultsYFitObject:
    """Collector for y-space fit results.

    Gathers the fitted J(y) data, resolution, best-fit parameters, and
    errors from the AnalysisDataService.  Saved as an ``.npz`` file for
    regression testing and post-hoc analysis.

    Attributes:
        finalRawDataY: DataY of the final corrected TOF workspace.
        finalRawDataE: DataE of the final corrected TOF workspace.
        HdataY: DataY of the mass₀-only workspace.
        YSpaceSymSumDataY: DataY of the (symmetrised) weighted-average
            J(y) workspace.
        YSpaceSymSumDataE: DataE of the same.
        resolution: Resolution Y values (summed).
        popt: Best-fit parameters from each optimiser, shape
            ``(n_optimisers, n_pars)``.
        perr: Errors from each optimiser, same shape.
        savePath: Path for the ``.npz`` output file.
        fitModel: Model identifier string.
    """

    def __init__(
        self, ic: Any, yFitIC: Any, wsFinalName: str, wsYSpaceAvgName: str
    ) -> None:
        # Extract most relevant information from ws
        wsFinal = mtd[wsFinalName]
        wsResSum = mtd[wsFinalName + "_Resolution_Sum"]

        wsJoYAvg = mtd[wsYSpaceAvgName]
        wsSubMassName = wsYSpaceAvgName.split("_JoY_")[0]
        wsMass0 = mtd[wsSubMassName]

        self.finalRawDataY = wsFinal.extractY()
        self.finalRawDataE = wsFinal.extractE()
        self.HdataY = wsMass0.extractY()
        self.YSpaceSymSumDataY = wsJoYAvg.extractY()
        self.YSpaceSymSumDataE = wsJoYAvg.extractE()
        self.resolution = wsResSum.extractY()

        # Extract best fit parameters from workspaces
        poptList = []
        perrList = []
        try:
            wsFitMinuit = mtd[wsJoYAvg.name() + "_Fitted_Minuit_Parameters"]
            poptList.append(wsFitMinuit.column("Value"))
            perrList.append(wsFitMinuit.column("Error"))
        except: pass
        try:
            wsFitLM = mtd[wsJoYAvg.name() + "_Fitted_Levenberg-Marquardt_Parameters"]
            poptList.append(wsFitLM.column("Value"))
            perrList.append(wsFitLM.column("Error"))
        except: pass
        try:
            wsFitSimplex = mtd[wsJoYAvg.name() + "_Fitted_Simplex_Parameters"]
            poptList.append(wsFitSimplex.column("Value"))
            perrList.append(wsFitSimplex.column("Error"))
        except: pass

        # Number of parameters might not be the same, need to add zeros to some lists to match length
        maxLen = max([len(l) for l in poptList])
        for pList in [poptList, perrList]:
            for l in pList:
                while len(l) < maxLen:
                    l.append(0)
        
        popt = np.array(poptList)
        perr = np.array(perrList)

        self.popt = popt
        self.perr = perr

        self.savePath = ic.ySpaceFitSavePath
        self.fitModel = yFitIC.fitModel


    def save(self) -> None:
        """Save all result arrays to an ``.npz`` file at ``self.savePath``."""

        np.savez(self.savePath,
                 YSpaceSymSumDataY=self.YSpaceSymSumDataY,
                 YSpaceSymSumDataE=self.YSpaceSymSumDataE,
                 resolution=self.resolution, 
                 HdataY=self.HdataY,
                 finalRawDataY=self.finalRawDataY, 
                 finalRawDataE=self.finalRawDataE,
                 popt=self.popt, 
                 perr=self.perr)


def runGlobalFit(
    wsYSpace: Any, wsRes: Any, IC: Any, yFitIC: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a simultaneous (global) fit across detector groups.

    Groups detectors using k-means clustering in (L1, θ) space,
    computes a summed ``cost.LeastSquares`` over all groups (with
    shared line-shape parameters), and minimises with iMinuit.

    Args:
        wsYSpace: Per-spectrum J(y) workspace.
        wsRes: Per-spectrum resolution workspace.
        IC: Completed initial-conditions object.
        yFitIC: Y-space fit configuration with ``nGlobalFitGroups``,
            ``fitModel``, ``symmetrisationFlag``, and ``showPlots``.

    Returns:
        A 2-tuple ``(values, errors)`` of best-fit parameters and
        their Hessian errors as NumPy arrays.
    """

    print("\nRunning GLobal Fit ...\n")

    dataX, dataY, dataE, dataRes, instrPars = extractData(wsYSpace, wsRes, IC)   
    dataX, dataY, dataE, dataRes, instrPars = takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars)

    idxList = groupDetectors(instrPars, yFitIC)
    dataX, dataY, dataE, dataRes = avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList, yFitIC)

    if yFitIC.symmetrisationFlag:  
        dataY, dataE = weightedSymArr(dataY, dataE)

    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitModel)   
    
    totCost = 0
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):
        totCost += calcCostFun(model, i, x, y, yerr, res, sharedPars)
    
    defaultPars["y0"] = 0    # Introduce default parameter for convolved model

    assert len(describe(totCost)) == len(sharedPars) + len(dataY)*(len(defaultPars)-len(sharedPars)), f"Wrong parameters for Global Fit:\n{describe(totCost)}"
   
    # Minuit Fit with global cost function and local+global parameters
    initPars = minuitInitialParameters(defaultPars, sharedPars, len(dataY))

    print("\nRunning Global Fit ...\n")
    m = Minuit(totCost, **initPars)

    for i in range(len(dataY)):     # Set limits for unshared parameters
        m.limits["A"+str(i)] = (0, np.inf)   

    if yFitIC.fitModel=="DOUBLE_WELL":  
        m.limits["d"] = (0, np.inf)     # Shared parameters
        m.limits["R"] = (0, np.inf) 

    t0 = time.time()
    if yFitIC.fitModel=="SINGLE_GAUSSIAN":
        m.simplex()
        m.migrad() 

    else:
        totSig = describe(totCost)   # This signature has 'x' already removed
        sharedIdxs = [totSig.index(shPar) for shPar in sharedPars]
        nCostFunctions = len(totCost)   # Number of individual cost functions
        x = dataX[0]

        def constr(*pars):
            """
            Constraint for positivity of non Gaussian function.
            Input: All parameters defined in global cost function.
            x is the range for each individual cost fun, defined outside function.
            Builds array with all constraints from individual functions.
            """

            sharedPars = [pars[i] for i in sharedIdxs]    # sigma1, c4, c6 in original GC
            unsharedPars = np.delete(pars, sharedIdxs, None)
            unsharedParsSplit = np.split(unsharedPars, nCostFunctions)   # Splits unshared parameters per individual cost fun

            joinedGC = np.zeros(nCostFunctions * x.size)  
            for i, unshParsModel in enumerate(unsharedParsSplit):    # Attention to format of unshared and shared parameters when calling model
                joinedGC[i*x.size : (i+1)*x.size] = model(x, *unshParsModel[1:], *sharedPars)   # Intercept is first of unshared parameters 
                 
            return joinedGC

        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constr, 0, np.inf))
    
    t1 = time.time()
    print(f"\nTime of fitting: {t1-t0:.2f} seconds")
    
    # Explicitly calculate errors
    m.hesse()

    chi2 = m.fval / (np.sum(dataE!=0)-m.nfit)   # Number of non zero points (considered in the fit) minus no of parameters
    print(f"Value of Chi2/ndof: {chi2:.2f}")
    print(f"Migrad Minimum valid: {m.valid}")

    print("\nResults of Global Fit:\n")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p:>7s} = {v:>8.4f} \u00B1 {e:<8.4f}")
    print("\n")

    if yFitIC.showPlots:
        plotGlobalFit(dataX, dataY, dataE, m, totCost, wsYSpace.name())
    
    return np.array(m.values), np.array(m.errors)     # Pass into array to store values in variable


def extractData(
    ws: Any, wsRes: Any, ic: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract data arrays and instrument parameters for the global fit.

    Args:
        ws: Per-spectrum J(y) workspace.
        wsRes: Per-spectrum resolution workspace.
        ic: Completed initial-conditions object.

    Returns:
        A 5-tuple ``(dataX, dataY, dataE, dataRes, instrPars)``.
    """

    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    dataRes = wsRes.extractY()
    instrPars = loadInstrParsFileIntoArray(ic)
    assert len(instrPars) == len(dataY), "Load of IP file not working correctly, probable issue with indexing."
    return dataX, dataY, dataE, dataRes, instrPars    


def loadInstrParsFileIntoArray(ic: Any) -> np.ndarray:
    """Load instrument parameters for the spectrum range in *ic*.

    Args:
        ic: Completed initial-conditions object with
            ``InstrParsPath``, ``firstSpec``, and ``lastSpec``.

    Returns:
        Instrument parameters array, shape
        ``(n_selected_spectra, 6)``.
    """

    data = np.loadtxt(ic.InstrParsPath, dtype=str)[1:].astype(float)
    spectra = data[:, 0]
    select_rows = np.where((spectra >= ic.firstSpec) & (spectra <= ic.lastSpec))
    instrPars = data[select_rows]
    return instrPars


def takeOutMaskedSpectra(
    dataX: np.ndarray,
    dataY: np.ndarray,
    dataE: np.ndarray,
    dataRes: np.ndarray,
    instrPars: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove fully masked (all-zero) spectra from all arrays.

    Args:
        dataX: X values, shape ``(n_spectra, n_bins)``.
        dataY: Y values, same shape.
        dataE: Error values, same shape.
        dataRes: Resolution values, same shape.
        instrPars: Instrument parameters, shape ``(n_spectra, 6)``.

    Returns:
        The five input arrays with masked rows removed.
    """
    zerosRowMask = np.all(dataY==0, axis=1)
    dataY = dataY[~zerosRowMask]
    dataE = dataE[~zerosRowMask]
    dataX = dataX[~zerosRowMask]
    dataRes = dataRes[~zerosRowMask]
    instrPars = instrPars[~zerosRowMask]
    return dataX, dataY, dataE, dataRes, instrPars 

# ------- Groupings 

def groupDetectors(ipData: np.ndarray, yFitIC: Any) -> List[List[int]]:
    """Group detectors using k-means clustering in (L1, θ) space.

    Normalises L1 and θ, applies k-means, and returns a list of index
    groups.

    Args:
        ipData: Instrument parameters (unmasked), shape
            ``(n_spectra, 6)``.
        yFitIC: Y-space fit configuration with ``nGlobalFitGroups``
            and ``showPlots``.

    Returns:
        A list of lists, each containing the row indices belonging to
        one detector group.
    """

    checkNGroupsValid(yFitIC, ipData)

    print(f"\nNumber of groups: {yFitIC.nGlobalFitGroups}")

    L1 = ipData[:, -1].copy()
    theta = ipData[:, 2].copy()  

    # Normalize  ranges to similar values, needed for clustering
    L1 /= np.sum(L1)       
    theta /= np.sum(theta)

    L1 *= 2           # Bigger weight to L1

    points = np.vstack((L1, theta)).T
    assert points.shape == (len(L1), 2), "Wrong shape."
    # Initial centers of groups
    startingIdxs = np.linspace(0, len(points)-1, yFitIC.nGlobalFitGroups).astype(int)
    centers = points[startingIdxs, :]    # Centers of cluster groups, NOT fitting parameter

    if False:    # Set to True to investigate problems with groupings
        plotDetsAndInitialCenters(L1, theta, centers)

    clusters = kMeansClustering(points, centers)
    idxList = formIdxList(clusters)

    if yFitIC.showPlots:
        fig, ax = plt.subplots(tight_layout=True, subplot_kw={'projection':'mantid'})  
        fig.canvas.setWindowTitle("Grouping of detectors")
        plotFinalGroups(ax, ipData, idxList)
        fig.show()
    return idxList


def checkNGroupsValid(yFitIC: Any, ipData: np.ndarray) -> None:
    """Validate the number of groups for the global fit.

    Converts ``"ALL"`` to the actual number of spectra.

    Args:
        yFitIC: Y-space fit configuration with ``nGlobalFitGroups``.
        ipData: Instrument parameters array (used for length).

    Raises:
        AssertionError: If the number is non-positive or exceeds
            the number of unmasked spectra.
    """

    nSpectra = len(ipData)  # Number of spectra in the workspace

    if (yFitIC.nGlobalFitGroups=="ALL"):
        yFitIC.nGlobalFitGroups = nSpectra
    else:
        assert type(yFitIC.nGlobalFitGroups)==int, "Number of global groups needs to be an integer."
        assert yFitIC.nGlobalFitGroups<=nSpectra, "Number of global groups needs to be less or equal to the no of unmasked spectra."
        assert yFitIC.nGlobalFitGroups>0, "NUmber of global groups needs to be bigger than zero"
    return 


def kMeansClustering(
    points: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """K-means clustering in 2-D (L1, θ) space.

    Iteratively assigns each point to its closest centre and
    recalculates centres until convergence.

    Args:
        points: Data points, shape ``(n_points, 2)``.
        centers: Initial cluster centres, shape ``(k, 2)``.

    Returns:
        Cluster assignments, shape ``(n_points,)`` with integer
        labels 0 … k-1.
    """

    prevCenters = centers   # Starting centers
    while  True:
        clusters = closestCenter(points, prevCenters)   # Form groups by assigning points to their closest center
        centers = calculateCenters(points, clusters)    # Recalculate centers of new groups

        if np.all(centers == prevCenters):
            break

        assert np.isfinite(centers).all(), f"Invalid centers found:\n{centers}\nTry a different number for the groupings."

        prevCenters = centers

    clusters = closestCenter(points, centers)
    return clusters


def closestCenter(
    points: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Assign each point to its nearest cluster centre.

    Args:
        points: Data points, shape ``(n_points, 2)``.
        centers: Cluster centres, shape ``(k, 2)``.

    Returns:
        Cluster assignments, shape ``(n_points,)``.
    """

    clusters = np.zeros(len(points))
    for p in range(len(points)):   # Iterate over each point
        
        distMin = np.inf    # To be replaced in first iteration
        
        for i in range(len(centers)):  # Assign closest center to point

            dist = pairDistance(points[p], centers[i])

            if dist < distMin:      # Store minimum found
                distMin = dist
                closeCenter = i

        clusters[p] = closeCenter
    return clusters


def pairDistance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        Scalar distance.
    """
    return np.sqrt(np.sum(np.square(p1-p2)))


def calculateCenters(
    points: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    """Recalculate cluster centres from current assignments.

    Args:
        points: Data points, shape ``(n_points, 2)``.
        clusters: Cluster assignments, shape ``(n_points,)``.

    Returns:
        Updated centres, shape ``(k, 2)``.
    """

    nGroups = len(np.unique(clusters))

    centers = np.zeros((nGroups, 2))
    for i in range(nGroups):
        centers[i, :] = np.mean(points[clusters==i, :], axis=0)  # If cluster i is not present, returns nan
    return centers


def formIdxList(clusters: np.ndarray) -> List[List[int]]:
    """Convert cluster assignment array to a list of index lists.

    Args:
        clusters: Integer cluster labels, shape ``(n_points,)``.

    Returns:
        A list of lists, one per cluster, containing the indices of
        the assigned points.
    """

    idxList = []
    for i in np.unique(clusters):
        idxs = np.argwhere(clusters==i).flatten()
        idxList.append(list(idxs))

    # Print groupings information
    print("\nGroups formed successfully:\n")
    groupLen = np.array([len(group) for group in idxList])
    unique, counts = np.unique(groupLen, return_counts=True)
    for length, no in zip(unique, counts):
        print(f"{no} groups with {length} detectors.")

    return idxList


def plotDetsAndInitialCenters(
    L1: np.ndarray, theta: np.ndarray, centers: np.ndarray
) -> None:
    """Debug plot of detector positions and initial k-means centroids.

    Args:
        L1: Normalised L1 values.
        theta: Normalised theta values.
        centers: Initial centroids, shape ``(k, 2)``.
    """
    fig, ax = plt.subplots(tight_layout=True, subplot_kw={'projection':'mantid'})  
    fig.canvas.setWindowTitle("Starting centroids for groupings")
    ax.scatter(L1, theta, alpha=0.3, color="r", label="Detectors")
    ax.scatter(centers[:, 0], centers[:, 1], color="k", label="Starting centroids")
    ax.axes.xaxis.set_ticks([])  # Numbers plotted do not correspond to real numbers, so hide them
    ax.axes.yaxis.set_ticks([]) 
    ax.set_xlabel("L1")
    ax.set_ylabel("Theta")
    ax.legend()
    fig.show()


def plotFinalGroups(ax: Any, ipData: np.ndarray, idxList: List[List[int]]) -> None:
    """Plot detector groupings on an axis.

    Args:
        ax: Matplotlib axis.
        ipData: Instrument parameters, shape ``(n_spectra, 6)``.
        idxList: List of index lists from ``formIdxList``.
    """

    for i, idxs in enumerate(idxList):
        L1 = ipData[idxs, -1]
        theta = ipData[idxs, 2]
        ax.scatter(L1, theta, label=f"Group {i}")

        dets = ipData[idxs, 0]
        for det, x, y in zip(dets, L1, theta):
            ax.text(x, y, str(int(det)), fontsize=8)

    ax.set_xlabel("L1")
    ax.set_ylabel("Theta")
    ax.legend()
    return

# --------- Weighted Avg of detectors

def avgWeightDetGroups(
    dataX: np.ndarray,
    dataY: np.ndarray,
    dataE: np.ndarray,
    dataRes: np.ndarray,
    idxList: List[List[int]],
    yFitIC: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted average of data within each detector group.

    Dispatches to ``avgGroupsWithBins`` (NAN masking) or
    ``avgGroupsOverCols`` (standard/NCP masking).

    Args:
        dataX: X values (unmasked), shape ``(n_spectra, n_bins)``.
        dataY: Y values, same shape.
        dataE: Error values, same shape.
        dataRes: Resolution values, same shape.
        idxList: Detector groupings from ``groupDetectors``.
        yFitIC: Y-space fit configuration with ``maskTypeProcedure``.

    Returns:
        A 4-tuple ``(wDataX, wDataY, wDataE, wDataRes)`` of
        group-averaged arrays, shape ``(n_groups, n_bins)``.
    """
    assert ~np.any(np.all(dataY==0, axis=1)), f"Input data should not include masked spectra at: {np.argwhere(np.all(dataY==0, axis=1))}"

    if (yFitIC.maskTypeProcedure=="NAN"): 
        return avgGroupsWithBins(dataX, dataY, dataE, dataRes, idxList, yFitIC)
    
    # Use Default for unmasked or NCP masked
    return avgGroupsOverCols(dataX, dataY, dataE, dataRes, idxList)


def avgGroupsOverCols(
    dataX: np.ndarray,
    dataY: np.ndarray,
    dataE: np.ndarray,
    dataRes: np.ndarray,
    idxList: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted average per group when data is already rebinned.

    Args:
        dataX: X values, shape ``(n_spectra, n_bins)``.
        dataY: Y values, same shape.
        dataE: Error values, same shape.
        dataRes: Resolution values, same shape.
        idxList: Detector groupings.

    Returns:
        Group-averaged arrays, shape ``(n_groups, n_bins)``.
    """

    wDataX, wDataY, wDataE, wDataRes = initiateZeroArr((len(idxList), len(dataY[0])))

    for i, idxs in enumerate(idxList):
        groupX, groupY, groupE, groupRes = extractArrByIdx(dataX, dataY, dataE, dataRes, idxs)
        assert len(groupY) > 0, "Group with zero detectors found, invalid."

        if len(groupY) == 1:   # Cannot use weight avg in single spec, wrong results
            meanY, meanE = groupY, groupE
            meanRes = groupRes

        else:
            meanY, meanE = weightedAvgArr(groupY, groupE)
            meanRes = np.nanmean(groupRes, axis=0)   # Nans are not present but safeguard

        assert np.all(groupX[0] == np.mean(groupX, axis=0)), "X values should not change with groups"
        
        for wsData, mean in zip([wDataX, wDataY, wDataE, wDataRes], [groupX[0], meanY, meanE, meanRes]):
            wsData[i] = mean
    
    assert ~np.any(np.all(wDataY==0, axis=1)), f"Some avg weights in groups are not being performed:\n{np.argwhere(np.all(wDataY==0, axis=1))}"
    return wDataX, wDataY, wDataE, wDataRes


def avgGroupsWithBins(
    dataX: np.ndarray,
    dataY: np.ndarray,
    dataE: np.ndarray,
    dataRes: np.ndarray,
    idxList: List[List[int]],
    yFitIC: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted average per group when NAN bin-masking is active.

    Uses ``weightedAvgXBinsArr`` to handle multiple dataY per bin
    centre.

    Args:
        dataX: X values, shape ``(n_spectra, n_pts)``.
        dataY: Y values, same shape.
        dataE: Error values, same shape.
        dataRes: Resolution values, same shape.
        idxList: Detector groupings.
        yFitIC: Y-space fit configuration.

    Returns:
        Group-averaged arrays, shape ``(n_groups, n_bins)``.
    """

    # Build range to average over
    meanX = buildXRangeFromRebinPars(yFitIC)  

    wDataX, wDataY, wDataE, wDataRes = initiateZeroArr((len(idxList), len(meanX)))
    for i, idxs in enumerate(idxList):
        groupX, groupY, groupE, groupRes = extractArrByIdx(dataX, dataY, dataE, dataRes, idxs)

        meanY, meanE = weightedAvgXBinsArr(groupX, groupY, groupE, meanX)
        
        meanRes = np.nanmean(groupRes, axis=0)   # Nans are not present but safeguard
        
        for wsData, mean in zip([wDataX, wDataY, wDataE, wDataRes], [meanX, meanY, meanE, meanRes]):
            wsData[i] = mean
    
    return wDataX, wDataY, wDataE, wDataRes


def initiateZeroArr(
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create four zero-filled arrays of the given shape.

    Args:
        shape: ``(n_groups, n_bins)``.

    Returns:
        A 4-tuple ``(wDataX, wDataY, wDataE, wDataRes)``.
    """
    wDataX = np.zeros(shape)
    wDataY = np.zeros(shape)
    wDataE = np.zeros(shape)
    wDataRes = np.zeros(shape)  
    return  wDataX, wDataY, wDataE, wDataRes


def extractArrByIdx(
    dataX: np.ndarray,
    dataY: np.ndarray,
    dataE: np.ndarray,
    dataRes: np.ndarray,
    idxs: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select rows of multiple arrays by index list.

    Args:
        dataX: X values, shape ``(n_spectra, n_bins)``.
        dataY: Y values, same shape.
        dataE: Error values, same shape.
        dataRes: Resolution values, same shape.
        idxs: Row indices to select.

    Returns:
        A 4-tuple of the selected sub-arrays.
    """
    groupE = dataE[idxs, :]
    groupY = dataY[idxs, :]
    groupX = dataX[idxs, :]
    groupRes = dataRes[idxs, :]
    return groupX, groupY, groupE, groupRes


def calcCostFun(
    model: Callable,
    i: int,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    res: np.ndarray,
    sharedPars: List[str],
) -> "GlobalNCPCostFunction":
    """Build a ``GlobalNCPCostFunction`` for one detector group.

    Constructs a convolved model, renames unshared parameters with
    suffix *i*, and returns the cost function to be summed into the
    global cost via ``CostSum``.

    Args:
        model: J(y) model callable.
        i: Group index (appended to unshared parameter names).
        x: Abscissa values, shape ``(n_bins,)``.
        y: Observed values, same shape.
        yerr: Errors, same shape.
        res: Resolution values, same shape.
        sharedPars: Parameter names shared across groups.

    Returns:
        A ``GlobalNCPCostFunction`` instance.
    """
   
    xDelta, resDense = oddPointsRes(x, res)
    def convolvedModel(xrange, y0, *pars):
        """Performs convolution first on high density grid and interpolates to desired x range"""
        return y0 + signal.convolve(model(xrange, *pars), resDense, mode="same") * xDelta

    signature = describe(model)[:]
    signature[1:1] = ["y0"]

    costSig = [key if key in sharedPars else key+str(i) for key in signature]

    # Select only valid data, i.e. when error is not 0 or nan or inf
    nonZeros = (yerr != 0) & ~np.isnan(yerr) & ~np.isinf(yerr) & ~np.isnan(y)
    xNZ = x[nonZeros]
    yNZ = y[nonZeros]
    yerrNZ = yerr[nonZeros]

    costFun = GlobalNCPCostFunction(xNZ, yNZ, yerrNZ, convolvedModel, costSig)
    return costFun


def minuitInitialParameters(
    defaultPars: Dict[str, float], sharedPars: List[str], nSpec: int
) -> Dict[str, float]:
    """Build the initial parameter dict for a global Minuit fit.

    Shared parameters appear once; unshared parameters are replicated
    with index suffixes ``0`` … ``nSpec-1``.

    Args:
        defaultPars: Default values for all model parameters.
        sharedPars: Names of parameters shared across groups.
        nSpec: Number of detector groups.

    Returns:
        A dict of initial values suitable for ``Minuit(**initPars)``.
    """
    
    initPars = {}
    # Populate with initial shared parameters
    for sp in sharedPars:
        initPars[sp] = defaultPars[sp]
    # Add initial unshared parameters
    unsharedPars = [key for key in defaultPars if key not in sharedPars]
    for up in unsharedPars:
        for i in range(nSpec):
            initPars[up+str(i)] = defaultPars[up]
    return initPars


def plotGlobalFit(
    dataX: np.ndarray,
    dataY: np.ndarray,
    dataE: np.ndarray,
    mObj: Minuit,
    totCost: Any,
    wsName: str,
) -> None:
    """Plot the global fit results per detector group.

    Skipped if more than 10 groups are present.

    Args:
        dataX: X values per group, shape ``(n_groups, n_bins)``.
        dataY: Y values per group, same shape.
        dataE: Error values per group, same shape.
        mObj: The ``Minuit`` object after the global fit.
        totCost: The summed cost function (iterable over groups).
        wsName: Base workspace name for the figure title.
    """

    if len(dataY) > 10:    
        print("\nToo many axes to show in figure, skipping the plot ...\n")
        return

    rows = 2
    fig, axs = plt.subplots(
        rows, 
        int(np.ceil(len(dataY)/rows)),
        figsize=(15, 8), 
        tight_layout=True,
        subplot_kw={'projection':'mantid'}
    )
    fig.canvas.setWindowTitle(wsName+"_Plot_of_Global_Fit")

    # Data used in Global Fit
    for i, (x, y, yerr, ax) in enumerate(zip(dataX, dataY, dataE, axs.flat)):
        ax.errorbar(x, y, yerr, fmt="k.", label=f"Data Group {i}") 

    # Global Fit 
    for x, costFun, ax in zip(dataX, totCost, axs.flat):
        signature = describe(costFun)

        values = mObj.values[signature]
        errors = mObj.errors[signature]

        yfit = costFun.model(x, *values)

        # Build a decent legend
        leg = []
        for p, v, e in zip(signature, values, errors):
            leg.append(f"${p} = {v:.3f} \pm {e:.3f}$")

        ax.fill_between(x, yfit, label="\n".join(leg), alpha=0.4)
        ax.legend()
    fig.show()
    return
