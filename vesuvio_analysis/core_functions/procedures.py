
from typing import Any, List, Optional, Tuple

from .analysis_functions import iterativeFitForDataReduction, switchFirstTwoAxis
from mantid.api import AnalysisDataService, mtd
from mantid.simpleapi import CreateEmptyTableWorkspace
import numpy as np


def runIndependentIterativeProcedure(
    IC: Any, clearWS: bool = True
) -> Tuple[Any, "resultsObject"]:
    """Run the iterative NCP fitting for a single scattering direction.

    Clears all Mantid workspaces (unless *clearWS* is ``False``) and
    delegates to ``iterativeFitForDataReduction``.

    Args:
        IC: Completed ``BackwardInitialConditions`` or
            ``ForwardInitialConditions`` object.
        clearWS: Clear the AnalysisDataService before starting.

    Returns:
        A 2-tuple ``(wsFinal, fittingResults)`` from
        ``iterativeFitForDataReduction``.
    """

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()
        
    return iterativeFitForDataReduction(IC)


def runJointBackAndForwardProcedure(
    bckwdIC: Any, fwdIC: Any, clearWS: bool = True
) -> Tuple[Any, Any, Any]:
    """Run the joint backward + forward iterative fitting procedure.

    Clears workspaces, then runs backward and forward scattering in
    sequence, passing mean widths and intensities from backward
    results into the forward initial conditions.

    Args:
        bckwdIC: Completed backward initial-conditions object.
        fwdIC: Completed forward initial-conditions object.
        clearWS: Clear the AnalysisDataService before starting.

    Returns:
        A 3-tuple ``(wsFinal, bckwdScatResults, fwdScatResults)``.

    Raises:
        AssertionError: If the IC objects are in the wrong order.
    """
    assert bckwdIC.modeRunning == "BACKWARD", "Missing backward IC, args usage: (bckwdIC, fwdIC)"
    assert fwdIC.modeRunning == "FORWARD", "Missing forward IC, args usage: (bckwdIC, fwdIC)"

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()

    return runJoint(bckwdIC, fwdIC)


def runPreProcToEstHRatio(
    bckwdIC: Any, fwdIC: Any
) -> Tuple[List[float], List[int]]:
    """Run a preliminary procedure to estimate the H-to-mass intensity ratio.

    Used when hydrogen is present but ``HToMassIdxRatio`` is unknown.
    Alternates backward and forward fits with zero MS iterations to
    converge on a ratio.  The estimated ratio is written to
    ``bckwdIC.HToMassIdxRatio``.

    Args:
        bckwdIC: Completed backward IC (mutated with the estimated
            H ratio).
        fwdIC: Completed forward IC.

    Returns:
        A 2-tuple ``(HRatios, massIdxs)`` — lists of H intensity
        ratios and corresponding mass indices at each iteration.

    Raises:
        AssertionError: If called during a bootstrap run.
    """

    assert bckwdIC.runningSampleWS == False, "Preliminary procedure not suitable for Bootstrap."
    fwdIC.runningPreliminary = True

    # Store original no of MS and set MS iterations to zero
    oriMS = []
    for IC in [bckwdIC, fwdIC]:
        oriMS.append(IC.noOfMSIterations)
        IC.noOfMSIterations = 0

    nIter = askUserNoOfIterations()
 
    HRatios = []   # List to store HRatios
    massIdxs = []
    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    for i in range(int(nIter)):    # Loop until convergence is achieved

        AnalysisDataService.clear()    # Clears all Workspaces

        # Update H ratio
        massIdx, HRatio = calculateHToMassIdxRatio(fwdScatResults)
        bckwdIC.HToMassIdxRatio = HRatio
        bckwdIC.massIdx = massIdx
        HRatios.append(HRatio)
        massIdxs.append(massIdx)

        wsFinal, bckwdScatResults, fwdScatResults = runJoint(bckwdIC, fwdIC)


    print(f"\nIdxs of masses for H ratio for each iteration: \n{massIdxs}")
    print(f"\nCorresponding H ratios: \n{HRatios}")

    fwdIC.runningPreliminary = False  # Change to default since end of preliminary procedure

    # Set original number of MS iterations
    for IC, ori in zip([bckwdIC, fwdIC], oriMS):
        IC.noOfMSIterations = ori

    # Update the H ratio with the best estimate, chages bckwdIC outside function
    massIdx, HRatio = calculateHToMassIdxRatio(fwdScatResults)
    bckwdIC.HToMassIdxRatio = HRatio
    bckwdIC.massIdx = massIdx
    HRatios.append(HRatio)
    massIdxs.append(massIdx)

    return HRatios, massIdxs


def createTableWSHRatios(HRatios: List[float], massIdxs: List[int]) -> None:
    """Store H-ratio convergence history in a Mantid TableWorkspace.

    Creates a table named ``"H_Ratios_From_Preliminary_Procedure"``
    in the AnalysisDataService.

    Args:
        HRatios: H intensity ratios at each iteration.
        massIdxs: Mass indices at each iteration.
    """
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace="H_Ratios_From_Preliminary_Procedure")
    tableWS.setTitle("H Ratios and Idxs at each iteration")
    tableWS.addColumn(type="int", name="iter")
    tableWS.addColumn(type="float", name="H Ratio")
    tableWS.addColumn(type="int", name="Mass Idx")
    for i, (hr, hi) in enumerate(zip(HRatios, massIdxs)):
        tableWS.addRow([i, hr, hi]) 
    return


def askUserNoOfIterations() -> int:
    """Prompt the user for the number of preliminary iterations.

    Returns:
        Number of iterations entered by the user.

    Raises:
        KeyboardInterrupt: If the user declines to run.
    """
    print("\nH was detected but HToMassIdxRatio was not provided.")
    print("\nSugested preliminary procedure:\n\nrun_forward\nfor n:\n    estimate_HToMassIdxRatio\n    run_backward\n    run_forward")
    userInput = input("\n\nDo you wish to run preliminary procedure to estimate HToMassIdxRatio? (y/n)") 
    if not((userInput=="y") or (userInput=="Y")): raise KeyboardInterrupt("Preliminary procedure interrupted.")
    
    nIter = int(input("\nHow many iterations do you wish to run? n="))
    return nIter
 

def calculateHToMassIdxRatio(fwdScatResults: Any) -> Tuple[int, float]:
    """Estimate the H intensity ratio from forward-scattering results.

    Finds the non-H mass with the highest mean intensity and computes
    ``H_intensity / max_non_H_intensity``.

    Args:
        fwdScatResults: ``resultsObject`` from the forward fit.

    Returns:
        A 2-tuple ``(massIdx, HRatio)`` — the index of the reference
        mass (in the backward mass list) and the ratio.

    Raises:
        AssertionError: If the reference mass intensity is zero.
    """
    fwdMeanIntensityRatios = fwdScatResults.all_mean_intensities[-1] 

    # To find idx of mass in backward scattering, take out first mass H
    fwdIntensitiesNoH = fwdMeanIntensityRatios[1:]

    massIdx = np.argmax(fwdIntensitiesNoH)   # Idex of forward inensities, which include H
    assert fwdIntensitiesNoH[massIdx] != 0, "Cannot estimate H intensity since maximum peak from backscattering is zero."
   
    HRatio = fwdMeanIntensityRatios[0] / fwdIntensitiesNoH[massIdx]

    return massIdx, HRatio


def runJoint(
    bckwdIC: Any, fwdIC: Any
) -> Tuple[Any, Any, Any]:
    """Execute backward then forward iterative fits in sequence.

    Passes mean widths and intensities from the backward fit into
    the forward initial conditions via ``setInitFwdParsFromBackResults``.

    Args:
        bckwdIC: Completed backward IC.
        fwdIC: Completed forward IC.

    Returns:
        A 3-tuple ``(wsFinal, bckwdScatResults, fwdScatResults)``.
    """
    wsFinal, bckwdScatResults = iterativeFitForDataReduction(bckwdIC)
    setInitFwdParsFromBackResults(bckwdScatResults, bckwdIC, fwdIC)
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, bckwdScatResults, fwdScatResults   


def setInitFwdParsFromBackResults(
    bckwdScatResults: Any, bckwdIC: Any, fwdIC: Any
) -> None:
    """Update forward IC with mean widths and intensities from backward results.

    When hydrogen is present, uses ``HToMassIdxRatio`` to compute the
    H intensity and fixes all non-H widths.  When hydrogen is absent,
    directly maps widths and intensities, fixing all widths except the
    first.

    Args:
        bckwdScatResults: ``resultsObject`` from the backward fit.
        bckwdIC: Completed backward IC (carries ``HToMassIdxRatio``
            and ``massIdx``).
        fwdIC: Completed forward IC (mutated in-place).
    """

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = bckwdScatResults.all_mean_widths[-1]
    backMeanIntensityRatios = bckwdScatResults.all_mean_intensities[-1] 

    if isHPresent(fwdIC.masses):

        assert len(backMeanWidths) == fwdIC.noOfMasses-1, "H Mass present, no of masses in front needs to be bigger than back by 1."

        # Use H ratio to calculate intensity ratios 
        HIntensity = bckwdIC.HToMassIdxRatio * backMeanIntensityRatios[bckwdIC.massIdx]
        # Add H intensity in the first idx
        initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
        # Normalize intensities
        initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)

        # Set calculated intensity ratios to forward scattering 
        fwdIC.initPars[0::3] = initialFwdIntensityRatios
        # Set forward widths from backscattering
        fwdIC.initPars[4::3] = backMeanWidths
        # Fix all widths except for H, i.e. the first one
        fwdIC.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))

    else:   # H mass not present anywhere

        assert len(backMeanWidths) == fwdIC.noOfMasses, "H Mass not present, no of masses needs to be the same for front and back scattering."

        # Set widths and intensity ratios
        fwdIC.initPars[1::3] = backMeanWidths       
        fwdIC.initPars[0::3] = backMeanIntensityRatios  

        if len(backMeanWidths) > 1:           # In the case of single mass, width is not fixed
            # Fix all widhts except first
            fwdIC.bounds[4::3] = backMeanWidths[1:][:, np.newaxis] * np.ones((1,2))   

    print("\nChanged initial conditions of forward scattering according to mean widhts and intensity ratios from backscattering.\n")
    return


def isHPresent(masses: np.ndarray) -> bool:
    """Check whether hydrogen is present in the mass array.

    Hydrogen is identified as a mass within 10% of 1 a.m.u.  It must
    be the first element of *masses* and appear exactly once.

    Args:
        masses: Atomic masses in a.m.u., shape ``(n_masses,)``.

    Returns:
        ``True`` if hydrogen is present, ``False`` otherwise.

    Raises:
        AssertionError: If hydrogen is not first, appears more than
            once, or is the only mass present.
    """

    Hmask = np.abs(masses-1)/1 < 0.1        # H mass whithin 10% of 1 au

    if np.any(Hmask):    # H present

        print("\nH mass detected.\n")
        assert len(Hmask) > 1, "When H is only mass present, run independent forward procedure, not joint."
        assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
        assert sum(Hmask) == 1, "More than one mass very close to H were detected."
        return True
    else:
        return False


