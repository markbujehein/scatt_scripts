import numpy as np
from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateEmptyTableWorkspace

from .analysis_functions import iterativeFitForDataReduction


def runIndependentIterativeProcedure(IC, clearWS=False):
    """
    Runs the iterative fitting of NCP.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays
    """

    # Removed automatic workspace clearing as it is destructive for unrelated data.
    if clearWS:
        AnalysisDataService.clear()

    return iterativeFitForDataReduction(IC)


def runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=False):
    assert bckwdIC.modeRunning == "BACKWARD", (
        "Missing backward IC, args usage: (bckwdIC, fwdIC)"
    )
    assert fwdIC.modeRunning == "FORWARD", (
        "Missing forward IC, args usage: (bckwdIC, fwdIC)"
    )

    if clearWS:
        AnalysisDataService.clear()

    return runJoint(bckwdIC, fwdIC)


def runJoint(bckwdIC, fwdIC):
    """
    Runs iterative procedure with alternating back and forward scattering.
    """

    assert not bckwdIC.runningSampleWS, (
        "Preliminary procedure not suitable for Bootstrap."
    )
    fwdIC = fwdIC.model_copy(update={"runningPreliminary": True})
    bckwdIC = bckwdIC.model_copy(update={"runningPreliminary": True})

    # Save initial fwd MS and GC flags
    oriMS = [fwdIC.noOfMSIterations, bckwdIC.noOfMSIterations]
    oriFlags = [
        fwdIC.MSCorrectionFlag,
        fwdIC.GammaCorrectionFlag,
        bckwdIC.MSCorrectionFlag,
        bckwdIC.GammaCorrectionFlag,
    ]

    fwdIC = fwdIC.model_copy(
        update={"MSCorrectionFlag": False, "GammaCorrectionFlag": False}
    )
    bckwdIC = bckwdIC.model_copy(
        update={"MSCorrectionFlag": False, "GammaCorrectionFlag": False}
    )

    nIter = 3  # Defaulting to 3 iterations
    print(f"Running preliminary procedure with {nIter} iterations.")

    HRatios = []  # List to store HRatios
    massIdxs = []
    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, fwdScatRes = iterativeFitForDataReduction(fwdIC)
    bckwdScatRes = None

    for _ in range(int(nIter)):
        # Update H ratio
        massIdx, HRatio = calculateHToMassIdxRatio(fwdScatRes)
        HRatios.append(HRatio)
        massIdxs.append(massIdx)
        bckwdIC = bckwdIC.model_copy(
            update={"HToMassIdxRatio": HRatio, "massIdx": massIdx}
        )

        # Run joint procedure step
        wsFinal, bckwdScatRes, fwdScatRes, bckwdIC, fwdIC = _runJointStep(
            bckwdIC, fwdIC
        )

    # Set back initial flags
    fwdIC = fwdIC.model_copy(
        update={
            "runningPreliminary": False,
            "noOfMSIterations": oriMS[0],
            "MSCorrectionFlag": oriFlags[0],
            "GammaCorrectionFlag": oriFlags[1],
        }
    )
    bckwdIC = bckwdIC.model_copy(
        update={
            "runningPreliminary": False,
            "noOfMSIterations": oriMS[1],
            "MSCorrectionFlag": oriFlags[2],
            "GammaCorrectionFlag": oriFlags[3],
        }
    )

    return wsFinal, bckwdScatRes, fwdScatRes, bckwdIC, fwdIC


def _runJointStep(bckwdIC, fwdIC):
    wsFinal, bckwdScatResults = iterativeFitForDataReduction(bckwdIC)
    fwdIC = setInitFwdParsFromBackResults(bckwdScatResults, bckwdIC, fwdIC)
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, bckwdScatResults, fwdScatResults, bckwdIC, fwdIC


def runPreProcToEstHRatio(bckwdIC, fwdIC):
    """
    Wrapper for joint procedure.
    """
    wsFinal, bckwdScatResults, fwdScatResults, bckwdIC, fwdIC = runJoint(bckwdIC, fwdIC)

    # Re-calculate HRatios and massIdxs to return
    massIdx, HRatio = calculateHToMassIdxRatio(fwdScatResults)
    HRatios = [HRatio]
    massIdxs = [massIdx]
    return HRatios, massIdxs, bckwdIC, fwdIC


def askUserNoOfIterations():
    """No longer used. Defaulting to pre-set values to enable automation."""
    return 3


def calculateHToMassIdxRatio(fwdScatResults):
    """
    Calculate H ratio to mass with highest peak.
    Returns idx of mass and corresponding H ratio.
    """
    fwdMeanIntensityRatios = fwdScatResults.all_mean_intensities[-1]

    # To find idx of mass in backward scattering, take out first mass H
    fwdIntensitiesNoH = fwdMeanIntensityRatios[1:]

    massIdx = np.argmax(
        fwdIntensitiesNoH
    )  # Idex of forward inensities, which include H
    assert fwdIntensitiesNoH[massIdx] != 0, (
        "Cannot estimate H intensity since maximum peak from backscattering is zero."
    )

    HRatio = fwdMeanIntensityRatios[0] / fwdIntensitiesNoH[massIdx]

    return massIdx, HRatio


def isHPresent(masses) -> bool:
    Hmask = np.abs(masses - 1) / 1 < 0.1  # H mass within 10% of 1 au

    if np.any(Hmask):  # H present
        print("\nH mass detected.\n")
        assert len(Hmask) > 1, (
            "When H is only mass present, run independent forward procedure, not joint."
        )
        assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
        assert sum(Hmask) == 1, "More than one mass very close to H were detected."
        return True
    else:
        return False


def createTableWSHRatios(HRatios, massIdxs):
    """
    Creates a Mantid TableWorkspace with the estimated H ratios.
    """

    tableWS = CreateEmptyTableWorkspace(
        OutputWorkspace="H_Ratios_From_Preliminary_Procedure"
    )
    tableWS.setTitle("H Ratios and Idxs at each iteration")
    tableWS.addColumn(type="int", name="iter")
    tableWS.addColumn(type="float", name="H Ratio")
    tableWS.addColumn(type="int", name="Mass Idx")
    for i, (hr, hi) in enumerate(zip(HRatios, massIdxs)):
        tableWS.addRow([i, hr, hi])
    return tableWS


def setInitFwdParsFromBackResults(bckwdScatResults, bckwdIC, fwdIC):
    """
    Used to pass mean widths and intensities from back scattering onto intial conditions of forward scattering.
    Checks if H is present and adjust the passing accordingly:
    If H present, use HToMassIdxRatio to recalculate intensities and fix only non-H widths.
    If H not present, widths and intensities are directly mapped and all widhts except first are fixed.
    """

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = bckwdScatResults.all_mean_widths[-1]
    backMeanIntensityRatios = bckwdScatResults.all_mean_intensities[-1]

    new_initPars = fwdIC.initPars.copy()
    new_bounds = fwdIC.bounds.copy()

    if isHPresent(fwdIC.masses):
        assert len(backMeanWidths) == fwdIC.noOfMasses - 1, (
            "H Mass present, no of masses in front needs to be bigger than back by 1."
        )

        # Use H ratio to calculate intensity ratios
        HIntensity = bckwdIC.HToMassIdxRatio * backMeanIntensityRatios[bckwdIC.massIdx]
        # Add H intensity in the first idx
        initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
        # Normalize intensities
        initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)

        # Set calculated intensity ratios to forward scattering
        new_initPars[0::3] = initialFwdIntensityRatios
        # Set forward widths from backscattering
        new_initPars[4::3] = backMeanWidths
        # Fix all widths except for H, i.e. the first one
        new_bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1, 2))

    else:  # H mass not present anywhere
        assert len(backMeanWidths) == fwdIC.noOfMasses, (
            "H Mass not present, no of masses needs to be same for front and back scattering."
        )

        # Set widths and intensity ratios
        new_initPars[1::3] = backMeanWidths
        new_initPars[0::3] = backMeanIntensityRatios

        if len(backMeanWidths) > 1:  # In the case of single mass, width is not fixed
            # Fix all widhts except first
            new_bounds[4::3] = backMeanWidths[1:][:, np.newaxis] * np.ones((1, 2))

    print("\nChanged initial conditions toward forward scattering.\n")
    return fwdIC.model_copy(update={"initPars": new_initPars, "bounds": new_bounds})
