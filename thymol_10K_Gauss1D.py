import time
from pathlib import Path

import numpy as np
from vesuvio_analysis.core_functions.bootstrap_analysis import (
    runAnalysisOfStoredBootstrap,
)
from vesuvio_analysis.core_functions.run_script import runScript

scriptName = Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = (
    Path(__file__).absolute().parent / "experiments" / scriptName
)  # Path to the repository
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    runs = "50888-50900"  # The numbers of the runs to be analysed
    empty_runs = "51382-51415"  # The numbers of the empty runs to be subtracted
    spectra = "3-134"  # Spectra to be analysed
    mode = "DoubleDifference"
    ipfile = (
        ipFilesPath / "IP_2024_10_31_E1_variable_unmasked.par"
    )  # Name of ip file in ip_files folder


class LoadVesuvioFrontParameters:  # Same as previous class but for forward ws
    runs = "50888-50900"  # The numbers of the runs to be analysed
    empty_runs = "51382-51415"  # The numbers of the empty runs to be subtracted
    spectra = "135-182"
    mode = "SingleDifference"
    ipfile = ipFilesPath / "IP_2024_10_31_E1_variable_unmasked.par"


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""

    vertical_width, horizontal_width, thickness = (
        0.1,
        0.1,
        0.001,
    )  # Sample slab parameters, expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):
    subEmptyFromRaw = True  # Subtracts Empty WS from Raw WS
    scaleEmpty = 1  # Scaling factor
    scaleRaw = 1  # Scaling factor

    # Ratio of H peak to C

    # C10H14O:  14*82: 10*5.71: 4.232

    # HToMassIdxRatio = 20.1050788

    HToMassIdxRatio = None  # Set to None either when H not present or ratio not known
    massIdx = 1  # Idx of mass to take the ratio with, idx is relative to backward scattering masses

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])

    initPars = np.array(
        [
            # Intensities, NCP widths, NCP centers
            1,
            4.9,
            0.0,
            1,
            4.9,
            0.0,
            1,
            9.27,
            0.0,
        ]
    )
    bounds = np.array(
        [
            [0, np.nan],
            [1.573, 12],
            [-10, 10],
            [0, np.nan],
            [1.816, 12],
            [-10, 10],
            [0, np.nan],
            [9.27, 9.27],
            [-10, 10],
        ]
    )

    # C10H14O:  14*82: 10*5.71: 4.232

    # constraints =  ({'type': 'eq', 'fun': lambda par:  4.232*par[0] - 57.1*par[3]})

    constraints = []

    noOfMSIterations = 2  # Number of MS corrections, 0 is no correction
    firstSpec = 3  # 3
    lastSpec = 134  # 134

    maskedSpecAllNo = np.array([18, 34, 45, 52, 62])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    tofBinning = "110,1.,500"
    maskTOFRange = None  # TOF Range for the resonance peak

    transmission_guess = 0.6  # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.0e5  # Used in MS correction


class ForwardInitialConditions(GeneralInitialConditions):  # Same structure as above
    subEmptyFromRaw = False
    scaleEmpty = 1
    scaleRaw = 1

    # Masses, instrument parameters and initial fitting parameters

    masses = np.array([1.0079, 12, 16, 27])

    initPars = np.array(
        [
            # Intensities, NCP widths, NCP centers
            1,
            5,
            0.0,
            1,
            4.9,
            0.0,
            1,
            4.9,
            0.0,
            1,
            9.27,
            0.0,
        ]
    )
    bounds = np.array(
        [
            [0, np.nan],
            [4, 6],
            [-1, 1],
            [0, np.nan],
            [1.573, 12],
            [-10, 10],
            [0, np.nan],
            [1.816, 12],
            [-10, 10],
            [0, np.nan],
            [9.27, 9.27],
            [-10, 10],
        ]
    )

    # C10H14O:  14*82: 10*5.71: 4.232

    # constraints = ({'type': 'eq', 'fun': lambda par:  par[0] - 20.1050788*par[3]},{'type': 'eq', 'fun': lambda par:  4.232*par[3] - 57.1*par[6]})

    constraints = []

    noOfMSIterations = 2
    firstSpec = 135  # 144
    lastSpec = 182  # 182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = maskedSpecAllNo = np.array([173, 174, 178])

    tofBinning = "110,1,430"
    maskTOFRange = None  # TOF Range for the resonance peak

    transmission_guess = 0.87  # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.0e5  # Used in MS correction


class YSpaceFitInitialConditions:
    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"  # Needs to be symetric
    fitModel = "SINGLE_GAUSSIAN"  # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6', 'DOUBLE_WELL', 'ANSIO_GAUSSIAN'
    runMinos = True
    globalFit = True  # Performs global fit with Minuit by default
    nGlobalFitGroups = 3  # Number or string "ALL"
    maskTypeProcedure = "NAN"  # Options: 'NCP', 'NAN', None


class UserScriptControls:
    runRoutine = True

    # Choose main procedure to run
    procedure = "JOINT"  # Options: None, "BACKWARD", "FORWARD", "JOINT"
    # Choose on which ws to perform the fit in y space
    fitInYSpace = (
        "FORWARD"  # "FORWARD"    # Options: None, "BACKWARD", "FORWARD", "JOINT"
    )


class BootstrapInitialConditions:
    runBootstrap = False

    procedure = "BACKWARD"
    fitInYSpace = None  # "FORWARD"

    bootstrapType = (
        "BOOT_RESIDUALS"  # Options: "JACKKNIFE", "BOOT_RESIDUALS", "BOOT_GAUSS_ERRS"
    )
    nSamples = 650  # Used if running Bootstrap, otherwise code ignores it
    skipMSIterations = False  # Each replica runs with no MS or Gamma corrections
    userConfirmation = (
        True  # Asks user to confirm procedure, will probably be deleted in the future
    )


class BootstrapAnalysis:
    runAnalysis = False  # Controls whether or not analysis is run

    # Choose whether to filter averages as done in original procedure
    filterAvg = True  # True discards some unreasonable values of widths and intensities

    # Flags below control the plots to show
    plotRawWidthsIntensities = False
    plotMeanWidthsIntensities = True
    plotMeansEvolution = False
    plot2DHists = False
    plotYFitHists = True


# Initialize classes and run script below
#  ------------- Not for users ----------------

start_time = time.time()

wsBackIC = LoadVesuvioBackParameters
wsFrontIC = LoadVesuvioFrontParameters
bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yFitIC = YSpaceFitInitialConditions
bootIC = BootstrapInitialConditions
userCtr = UserScriptControls

runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

end_time = time.time()
print("\nRunning time: ", end_time - start_time, " seconds")

analysisIC = BootstrapAnalysis

runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr)
