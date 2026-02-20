from vesuvio_analysis.core_functions.ICHelpers import completeICFromInputs
import numpy as np
from pathlib import Path

ipFilesPath = Path(__file__).absolute().parent.parent / "ip_files"
ipFilePath = ipFilesPath / "ip2018_3.par"  


class LoadVesuvioBackParameters:
    runs="43066-43076"         # 77K         # The numbers of the runs to be analysed
    empty_runs="41876-41923"   # 77K         # The numbers of the empty runs to be subtracted
    spectra='3-134'                          # Spectra to be analysed
    mode='DoubleDifference'
    ipfile=str(ipFilesPath / "ip2019.par")   

    subEmptyFromRaw = True         # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1       
    scaleRaw = 1 


class LoadVesuvioFrontParameters:
    runs='43066-43076'         # 100K        # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
    spectra='144-182'                        # Spectra to be analysed
    mode='SingleDifference'
    ipfile=str(ipFilesPath / "ip2018_3.par") 

    subEmptyFromRaw = False         # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1
    scaleRaw = 1 


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):
    InstrParsPath = ipFilesPath / "ip2018_3.par" 

    HToMassIdxRatio = 19.0620008206  # Set to zero or None when H is not present
    massIdx = 0   # Idx of mass to take the ratio with
    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])
    # noOfMasses = len(masses)

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers   
            1, 12, 0.,    
            1, 12, 0.,   
            1, 12.5, 0.    
        ])
    bounds = np.array([
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [11, 14], [-3, 1]
        ])
    constraints = ()

    noOfMSIterations = 0     #4
    firstSpec = 3    #3
    lastSpec = 13    #134

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # # Parameters of workspaces in input_ws
    tofBinning='275.,1.,420'                    # Binning of ToF spectra
    maskTOFRange = None

    # Only when running tests, to match original script
    runHistData = True
    normVoigt = False


class ForwardInitialConditions(GeneralInitialConditions):

    InstrParsPath = ipFilePath

    masses = np.array([1.0079, 12, 16, 27]) 
    # noOfMasses = len(masses)

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers  
        1, 4.7, 0, 
        1, 12.71, 0.,    
        1, 8.76, 0.,   
        1, 13.897, 0.    
    ])
    bounds = np.array([
        [0, np.nan], [3, 6], [-3, 1],
        [0, np.nan], [12.71, 12.71], [-3, 1],
        [0, np.nan], [8.76, 8.76], [-3, 1],
        [0, np.nan], [13.897, 13.897], [-3, 1]
    ])
    constraints = ()

    noOfMSIterations = 1   #4
    firstSpec = 164   #144
    lastSpec = 175    #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 179])

    tofBinning="110,1.,430"                 # Binning of ToF spectra
    maskTOFRange = None

    # Only when running tests
    runHistData = True
    normVoigt = False
    

class YSpaceFitInitialConditions:
    showPlots = False
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    fitModel = "SINGLE_GAUSSIAN"      
    runMinos = True
    globalFit = None 
    nGlobalFitGroups = 4
    maskTypeProcedure = None

# userControls and bootIC defined in corresponding test scripts

wsFrontIC = LoadVesuvioFrontParameters
wsBackIC = LoadVesuvioBackParameters     # THIS WAS SET TO FRONT

fwdIC = ForwardInitialConditions
bckwdIC = BackwardInitialConditions
yFitIC = YSpaceFitInitialConditions

scriptName = "tests"
