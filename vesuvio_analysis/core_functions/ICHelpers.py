
from typing import Any, Dict, List, Optional, Tuple

from random import sample
from mantid.simpleapi import LoadVesuvio, SaveNexus
from pathlib import Path
import numpy as np
import json
from vesuvio_analysis.core_functions.ic_validation import (
    shadowValidateBackwardInitialConditions,
)
currentPath = Path(__file__).absolute().parent
experimentsPath = currentPath / ".."/ ".." / "experiments"


def completeICFromInputs(IC: Any, scriptName: str, wsIC: Any) -> None:
    """Populate derived attributes on an initial-conditions class.

    Determines the scattering direction (``BACKWARD`` / ``FORWARD``),
    constructs naming conventions, resolves input/output file paths,
    and ensures that cached Nexus workspaces exist (calling
    ``LoadVesuvio`` + ``SaveNexus`` if not).  Sets default values for
    optional attributes (``runHistData``, ``normVoigt``, etc.).

    This function mutates *IC* in-place by adding or overwriting class
    attributes.

    Args:
        IC: ``BackwardInitialConditions`` or
            ``ForwardInitialConditions`` class to be completed.
        scriptName: Base name of the submission script (without
            ``.py``).
        wsIC: ``LoadVesuvioBackParameters`` or
            ``LoadVesuvioFrontParameters`` class with run numbers,
            spectra, mode, and ip-file.

    Raises:
        ValueError: If the spectrum range does not fall entirely within
            one detector bank.
        AssertionError: If ``lastSpec <= firstSpec``.
    """

    assert IC.lastSpec > IC.firstSpec, "Last spectrum needs to be bigger than first spectrum"
    assert ((IC.lastSpec<135) & (IC.firstSpec<135)) | ((IC.lastSpec>=135) & (IC.firstSpec>=135)), "First and last spec need to be both in Back or Front scattering."

    if IC.lastSpec <= 134:
        IC.modeRunning = "BACKWARD"
    elif IC.firstSpec >= 135:
        IC.modeRunning = "FORWARD"
    else:
        raise ValueError("Invalid first and last spectra input.")

    IC.name = scriptName+"_"+IC.modeRunning+"_"

    IC.masses = IC.masses.astype(float)
    IC.noOfMasses = len(IC.masses)
    if IC.modeRunning == "BACKWARD":
        shadowValidateBackwardInitialConditions(IC)

    IC.maskedSpecNo = IC.maskedSpecAllNo[(IC.maskedSpecAllNo>=IC.firstSpec) & (IC.maskedSpecAllNo<=IC.lastSpec)]
    IC.maskedDetectorIdx = IC.maskedSpecNo - IC.firstSpec

    # Extract some attributes from wsIC
    IC.mode = wsIC.mode
    # IC.subEmptyFromRaw = wsIC.subEmptyFromRaw
    # IC.scaleEmpty = wsIC.scaleEmpty
    # IC.scaleRaw = wsIC.scaleRaw
    
    # When attribute InstrParsPath is not present, set it equal to path from wsIC
    try:    
        r = IC.InstrParsPath    # If present, leave it unaltered
    except AttributeError:
        IC.InstrParsPath = wsIC.ipfile

    # Sort out input and output paths
    rawPath, emptyPath = inputDirsForSample(wsIC, scriptName)

    if (not rawPath.is_file()) or (not emptyPath.is_file()):

        rawPath.parent.mkdir(parents=True, exist_ok=True)
        assert rawPath.parent == emptyPath.parent, "Raw and Empty workspaces not set up to be saved under the same directory"
        print(f"\nWorkspaces not found, will save new workspaces in: {rawPath.parent.name}")

        saveWSFromLoadVesuvio(wsIC, rawPath, emptyPath)
    
    IC.userWsRawPath = rawPath
    IC.userWsEmptyPath = emptyPath

    setOutputDirsForSample(IC, scriptName)
    
    # Do not run bootstrap sample, by default
    IC.runningSampleWS = False

    # Store script name
    IC.scriptName = scriptName

    # Default not running preliminary procedure to estimate HToMass0Ratio
    IC.runningPreliminary = False
    
    # Set directories for figures
    figSavePath = experimentsPath / scriptName /"figures" 
    figSavePath.mkdir(exist_ok=True)
    IC.figSavePath = figSavePath

    # Create default of not running original version with histogram data
    try:
        t = IC.runHistData
    except AttributeError:
        IC.runHistData = False

    # Norm voigt except when comparing with tests
    try:
        d = IC.normVoigt
    except AttributeError:
        IC.normVoigt = True

    return 


def inputDirsForSample(wsIC: Any, sampleName: str) -> Tuple[Path, Path]:
    """Resolve raw and empty workspace file paths for a sample.

    Searches existing directories for a matching JSON log file.  If no
    match is found, returns paths in a new versioned directory.

    Args:
        wsIC: Load-workspace parameter class with ``runs``,
            ``empty_runs``, ``spectra``, ``mode``, and ``ipfile``.
        sampleName: Base name of the experiment.

    Returns:
        A 2-tuple ``(rawPath, emptyPath)`` of ``Path`` objects.
    """
    inputWSPath = experimentsPath / sampleName / "input_ws"
    inputWSPath.mkdir(parents=True, exist_ok=True)

    runningMode = identifyRunningMode(wsIC)

    rawWSName, emptyWSName = nameRawEmptyWS(sampleName, runningMode)

    newWSDir = defaultNewWSDirectory(inputWSPath, runningMode)

    rawPath = newWSDir / rawWSName
    emptyPath = newWSDir / emptyWSName

    currLoadWSDict = convertLoadWSICToDict(wsIC)

    for filePath in inputWSPath.rglob('*' + runningMode + '.json'):
        storedDict = json.load(open(filePath))

        if currLoadWSDict == storedDict:   # Ignores order

            storedWSDir = filePath.parent

            rawPath = storedWSDir / rawWSName
            emptyPath = storedWSDir / emptyWSName
            print(f"\nFound {runningMode} workspaces with matching inputs in: {str(storedWSDir.name)}")

    return rawPath, emptyPath


def identifyRunningMode(wsIC: Any) -> str:
    """Determine the scattering direction from the spectrum range.

    Args:
        wsIC: Load-workspace parameter class with ``spectra``.

    Returns:
        ``"backward"`` or ``"forward"``.

    Raises:
        ValueError: If the spectrum range is invalid.
    """
    if int(wsIC.spectra.split("-")[1]) < 135:
        runningMode = "backward"
    elif int(wsIC.spectra.split("-")[0]) >= 135:
        runningMode = "forward"
    else:
        raise ValueError("Problem in loading workspaces: invalid range of spectra.")
    return runningMode


def nameRawEmptyWS(sampleName: str, runningMode: str) -> Tuple[str, str]:
    """Generate Nexus file names for raw and empty workspaces.

    Args:
        sampleName: Base name of the experiment.
        runningMode: ``"backward"`` or ``"forward"``.

    Returns:
        A 2-tuple ``(rawWSName, emptyWSName)``.
    """
    rawWSName = sampleName + "_raw_" + runningMode + ".nxs"
    emptyWSName = sampleName + "_empty_" + runningMode + ".nxs"
    return rawWSName, emptyWSName


def defaultNewWSDirectory(inputWSPath: Path, runningMode: str) -> Path:
    """Generate the next versioned directory for cached workspaces.

    Args:
        inputWSPath: Parent directory for workspace storage.
        runningMode: ``"backward"`` or ``"forward"``.

    Returns:
        Path to the new versioned directory.
    """
    wsDirs = inputWSPath.glob(f'{runningMode}*/')
    versionNums = [float(dir.name.split('_')[-1]) for dir in wsDirs]
    versionNums = [0.0] if not versionNums else versionNums    # Take care of empty list
    newDirName = runningMode + '_' + str(max(versionNums) + 1)
    newWSDir = inputWSPath / newDirName
    return newWSDir


def setOutputDirsForSample(IC: Any, sampleName: str) -> None:
    """Set ``.npz`` output paths on the initial-conditions object.

    Constructs file names incorporating the spectrum range, number of
    MS iterations, and correction flags.  Sets ``IC.resultsSavePath``
    and ``IC.ySpaceFitSavePath``.

    Args:
        IC: Initial-conditions object to mutate.
        sampleName: Base name of the experiment.
    """
    outputPath = experimentsPath / sampleName / "output_npz_for_testing"
    outputPath.mkdir(parents=True, exist_ok=True)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_MS"
    if IC.GammaCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_GC"

    fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"+".npz"
    fileNameYSpace = fileName + "_ySpaceFit"+".npz"

    IC.resultsSavePath = outputPath / fileName
    IC.ySpaceFitSavePath = outputPath / fileNameYSpace
    return


def saveWSFromLoadVesuvio(wsIC: Any, rawPath: Path, emptyPath: Path) -> None:
    """Load raw and empty VESUVIO data and cache as Nexus files.

    Calls Mantid ``LoadVesuvio`` for both raw and empty runs, then
    ``SaveNexus`` to the provided paths.  Also saves a JSON log file
    recording the load parameters.

    Args:
        wsIC: Load-workspace parameter class.
        rawPath: Output path for the raw Nexus file.
        emptyPath: Output path for the empty Nexus file.
    """
    
    print(f"\nLoading and storing workspace sample runs: {wsIC.runs}\n")

    rawVesuvio = LoadVesuvio(
        Filename=wsIC.runs,
        SpectrumList=wsIC.spectra,
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile),
        OutputWorkspace=rawPath.name
        )

    SaveNexus(rawVesuvio, str(rawPath))
    print(f"\nRaw workspace stored locally under {rawPath.parent.name}\n")

    emptyVesuvio = LoadVesuvio(
        Filename=wsIC.empty_runs,
        SpectrumList=wsIC.spectra,
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile),
        OutputWorkspace=emptyPath.name
        )

    SaveNexus(emptyVesuvio, str(emptyPath))
    print(f"\nRaw workspace stored locally under {emptyPath.parent.name}\n")

    wsLogNameFile = rawPath.name.replace('_raw_', '_').replace('.nxs', '.json')
    saveJsonFile(rawPath.parent, wsLogNameFile, wsIC)
    return


def saveJsonFile(parentDir: Path, fileName: str, wsIC: Any) -> None:
    """Save load-workspace parameters as a JSON file.

    Args:
        parentDir: Directory to write the file.
        fileName: JSON file name.
        wsIC: Load-workspace parameter class.
    """
    savePath = parentDir / fileName
    currLoadWSDict = convertLoadWSICToDict(wsIC)
    json.dump(currLoadWSDict, open(savePath, 'w'))
    return


def completeBootIC(bootIC: Any, bckwdIC: Any, fwdIC: Any, yFitIC: Any) -> None:
    """Populate derived attributes on the bootstrap initial-conditions class.

    Sets up output directories and file paths for bootstrap/jackknife
    data.  Does nothing if ``bootIC.runBootstrap`` is ``False``.

    Args:
        bootIC: ``BootstrapInitialConditions`` class.
        bckwdIC: Completed backward IC (used for naming).
        fwdIC: Completed forward IC.
        yFitIC: Y-space fit configuration.
    """
    if not bootIC.runBootstrap:
        return

    try:    # Assume it is not running a test if atribute is not found
        reading = bootIC.runningTest
    except AttributeError:
        bootIC.runningTest = False

    setBootstrapDirs(bckwdIC, fwdIC, bootIC, yFitIC)
    return


def setBootstrapDirs(bckwdIC: Any, fwdIC: Any, bootIC: Any, yFitIC: Any) -> None:
    """Create bootstrap output directories and set save paths.

    Organises output by bootstrap type (jackknife vs. bootstrap) and
    whether MS iterations are skipped.  Writes a log header file if
    one does not already exist.

    Args:
        bckwdIC: Completed backward IC.
        fwdIC: Completed forward IC.
        bootIC: Bootstrap configuration.
        yFitIC: Y-space fit configuration.
    """

    # Select script name and experiments path
    sampleName = bckwdIC.scriptName   # Name of sample currently running
    experimentsPath = currentPath/".."/".."/"experiments"
    
    # Used to store running times required to estimate Bootstrap total run time.
    bootIC.runTimesPath = experimentsPath / sampleName / "running_times.txt"

    # Make bootstrap and jackknife data directories
    if bootIC.bootstrapType=="JACKKNIFE":
        bootPath = experimentsPath / sampleName / "jackknife_data"
    else:
        bootPath = experimentsPath / sampleName / "bootstrap_data"
    bootPath.mkdir(exist_ok=True)

    # Folders for skipped and unskipped MS
    if bootIC.skipMSIterations:
        dataPath = bootPath / "skip_MS_corrections"
    else:
        dataPath = bootPath / "with_MS_corrections"
    dataPath.mkdir(exist_ok=True)

    # Create text file for logs
    logFilePath = dataPath / "data_files_log.txt"
    if not(logFilePath.is_file()):
        with open(logFilePath, "w") as txtFile:
            txtFile.write(header_string())

    for IC in [bckwdIC, fwdIC]:    # Make save paths for .npz files
        bootName, bootNameYFit = genBootFilesName(IC, bootIC)

        IC.bootSavePath = dataPath / bootName          # works because modeRunning has same strings as procedure
        IC.bootYFitSavePath = dataPath / bootNameYFit

        IC.logFilePath = logFilePath
        IC.bootSavePathLog = logString(bootName, IC, yFitIC, bootIC, isYFit=False)
        IC.bootYFitSavePathLog = logString(bootNameYFit, IC, yFitIC, bootIC, isYFit=True)
    return 


def genBootFilesName(IC: Any, bootIC: Any) -> Tuple[str, str]:
    """Generate ``.npz`` file names for bootstrap NCP and y-space results.

    Args:
        IC: Completed backward or forward IC.
        bootIC: Bootstrap configuration with ``nSamples`` and
            ``bootstrapType``.

    Returns:
        A 2-tuple ``(bootName, bootNameYFit)`` of file name strings.
    """

    nSamples = bootIC.nSamples
    if bootIC.bootstrapType=="JACKKNIFE": 
        nSamples = 3 if bootIC.runningTest else noOfHistsFromTOFBinning(IC)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_MS"
    if IC.GammaCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_GC"

    fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"
    bootName = fileName + f"_nsampl_{nSamples}"+".npz"
    bootNameYFit = fileName + "_ySpaceFit" + f"_nsampl_{nSamples}"+".npz"
    return bootName, bootNameYFit


def header_string() -> str:
    """Return the header text for the bootstrap log file.

    Returns:
        Multi-line header describing the log file format.
    """

    return """
    This file contains some information about each data file in the folder.
    ncp data file: boot type | procedure | tof binning | masked tof range.
    yspace fit data file: boot type | procedure | symmetrisation | rebin pars | fit model | mask type
    """

def logString(bootDataName: str, IC: Any, yFitIC: Any, bootIC: Any, isYFit: bool) -> str:
    """Build a single-line log entry for a bootstrap data file.

    Args:
        bootDataName: File name of the bootstrap data.
        IC: Completed initial-conditions object.
        yFitIC: Y-space fit configuration.
        bootIC: Bootstrap configuration.
        isYFit: ``True`` for y-space fit logs, ``False`` for NCP logs.

    Returns:
        Formatted log string.
    """
    if isYFit:
        log = (bootDataName+" : "+bootIC.bootstrapType+
        " | "+str(bootIC.fitInYSpace)+
        " | "+str(yFitIC.symmetrisationFlag)+
        " | "+yFitIC.rebinParametersForYSpaceFit+
        " | "+yFitIC.fitModel+
        " | "+str(yFitIC.maskTypeProcedure))
    else:
        log = (bootDataName+" : "+bootIC.bootstrapType+
        " | "+str(bootIC.procedure)+
        " | "+IC.tofBinning+
        " | "+str(IC.maskTOFRange))
    return log


def noOfHistsFromTOFBinning(IC: Any) -> int:
    """Calculate the number of histogram bins from the TOF binning string.

    Args:
        IC: Initial-conditions object with ``tofBinning``.

    Returns:
        Number of histogram bins (excluding the last column).
    """
    start, spacing, end = [int(float(s)) for s in IC.tofBinning.split(",")]  # Convert first to float and then to int because of decimal points
    return int((end-start)/spacing) - 1 # To account for last column being ignored


def buildFinalWSName(scriptName: str, procedure: str, IC: Any) -> str:
    """Construct the name of the corrected workspace from the last iteration.

    The name follows the convention
    ``"{scriptName}_{procedure}_{noOfMSIterations}"``.

    Args:
        scriptName: Base name of the submission script.
        procedure: ``"BACKWARD"`` or ``"FORWARD"``.
        IC: Completed initial-conditions object with
            ``noOfMSIterations``.

    Returns:
        The workspace name string.
    """
    # Format of corrected ws from last iteration
    name = scriptName + "_" + procedure + "_" + str(IC.noOfMSIterations)
    return name 

def completeYFitIC(yFitIC: Any, sampleName: str) -> None:
    """Set the figure save path on the y-space fit configuration.

    Args:
        yFitIC: ``YSpaceFitInitialConditions`` class to mutate.
        sampleName: Base name of the experiment.
    """
    # Set directories for figures

    figSavePath = experimentsPath / sampleName /  "figures" 
    figSavePath.mkdir(exist_ok=True)
    yFitIC.figSavePath = figSavePath
    return

def convertLoadWSICToDict(wsIC: Any) -> Dict[str, str]:
    """Convert load-workspace parameters to a JSON-serialisable dict.

    Args:
        wsIC: Load-workspace parameter class.

    Returns:
        A dict with string keys and string values.
    """
    load_ws_params = {}
    for attr in ["runs", "empty_runs", "spectra", "mode", "ipfile" ]:
        load_ws_params[attr] = str(getattr(wsIC, attr))      # str -> str, PosixPath -> str
    return load_ws_params
