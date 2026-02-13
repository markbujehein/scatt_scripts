import json
from pathlib import Path

from mantid.simpleapi import LoadVesuvio, SaveNexus

from .models import (
    BootstrapInitialConditions,
    InitialConditions,
    LoadVesuvioParameters,
    YSpaceFitInitialConditions,
)

# Use Current Working Directory (CWD) instead of library location
# This ensures outputs go to your project folder, not the library source
experimentsPath = Path.cwd() / "experiments"


def completeICFromInputs(
    IC: InitialConditions, scriptName: str, wsIC: LoadVesuvioParameters
) -> InitialConditions:
    """Assigns new methods to the initial conditions class from the inputs of that class"""

    assert IC.lastSpec > IC.firstSpec, (
        "Last spectrum needs to be bigger than first spectrum"
    )
    assert ((IC.lastSpec < 135) and (IC.firstSpec < 135)) or (
        (IC.lastSpec >= 135) and (IC.firstSpec >= 135)
    ), "First and last spec need to be both in Back or Front scattering."

    if IC.lastSpec <= 134:
        runningMode = "backward"
        modeRunning = "BACKWARD"
    elif IC.firstSpec >= 135:
        runningMode = "forward"
        modeRunning = "FORWARD"
    else:
        raise ValueError("Invalid first and last spectra input.")

    name = scriptName + "_" + modeRunning + "_"

    masses = IC.masses.astype(float)

    maskedSpecNo = IC.maskedSpecAllNo[
        (IC.maskedSpecAllNo >= IC.firstSpec) & (IC.maskedSpecAllNo <= IC.lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - IC.firstSpec

    # Sort out input and output paths
    rawPath, emptyPath = inputDirsForSample(wsIC, scriptName, runningMode)

    if (not rawPath.is_file()) or (not emptyPath.is_file()):
        rawPath.parent.mkdir(parents=True, exist_ok=True)
        assert rawPath.parent == emptyPath.parent, (
            "Raw and Empty workspaces not set up to be saved under same directory"
        )
        print(
            f"\nWorkspaces not found, will save new workspaces in: {rawPath.parent.name}"
        )

        saveWSFromLoadVesuvio(wsIC, rawPath, emptyPath)

    updated_ic = IC.model_copy(
        update={
            "modeRunning": modeRunning,
            "name": name,
            "masses": masses,
            "maskedSpecAllNo": maskedSpecNo,  # NOTE: Check if we should override maskedSpecAllNo or use a new field
            "userWsRawPath": rawPath,
            "userWsEmptyPath": emptyPath,
            "scriptName": scriptName,
            "runningSampleWS": False,
            "runningPreliminary": False,
            "InstrParsPath": wsIC.ipfile
            if IC.InstrParsPath is None
            else IC.InstrParsPath,
            "maskedDetectorIdx": maskedDetectorIdx,
        }
    )

    updated_ic = setOutputDirsForSample(updated_ic, scriptName)

    # Set directories for figures
    figSavePath = experimentsPath / scriptName / "figures"
    figSavePath.mkdir(exist_ok=True)

    return updated_ic.model_copy(update={"figSavePath": figSavePath})


def inputDirsForSample(wsIC: LoadVesuvioParameters, sampleName: str, runningMode: str):
    inputWSPath = experimentsPath / sampleName / "input_ws"
    inputWSPath.mkdir(parents=True, exist_ok=True)

    rawWSName, emptyWSName = nameRawEmptyWS(sampleName, runningMode)

    newWSDir = defaultNewWSDirectory(inputWSPath, runningMode)

    rawPath = newWSDir / rawWSName
    emptyPath = newWSDir / emptyWSName

    currLoadWSDict = convertLoadWSICToDict(wsIC)

    for filePath in inputWSPath.rglob("*" + runningMode + ".json"):
        storedDict = json.load(open(filePath))

        if currLoadWSDict == storedDict:  # Ignores order
            storedWSDir = filePath.parent

            rawPath = storedWSDir / rawWSName
            emptyPath = storedWSDir / emptyWSName
            print(
                f"\nFound {runningMode} workspaces with matching inputs in: "
                f"{str(storedWSDir.name)}"
            )

    return rawPath, emptyPath


def nameRawEmptyWS(sampleName, runningMode):
    rawWSName = sampleName + "_raw_" + runningMode + ".nxs"
    emptyWSName = sampleName + "_empty_" + runningMode + ".nxs"
    return rawWSName, emptyWSName


def defaultNewWSDirectory(inputWSPath, runningMode):
    wsDirs = inputWSPath.glob(f"{runningMode}*/")
    versionNums = [float(dir.name.split("_")[-1]) for dir in wsDirs]
    versionNums = [0.0] if not versionNums else versionNums  # Take care of empty list
    newDirName = runningMode + "_" + str(max(versionNums) + 1)
    newWSDir = inputWSPath / newDirName
    return newWSDir


def setOutputDirsForSample(IC: InitialConditions, scriptName: str) -> InitialConditions:
    outputPath = experimentsPath / scriptName / "output_npz_for_testing"
    outputPath.mkdir(parents=True, exist_ok=True)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag and (IC.noOfMSIterations > 0):
        corr += "_MS"
    if IC.GammaCorrectionFlag and (IC.noOfMSIterations > 0):
        corr += "_GC"

    fileName = (
        f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}" + ".npz"
    )
    fileNameYSpace = fileName + "_ySpaceFit" + ".npz"

    updated_ic = IC.model_copy(
        update={
            "resultsSavePath": outputPath / fileName,
            "ySpaceFitSavePath": outputPath / fileNameYSpace,
        }
    )
    return updated_ic


def saveWSFromLoadVesuvio(wsIC: LoadVesuvioParameters, rawPath: Path, emptyPath: Path):
    print(f"\nLoading and storing workspace sample runs: {wsIC.runs}\n")

    rawVesuvio = LoadVesuvio(
        Filename=wsIC.runs,
        SpectrumList=wsIC.spectra,
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile),
        OutputWorkspace=rawPath.name,
    )

    SaveNexus(rawVesuvio, str(rawPath))
    print(f"\nRaw workspace stored locally under {rawPath.parent.name}\n")

    emptyVesuvio = LoadVesuvio(
        Filename=wsIC.empty_runs,
        SpectrumList=wsIC.spectra,
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile),
        OutputWorkspace=emptyPath.name,
    )

    SaveNexus(emptyVesuvio, str(emptyPath))
    print(f"\nRaw workspace stored locally under {emptyPath.parent.name}\n")

    wsLogNameFile = rawPath.name.replace("_raw_", "_").replace(".nxs", ".json")
    saveJsonFile(rawPath.parent, wsLogNameFile, wsIC)
    return


def saveJsonFile(parentDir, fileName, wsIC):
    savePath = parentDir / fileName
    currLoadWSDict = convertLoadWSICToDict(wsIC)
    json.dump(currLoadWSDict, open(savePath, "w"))
    return


def completeBootIC(
    bootIC: BootstrapInitialConditions,
    bckwdIC: InitialConditions,
    fwdIC: InitialConditions,
    yFitIC: YSpaceFitInitialConditions,
) -> tuple[BootstrapInitialConditions, InitialConditions, InitialConditions]:
    if not bootIC.runBootstrap:
        return bootIC, bckwdIC, fwdIC

    updated_bootIC, updated_bckwdIC, updated_fwdIC = setBootstrapDirs(
        bckwdIC, fwdIC, bootIC, yFitIC
    )
    return updated_bootIC, updated_bckwdIC, updated_fwdIC


def setBootstrapDirs(
    bckwdIC: InitialConditions,
    fwdIC: InitialConditions,
    bootIC: BootstrapInitialConditions,
    yFitIC: YSpaceFitInitialConditions,
):
    """Form bootstrap output data paths"""

    # Select script name and experiments path
    sampleName = bckwdIC.scriptName  # Name of sample currently running

    # Used to store running times required to estimate Bootstrap total run time.
    runTimesPath = experimentsPath / sampleName / "running_times.txt"

    # Make bootstrap and jackknife data directories
    if bootIC.bootstrapType == "JACKKNIFE":
        bootPath = experimentsPath / sampleName / "jackknife_data"
    else:
        bootPath = experimentsPath / sampleName / "bootstrap_data"
    bootPath.mkdir(parents=True, exist_ok=True)

    # Folders for skipped and unskipped MS
    if bootIC.skipMSIterations:
        dataPath = bootPath / "skip_MS_corrections"
    else:
        dataPath = bootPath / "with_MS_corrections"
    dataPath.mkdir(parents=True, exist_ok=True)

    # Create text file for logs
    logFilePath = dataPath / "data_files_log.txt"
    if not (logFilePath.is_file()):
        with open(logFilePath, "w") as txtFile:
            txtFile.write(header_string())

    updated_ICs = []
    for IC in [bckwdIC, fwdIC]:  # Make save paths for .npz files
        bootName, bootNameYFit = genBootFilesName(IC, bootIC)

        updated_ICs.append(
            IC.model_copy(
                update={
                    "bootSavePath": dataPath / bootName,
                    "bootYFitSavePath": dataPath / bootNameYFit,
                    "logFilePath": logFilePath,
                    "bootSavePathLog": logString(
                        bootName, IC, yFitIC, bootIC, isYFit=False
                    ),
                    "bootYFitSavePathLog": logString(
                        bootNameYFit, IC, yFitIC, bootIC, isYFit=True
                    ),
                }
            )
        )

    updated_bootIC = bootIC.model_copy(update={"runTimesPath": runTimesPath})

    return updated_bootIC, updated_ICs[0], updated_ICs[1]


def genBootFilesName(IC: InitialConditions, bootIC: BootstrapInitialConditions):
    """Generates save file name for either BACKWARD or FORWARD class"""

    nSamples = bootIC.nSamples
    if bootIC.bootstrapType == "JACKKNIFE":
        nSamples = 3 if bootIC.runningTest else noOfHistsFromTOFBinning(IC)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag and (IC.noOfMSIterations > 0):
        corr += "_MS"
    if IC.GammaCorrectionFlag and (IC.noOfMSIterations > 0):
        corr += "_GC"

    fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"
    bootName = fileName + f"_nsampl_{nSamples}" + ".npz"
    bootNameYFit = fileName + "_ySpaceFit" + f"_nsampl_{nSamples}" + ".npz"
    return bootName, bootNameYFit


def header_string():
    return """
    This file contains some information about each data file in the folder.
    ncp data file: boot type | procedure | tof binning | masked tof range.
    yspace fit data file: boot type | procedure | symmetrisation | rebin pars | fit model | mask type
    """


def logString(bootDataName, IC, yFitIC, bootIC, isYFit):
    if isYFit:
        log = (
            bootDataName
            + " : "
            + bootIC.bootstrapType
            + " | "
            + str(bootIC.fitInYSpace)
            + " | "
            + str(yFitIC.symmetrisationFlag)
            + " | "
            + yFitIC.rebinParametersForYSpaceFit
            + " | "
            + yFitIC.fitModel
            + " | "
            + str(yFitIC.maskTypeProcedure)
        )
    else:
        log = (
            bootDataName
            + " : "
            + bootIC.bootstrapType
            + " | "
            + str(bootIC.procedure)
            + " | "
            + IC.tofBinning
            + " | "
            + str(IC.maskTOFRange)
        )
    return log


def noOfHistsFromTOFBinning(IC: InitialConditions):
    start, spacing, end = [
        int(float(s)) for s in IC.tofBinning.split(",")
    ]  # Convert first to float and then to int because of decimal points
    return int((end - start) / spacing) - 1  # To account for last column being ignored


def buildFinalWSName(scriptName: str, procedure: str, IC: InitialConditions):
    # Format of corrected ws from last iteration
    name = scriptName + "_" + procedure + "_" + str(IC.noOfMSIterations)
    return name


def completeYFitIC(
    yFitIC: YSpaceFitInitialConditions, sampleName: str
) -> YSpaceFitInitialConditions:
    # Set directories for figures
    figSavePath = experimentsPath / sampleName / "figures"
    figSavePath.mkdir(parents=True, exist_ok=True)

    return yFitIC.model_copy(update={"figSavePath": figSavePath})


def convertLoadWSICToDict(wsIC: LoadVesuvioParameters):
    load_ws_params = {}
    for attr in ["runs", "empty_runs", "spectra", "mode", "ipfile"]:
        load_ws_params[attr] = str(getattr(wsIC, attr))  # str -> str, PosixPath -> str
    return load_ws_params
