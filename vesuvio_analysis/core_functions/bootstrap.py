import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mantid.api import mtd
from mantid.simpleapi import CloneWorkspace, Load, SaveNexus, SumSpectra
from scipy import stats

from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.ICHelpers import (
    buildFinalWSName,
    noOfHistsFromTOFBinning,
)
from vesuvio_analysis.core_functions.procedures import (
    runIndependentIterativeProcedure,
    runJointBackAndForwardProcedure,
)

plt.style.use("ggplot")
currentPath = Path(__file__).parent.absolute()


def runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC, scriptName):
    checkValidInput(bootIC)

    # Check to see if those directories exits already
    checkOutputDirExists(bckwdIC, fwdIC, bootIC)
    askUserConfirmation(bckwdIC, fwdIC, bootIC)

    if bootIC.bootstrapType == "JACKKNIFE":
        return JackknifeProcedure(bckwdIC, fwdIC, bootIC, yFitIC, scriptName)

    return bootstrapProcedure(bckwdIC, fwdIC, bootIC, yFitIC, scriptName)


def checkValidInput(bootIC):
    boot = bootIC.bootstrapType
    assert boot in ["JACKKNIFE", "BOOT_GAUSS_ERRS", "BOOT_RESIDUALS"], (
        "bootstrapType not recognized. Options: 'JACKKNIFE', 'BOOT_GAUSS_ERRS', 'BOOT_RESIDUALS'"
    )


def checkOutputDirExists(bckwdIC, fwdIC, bootIC):
    if bootIC.runningTest or bootIC.allowOverwrite:
        return

    proc = bootIC.procedure
    if (proc == "BACKWARD") or (proc == "JOINT"):
        checkOutDirIC(bckwdIC, bootIC)
    if (proc == "FORWARD") or (proc == "JOINT"):
        checkOutDirIC(fwdIC, bootIC)
    return


def checkOutDirIC(IC, bootIC):
    if IC.bootSavePath.is_file() or IC.bootYFitSavePath.is_file():
        print(
            f"\nOutput data files were detected:"
            f"\n{IC.bootSavePath.name}\n{IC.bootYFitSavePath.name}"
            f"\nAborting Run of Bootstrap to prevent overwriting data."
            f"\nTo avoid this issue you can change the number of samples to run, "
            f"or set 'allowOverwrite' to True in BootstrapInitialConditions."
        )
        raise FileExistsError(
            f"Output results already exist at {IC.bootSavePath}. "
            "Aborted Bootstrap to prevent overwrite."
        )
    return


def JackknifeProcedure(bckwdIC, fwdIC, bootIC, yFitIC, scriptName):
    assert bootIC.procedure is not None

    proc = bootIC.procedure
    if (proc == "FORWARD") or (proc == "BACKWARD"):
        return bootstrapProcedure(bckwdIC, fwdIC, bootIC, yFitIC, scriptName)

    elif proc == "JOINT":  # Do the Jackknife procedure separately
        # Run original procedure to change fwdIC from running backward
        parentResults, parentWSnNCPs, bckwdIC, fwdIC, yFitIC = (
            runOriginalBeforeBootstrap(bckwdIC, fwdIC, bootIC, yFitIC)
        )

        bckwd_bootIC = bootIC.model_copy(
            update={"procedure": "BACKWARD", "fitInYSpace": "BACKWARD"}
        )
        bckwdJackRes = bootstrapProcedure(
            bckwdIC, fwdIC, bckwd_bootIC, yFitIC, scriptName
        )

        fwd_bootIC = bootIC.model_copy(
            update={"procedure": "FORWARD", "fitInYSpace": "FORWARD"}
        )
        fwdJackRes = bootstrapProcedure(
            bckwd_bootIC, fwdIC, fwd_bootIC, yFitIC, scriptName
        )

        return {**bckwdJackRes, **fwdJackRes}  # For consistency
    else:
        raise ValueError("Bootstrap procedure not recognized.")


def bootstrapProcedure(bckwdIC, fwdIC, bootIC, yFitIC, scriptName):
    """
    Main algorithm for the Bootstrap.
    Allows for Jackknife or Bootstrap depending on bool flag set in bootIC.
    Chooses fast or slow (correct) version of bootstrap depending on flag set in bootIC.
    Performs either independent or joint procedure depending of len(inputIC).
    """
    if bootIC.bootstrapType == "JACKKNIFE":
        assert bootIC.procedure != "JOINT", (
            "'JOINT' mode should not have reached Jackknife here."
        )

    parentResults, parentWSnNCPs, bckwdIC, fwdIC, yFitIC = runOriginalBeforeBootstrap(
        bckwdIC, fwdIC, bootIC, yFitIC
    )
    corrCoefs = autoCorrResiduals(parentWSnNCPs)

    nSamples = chooseNSamples(bootIC, parentWSnNCPs)

    bootResults = initializeResults(parentResults, nSamples, corrCoefs)
    saveBootstrapLogs(bootResults, bckwdIC, fwdIC)
    parentWSNCPSavePaths = convertWSToSavePaths(parentWSnNCPs, scriptName)

    iStart, iEnd = chooseLoopRange(bootIC, nSamples)

    # Form each bootstrap workspace and run ncp fit with MS corrections
    for i in range(iStart, iEnd):
        plt.close("all")

        try:
            sampleInputWS, parentWS = createSampleWS(
                parentWSNCPSavePaths, i, bootIC
            )  # Creates ith sample
        except JackMaskCol:
            continue  # If Jackknife column already masked, skip to next column

        # Update models for this sample
        curr_bckwdIC, curr_fwdIC = formSampleIC(
            bckwdIC, fwdIC, bootIC, sampleInputWS, parentWS
        )
        try:
            iterResults = runMainProcedure(
                curr_bckwdIC, curr_fwdIC, bootIC, yFitIC
            )  # Conversion to YSpace with masked column
        except AssertionError:
            continue  # If the procedure fails, skip to next iteration

        storeBootIter(bootResults, i, iterResults)  # Stores results for each iteration
        saveBootstrapResults(bootResults, curr_bckwdIC, curr_fwdIC)
    return bootResults


def askUserConfirmation(bckwdIC, fwdIC, bootIC):
    """Estimates running time for all samples and asks the user to confirm the run."""

    if not (bootIC.userConfirmation):  # Skip user confirmation
        return

    tDict = storeRunnningTime(
        fwdIC, bckwdIC, bootIC
    )  # Run times file path stores in bootIC

    proc = bootIC.procedure
    runTime = 0
    if (proc == "BACKWARD") or (proc == "JOINT"):
        runTime += calcRunTime(bckwdIC, tDict["tBackNoMS"], tDict["tBackPerMS"], bootIC)

    if (proc == "FORWARD") or (proc == "JOINT"):
        runTime += calcRunTime(fwdIC, tDict["tFowNoMS"], tDict["tFowPerMS"], bootIC)

    print(f"\nEstimated time for Bootstrap procedure: {runTime / 60:.1f} hours.\n")
    # Interactive confirmation removed for automation.
    return


def storeRunnningTime(fwdIC, bckwdIC, bootIC):
    """Used to write run times to txt file."""

    savePath = bootIC.runTimesPath

    if not (savePath.is_file()):
        with open(savePath, "w") as txtFile:
            txtFile.write(
                "This file stores run times to estimate Bootstrap total run time."
            )
            txtFile.write("\nTime in minutes.\n\n")

    resDict = {}
    with open(savePath, "r") as txtFile:
        for line in txtFile:
            if line[0] == "{":  # If line contains dictionary
                resDict = eval(line)

    if len(resDict) < 4:
        if bootIC.userConfirmation:
            print(
                "Did not find necessary information to estimate runtime. Running short routine to store an estimate..."
            )
        resDict = buildRunTimes(fwdIC, bckwdIC)

        with open(savePath, "a") as txtFile:
            print(resDict, file=txtFile)

    return resDict


def buildRunTimes(fwdIC, bckwdIC):
    resDict = {}
    for IC, mode in zip([bckwdIC, fwdIC], ["Back", "Fow"]):
        for NIter, key in zip([0, 1], ["NoMS", "PerMS"]):
            curr_IC = IC.model_copy(update={"noOfMSIterations": NIter})
            t0 = time.time()
            runIndependentIterativeProcedure(curr_IC)
            t1 = time.time()
            resDict["t" + mode + key] = (t1 - t0) / 60

        # Correct times of only MS by subtacting time spend on fitting ncps
        resDict["t" + mode + "PerMS"] -= 2 * resDict["t" + mode + "NoMS"]

    return resDict


def calcRunTime(IC, tNoMS, tPerMS, bootIC):
    if bootIC.skipMSIterations:
        timePerSample = tNoMS
    else:
        timePerSample = tNoMS + (IC.noOfMSIterations) * (tNoMS + tPerMS)

    nSamples = bootIC.nSamples
    if bootIC.bootstrapType == "JACKKNIFE":
        nSamples = 3 if bootIC.runningTest else noOfHistsFromTOFBinning(IC)

    return nSamples * timePerSample


def chooseLoopRange(bootIC, nSamples):
    iStart = 0
    iEnd = nSamples
    if bootIC.bootstrapType == "JACKKNIFE" and bootIC.runningTest:
        iStart = int(nSamples / 2)
        iEnd = iStart + 3
    return iStart, iEnd


def runOriginalBeforeBootstrap(bckwdIC, fwdIC, bootIC, yFitIC):
    """Runs unaltered procedure to store parent results and select parent ws"""
    bckwdIC, fwdIC, yFitIC = setICsToDefault(bckwdIC, fwdIC, yFitIC)
    parentResults = runMainProcedure(bckwdIC, fwdIC, bootIC, yFitIC)
    parentWSnNCPs = selectParentWorkspaces(bckwdIC, fwdIC, bootIC)

    return parentResults, parentWSnNCPs, bckwdIC, fwdIC, yFitIC


def chooseNSamples(bootIC, parentWSnNCPs: dict):
    """
    Returns number of samples to run.
    If Jackknife is running, no of samples is the number of bins in the workspace."""

    nSamples = bootIC.nSamples
    if bootIC.bootstrapType == "JACKKNIFE":
        assert len(parentWSnNCPs) == 2, (
            "Running Jackknife, supports only one IC at a time."
        )
        if bootIC.procedure == "FORWARD":
            key = "fwdNCP"
        elif bootIC.procedure == "BACKWARD":
            key = "bckwdNCP"

        nSamples = (
            parentWSnNCPs[key].blocksize()
        )  # Number of cols from ncp workspace, accounts for missing last col or not
    return nSamples


def setICsToDefault(bckwdIC, fwdIC, yFitIC):
    """Disables some features of yspace fit, makes sure the default"""

    yFit_update = {}
    if yFitIC.runMinos:
        yFit_update["runMinos"] = False
    if yFitIC.globalFit:
        yFit_update["globalFit"] = False
    if yFitIC.showPlots:
        yFit_update["showPlots"] = False

    if yFit_update:
        yFitIC = yFitIC.model_copy(update=yFit_update)

    if bckwdIC.runningSampleWS:
        bckwdIC = bckwdIC.model_copy(update={"runningSampleWS": False})
    if fwdIC.runningSampleWS:
        fwdIC = fwdIC.model_copy(update={"runningSampleWS": False})

    return bckwdIC, fwdIC, yFitIC


def runMainProcedure(bckwdIC, fwdIC, bootIC, yFitIC):
    """Decides main procedure to run based on the initial conditions offered as inputs."""

    resultsDict = {}

    if (bootIC.procedure == "FORWARD") | (bootIC.procedure == "BACKWARD"):
        for mode, IC, key in zip(
            ["FORWARD", "BACKWARD"], [fwdIC, bckwdIC], ["fwd", "bckwd"]
        ):
            if bootIC.procedure == mode:
                wsFinal, bckwdScatRes = runIndependentIterativeProcedure(
                    IC, clearWS=False
                )
                resultsDict[key + "Scat"] = bckwdScatRes

                if bootIC.bootstrapType == "JACKKNIFE":
                    yFitIC.maskTypeProcedure = (
                        "NAN"  # Enable NAN averaging in y-space fit
                    )

                bckwdYFitRes = fitInYSpaceProcedure(yFitIC, IC, wsFinal)
                resultsDict[key + "YFit"] = bckwdYFitRes

    elif bootIC.procedure == "JOINT":
        ws, bckwdScatRes, fwdScatRes, bckwdIC, fwdIC = runJointBackAndForwardProcedure(
            bckwdIC, fwdIC, clearWS=False
        )
        resultsDict["bckwdScat"] = bckwdScatRes
        resultsDict["fwdScat"] = fwdScatRes

        for mode, IC, key in zip(
            ["FORWARD", "BACKWARD"], [fwdIC, bckwdIC], ["fwd", "bckwd"]
        ):
            if (bootIC.fitInYSpace == mode) | (bootIC.fitInYSpace == "JOINT"):
                wsName = buildFinalWSName(IC.scriptName, mode, IC)
                fwdYFitRes = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
                resultsDict[key + "YFit"] = fwdYFitRes
    else:
        raise ValueError("Bootstrap procedure not recognized.")

    return resultsDict


def selectParentWorkspaces(bckwdIC, fwdIC, bootIC):
    """
    Selects parent workspace from which the Bootstrap replicas will be created.
    If fast mode, the parent ws is the final ws after MS corrections.
    """
    parentWSnNCPsDict = {}

    for mode, IC, key in zip(
        ["FORWARD", "BACKWARD"], [fwdIC, bckwdIC], ["fwd", "bckwd"]
    ):
        if (bootIC.procedure == mode) | (bootIC.procedure == "JOINT"):
            wsIter = (
                str(IC.noOfMSIterations) if bootIC.skipMSIterations else "0"
            )  # In case of skipping MS, select very last corrected ws

            parentWS = mtd[IC.name + wsIter]
            parentNCP = mtd[parentWS.name() + "_TOF_Fitted_Profiles"]

            parentWSnNCPsDict[key + "WS"] = parentWS
            parentWSnNCPsDict[key + "NCP"] = parentNCP

    return parentWSnNCPsDict


def autoCorrResiduals(parentWSnNCP: dict):
    """
    Calculates the self-correlation of residuals for each spectrum.
    """
    corrCoefs = {}
    for mode in ["bckwd", "fwd"]:
        try:  # Look for workspaces in dictionary, skip if not present
            parentWS = parentWSnNCP[mode + "WS"]
            parentNCP = parentWSnNCP[mode + "NCP"]
        except KeyError:
            continue

        totNcp = parentNCP.extractY()[:, :]
        dataY = parentWS.extractY()[:, : totNcp.shape[1]]  # Missing last column or not
        residuals = dataY - totNcp

        lag = 1  # For lag-plot of self-correlation
        corr = np.zeros((len(residuals), 2))
        for i, rowRes in enumerate(residuals):
            corr[i] = stats.pearsonr(rowRes[:-lag], rowRes[lag:])

        corrCoefs[mode + "Scat"] = corr
    return corrCoefs


def initializeResults(parentResults: dict, nSamples, corrCoefs):
    """
    Initializes a list with objects to store output data.
    [BootBackResults, BootFrontResults, BootYSpaceResults]
    """
    bootResultObjs = {}

    for key in ["fwd", "bckwd"]:
        if key + "Scat" in parentResults:
            bootResultObjs[key + "Scat"] = BootScattResults(
                parentResults[key + "Scat"], nSamples, corrCoefs[key + "Scat"]
            )

        if key + "YFit" in parentResults:
            bootResultObjs[key + "YFit"] = BootYFitResults(
                parentResults[key + "YFit"], nSamples
            )
    return bootResultObjs


class BootScattResults:
    def __init__(self, parentResults, nSamples, corr):
        self.parentResult = parentResults.all_spec_best_par_chi_nit[-1]
        self.bootSamples = np.full((nSamples, *self.parentResult.shape), np.nan)
        self.corrResiduals = corr

    def storeBootIterResults(self, j, bootResult):
        self.bootSamples[j] = bootResult.all_spec_best_par_chi_nit[-1]

    def saveResults(self, IC):
        np.savez(
            IC.bootSavePath,
            boot_samples=self.bootSamples,
            parent_result=self.parentResult,
            corr_residuals=self.corrResiduals,
        )

    def saveLog(self, IC):
        with open(IC.logFilePath, "a") as logFile:
            logFile.write("\n" + IC.bootSavePathLog)


class BootYFitResults:
    def __init__(self, parentResults, nSamples):
        self.parentPopt = parentResults.popt
        self.parentPerr = parentResults.perr
        self.bootSamples = np.full((nSamples, *self.parentPopt.shape), np.nan)

    def storeBootIterResults(self, j, bootResult):
        self.bootSamples[j] = bootResult.popt

    def saveResults(self, IC):
        np.savez(
            IC.bootYFitSavePath,
            boot_samples=self.bootSamples,
            parent_popt=self.parentPopt,
            parent_perr=self.parentPerr,
        )

    def saveLog(self, IC):
        with open(IC.logFilePath, "a") as logFile:
            logFile.write("\n" + IC.bootYFitSavePathLog)


def storeBootIter(bootResultObjs: dict, j: int, bootIterResults: dict):
    for key in bootResultObjs:
        bootResultObjs[key].storeBootIterResults(j, bootIterResults[key])
    return


def saveBootstrapResults(bootResultObjs: dict, bckwdIC, fwdIC):
    for key, IC in zip(["bckwd", "fwd"], [bckwdIC, fwdIC]):
        for res in ["Scat", "YFit"]:
            if key + res in bootResultObjs:
                bootResultObjs[key + res].saveResults(IC)
    return


def saveBootstrapLogs(bootResultObjs: dict, bckwdIC, fwdIC):
    for key, IC in zip(["bckwd", "fwd"], [bckwdIC, fwdIC]):
        for res in ["Scat", "YFit"]:
            if key + res in bootResultObjs:
                bootResultObjs[key + res].saveLog(IC)
    return


def convertWSToSavePaths(parentWSnNCPs: dict, scriptName):
    savePaths = {}
    for key in parentWSnNCPs:
        savePaths[key] = saveWorkspacesLocally(parentWSnNCPs[key], scriptName)
    return savePaths


def saveWorkspacesLocally(ws, scriptName):
    keys = ws.name().split("_")
    saveName = "Parent"

    if "FORWARD" in keys:
        saveName += "_Front"
    elif "BACKWARD" in keys:
        saveName += "_Back"

    if "Profiles" in keys:
        saveName += "_NCP"

    saveName += ".nxs"

    # FIXED: Save bootstrap results to the experiments folder in the project root
    # We create a 'bootstrap_ws' folder inside the specific experiment folder is usually better,
    # but matching the original logic relative to CWD:
    ## OLD CODE:
    # savePath = currentPath / "bootstrap_ws" / saveName
    ## NEW CODE:
    savePath = Path.cwd() / "experiments" / scriptName / "bootstrap_ws" / saveName

    SaveNexus(ws, str(savePath))
    return savePath


def createSampleWS(parentWSNCPSavePaths: dict, j: int, bootIC):
    boot = bootIC.bootstrapType
    if boot == "JACKKNIFE":
        return createJackknifeWS(parentWSNCPSavePaths, j)
    elif boot == "BOOT_RESIDUALS":
        return createBootstrapWS(parentWSNCPSavePaths)
    elif boot == "BOOT_GAUSS_ERRS":
        return createBootstrapWS(parentWSNCPSavePaths, drawGauss=True)


def createBootstrapWS(parentWSNCPSavePaths: dict, drawGauss=False):
    """
    Creates bootstrap ws replica.
    Inputs: Experimental (parent) workspace and corresponding NCP total fit
    """

    bootInputWS = {}
    parentInputWS = {}
    for key in ["bckwd", "fwd"]:
        try:
            parentWSPath = parentWSNCPSavePaths[key + "WS"]
            totNcpWSPath = parentWSNCPSavePaths[key + "NCP"]
        except KeyError:
            continue

        parentWS, totNcpWS = loadWorkspacesFromPath(parentWSPath, totNcpWSPath)

        totNcp = totNcpWS.extractY()[:, :]
        dataY = parentWS.extractY()[:, : totNcp.shape[1]]  # Missing last col or not
        dataE = parentWS.extractE()[:, : totNcp.shape[1]]

        # Filter out masked columns
        maskCols = np.all(dataY == 0, axis=0)
        dataY, totNcp, dataE = (
            dataY[:, ~maskCols],
            totNcp[:, ~maskCols],
            dataE[:, ~maskCols],
        )

        # Draw DataY from Gaussian distribution
        if drawGauss:
            bootDataY = np.random.normal(dataY, dataE)  # Mean at dataY, width dataE

        else:  # Default, resample residuals
            residuals = dataY - totNcp
            bootRes = bootstrapResidualsSample(residuals)
            bootDataY = totNcp + bootRes

        # Add masked columns as in parent workspace
        fullBootDataY = np.zeros((len(bootDataY), len(maskCols)))
        fullBootDataY[:, ~maskCols] = bootDataY  # Set non-masked values

        # Pass dataY onto workspace
        wsBoot = CloneWorkspace(
            parentWS, OutputWorkspace=parentWS.name() + "_Bootstrap"
        )
        for i, row in enumerate(fullBootDataY):
            wsBoot.dataY(i)[: len(row)] = row  # Last column will be ignored or not
            if drawGauss:
                wsBoot.dataE(i)[:] = np.zeros(wsBoot.readE(i).size)

        assert ~np.all(wsBoot.extractY() == parentWS.extractY()), (
            "Bootstrap data not being correctly passed onto ws."
        )

        bootInputWS[key + "WS"] = wsBoot
        parentInputWS[key + "WS"] = parentWS
        parentInputWS[key + "NCP"] = totNcpWS
    return bootInputWS, parentInputWS


def bootstrapResidualsSample(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))  # [low, high)
        bootRes[i] = res[rowIdxs]
    return bootRes


def createJackknifeWS(parentWSNCPSavePaths: list, j: int):
    """
    Creates jackknife ws replicas.
    Inputs: Experimental (parent) workspace and corresponding NCP total fit
    """

    jackInputWS = {}
    parentInputWS = {}
    # Jackknife does not have 'JOINT' option
    # Careful with this step if in future Jackknife allows for 'JOINT' internally
    assert len(parentWSNCPSavePaths) == 2, (
        "Jackknife can only allow either forward or backward at a time."
    )
    for key in ["bckwd", "fwd"]:  # Only one iteration is selected at a time
        try:
            parentWSPath = parentWSNCPSavePaths[key + "WS"]
            totNcpWSPath = parentWSNCPSavePaths[key + "NCP"]
        except KeyError:
            continue

        parentWS, totNcpWS = loadWorkspacesFromPath(parentWSPath, totNcpWSPath)

        dataY = parentWS.extractY()

        jackDataY = dataY.copy()

        # Skip Jackknife procedure on columns that are already masked
        if np.all(jackDataY[:, j] == 0):
            raise JackMaskCol

        jackDataY[:, j] = 0  # Masks j collumn with zeros
        # DataE is not masked intentionally, to preserve errors that are used in the normalization of averaged NaN profile

        wsJack = CloneWorkspace(
            parentWS, OutputWorkspace=parentWS.name() + "_Jackknife"
        )
        for i, yRow in enumerate(jackDataY):
            wsJack.dataY(i)[:] = yRow  # Last column will be ignored in ncp fit anyway

        assert np.all(wsJack.extractY() == jackDataY), (
            "Bootstrap data not being correctly passed onto ws."
        )

        jackInputWS[key + "WS"] = wsJack
        parentInputWS[key + "WS"] = parentWS
        parentInputWS[key + "NCP"] = totNcpWS
    return jackInputWS, parentInputWS


class JackMaskCol(Exception):
    """
    Custom exception used only to flag and skip a Jackknife iteration
    for a column that is already masked.
    """

    pass


def loadWorkspacesFromPath(*savePaths):
    wsList = []
    for path in savePaths:
        saveName = path.name.split(".")[0]
        ws = Load(str(path), OutputWorkspace=saveName)
        SumSpectra(ws, OutputWorkspace=ws.name() + "_Sum")
        wsList.append(ws)

    return wsList


def formSampleIC(bckwdIC, fwdIC, bootIC, sampleInputWS: dict, parentWS: dict):
    """Adds attributes to initial conditions to start procedure with sample ws."""

    new_fwdIC = fwdIC
    new_bckwdIC = bckwdIC

    if (bootIC.procedure == "FORWARD") or (bootIC.procedure == "JOINT"):
        update = {
            "runningSampleWS": True,
            "sampleWS": sampleInputWS["fwdWS"],
            "parentWS": parentWS["fwdWS"],
        }
        if bootIC.skipMSIterations:
            update["noOfMSIterations"] = 0
        new_fwdIC = fwdIC.model_copy(update=update)

    if (bootIC.procedure == "BACKWARD") or (bootIC.procedure == "JOINT"):
        update = {
            "runningSampleWS": True,
            "sampleWS": sampleInputWS["bckwdWS"],
            "parentWS": parentWS["bckwdWS"],
        }
        if bootIC.skipMSIterations:
            update["noOfMSIterations"] = 0
        new_bckwdIC = bckwdIC.model_copy(update=update)

    return new_bckwdIC, new_fwdIC
