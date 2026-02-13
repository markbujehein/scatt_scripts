from mantid.api import mtd

from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.ICHelpers import (
    buildFinalWSName,
    completeBootIC,
    completeICFromInputs,
    completeYFitIC,
)
from vesuvio_analysis.core_functions.procedures import (
    createTableWSHRatios,
    isHPresent,
    runIndependentIterativeProcedure,
    runJointBackAndForwardProcedure,
    runPreProcToEstHRatio,
)


def runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC):
    # Set extra attributes from user attributes
    fwdIC = completeICFromInputs(fwdIC, scriptName, wsFrontIC)
    bckwdIC = completeICFromInputs(bckwdIC, scriptName, wsBackIC)
    bootIC, bckwdIC, fwdIC = completeBootIC(bootIC, bckwdIC, fwdIC, yFitIC)
    yFitIC = completeYFitIC(yFitIC, scriptName)

    # Ensure bootIC inherits procedure and fitInYSpace from userCtr if missing
    # but only if bootstrapping is requested
    if bootIC.runBootstrap:
        update = {}
        if bootIC.procedure is None:
            update["procedure"] = userCtr.procedure
        if bootIC.fitInYSpace is None:
            update["fitInYSpace"] = userCtr.fitInYSpace

        if update:
            bootIC = bootIC.model_copy(update=update)

    checkInputs(userCtr)
    checkInputs(bootIC)
    assert not (userCtr.runRoutine and bootIC.runBootstrap), (
        "Main routine and bootstrap both set to run!"
    )

    def runProcedure():
        proc = userCtr.procedure  # Shorthad to make it easier to read

        if proc is None:
            return

        nonlocal bckwdIC, fwdIC
        ranPreliminary = False
        if (proc == "BACKWARD") or (proc == "JOINT"):
            if isHPresent(fwdIC.masses) and (bckwdIC.HToMassIdxRatio is None):
                HRatios, massIdxs, bckwdIC, fwdIC = runPreProcToEstHRatio(
                    bckwdIC, fwdIC
                )
                ranPreliminary = True
            assert isHPresent(fwdIC.masses) != (bckwdIC.HToMassIdxRatio is None), (
                "When H is not present, HToMassIdxRatio has to be set to None"
            )

        if proc == "BACKWARD":
            res = runIndependentIterativeProcedure(bckwdIC)
        if proc == "FORWARD":
            res = runIndependentIterativeProcedure(fwdIC)
        if proc == "JOINT":
            res, bckwdScatResults, fwdScatResults, bckwdIC, fwdIC = (
                runJointBackAndForwardProcedure(bckwdIC, fwdIC)
            )

        # If preliminary procedure ran, make TableWS with H ratios values
        if ranPreliminary:
            createTableWSHRatios(HRatios, massIdxs)
        return res

    # Names of workspaces to be fitted in y space
    wsNames = []
    ICs = []
    for mode, IC in zip(["BACKWARD", "FORWARD"], [bckwdIC, fwdIC]):
        if (userCtr.fitInYSpace == mode) or (userCtr.fitInYSpace == "JOINT"):
            wsNames.append(buildFinalWSName(scriptName, mode, IC))
            ICs.append(IC)

    # If bootstrap is not None, run bootstrap procedure and finish
    if bootIC.runBootstrap:
        assert (
            (bootIC.procedure == "FORWARD")
            or (bootIC.procedure == "BACKWARD")
            or (bootIC.procedure == "JOINT")
        ), "Invalid Bootstrap procedure."
        return runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC, scriptName), None

    # Default workflow for procedure + fit in y space
    if userCtr.runRoutine:
        # Check if final ws are loaded:
        wsInMtd = [ws in mtd for ws in wsNames]  # Bool list
        if (len(wsInMtd) > 0) and all(
            wsInMtd
        ):  # When wsName is empty list, loop doesn't run
            for wsName, IC in zip(wsNames, ICs):
                resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
            return None, resYFit  # To match return below.

        checkUserClearWS()  # Check if user is OK with cleaning all workspaces
        res = runProcedure()

        resYFit = None
        for wsName, IC in zip(wsNames, ICs):
            resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])

        return res, resYFit  # Return results used only in tests


def checkUserClearWS():
    """If any workspace is loaded, log a warning about cleaning all workspaces."""

    if len(mtd) != 0:
        print("WARNING: This action will clean all current workspaces to start anew.")
    return


def checkInputs(crtIC):
    try:
        is_active = crtIC.runRoutine
    except AttributeError:
        is_active = crtIC.runBootstrap

    if not is_active:
        return

    # For active runs, procedure MUST be one of the options (not None)
    assert crtIC.procedure in [
        "BACKWARD",
        "FORWARD",
        "JOINT",
    ], f"Procedure '{crtIC.procedure}' not recognized or missing."

    # fitInYSpace is optional but must be valid if present
    assert crtIC.fitInYSpace in [
        "BACKWARD",
        "FORWARD",
        "JOINT",
        None,
    ], f"fitInYSpace '{crtIC.fitInYSpace}' not recognized."

    if (crtIC.procedure != "JOINT") and (crtIC.fitInYSpace is not None):
        assert crtIC.procedure == crtIC.fitInYSpace, (
            f"Procedure '{crtIC.procedure}' and fitInYSpace '{crtIC.fitInYSpace}' "
            "must match for non-JOINT runs."
        )
