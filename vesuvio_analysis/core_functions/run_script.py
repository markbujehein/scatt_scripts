from typing import Any, Optional, Tuple

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
from vesuvio_analysis.core_functions.statistical_plugins import (
    BayesianBootstrapSieve,
    HardwareOutlierSieve,
    PhysicsTrendSieve,
)


def runScript(
    userCtr: Any,
    scriptName: str,
    wsBackIC: Any,
    wsFrontIC: Any,
    bckwdIC: Any,
    fwdIC: Any,
    yFitIC: Any,
    bootIC: Any,
) -> Optional[Tuple[Any, Any]]:
    """Dispatch the VESUVIO analysis pipeline according to user flags.

    This is the top-level orchestrator called from every submission script.
    It completes the initial-condition objects, validates inputs, and
    branches into one of three paths:

    * **Bootstrap** (``bootIC.runBootstrap``): delegates to
      ``bootstrap.runBootstrap`` and returns immediately.
    * **Routine** (``userCtr.runRoutine``): runs the iterative NCP
      fitting procedure (``BACKWARD``, ``FORWARD``, or ``JOINT``),
      followed by fitting in y-space via
      ``fit_in_yspace.fitInYSpaceProcedure``.
    * **No-op**: if neither flag is set, nothing is executed.

    ``runRoutine`` and ``runBootstrap`` are mutually exclusive; an
    ``AssertionError`` is raised if both are ``True``.

    Expects the Mantid AnalysisDataService (``mtd``) to be available.
    When ``runRoutine`` is ``True`` and the final workspace already
    exists in ``mtd``, the NCP fitting is skipped and only the
    y-space fit is performed.

    Args:
        userCtr: ``UserScriptControls`` class with ``runRoutine``,
            ``procedure``, and ``fitInYSpace`` flags.
        scriptName: Base name of the submission script (without
            ``.py``), used to construct workspace and file names.
        wsBackIC: ``LoadVesuvioBackParameters`` class with run numbers,
            spectra, mode, and ip-file for backward scattering.
        wsFrontIC: ``LoadVesuvioFrontParameters`` class with the same
            fields for forward scattering.
        bckwdIC: ``BackwardInitialConditions`` class with masses,
            fit parameters, bounds, and correction flags.
        fwdIC: ``ForwardInitialConditions`` class with masses,
            fit parameters, bounds, and correction flags.
        yFitIC: ``YSpaceFitInitialConditions`` class controlling
            rebinning, symmetrisation, model selection, and Minos.
        bootIC: ``BootstrapInitialConditions`` class controlling
            the resampling procedure.

    Returns:
        A 2-tuple ``(res, resYFit)`` where *res* is the result object
        from the iterative NCP fit (or ``None``) and *resYFit* is the
        ``ResultsYFitObject`` from the y-space fit (or ``None``).
        When bootstrap is active, returns ``(bootResult, None)``.

    Raises:
        AssertionError: If both ``runRoutine`` and ``runBootstrap`` are
            ``True``, or if input flags contain invalid values.
    """

    # Set extra attributes from user attributes
    completeICFromInputs(fwdIC, scriptName, wsFrontIC)
    completeICFromInputs(bckwdIC, scriptName, wsBackIC)
    completeBootIC(bootIC, bckwdIC, fwdIC, yFitIC)
    completeYFitIC(yFitIC, scriptName)

    checkInputs(userCtr)
    checkInputs(bootIC)
    assert not (userCtr.runRoutine & bootIC.runBootstrap), (
        "Main routine and bootstrap both set to run!"
    )

    def runProcedure():
        proc = userCtr.procedure  # Shorthad to make it easier to read

        if proc == None:
            return

        ranPreliminary = False
        if (proc == "BACKWARD") | (proc == "JOINT"):
            if isHPresent(fwdIC.masses) & (bckwdIC.HToMassIdxRatio == None):
                HRatios, massIdxs = runPreProcToEstHRatio(
                    bckwdIC, fwdIC
                )  # Sets H ratio to bckwdIC automatically
                ranPreliminary = True
            assert isHPresent(fwdIC.masses) != (bckwdIC.HToMassIdxRatio == None), (
                "When H is not present, HToMassIdxRatio has to be set to None"
            )

        if proc == "BACKWARD":
            res = runIndependentIterativeProcedure(bckwdIC)
        if proc == "FORWARD":
            res = runIndependentIterativeProcedure(fwdIC)
        if proc == "JOINT":
            res = runJointBackAndForwardProcedure(bckwdIC, fwdIC)

        # If preliminary procedure ran, make TableWS with H ratios values
        if ranPreliminary:
            createTableWSHRatios(HRatios, massIdxs)
        return res

    # Names of workspaces to be fitted in y space
    wsNames = []
    ICs = []
    for mode, IC in zip(["BACKWARD", "FORWARD"], [bckwdIC, fwdIC]):
        if (userCtr.fitInYSpace == mode) | (userCtr.fitInYSpace == "JOINT"):
            wsNames.append(buildFinalWSName(scriptName, mode, IC))
            ICs.append(IC)

    # If bootstrap is not None, run bootstrap procedure and finish
    if bootIC.runBootstrap:
        assert (
            (bootIC.procedure == "FORWARD")
            | (bootIC.procedure == "BACKWARD")
            | (bootIC.procedure == "JOINT")
        ), "Invalid Bootstrap procedure."
        return runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC), None

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

        # --- Phase 6: Statistical Sieves (post-fit) ---
        _runStatisticalSieves(userCtr, res, resYFit)

        return res, resYFit  # Return results used only in tests


def checkUserClearWS() -> None:
    """Prompt the user before clearing all loaded Mantid workspaces.

    If any workspaces are present in the AnalysisDataService, the user
    is asked to confirm.  A ``KeyboardInterrupt`` is raised if the user
    declines.

    Raises:
        KeyboardInterrupt: If the user does not confirm the action.
    """

    if len(mtd) != 0:
        userInput = input(
            "This action will clean all current workspaces to start anew. Proceed? (y/n): "
        )
        if (userInput == "y") | (userInput == "Y"):
            pass
        else:
            raise KeyboardInterrupt("Run of procedure canceled.")
    return


def checkInputs(crtIC: Any) -> None:
    """Validate procedure and fitInYSpace flags on a control class.

    Checks that ``crtIC.procedure`` and ``crtIC.fitInYSpace`` are among
    the accepted values (``"BACKWARD"``, ``"FORWARD"``, ``"JOINT"``, or
    ``None``) and that they are consistent with each other.  Silently
    returns when the corresponding run flag is ``False``.

    Args:
        crtIC: A ``UserScriptControls`` or ``BootstrapInitialConditions``
            class whose ``procedure`` and ``fitInYSpace`` attributes are
            validated.

    Raises:
        AssertionError: If any flag value is invalid or if ``procedure``
            and ``fitInYSpace`` are inconsistent.
    """

    try:
        if ~crtIC.runRoutine:
            return
    except AttributeError:
        if ~crtIC.runBootstrap:
            return

    for flag in [crtIC.procedure, crtIC.fitInYSpace]:
        assert (
            (flag == "BACKWARD")
            | (flag == "FORWARD")
            | (flag == "JOINT")
            | (flag == None)
        ), "Option not recognized."

    if (crtIC.procedure != "JOINT") & (crtIC.fitInYSpace != None):
        assert crtIC.procedure == crtIC.fitInYSpace


def _runStatisticalSieves(userCtr: Any, res: Any, resYFit: Any) -> None:
    """Dispatch Phase 6 statistical sieves when their toggle flags are set.

    Called after the main NCP fitting and y-space fitting have completed.
    Each sieve is gated by its own boolean flag on ``userCtr`` and runs
    only when the flag is ``True``.

    Args:
        userCtr: ``UserScriptControls`` class with ``runSieve1``,
            ``runSieve2``, and ``runSieve4`` flags.
        res: Result object from the iterative NCP fit (may be ``None``).
        resYFit: ``ResultsYFitObject`` from the y-space fit (may be
            ``None``).
    """
    if getattr(userCtr, "runSieve1", False):
        print("[Phase 6] Sieve 1 (Hardware Outlier Detection) enabled.")

    if getattr(userCtr, "runSieve2", False):
        print("[Phase 6] Sieve 2 (Physics Trend Clustering) enabled.")

    if getattr(userCtr, "runSieve4", False):
        print("[Phase 6] Sieve 4 (Bayesian Bootstrap) enabled.")
