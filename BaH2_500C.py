import time
import numpy as np
from pathlib import Path
from typing import Optional
from vesuvio_analysis.core_functions.bootstrap_analysis import runAnalysisOfStoredBootstrap
from vesuvio_analysis.core_functions.run_script import runScript

scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to the repository
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    """Parameters for loading backward-scattering VESUVIO data via LoadVesuvio.

    Defines run numbers, spectra range, difference mode, and instrument
    parameter file for the backward detector bank. Consumed by
    ``ICHelpers.completeICFromInputs`` which calls ``LoadVesuvio`` (Mantid)
    and caches the result as a Nexus file.

    Attributes:
        runs: Run number range for the sample measurement.
        empty_runs: Run number range for the empty-cell measurement.
        spectra: Spectrum range string (backward bank: 3–134).
        mode: Differencing mode passed to LoadVesuvio.
        ipfile: Path to the instrument parameter ``.par`` file.
    """

    runs: str = '38531-38542'
    empty_runs: str = '38373-38387'
    spectra: str = '3-134'
    mode: str = "DoubleDifference"
    ipfile: Path = ipFilesPath / 'ip2018_3.par'


class LoadVesuvioFrontParameters:
    """Parameters for loading forward-scattering VESUVIO data via LoadVesuvio.

    Defines run numbers, spectra range, difference mode, and instrument
    parameter file for the forward detector bank. Consumed by
    ``ICHelpers.completeICFromInputs`` which calls ``LoadVesuvio`` (Mantid)
    and caches the result as a Nexus file.

    Attributes:
        runs: Run number range for the sample measurement.
        empty_runs: Run number range for the empty-cell measurement.
        spectra: Spectrum range string (forward bank: 135–182).
        mode: Differencing mode passed to LoadVesuvio.
        ipfile: Path to the instrument parameter ``.par`` file.
    """

    runs: str = '38543-38564'
    empty_runs: str = '38373-38387'
    spectra: str = '135-182'
    mode: str = "SingleDifference"
    ipfile: Path = ipFilesPath / "ip2018_3.par"


class GeneralInitialConditions:
    """Sample slab geometry shared by both backward and forward scattering.

    These dimensions define the cuboid used by ``VesuvioThickness`` and
    ``VesuvioCalculateMS`` for the multiple-scattering correction.

    Attributes:
        vertical_width: Slab height in metres.
        horizontal_width: Slab width in metres.
        thickness: Slab thickness in metres.
    """

    vertical_width: float = 0.1
    horizontal_width: float = 0.1
    thickness: float = 0.001


class BackwardInitialConditions(GeneralInitialConditions):
    """Initial conditions for the backward-scattering iterative NCP fit.

    Consumed by ``analysis_functions.iterativeFitForDataReduction``.
    Contains atomic masses, starting NCP parameters (intensity, width,
    centre), optimiser bounds, and flags controlling multiple-scattering
    and gamma-background corrections.

    Attributes:
        subEmptyFromRaw: Whether to subtract the empty-cell workspace.
        scaleEmpty: Multiplicative scale factor applied to empty workspace.
        scaleRaw: Multiplicative scale factor applied to raw workspace.
        HToMassIdxRatio: Intensity ratio of H to the mass at ``massIdx``.
            Set to ``None`` when H is absent or the ratio is unknown.
        massIdx: Index into ``masses`` used with ``HToMassIdxRatio``.
        masses: Atomic masses in a.m.u., shape ``(n_masses,)``.
        initPars: Flat array of starting [intensity, width, centre] triplets
            for each mass, shape ``(3 * n_masses,)``.
        bounds: Optimiser bounds per parameter, shape ``(3 * n_masses, 2)``.
        constraints: Tuple of ``scipy.optimize`` constraint dicts (empty
            for unconstrained fits).
        noOfMSIterations: Number of multiple-scattering correction
            iterations.  The total number of NCP fits is
            ``noOfMSIterations + 1``.
        firstSpec: First spectrum number in the workspace to fit.
        lastSpec: Last spectrum number in the workspace to fit.
        maskedSpecAllNo: Spectrum numbers to mask, shape ``(n_masked,)``.
        MSCorrectionFlag: Enable multiple-scattering correction.
        GammaCorrectionFlag: Enable gamma-background correction.
        tofBinning: Mantid-style rebinning string ``"start,step,end"``.
        maskTOFRange: Optional TOF range ``"start, end"`` for masking
            resonance peaks, or ``None`` to skip.
        transmission_guess: Estimated sample transmission for
            ``VesuvioThickness``.
        multiple_scattering_order: Maximum scattering order for
            ``VesuvioCalculateMS``.
        number_of_events: Number of Monte-Carlo events for
            ``VesuvioCalculateMS``.
    """

    subEmptyFromRaw: bool = True
    scaleEmpty: float = 0.9
    scaleRaw: int = 1

    HToMassIdxRatio: Optional[int] = 21  # Set to None when either unknown or H not present
    massIdx: int = 0

    # Masses, instrument parameters and initial fitting parameters
    masses: np.ndarray = np.array([16, 27, 28, 93, 137.3])

    initPars: np.ndarray = np.array([
    # Intensities, NCP widths, NCP centers
            1,  9,   0.,
            1,  13,  0.,
            1,  13,  0.,
            1,  20,  0.,
            1,  20,  0.
        ])
    bounds: np.ndarray = np.array([
            [0, np.nan], [7, 14], [-3, 1],
            [0, np.nan], [12, 14], [-3, 1],
            [0, np.nan], [10, 20], [-3, 1],
            [0, np.nan], [10, 40], [-3, 1],
            [0, np.nan], [10, 40], [-3, 1]
        ])
    constraints: tuple = ()

    noOfMSIterations: int = 1     #4
    firstSpec: int = 3    #3
    lastSpec: int = 134   #134

    maskedSpecAllNo: np.ndarray = np.array([18, 34, 42, 62])

    # Boolean Flags to control script
    MSCorrectionFlag: bool = True
    GammaCorrectionFlag: bool = False

    # # Parameters of workspaces in input_ws
    tofBinning: str = "110,1.,420"                    # Binning of ToF spectra
    maskTOFRange = None     # TOF Range for the resonance peak

    transmission_guess: float = 0.9        # Experimental value from VesuvioTransmission
    multiple_scattering_order: int = 2
    number_of_events: float = 1.e5


class ForwardInitialConditions(GeneralInitialConditions):
    """Initial conditions for the forward-scattering iterative NCP fit.

    Consumed by ``analysis_functions.iterativeFitForDataReduction``.
    Same structure as ``BackwardInitialConditions`` but configured for
    the forward detector bank and typically includes hydrogen as the
    first mass.

    Attributes:
        subEmptyFromRaw: Whether to subtract the empty-cell workspace.
        scaleEmpty: Multiplicative scale factor applied to empty workspace.
        scaleRaw: Multiplicative scale factor applied to raw workspace.
        masses: Atomic masses in a.m.u., shape ``(n_masses,)``.  The
            first entry is hydrogen (1.0079) when H is present.
        initPars: Flat array of starting [intensity, width, centre] triplets
            for each mass, shape ``(3 * n_masses,)``.
        bounds: Optimiser bounds per parameter, shape ``(3 * n_masses, 2)``.
        constraints: Tuple of ``scipy.optimize`` constraint dicts.
        noOfMSIterations: Number of MS correction iterations.
        firstSpec: First spectrum number to fit.
        lastSpec: Last spectrum number to fit.
        MSCorrectionFlag: Enable multiple-scattering correction.
        GammaCorrectionFlag: Enable gamma-background correction.
        maskedSpecAllNo: Spectrum numbers to mask, shape ``(n_masked,)``.
        tofBinning: Mantid-style rebinning string.
        maskTOFRange: Optional TOF range for masking resonance peaks.
        transmission_guess: Estimated sample transmission.
        multiple_scattering_order: Maximum scattering order for MS.
        number_of_events: Number of Monte-Carlo events for MS.
    """

    subEmptyFromRaw: bool = True
    scaleEmpty: float = 0.9
    scaleRaw: int = 1

    masses: np.ndarray = np.array([1.0079, 16, 27, 28, 93, 137.3])

    initPars: np.ndarray = np.array([
    # Intensities, NCP widths, NCP centers
            10, 5, 0.,
            1,  9,   0.,
            1,  13,  0.,
            1,  13,  0.,
            1,  20,  0.,
            1,  20,  0.
    ])
    bounds: np.ndarray = np.array([
            [0, np.nan], [3, 6], [-1.5, 0.5],
            [0, np.nan], [7, 14], [-3, 1],
            [0, np.nan], [12, 14], [-3, 1],
            [0, np.nan], [10, 20], [-3, 1],
            [0, np.nan], [10, 40], [-3, 1],
            [0, np.nan], [10, 40], [-3, 1]
    ])
    constraints: tuple = ()

    noOfMSIterations: int = 0   #4
    firstSpec: int = 135   #135
    lastSpec: int = 147   #182

    # Boolean Flags to control script
    MSCorrectionFlag: bool = True
    GammaCorrectionFlag: bool = True

    maskedSpecAllNo: np.ndarray = np.array([171, 172, 173, 174])

    tofBinning: str = "110,1.,420"                 # Binning of ToF spectra
    maskTOFRange: str = "150, 163"     # TOF Range for the resonance peak

    transmission_guess: float = 0.9        # Experimental value from VesuvioTransmission
    multiple_scattering_order: int = 2
    number_of_events: float = 1.e5


class YSpaceFitInitialConditions:
    """Configuration for the momentum-space (y-space) J(y) fit.

    Consumed by ``fit_in_yspace.fitInYSpaceProcedure``. Controls the
    rebinning, symmetrisation, fit model selection, and error-analysis
    options (Minos, global fit).

    Attributes:
        showPlots: Display interactive Matplotlib figures.
        symmetrisationFlag: Symmetrise J(y) about y = 0 before fitting.
        rebinParametersForYSpaceFit: Mantid rebin string
            ``"start, step, end"``; must be symmetric about zero.
        fitModel: Model identifier for the J(y) line-shape. Options:
            ``'SINGLE_GAUSSIAN'``, ``'GC_C4'``, ``'GC_C6'``,
            ``'GC_C4_C6'``, ``'DOUBLE_WELL'``, ``'ANSIO_GAUSSIAN'``,
            ``'MULTIVARIATE_GAUSSIAN'``.
        runMinos: Run Minos asymmetric-error analysis after the fit.
        globalFit: Perform a simultaneous fit across detector groups.
        nGlobalFitGroups: Number of k-means detector groups, or
            ``"ALL"`` for one group per unmasked spectrum.
        maskTypeProcedure: Strategy for handling masked TOF bins in
            y-space. Options: ``'NCP'`` (fill with NCP), ``'NAN'``
            (bin-by-bin weighted average), or ``None``.
    """

    showPlots: bool = True
    symmetrisationFlag: bool = True
    rebinParametersForYSpaceFit: str = "-25, 0.5, 25"    # Needs to be symetric
    fitModel: str = "SINGLE_GAUSSIAN" #"DOUBLE_WELL"   # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6', 'DOUBLE_WELL', 'DOUBLE_WELL_ANSIO'
    runMinos: bool = False
    globalFit: bool = False
    nGlobalFitGroups: int = 4         # Number or string "ALL"
    maskTypeProcedure: str = "NAN" #"NAN" # "NCP"    # Options: 'NCP', 'NAN', None



class UserScriptControls:
    """Top-level boolean switches that select the analysis workflow.

    Consumed by ``run_script.runScript`` to decide which procedures to
    execute.  ``runRoutine`` and ``BootstrapInitialConditions.runBootstrap``
    are mutually exclusive.

    Attributes:
        runRoutine: Enable the full NCP fitting + y-space fitting
            pipeline.  When ``False``, processing is skipped unless
            bootstrap is active.
        procedure: Scattering direction for the main iterative fit.
            Options: ``None``, ``"BACKWARD"``, ``"FORWARD"``, ``"JOINT"``.
        fitInYSpace: Scattering direction(s) whose final workspace is
            fitted in y-space. Options: ``None``, ``"BACKWARD"``,
            ``"FORWARD"``, ``"JOINT"``.
        runningTest: When ``True``, activates the fast-track execution
            path: MS/Gamma iterations are capped at 1, bootstrap samples
            are capped at 3, and iMinuit uses a loose convergence
            tolerance.  Intended for smoke tests and CI validation.
        runOutlierDetection: Enable PCA hardware-outlier detection.
        removeOutliers: When True and outliers are detected, mask them
            from the workspace before clustering and global fit.
        runPhysicsClustering: Enable DBSCAN physics-trend clustering.
            Dynamically overwrites ``nGlobalFitGroups`` when enabled.
    """

    runRoutine: bool = False

    # Choose main procedure to run
    procedure: str = "FORWARD"   # Options: None, "BACKWARD", "FORWARD", "JOINT"
    # Choose on which ws to perform the fit in y space
    fitInYSpace: str = "FORWARD"   # Options: None, "BACKWARD", "FORWARD", "JOINT"

    # Fast-track flag: when True, truncates expensive operations for smoke testing
    runningTest: bool = False

    # Phase 6 pre-fit statistical analysis toggles
    runOutlierDetection: bool = False    # UMAP hardware-outlier detection (replaces PCA)
    removeOutliers: bool = False         # Mask detected outliers from workspace
    runPhysicsClustering: bool = False   # DBSCAN physics-trend clustering → dynamic nGlobalFitGroups

    # UMAP hyperparameters for outlier detection dimensionality reduction.
    # UMAP preserves local topological structure of spectroscopic data
    # (McInnes, Healy & Melville, 2018, arXiv:1802.03426).
    umapNNeighbors: int = 15    # Neighbourhood size — balances local vs global structure
    umapMinDist: float = 0.1    # Minimum embedding distance — controls cluster compactness
    umapNComponents: int = 2    # Embedding dimensionality

    # Output verbosity: True = headers, footers, agreement summary; False = silent
    verbose: bool = True


class BootstrapInitialConditions:
    """Configuration for the bootstrap / jackknife resampling procedure.

    Consumed by ``bootstrap.runBootstrap``.  Mutually exclusive with
    ``UserScriptControls.runRoutine``.

    Attributes:
        runBootstrap: Enable the bootstrap resampling workflow.
        procedure: Scattering direction for each bootstrap replica.
        fitInYSpace: Direction whose y-space fit is stored per replica.
        bootstrapType: Resampling strategy. Options:
            ``"JACKKNIFE"``, ``"BOOT_RESIDUALS"``, ``"BOOT_GAUSS_ERRS"``,
            ``"BOOT_BAYESIAN"``.  The first three perform iterative
            re-fitting.  ``"BOOT_BAYESIAN"`` uses fast Dirichlet-weighted
            resampling without re-fitting.
        nSamples: Number of bootstrap replicas to generate.
        skipMSIterations: Skip MS correction iterations inside each
            replica for speed.
        userConfirmation: Prompt the user for confirmation before
            starting.
        runningTest: When ``True``, limits the number of jackknife
            samples for fast testing.
    """

    runBootstrap: bool = False

    procedure: str = "FORWARD"
    fitInYSpace: str = "FORWARD"

    bootstrapType: str = "BOOT_GAUSS_ERRS"    # Options: "JACKKNIFE", "BOOT_RESIDUALS", "BOOT_GAUSS_ERRS", "BOOT_BAYESIAN"
    nSamples: int = 1
    skipMSIterations: bool = False
    userConfirmation: bool = True
    runningTest: bool = False


class BootstrapAnalysis:
    """Post-hoc analysis and plotting flags for stored bootstrap results.

    Consumed by ``bootstrap_analysis.runAnalysisOfStoredBootstrap``.
    Controls which summary statistics and plots are produced from
    previously saved ``.npz`` bootstrap data.

    Attributes:
        runAnalysis: Enable the post-hoc analysis.
        filterAvg: Apply sigma-clipping to discard outlier widths
            and intensities before averaging.
        plotRawWidthsIntensities: Show raw per-replica width/intensity
            distributions.
        plotMeanWidthsIntensities: Show mean width/intensity
            distributions.
        plotMeansEvolution: Show convergence of means across replicas.
        plot2DHists: Show 2-D histograms of width vs. intensity.
        plotYFitHists: Show histograms of y-space fit parameters.
    """

    # Flag below controls whether or not analysis is run
    runAnalysis: bool = False

    # Choose whether to filter averages as done in original procedure
    filterAvg: bool = False       # True discards some unreasonable values of widths and intensities

    # Flags below control the plots to show
    plotRawWidthsIntensities: bool = True
    plotMeanWidthsIntensities: bool = True
    plotMeansEvolution: bool = False
    plot2DHists: bool = False
    plotYFitHists: bool = True


# Initialize classes and run script below
# Not for useers

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
_elapsed = end_time - start_time
_m, _s = divmod(_elapsed, 60)
print(f"\nTotal Running Time: {_elapsed:.2f} seconds ({int(_m)}m {int(_s)}s)")

analysisIC = BootstrapAnalysis

runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr)
