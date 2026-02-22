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
    """Backward-scattering Vesuvio data loading parameters for Thymol at 10 K.

    Attributes:
        runs: ISIS run numbers for the Thymol 10 K backward-scattering measurement.
        empty_runs: ISIS run numbers for the empty-can background subtraction.
        spectra: Detector spectra range covering the backward bank (detectors 3–134).
        mode: Vesuvio difference mode; 'DoubleDifference' cancels the foil-in
            and foil-out backgrounds for the backward bank.
        ipfile: Path to the instrument parameter file used by LoadVesuvio (Mantid).
    """

    runs = "50888-50900"
    empty_runs = "51382-51415"
    spectra = "3-134"
    mode = "DoubleDifference"
    ipfile = ipFilesPath / "IP_2024_10_31_E1_variable_unmasked.par"


class LoadVesuvioFrontParameters:
    """Forward-scattering Vesuvio data loading parameters for Thymol at 10 K.

    Attributes:
        runs: ISIS run numbers for the Thymol 10 K forward-scattering measurement.
        empty_runs: ISIS run numbers for the empty-can background subtraction.
        spectra: Detector spectra range covering the forward bank (detectors 135–182).
        mode: Vesuvio difference mode; 'SingleDifference' is used for the forward
            bank where only one foil state is available.
        ipfile: Path to the instrument parameter file used by LoadVesuvio (Mantid).
    """

    runs = "50888-50900"
    empty_runs = "51382-51415"
    spectra = "135-182"
    mode = "SingleDifference"
    ipfile = ipFilesPath / "IP_2024_10_31_E1_variable_unmasked.par"


class GeneralInitialConditions:
    """Initial conditions shared by both backward and forward scattering analyses.

    Defines the sample slab geometry used by the multiple-scattering and
    gamma-background corrections (VesuvioCalculateMS / VesuvioThickness in Mantid).

    Attributes:
        vertical_width: Vertical extent of the sample slab in metres.
        horizontal_width: Horizontal extent of the sample slab in metres.
        thickness: Thickness of the sample slab along the beam direction in metres.
        preliminaryNoOfIterations: Number of iterations for automatic H-ratio
            convergence when hydrogen is detected but HToMassIdxRatio is unknown.
            Applied only in JOINT procedures.
    """

    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in metres
    
    # Preliminary procedure configuration (used when H-ratio estimation is needed)
    preliminaryNoOfIterations = 3  # Number of iterations for automatic H-ratio convergence when H is detected but HToMassIdxRatio is unknown


class BackwardInitialConditions(GeneralInitialConditions):
    """Backward-scattering NCP fitting configuration for Thymol at 10 K.

    Inherits slab geometry from GeneralInitialConditions.  Configures the
    iterative NCP fit on the backward bank (spectra 3–134).

    The sample is Thymol (C10H14O).  At 10 K the hydrogen is assumed
    inaccessible in the backward scattering, so HToMassIdxRatio is set to
    None.  The three fitted masses correspond to:
        index 0 → Carbon  (mass 12 u)
        index 1 → Oxygen  (mass 16 u)
        index 2 → Aluminium can (mass 27 u)

    Attributes:
        subEmptyFromRaw: If True, the empty-can workspace is subtracted from
            the raw data workspace before fitting.
        scaleEmpty: Scaling factor applied to the empty-can workspace.
        scaleRaw: Scaling factor applied to the raw data workspace.
        HToMassIdxRatio: Ratio of the hydrogen peak intensity to the reference
            mass peak.  Set to None because H is not resolved in the backward bank.
        massIdx: Index of the reference mass used for the H-ratio constraint.
        masses: Array of nuclear masses (u) for the fitted components.
        initPars: Flat array of initial fitting parameters ordered as
            [intensity, NCP_width, NCP_centre] for each mass.
        bounds: Array of (lower, upper) bounds for each element of initPars.
            Use np.nan to indicate an unbounded limit.
        constraints: Scipy-style equality/inequality constraints passed to
            scipy.optimize.minimize.  Empty list means unconstrained.
        noOfMSIterations: Number of multiple-scattering correction iterations.
            0 disables the correction entirely.
        firstSpec: First spectrum index included in the fit (inclusive).
        lastSpec: Last spectrum index included in the fit (inclusive).
        maskedSpecAllNo: Absolute spectrum numbers permanently excluded from
            the fit (e.g., noisy or dead detectors).
        MSCorrectionFlag: If True, enables the multiple-scattering correction.
        GammaCorrectionFlag: If True, enables the gamma-background correction.
        tofBinning: Rebin parameters for the ToF axis ('start,step,end' in μs).
        maskTOFRange: ToF range to mask for resonance peaks; None means no mask.
        transmission_guess: Estimated neutron transmission of the sample,
            used in the multiple-scattering correction.
        multiple_scattering_order: Maximum order of multiple scattering to model.
        number_of_events: Number of Monte Carlo events for the MS correction.
    """

    subEmptyFromRaw = True
    scaleEmpty = 1
    scaleRaw = 1

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

    # constraints =  ({'type': 'eq', 'fun': lambda par:  4.232*par[0] - 57.1*par[3]})
    constraints = []

    noOfMSIterations = 0  # Number of MS corrections, 0 is no correction
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

    runHistData = False


class ForwardInitialConditions(GeneralInitialConditions):
    """Forward-scattering NCP fitting configuration for Thymol at 10 K.

    Inherits slab geometry from GeneralInitialConditions.  Configures the
    iterative NCP fit on the forward bank (spectra 135–182).

    At 10 K hydrogen recoil is well resolved in the forward bank.  The four
    fitted masses correspond to:
        index 0 → Hydrogen (mass 1.0079 u)  — primary interest for J(y) analysis
        index 1 → Carbon   (mass 12 u)
        index 2 → Oxygen   (mass 16 u)
        index 3 → Aluminium can (mass 27 u)

    Attributes:
        subEmptyFromRaw: If True, the empty-can workspace is subtracted from
            the raw data workspace before fitting.
        scaleEmpty: Scaling factor applied to the empty-can workspace.
        scaleRaw: Scaling factor applied to the raw data workspace.
        masses: Array of nuclear masses (u) for the fitted components.
        initPars: Flat array of initial fitting parameters ordered as
            [intensity, NCP_width, NCP_centre] for each mass.
        bounds: Array of (lower, upper) bounds for each element of initPars.
            Use np.nan to indicate an unbounded limit.
        constraints: Scipy-style equality/inequality constraints passed to
            scipy.optimize.minimize.  Empty list means unconstrained.
        noOfMSIterations: Number of multiple-scattering correction iterations.
        firstSpec: First spectrum index included in the fit (inclusive).
        lastSpec: Last spectrum index included in the fit (inclusive).
        maskedSpecAllNo: Absolute spectrum numbers permanently excluded from
            the fit (e.g., noisy or dead detectors).
        MSCorrectionFlag: If True, enables the multiple-scattering correction.
        GammaCorrectionFlag: If True, enables the gamma-background correction.
        tofBinning: Rebin parameters for the ToF axis ('start,step,end' in μs).
        maskTOFRange: ToF range to mask for resonance peaks; None means no mask.
        transmission_guess: Estimated neutron transmission of the sample.
        multiple_scattering_order: Maximum order of multiple scattering to model.
        number_of_events: Number of Monte Carlo events for the MS correction.
    """

    subEmptyFromRaw = False
    scaleEmpty = 1
    scaleRaw = 1
    HToMassIdxRatio = 20.1050788

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

    # constraints = ({'type': 'eq', 'fun': lambda par:  par[0] - 20.1050788*par[3]},{'type': 'eq', 'fun': lambda par:  4.232*par[3] - 57.1*par[6]})
    constraints = []

    noOfMSIterations = 0
    firstSpec = 135  # 144
    lastSpec = 182  # 182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 178])

    tofBinning = "110,1,430"
    maskTOFRange = None  # TOF Range for the resonance peak

    transmission_guess = 0.87  # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.0e5  # Used in MS correction

    runHistData = False


class YSpaceFitInitialConditions:
    """Y-space fitting configuration for the Thymol 10 K hydrogen J(y) analysis.

    Controls the fit of the momentum distribution J(y) after the NCP fitting
    pipeline has isolated the hydrogen recoil peak and converted to y-space
    via the y-scaling relation J(y) = (M / ℏq) * (E - E_recoil).

    Attributes:
        showPlots: If True, generate and display diagnostic plots after the fit.
        symmetrisationFlag: If True, symmetrise the J(y) spectrum about y = 0
            before fitting to enforce time-reversal symmetry.
        rebinParametersForYSpaceFit: Rebin string ('start, step, end') in
            inverse ångströms applied to the averaged y-space spectrum.
            Must be symmetric about zero.
        fitModel: Name of the model function used to fit J(y).
            Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6',
            'DOUBLE_WELL', 'ANSIO_GAUSSIAN'.
        runMinos: If True, run MINOS error analysis in Minuit after MIGRAD
            to obtain asymmetric confidence intervals.
        globalFit: If True, perform a simultaneous global fit across all
            detector groups using iMinuit's CostSum mechanism.
        nGlobalFitGroups: Number of detector groups for the global fit.
            Can also be the string 'ALL' to use every detector individually.
        maskTypeProcedure: Method used to handle masked bins during fitting.
            Options: 'NCP' (replace with NCP estimate), 'NAN' (leave as NaN),
            None (no masking applied).
    """

    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"  # Needs to be symmetric
    fitModel = "SINGLE_GAUSSIAN"  # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6', 'DOUBLE_WELL', 'ANSIO_GAUSSIAN'
    runMinos = True
    globalFit = True  # Performs global fit with Minuit by default
    nGlobalFitGroups = 3  # Number or string "ALL"
    maskTypeProcedure = "NAN"  # Options: 'NCP', 'NAN', None


class UserScriptControls:
    """Top-level execution flags controlling which pipeline stages are active.

    Attributes:
        runRoutine: If True, execute the full NCP fitting and y-space fitting
            pipeline.  Mutually exclusive with BootstrapInitialConditions.runBootstrap.
        procedure: Selects the scattering direction(s) processed by the NCP
            fitting routine.
            Options: None, 'BACKWARD', 'FORWARD', 'JOINT'.
        fitInYSpace: Selects which final workspace is passed to the y-space
            fitting procedure.  Can differ from procedure.
            Options: None, 'BACKWARD', 'FORWARD', 'JOINT'.
        runningTest: If True, truncates expensive operations (iterations,
            bootstrap samples) for fast smoke-testing.  Propagated to all IC
            objects by run_script.runScript().
        runOutlierDetection: If True, runs UMAP-based hardware-outlier
            detection as part of the Phase 6 pre-fit statistical analysis
            stage.  UMAP preserves the non-linear manifold structure of
            TOF spectroscopic data (McInnes et al., 2018).
        removeOutliers: If True **and** ``runOutlierDetection`` is True,
            dynamically masks the detected outlier spectra in the Mantid
            workspace so they are excluded from clustering and the global fit.
        runPhysicsClustering: If True, runs DBSCAN physics-trend clustering
            as part of the Phase 6 pre-fit analysis.  When enabled, the
            number of valid clusters dynamically overwrites
            ``YSpaceFitInitialConditions.nGlobalFitGroups``.
        umapNNeighbors: UMAP neighbourhood size — controls local vs global
            structure preservation.  Default: 15.
        umapMinDist: UMAP minimum embedding distance — controls cluster
            compactness.  Default: 0.1.
        umapNComponents: UMAP embedding dimensionality.  Default: 2.
        verbose: If True, print pipeline headers, footers, and the optimizer
            agreement summary.  If False, suppress all informational output
            (errors and warnings are always shown).
    """

    runRoutine = True

    # Choose main procedure to run
    procedure = "JOINT"  # Options: None, "BACKWARD", "FORWARD", "JOINT"
    # Choose on which ws to perform the fit in y space
    fitInYSpace = "FORWARD"  # Options: None, "BACKWARD", "FORWARD", "JOINT"

    # Fast-track flag: when True, truncates expensive operations for smoke testing
    runningTest: bool = False

    # Phase 6 pre-fit statistical analysis toggles
    runOutlierDetection: bool = False    # UMAP hardware-outlier detection (replaces PCA)
    removeOutliers: bool = False         # Mask detected outliers from workspace
    runPhysicsClustering: bool = True    # DBSCAN physics-trend clustering → dynamic nGlobalFitGroups

    # UMAP hyperparameters for outlier detection dimensionality reduction.
    # UMAP preserves local topological structure of spectroscopic data
    # (McInnes, Healy & Melville, 2018, arXiv:1802.03426).
    umapNNeighbors: int = 15    # Neighbourhood size — balances local vs global structure
    umapMinDist: float = 0.1    # Minimum embedding distance — controls cluster compactness
    umapNComponents: int = 2    # Embedding dimensionality

    # Output verbosity: True = headers, footers, agreement summary; False = silent
    verbose: bool = True


class BootstrapInitialConditions:
    """Bootstrap / jackknife resampling configuration.

    Attributes:
        runBootstrap: If True, redirect execution to the bootstrap resampling
            procedure instead of the standard routine.  Mutually exclusive
            with UserScriptControls.runRoutine.
        procedure: Scattering direction processed by each bootstrap replica.
            Options: None, 'BACKWARD', 'FORWARD', 'JOINT'.
        fitInYSpace: Selects which workspace is fitted in y-space for each
            bootstrap replica.
            Options: None, 'BACKWARD', 'FORWARD', 'JOINT'.
        bootstrapType: Resampling strategy to use.
            Options: 'JACKKNIFE', 'BOOT_RESIDUALS', 'BOOT_GAUSS_ERRS',
            'BOOT_BAYESIAN'.  The first three perform iterative re-fitting
            of bootstrap replicas.  'BOOT_BAYESIAN' uses fast
            Dirichlet-weighted resampling of NCP residuals without
            re-fitting.
        nSamples: Number of bootstrap replicas to generate.
        skipMSIterations: If True, each replica skips the multiple-scattering
            and gamma-background corrections to reduce computation time.
        userConfirmation: If True, prompt the user to confirm the procedure
            before starting the (potentially long) bootstrap loop.
        runningTest: If True, suppress side-effects (file I/O, plots) so the
            bootstrap procedure can be exercised inside unit tests.
    """

    runBootstrap = False

    procedure = "BACKWARD"
    fitInYSpace = "FORWARD"

    bootstrapType: str = "BOOT_BAYESIAN"    # Options: "JACKKNIFE", "BOOT_RESIDUALS", "BOOT_GAUSS_ERRS", "BOOT_BAYESIAN"
    nSamples = 100  # Used if running Bootstrap, otherwise code ignores it. 500-1000 is a reasonable number for production runs
    skipMSIterations = False  # Each replica runs with no MS or Gamma corrections
    userConfirmation = (
        False  # Asks user to confirm procedure, will probably be deleted in the future
    )
    runningTest = False


class BootstrapAnalysis:
    """Post-hoc analysis configuration for stored bootstrap results.

    Run independently after runScript() returns by calling
    runAnalysisOfStoredBootstrap().  Does not interact with the runRoutine
    or runBootstrap pipelines.

    Attributes:
        runAnalysis: If True, execute the analysis of previously stored
            bootstrap output files.
        filterAvg: If True, apply the original outlier-rejection filter to
            widths and intensities before computing averages.
        plotRawWidthsIntensities: If True, plot all per-replica width and
            intensity distributions without averaging.
        plotMeanWidthsIntensities: If True, plot the mean widths and
            intensities with their bootstrap uncertainties.
        plotMeansEvolution: If True, plot the evolution of mean values as a
            function of the bootstrap sample index.
        plot2DHists: If True, plot 2-D joint histograms of widths and
            intensities.
        plotYFitHists: If True, plot histograms of the y-space fit parameters
            obtained from each bootstrap replica.
    """

    runAnalysis = False

    # Choose whether to filter averages as done in original procedure
    filterAvg = True  # True discards some unreasonable values of widths and intensities

    # Flags below control the plots to show
    plotRawWidthsIntensities = True
    plotMeanWidthsIntensities = True
    plotMeansEvolution = True
    plot2DHists = True
    plotYFitHists = True


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

analysisIC = BootstrapAnalysis

runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr)

end_time = time.time()
_elapsed = end_time - start_time
_m, _s = divmod(_elapsed, 60)
print(f"\nTotal Running Time: {_elapsed:.2f} seconds ({int(_m)}m {int(_s)}s)")