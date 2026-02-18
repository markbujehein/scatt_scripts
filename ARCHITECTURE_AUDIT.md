# Architectural Audit: `scatt_scripts` VESUVIO Analysis Pipeline

> **Date:** 2026-02-18
> **Scope:** Comprehensive data-flow map, module interaction analysis, and refactoring strategy.
> **Branch:** `dev` (to be created for implementation)

---

## 1. Entry-Point: `BaH2_500C.py`

The user submission script defines **seven** parameter classes that fully configure a DINS
analysis run, then dispatches into two top-level functions:

```
BaH2_500C.py
├── Parameter Classes
│   ├── LoadVesuvioBackParameters   — run numbers, spectra, mode, ip-file (backward)
│   ├── LoadVesuvioFrontParameters  — run numbers, spectra, mode, ip-file (forward)
│   ├── GeneralInitialConditions    — slab geometry shared by both scattering directions
│   ├── BackwardInitialConditions   — masses, initPars, bounds, MS/GC flags, H-ratio
│   ├── ForwardInitialConditions    — masses, initPars, bounds, MS/GC flags
│   ├── YSpaceFitInitialConditions  — fitModel, rebinPars, symmetrisation, Minos, global fit
│   ├── UserScriptControls          — runRoutine, procedure, fitInYSpace,
│   │                                  runOutlierDetection, runPhysicsClustering,
│   │                                  runBayesianBootstrap
│   ├── BootstrapInitialConditions  — runBootstrap, bootstrapType, nSamples
│   └── BootstrapAnalysis           — runAnalysis, plot flags
│
├── runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)
└── runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr)
```

### Parameter Dispatch Summary

| Class | Key Parameters | Consumed By |
|---|---|---|
| `LoadVesuvioBackParameters` | `runs`, `empty_runs`, `spectra`, `mode`, `ipfile` | `ICHelpers.completeICFromInputs` → `LoadVesuvio` (Mantid) |
| `LoadVesuvioFrontParameters` | Same as above for forward | Same |
| `BackwardInitialConditions` | `masses`, `initPars`, `bounds`, `HToMassIdxRatio`, `noOfMSIterations`, `MSCorrectionFlag`, `GammaCorrectionFlag` | `analysis_functions.iterativeFitForDataReduction` |
| `ForwardInitialConditions` | Same (no H-ratio) | Same |
| `YSpaceFitInitialConditions` | `fitModel`, `rebinParametersForYSpaceFit`, `symmetrisationFlag`, `runMinos`, `globalFit` | `fit_in_yspace.fitInYSpaceProcedure` |
| `UserScriptControls` | `runRoutine`, `procedure`, `fitInYSpace`, `runOutlierDetection`, `runPhysicsClustering`, `runBayesianBootstrap` | `run_script.runScript` (branching logic), `run_script._runStatisticalAnalysis` (Phase 6) |
| `BootstrapInitialConditions` | `runBootstrap`, `bootstrapType`, `nSamples` | `bootstrap.runBootstrap` |

---

## 2. `runScript()` Logical Flow

**File:** `vesuvio_analysis/core_functions/run_script.py`

```
runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)
│
├── 1. completeICFromInputs(fwdIC, ...)   ← Sets modeRunning, loads/saves Nexus via LoadVesuvio
├── 2. completeICFromInputs(bckwdIC, ...) ← Same for backward
├── 3. completeBootIC(bootIC, ...)        ← Sets bootstrap dirs
├── 4. completeYFitIC(yFitIC, ...)        ← Sets figure paths
├── 5. checkInputs(userCtr)
├── 6. checkInputs(bootIC)
├── 7. assert not(runRoutine & runBootstrap)
│
├── [A] If bootIC.runBootstrap:
│   └── return runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC)
│
└── [B] If userCtr.runRoutine:
    ├── Check if final ws already in mtd → skip to fitInYSpace
    ├── checkUserClearWS()
    ├── runProcedure()  ← inner function
    │   ├── procedure == "BACKWARD" → runIndependentIterativeProcedure(bckwdIC)
    │   ├── procedure == "FORWARD"  → runIndependentIterativeProcedure(fwdIC)
    │   └── procedure == "JOINT"    → runJointBackAndForwardProcedure(bckwdIC, fwdIC)
    │       └── (if H present & HToMassIdxRatio is None → runPreProcToEstHRatio first)
    │
    ├── for each wsName in [BACKWARD, FORWARD, JOINT as requested]:
    │   └── fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
    │
    └── _runStatisticalAnalysis(userCtr, res, bckwdIC, fwdIC)  ← Phase 6 (post-fit)
        ├── [if runOutlierDetection]  → HardwareOutlierDetector.fit_predict(spectra)
        ├── [if runPhysicsClustering] → PhysicsTrendClusterer.fit_predict([L1, theta])
        └── [if runBayesianBootstrap] → BayesianBootstrap.compute_weighted_residuals(residuals)
```

### Critical Boolean Gates

| Flag | Effect |
|---|---|
| `userCtr.runRoutine == True` | Enables the full NCP fitting + y-space fitting pipeline |
| `userCtr.runRoutine == False` | Skips all processing (no-op unless bootstrap is on) |
| `bootIC.runBootstrap == True` | Redirects to bootstrap resampling procedure (mutually exclusive with runRoutine) |
| `userCtr.procedure` | Selects BACKWARD / FORWARD / JOINT scattering direction |
| `userCtr.fitInYSpace` | Selects which final ws to fit in y-space (can differ from procedure) |
| `IC.MSCorrectionFlag` | Enables multiple-scattering correction in iterative loop |
| `IC.GammaCorrectionFlag` | Enables gamma-background correction |
| `userCtr.runOutlierDetection` | Enables PCA + EllipticEnvelope hardware outlier detection (Phase 6; requires `runRoutine == True`) |
| `userCtr.runPhysicsClustering` | Enables DBSCAN physics-trend clustering on (L1, θ) features (Phase 6; requires `runRoutine == True`) |
| `userCtr.runBayesianBootstrap` | Enables Dirichlet-weighted Bayesian Bootstrap on NCP residuals (Phase 6; requires `runRoutine == True`) |
| `analysisIC.runAnalysis` | Enables post-hoc analysis of stored bootstrap results (independent of runRoutine/runBootstrap) |

#### Phase 6 Activation Rules

The Phase 6 statistical analysis steps (`_runStatisticalAnalysis`) are only executed when
**all** of the following conditions are met:

1. `userCtr.runRoutine == True` — the main fitting pipeline must be active
2. At least one of `runOutlierDetection`, `runPhysicsClustering`, or `runBayesianBootstrap` is `True`
3. `res is not None` — the fitting procedure must have produced results

When `bootIC.runBootstrap == True`, the Phase 6 pipeline is **not** executed because
`runBootstrap` and `runRoutine` are mutually exclusive.  When `analysisIC.runAnalysis == True`,
it runs independently after `runScript()` returns and does not interact with Phase 6.

---

## 3. Module Interaction Map

### 3.1 Core Fitting Pipeline (`analysis_functions.py`)

```
iterativeFitForDataReduction(IC)
│
├── createTableInitialParameters(IC)
├── loadRawAndEmptyWsFromUserPath(IC)      ← Load from Nexus cache (LoadVesuvio already done in ICHelpers)
│   ├── Load → Rebin → Scale → SumSpectra
│   └── (if subEmptyFromRaw) Load empty → Rebin → Scale → Minus
├── cropAndMaskWorkspace(IC, ws)           ← CropWorkspace + MaskDetectors
│
├── for iteration in 0 .. noOfMSIterations:
│   │
│   ├── fitNcpToWorkspace(IC, ws)          ← THE CORE FIT
│   │   ├── extractWS(ws) → dataX, dataY, dataE
│   │   ├── prepareFitArgs(IC, dataX)
│   │   │   ├── loadInstrParsFileIntoArray()
│   │   │   ├── loadResolutionPars()
│   │   │   ├── calculateKinematicsArrays()    ← v0, E0, deltaE, deltaQ
│   │   │   └── convertDataXToYSpacesForEachMass()  ← y-scaling physics
│   │   │
│   │   ├── fitNcpToArray(IC, ...)         ← loops over spectra
│   │   │   └── fitNcpToSingleSpec(...)    ← scipy.optimize.minimize (SLSQP)
│   │   │       └── errorFunction(pars, ...)
│   │   │           └── calculateNcpSpec(IC, pars, ...)  ← synthetic C(t)
│   │   │               ├── prepareArraysFromPars()
│   │   │               ├── caculateResolutionForEachMass()
│   │   │               │   ├── kinematicsAtYCenters()
│   │   │               │   ├── calcGaussianResolution()  ★ BOTTLENECK
│   │   │               │   └── calcLorentzianResolution() ★ BOTTLENECK
│   │   │               ├── pseudoVoigt()
│   │   │               ├── numericalThirdDerivative()     ← FSE term
│   │   │               └── ncpForEachMass = intensity * (J(y)+FSE) * E0 * ... / deltaQ
│   │   │
│   │   ├── calculateNcpArr(IC, bestFitPars, ...)  ← recalculate NCP from best-fit
│   │   └── createNcpWorkspaces(...)       ← store in Mantid ws
│   │
│   ├── extractMeans() → meanWidths, meanIntensityRatios
│   │   └── calculateMeansAndStds()
│   │       └── filterWidthsAndIntensities()  ← outlier rejection (σ-clipping)
│   │
│   ├── [if not last iteration]:
│   │   ├── replaceZerosWithNCP()          ← fill masked bins
│   │   ├── MSCorrectionFlag → createWorkspacesForMSCorrection()
│   │   │   ├── createSlabGeometry()
│   │   │   ├── VesuvioThickness (Mantid)
│   │   │   └── VesuvioCalculateMS (Mantid)
│   │   └── GammaCorrectionFlag → createWorkspacesForGammaCorrection()
│   │       └── VesuvioCalculateGammaBackground (Mantid)
│   │
│   └── Minus(ws, correction) → next iteration ws
│
└── return wsFinal, resultsObject(IC)
```

### 3.2 Y-Space Fitting Pipeline (`fit_in_yspace.py`)

```
fitInYSpaceProcedure(yFitIC, IC, wsTOF)
│
├── extractNCPFromWorkspaces(wsTOF, IC)
├── calculateMantidResolutionFirstMass(IC, yFitIC, wsTOF)  ← VesuvioResolution (Mantid)
├── subtractAllMassesExceptFirst(IC, wsTOF, ncpForEachMass) ← isolate mass₀
├── ySpaceReduction(wsTOFMass0, mass0, yFitIC, ncp)
│   ├── ConvertToYSpace (Mantid)
│   ├── Rebin + Integration + Normalise (or NAN-binning path)
│   └── weightedAvgCols / weightedAvgXBins
├── [symmetrisationFlag] → symmetrizeWs()
│
├── fitProfileMinuit(yFitIC, wsJoYAvg, wsRes)    ★ iMINUIT FIT
│   ├── selectModelAndPars(fitModel)              ← model factory
│   ├── oddPointsRes(resX, resY) → xDelta, resDense
│   ├── convolvedModel = y0 + convolve(model, res)
│   ├── MyLeastSquares (class) OR cost.LeastSquares
│   ├── Minuit(costFun, **defaultPars)
│   │   ├── m.simplex()
│   │   ├── m.migrad() / m.scipy(constraints=...)
│   │   └── m.hesse()
│   ├── jacobi.propagate → confidence band
│   └── [runMinos] → runMinos() / runAndPlotManualMinos()
│
├── fitProfileMantidFit(yFitIC, wsJoYAvg, wsRes) ← Mantid Fit (parallel validation)
│   └── Fit(Function=..., Minimizer='Levenberg-Marquardt'/'Simplex')
│
├── [globalFit] → runGlobalFit(wsJoY, wsRes, IC, yFitIC)
│   ├── extractData → takeOutMaskedSpectra → groupDetectors (k-means)
│   ├── avgWeightDetGroups → symmetrise
│   ├── calcCostFun per group → sum to totCost
│   └── Minuit(totCost, **initPars)
│
└── return ResultsYFitObject(IC, yFitIC, ...)
```

### 3.3 Bootstrap / Jackknife (`bootstrap.py`)

```
runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC)
│
├── runOriginalBeforeBootstrap() → parentResults, parentWSnNCPs
│   └── runMainProcedure() → runIndependentIterativeProcedure / runJointBackAndForwardProcedure
│       └── fitInYSpaceProcedure()
│
├── autoCorrResiduals(parentWSnNCPs)
├── for i in range(nSamples):
│   ├── createSampleWS(...)     ← JACKKNIFE / BOOT_RESIDUALS / BOOT_GAUSS_ERRS
│   ├── formSampleIC(...)       ← Inject sample ws into IC
│   ├── runMainProcedure(...)   ← Full iterative fit on replica
│   └── storeBootIter(...)      ← Accumulate results
│
└── saveBootstrapResults(...)
```

### 3.4 Statistical Post-Processing (`statistical_plugins.py`) — Phase 6

```
_runStatisticalAnalysis(userCtr, res, bckwdIC, fwdIC)
│
├── Gate: any_enabled = runOutlierDetection OR runPhysicsClustering OR runBayesianBootstrap
├── Gate: res is not None
│
├── Extract resultsObject(s) from res:
│   ├── BACKWARD/FORWARD → [(res[1], bckwdIC or fwdIC)]
│   └── JOINT            → [(res[1], bckwdIC), (res[2], fwdIC)]
│
├── For each (resultsObject, ic):
│   ├── spectra    = results.all_fit_workspaces[-1]    ← last-iteration fitted spectra
│   ├── ncp_total  = results.all_tot_ncp[-1]           ← last-iteration total NCP
│   │
│   ├── [runOutlierDetection]:
│   │   └── HardwareOutlierDetector(n_components=5, contamination=0.1)
│   │       ├── StandardScaler → PCA → EllipticEnvelope
│   │       └── labels: -1 = outlier, 0 = inlier
│   │
│   ├── [runPhysicsClustering]:
│   │   ├── loadInstrParsFileIntoArray(ic.InstrParsPath, ic.firstSpec, ic.lastSpec)
│   │   ├── features = [L1 (col 5), theta (col 2)]
│   │   └── PhysicsTrendClusterer(eps=0.5, min_samples=3)
│   │       ├── StandardScaler → DBSCAN
│   │       └── get_cluster_groups(labels) → {cluster_id: [indices]} (noise=-1 excluded)
│   │
│   └── [runBayesianBootstrap]:
│       ├── residuals = spectra - ncp_total
│       └── BayesianBootstrap(n_samples=1000, seed=42)
│           ├── Dirichlet(1,...,1) → weights, shape (n_samples, n_spectra)
│           └── compute_weighted_residuals → weights @ residuals
```

---

## 4. Identified Computational Bottlenecks

The following functions are called **per-spectrum** inside the inner `fitNcpToSingleSpec` →
`scipy.optimize.minimize` loop, which itself iterates many times per spectrum across
`noOfMSIterations+1` full passes:

| Function | Location | Hot Path | Numba Candidate |
|---|---|---|---|
| `calcGaussianResolution()` | `analysis_functions.py:639` | Called every cost-function evaluation | **YES** — pure NumPy arithmetic |
| `calcLorentzianResolution()` | `analysis_functions.py:673` | Same | **YES** — pure NumPy arithmetic |
| `pseudoVoigt()` | `analysis_functions.py:705` | Same | **YES** — Gaussian + Lorentzian |
| `numericalThirdDerivative()` | `analysis_functions.py:730` | Same | **YES** — stencil operation |
| `calculateNcpSpec()` | `analysis_functions.py:572` | Orchestrator of above | **YES** — can wrap entire function |
| `errorFunction()` | `analysis_functions.py:555` | Objective for scipy | Partial — needs `calculateNcpSpec` accelerated |
| `kinematicsAtYCenters()` | `analysis_functions.py:616` | Called per resolution calc | **YES** — index operations |
| `loadConstants()` | `analysis_functions.py:694` | Very frequent | Inline / cache with `@njit` |

**Non-candidates** (Mantid API calls, I/O, plotting):
`loadRawAndEmptyWsFromUserPath`, `cropAndMaskWorkspace`, `createMulScatWorkspaces`,
`VesuvioCalculateMS`, `VesuvioCalculateGammaBackground`, all plot functions.

---

## 5. Current Fitting Architecture

### 5.1 NCP Fitting (TOF domain — `analysis_functions.py`)

- **Optimizer:** `scipy.optimize.minimize` with method `'SLSQP'`
- **Cost function:** `errorFunction()` — a plain function returning scalar χ²
- **Per-spectrum:** Each of the ~130 spectra fitted independently in a loop
- **No iMinuit involvement** in this stage

### 5.2 Y-Space Fitting (`fit_in_yspace.py`)

- **Optimizer 1 (primary):** iMinuit via `Minuit(costFun, **defaultPars)`
  - Uses `cost.LeastSquares` (with errors) or `MyLeastSquares` class (without errors)
  - Constraint handling via `m.scipy(constraints=NonlinearConstraint(...))`
  - Error analysis: `m.hesse()`, optional `m.minos()`
- **Optimizer 2 (parallel validation):** Mantid `Fit` algorithm with `Levenberg-Marquardt` and `Simplex`
- **Global fit:** Summed `cost.LeastSquares` per detector group, single `Minuit` call

### 5.3 Existing `MyLeastSquares` Class (line 669)

Already implements the iMinuit class-based cost-function pattern:
```python
class MyLeastSquares:
    errordef = Minuit.LEAST_SQUARES
    def __init__(self, x, y, model): ...
    def __call__(self, *par): ...
    @property
    def ndata(self): ...
```
This is used **only** when `dataE` is all zeros (bootstrap without errors).

---

## 6. Refactoring Strategy

### 6.1 Phase 1 — Numba Acceleration of Resolution Functions

**Goal:** Accelerate the inner-loop bottleneck without changing any public API.

**Targets (in `analysis_functions.py`):**

1. **`calcGaussianResolution()`** → `@njit` (pure arithmetic on NumPy arrays)
2. **`calcLorentzianResolution()`** → `@njit`
3. **`pseudoVoigt()`** → `@njit` (needs `gaussian()` and `lorentizian()` [sic — Lorentzian] also `@njit`)
4. **`numericalThirdDerivative()`** → `@njit` (stencil / slicing pattern)
5. **`calculateNcpSpec()`** → `@njit` wrapper calling above functions
6. **`loadConstants()`** → inline as module-level constants or `@njit` returning tuple

**Approach:**
- Create a new module `vesuvio_analysis/core_functions/numba_routines.py`
- Port the pure-NumPy functions there with `@njit(cache=True)` decorators
- Import and use in `analysis_functions.py` as drop-in replacements
- Keep original implementations available behind a `USE_NUMBA` flag for regression testing
- The flag enables easy A/B comparison and ensures no physics is altered

**Constraints:**
- `@njit` functions cannot call Mantid API — only pure NumPy
- `pseudoVoigt` uses `np.trapz` — need to replace with manual trapezoidal rule in Numba
- `IC.normVoigt` boolean must be passed as a plain argument, not as an object attribute

### 6.2 Phase 2 — iMinuit Class-Based Cost Functions for NCP Fitting

**Goal:** Introduce iMinuit as a parallel optimizer for the NCP (TOF-domain) fit alongside
the existing `scipy.optimize.minimize(SLSQP)`.

**Design:**

```python
class NCPCostFunction:
    """
    iMinuit-compatible cost function for NCP fitting in TOF space.
    Implements __call__ with explicit parameter signature.
    """
    errordef = Minuit.LEAST_SQUARES

    def __init__(self, dataY, dataE, ySpaces, resPars, instrPars, kinArrays, ic):
        self.model_args = (ySpaces, resPars, instrPars, kinArrays, ic)
        self.dataY = dataY
        self.dataE = dataE
        # Build func_code from ic.initPars structure
        self.func_code = make_func_code(self._build_signature(ic))

    def _build_signature(self, ic):
        sig = []
        for i in range(ic.noOfMasses):
            sig.extend([f"I{i}", f"W{i}", f"C{i}"])
        return sig

    def __call__(self, *pars):
        # Reuses calculateNcpSpec internally
        ...
        return chi2_scalar

    @property
    def ndata(self):
        return np.sum(self.dataY != 0)
```

**Integration plan:**
- Add to `analysis_functions.py` or a new `iminuit_costs.py` module
- In `fitNcpToSingleSpec()`, run **both** scipy and iMinuit, compare results
- Store iMinuit results in a parallel column of the results table (e.g., `"Minuit Chi2"`)
- **Do NOT remove** `scipy.optimize.minimize` — it remains the primary optimizer for regression safety

### 6.3 Phase 3 — Unify Cost-Function Interface ✅

**Goal:** Unify the cost-function interface across the codebase.

- `MyLeastSquares` updated: replaced deprecated `func_code` with modern `_parameters` dict
- `NCPCostFunction` extended: added `ndata` property returning count of non-zero data points
- `GlobalNCPCostFunction` created: inherits from `cost.Cost`, supports `CostSum` for global fits
- `calcCostFun` refactored: returns `GlobalNCPCostFunction` instead of generic `cost.LeastSquares`
- All custom cost classes expose: `errordef`, `__call__`, `_parameters`, `ndata`
- Verification: `tests/test_interface_unification.py` (zero Mantid dependency, 23 tests)

**Documentation Verification** (scikit-hep.org/iminuit, v2.x):

The following findings from the official iminuit documentation at scikit-hep.org
confirm the correctness of the Phase 3 implementation:

1. **`_parameters` dict** — `describe()` docs (Method 1 in Notes): "Users are
   encouraged to use this mechanism to provide signatures for objects that
   otherwise would not have a detectable signature." Maps parameter names to
   `(lower, upper)` limit tuples or `None` for unbounded.  `func_code` is
   confirmed deprecated: "still supported for legacy code, but should not be
   used anymore in new code."

2. **`CostSum` shared-parameter merging** — `simultaneous_fits.html` tutorial:
   "The σ parameter is shared between the data sets."  Reference docs confirm
   "The parameters of CostSum are the union of all parameters of its
   constituents." Parameters with the same name are automatically shared.

3. **`ndata` / GoF metrics** — `Minuit.ndof` docs: "To support this feature,
   the cost function has to report the number of data points with a property
   called `ndata`."  `generic_least_squares.html` tutorial: "iminuit
   automatically reports the reduced chi2 value χ²/ndof if the cost function has
   `errordef` equal to `Minuit.LEAST_SQUARES` and reports the number of data
   points."  `CostSum.ndata` aggregates constituent `ndata` values.

4. **`errordef` attribute** — Reference docs: "If FCN has an attribute
   `errordef`, its value is used automatically."
   `Minuit.LEAST_SQUARES = 1.0`, `Minuit.LIKELIHOOD = 0.5`.

5. **Annotation propagation** — `generic_least_squares.html` best practice uses
   `describe(model, annotations=True)` to propagate type-annotation limits
   (e.g. `Annotated[float, 0:]`) through `_parameters`.  Applied to
   `MyLeastSquares` and `fitProfileMinuit`.

### 6.4 Phase 4 — Mantid Workspace Lifecycle Preservation

**Critical invariants that must not change:**

1. `LoadVesuvio` → `SaveNexus` happens once in `ICHelpers.completeICFromInputs` and is cached
2. `AnalysisDataService.clear()` is called at procedure boundaries — respect this
3. Workspace naming convention: `{scriptName}_{DIRECTION}_{iteration}` must be preserved
4. `SumSpectra`, `MaskDetectors`, `CropWorkspace` produce named workspaces consumed downstream
5. MS/GC corrections use `Minus(LHSWorkspace, RHSWorkspace)` on named workspaces
6. `fitInYSpaceProcedure` expects `mtd[wsName]` — the final corrected workspace must exist

**No Numba or iMinuit code may call Mantid algorithms.** All acceleration is confined to
the pure-NumPy computation layer between `extractWS()` and `passDataIntoWS()`.

### 6.5 Phase 5 — Numba Regression Testing ✅

**Goal:** Validate that Numba-accelerated resolution functions produce results identical
to the original NumPy implementations within floating-point tolerance.

- `tests/test_numba_regression.py` — Regression tests for `pseudoVoigt`, `calculateNcpSpec`
- Benchmarks compare Numba vs NumPy execution time
- Pre-existing failures (7×) due to `np.trapz` removal in NumPy 2.x are unrelated

### 6.6 Phase 6 — Statistical Post-Processing Pipeline ✅

**Goal:** Add a multi-stage statistical analysis pipeline that runs after the main
NCP fitting procedure, providing hardware outlier detection, physics-trend clustering,
and uncertainty quantification via Bayesian Bootstrap.

**Module:** `vesuvio_analysis/core_functions/statistical_plugins.py`

**Classes:**

1. **`HardwareOutlierDetector`** — Identifies broken detectors using PCA dimensionality
   reduction followed by robust covariance scoring (`EllipticEnvelope`).
   - Input: fitted spectra array, shape `(n_spectra, n_bins)`
   - Pipeline: `StandardScaler` → `PCA(n_components)` → `EllipticEnvelope(contamination)`
   - Output: labels array (`-1` = outlier, `0` = inlier)

2. **`PhysicsTrendClusterer`** — Groups detectors by physical features (flight-path L,
   scattering angle θ) using density-based clustering (`DBSCAN`).
   - Input: feature matrix `[L1, theta]`, shape `(n_spectra, 2)`
   - Pipeline: `StandardScaler` → `DBSCAN(eps, min_samples)`
   - Output: cluster labels; noise points (`-1`) explicitly excluded from `get_cluster_groups()`

3. **`BayesianBootstrap`** — Rubin-style Weighted Bayesian Bootstrap using symmetric
   Dirichlet(1, ..., 1) weights for fast residual resampling without re-fitting.
   - Input: residuals = spectra - ncp_total, shape `(n_spectra, n_bins)`
   - Pipeline: `Dirichlet(alpha=1)` → weight matrix → `weights @ residuals`
   - Output: weighted residual profiles, shape `(n_samples, n_bins)`

**Integration in `run_script.py`:**

- `_runStatisticalAnalysis(userCtr, res, bckwdIC, fwdIC)` is called post-fit within
  the `runRoutine` branch, after `fitInYSpaceProcedure` completes.
- Extracts `resultsObject` from the `res` tuple (handles both 2-tuple and 3-tuple formats).
- Each step is independently gated by its own boolean flag on `UserScriptControls`.
- Loads instrument parameters from IC objects (`ic.InstrParsPath`, `ic.firstSpec`, `ic.lastSpec`)
  for physics-trend clustering via `loadInstrParsFileIntoArray`.

**User-facing flags in `BaH2_500C.py` (`UserScriptControls`):**

| Flag | Default | Effect |
|---|---|---|
| `runOutlierDetection` | `False` | Run PCA + EllipticEnvelope on fitted spectra |
| `runPhysicsClustering` | `False` | Run DBSCAN on (L1, θ) instrument features |
| `runBayesianBootstrap` | `False` | Compute Dirichlet-weighted bootstrap residuals |

**Tests:** `tests/test_statistical_workflow.py` — 12 tests covering all three classes
against synthetic data (no Mantid dependency).

**Dependency:** `scikit-learn` added to `pyproject.toml`.

---

## 7. File-Level Dependency Graph

```
BaH2_500C.py
    │
    ├── run_script.py
    │   ├── ICHelpers.py           ← completeICFromInputs, buildFinalWSName, completeBootIC
    │   │   └── mantid.simpleapi   ← LoadVesuvio, SaveNexus
    │   │
    │   ├── procedures.py          ← runIndependentIterativeProcedure, runJoint, isHPresent
    │   │   └── analysis_functions.py  ← iterativeFitForDataReduction (THE CORE)
    │   │       ├── scipy.optimize     ← minimize(SLSQP) for NCP fitting
    │   │       ├── mantid.simpleapi   ← VesuvioCalculateMS, VesuvioCalculateGammaBackground
    │   │       └── fit_in_yspace.py   ← passDataIntoWS, replaceZerosWithNCP (utility imports)
    │   │
    │   ├── fit_in_yspace.py       ← fitInYSpaceProcedure
    │   │   ├── iminuit             ← Minuit, cost.LeastSquares, make_func_code
    │   │   ├── scipy.optimize      ← NonlinearConstraint (for constrained Minuit)
    │   │   ├── scipy.signal        ← convolve
    │   │   ├── jacobi              ← propagate (error propagation)
    │   │   └── mantid.simpleapi    ← ConvertToYSpace, VesuvioResolution, Fit
    │   │
    │   ├── statistical_plugins.py ← Phase 6: _runStatisticalAnalysis (post-fit)
    │   │   ├── sklearn.decomposition  ← PCA
    │   │   ├── sklearn.covariance     ← EllipticEnvelope
    │   │   ├── sklearn.cluster        ← DBSCAN
    │   │   ├── sklearn.preprocessing  ← StandardScaler
    │   │   └── analysis_functions.py  ← loadInstrParsFileIntoArray (for clustering features)
    │   │
    │   └── bootstrap.py           ← runBootstrap
    │       ├── procedures.py       ← reuses same fitting pipeline
    │       ├── fit_in_yspace.py    ← reuses y-space fitting
    │       └── mantid.simpleapi    ← CloneWorkspace, SaveNexus, Load
    │
    └── bootstrap_analysis.py      ← runAnalysisOfStoredBootstrap (post-hoc analysis)
        └── analysis_functions.py   ← calculateMeansAndStds, filterWidthsAndIntensities
```

---

## 8. Risk Assessment

| Risk | Mitigation |
|---|---|
| Numba JIT compilation changes floating-point results | Run existing tests with both paths; compare at `rtol=1e-6` |
| iMinuit finds different minima than scipy SLSQP | Keep both optimizers; compare chi² and parameters |
| Mantid workspace lifecycle broken | All Numba code operates only on extracted NumPy arrays |
| Bootstrap results change | Bootstrap calls the same `iterativeFitForDataReduction`; if inner functions are accelerated, results must match within tolerance |
| `@njit` incompatibility with object attributes | Pass all IC parameters as plain arrays/scalars, not class instances |
| Phase 6 outlier detection flags good detectors | `contamination` parameter is configurable; validate against known-good runs |
| Phase 6 DBSCAN finds wrong number of clusters | `eps` and `min_samples` tuned per instrument geometry; standardisation handles scale differences |
| Phase 6 runs when no fit results available | Gated by `res is not None` check; no-op when fitting is skipped |

---

## 9. Recommended Implementation Order

1. **Create `numba_routines.py`** with `@njit` versions of resolution functions ✅
2. **Add `NCPCostFunction` class** for iMinuit-based NCP fitting ✅
3. **Wire into `fitNcpToSingleSpec()`** with dual-optimizer logic
4. **Add regression tests** comparing scipy vs iMinuit results ✅
5. **Benchmark** Numba-accelerated vs original NumPy on a representative dataset ✅
6. **Document** performance results and any numerical differences
7. **Add `statistical_plugins.py`** with outlier detection, clustering, and bootstrap ✅
8. **Wire Phase 6** into `runScript()` as post-fit pipeline ✅

All work to be performed on the `dev` branch.
