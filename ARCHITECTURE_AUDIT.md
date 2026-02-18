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
│   ├── UserScriptControls          — runRoutine, procedure, fitInYSpace
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
| `UserScriptControls` | `runRoutine`, `procedure`, `fitInYSpace` | `run_script.runScript` (branching logic) |
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
    └── for each wsName in [BACKWARD, FORWARD, JOINT as requested]:
        └── fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
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
3. **`pseudoVoigt()`** → `@njit` (needs `gaussian()` and `lorentizian()` also `@njit`)
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

### 6.3 Phase 3 — Extend `MyLeastSquares` Pattern

**Goal:** Unify the cost-function interface across the codebase.

- The existing `MyLeastSquares` class in `fit_in_yspace.py:669` is already correct
- Extend the same pattern to:
  - `NCPCostFunction` (TOF domain, Phase 2)
  - Global-fit cost functions (already using `cost.LeastSquares`)
- Ensure all custom cost classes expose: `errordef`, `__call__`, `func_code`, `ndata`

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

---

## 9. Recommended Implementation Order

1. **Create `numba_routines.py`** with `@njit` versions of resolution functions
2. **Add `NCPCostFunction` class** for iMinuit-based NCP fitting
3. **Wire into `fitNcpToSingleSpec()`** with dual-optimizer logic
4. **Add regression tests** comparing scipy vs iMinuit results
5. **Benchmark** Numba-accelerated vs original NumPy on a representative dataset
6. **Document** performance results and any numerical differences

All work to be performed on the `dev` branch.
