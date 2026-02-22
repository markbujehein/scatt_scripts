# Full-Stack Statistics Toolchain — Integration Audit

> **Date:** 2026-02-22
> **Auditor:** Principal RSE (Automated Audit Agent)
> **Branch:** `dev` (via `copilot/audit-legacy-compliance-verification`)
> **Reference Document:** `ARCHITECTURE_AUDIT.md` (Sections 3.4, 6.6, 11)
> **Scope:** All commits to `dev` within the last 3 days; cross-referenced against
> the canonical pipeline architecture.

---

## Executive Summary

The "Full-Stack Stats Toolchain" (Phase 6) has been integrated into the
`vesuvio_analysis` pipeline with **generally strong architectural discipline**.
The three core statistical classes (`HardwareOutlierDetector`,
`PhysicsTrendClusterer`, `BayesianBootstrap`) are correctly isolated in
`statistical_plugins.py`, and the orchestration function
`_runStatisticalAnalysis()` in `run_script.py` is properly flag-gated.

**Overall Compliance: PASS (with advisory items)**

| Category | Status | Details |
|---|---|---|
| Phase I: Sequential Pipeline Integrity | ✅ PASS | 1 → 2 → 3 dependency chain preserved |
| Phase II-A: Main Process Flag Gating | ✅ PASS | All 3 tools gated by independent booleans |
| Phase II-B: Bayesian Bootstrap Separation | ✅ PASS | Cleanly separated from legacy bootstrap |
| Phase II-C: Plotting Integration | ⚠️ ADVISORY | Diagnostic plots defined but not invoked from pipeline |
| Legacy Bypass Verification | ✅ PASS | All flags default to `False`; legacy path untouched |
| `.npz` Contract Integrity | ✅ PASS | No structural changes to serialization format |
| Test Coverage | ✅ PASS | 25/25 tests pass (no Mantid dependency) |

**Critical Violations Found: 0**
**Advisory Items: 3** (documented below)

---

## Phase I: Logical Flow Verification

### Procedure 1 — Main Process (Reduction & Fitting)

**File:** `vesuvio_analysis/core_functions/run_script.py` (lines 30–270)

**Verification:**

1. **Entry dispatch:** `runScript()` branches correctly on `bootIC.runBootstrap`
   vs `userCtr.runRoutine` (lines 189, 211). Mutual exclusivity is enforced
   via `assert` at line 118.

2. **Procedure execution:** `runProcedure()` (line 151) delegates to
   `runIndependentIterativeProcedure()` or `runJointBackAndForwardProcedure()`
   depending on `userCtr.procedure`. This matches the architecture in
   `ARCHITECTURE_AUDIT.md` Section 2.

3. **Y-space fitting:** `fitInYSpaceProcedure()` runs after the iterative NCP
   fit (line 251). Both iMinuit and Mantid Fit are available as documented.

4. **Phase 6 integration:** `_runStatisticalAnalysis()` is called at line 260,
   **after** both NCP fitting and y-space fitting. This is the correct position
   — it is a post-fit analysis step that does not feed back into the fitting
   pipeline.

5. **`.npz` output:** The `StreamManager.save()` method in
   `stream_manager.py:272` uses `np.savez_compressed()`. The legacy bootstrap
   procedure saves via `np.savez()` in `bootstrap.py`. Both serialization
   paths are **unchanged** by the Phase 6 integration.

**Result: ✅ PASS — No deviations from the documented architecture.**

### Procedure 2 — Bootstrapping

**File:** `vesuvio_analysis/core_functions/bootstrap.py` (lines 14–25)

**Verification:**

1. **Entry gate:** `runBootstrap()` is called only when `bootIC.runBootstrap`
   is `True` (line 189 of `run_script.py`). It is mutually exclusive with
   `runRoutine` (assert at line 118).

2. **Bootstrap types:** Three types are supported: `JACKKNIFE`,
   `BOOT_GAUSS_ERRS`, `BOOT_RESIDUALS` (validated at line 29–31). No new
   types have been introduced.

3. **`.npz` dependency:** The bootstrap procedure generates its own `.npz`
   output via `saveBootstrapResults()`. It does **not** depend on a
   pre-existing `.npz` from Procedure 1 — instead, it calls
   `runOriginalBeforeBootstrap()` to produce the parent results within the
   same bootstrap call. This is consistent with the architecture.

4. **Error handling (Advisory):** `readBootData()` in `bootstrap_analysis.py`
   (line 92) uses bare `np.load(dataPath)` without a `try/except` block.
   If the `.npz` file is malformed, a raw NumPy exception propagates without
   a descriptive `logging.error`. The check for file existence (line 27 of
   `bootstrap_analysis.py`) uses `print()` rather than `logging.error()`.

**Result: ✅ PASS — Pipeline integrity preserved.**

**Advisory A1:** `readBootData()` and `runAnalysisOfStoredBootstrap()` should
use `logging.error()` instead of `print()` for missing/malformed `.npz` files.
See Remediation Plan item R1.

### Procedure 3 — Analysis & Validation

**File:** `vesuvio_analysis/core_functions/bootstrap_analysis.py` (lines 18–74)

**Verification:**

1. **`.npz` loading:** `runAnalysisOfStoredBootstrap()` checks
   `IC.bootSavePath.is_file()` before attempting to load (line 27). If the
   file is missing, it prints a message and `continue`s to the next IC.

2. **Y-space fit data:** `IC.bootYFitSavePath.is_file()` is checked
   separately (line 62). Missing files are reported and skipped.

3. **Physical validation:** `checkBootSamplesVSParent()` (line 36) compares
   bootstrap samples against parent results. The filtering logic in
   `filteredBootMeans()` (line 40) applies sigma-clipping when
   `analysisIC.filterAvg` is `True`.

4. **No Phase 6 interference:** The bootstrap analysis procedure does not
   import or reference any Phase 6 classes. It operates independently on
   the stored `.npz` data.

**Result: ✅ PASS — Validation pipeline correctly maps bootstrap distributions
back to original physical constraints.**

---

## Phase II: Statistical Upgrade & Legacy Compliance Audit

### A. Main Process (Procedure 1) — Flag-Gating Verification

**File:** `vesuvio_analysis/core_functions/run_script.py`, function
`_runStatisticalAnalysis()` (lines 441–549)

**Flag Verification Matrix:**

| Flag | Location | Default | Gate Check | Status |
|---|---|---|---|---|
| `runOutlierDetection` | `BaH2_500C.py:309` | `False` | `getattr(userCtr, 'runOutlierDetection', False)` at line 505 | ✅ COMPLIANT |
| `runPhysicsClustering` | `BaH2_500C.py:310` | `False` | `getattr(userCtr, 'runPhysicsClustering', False)` at line 517 | ✅ COMPLIANT |
| `runBayesianBootstrap` | `BaH2_500C.py:311` | `False` | `getattr(userCtr, 'runBayesianBootstrap', False)` at line 535 | ✅ COMPLIANT |

**Execution Path When All Flags Are `False`:**

```
_runStatisticalAnalysis(userCtr, res, bckwdIC, fwdIC)
│
├── any_enabled = False or False or False → False
├── if not any_enabled → return          ← IMMEDIATE EXIT
└── (no statistical code executes)
```

**Verification:** When all three Phase 6 flags are `False` (the default),
`_runStatisticalAnalysis()` returns immediately at line 479. No statistical
module is instantiated, no scikit-learn import occurs at runtime (the import
at line 465 happens but is never reached due to the early return pattern —
the import is **inside** the function body, not at module level, so it is
lazy-evaluated only when the function is actually called with
`any_enabled=True`).

**Critical Violation Check:** Are any Phase 6 functions hardcoded or run
unconditionally?

- `HardwareOutlierDetector`: Only instantiated inside `if getattr(userCtr, 'runOutlierDetection', False)` block. ✅
- `PhysicsTrendClusterer`: Only instantiated inside `if getattr(userCtr, 'runPhysicsClustering', False)` block. ✅
- `BayesianBootstrap`: Only instantiated inside `if getattr(userCtr, 'runBayesianBootstrap', False)` block. ✅

**Result: ✅ PASS — No hardcoded or unconditional Phase 6 execution found.
Legacy path is fully preserved.**

### B. Bootstrapping (Procedure 2 & 3) — Bayesian Bootstrap Separation

**Verification:**

1. **Bayesian Bootstrap location:** The `BayesianBootstrap` class resides in
   `statistical_plugins.py` (lines 172–237) and is **only** invoked from
   `_runStatisticalAnalysis()` in `run_script.py`. It is **not** integrated
   into the `bootstrap.py` resampling procedure.

2. **Architectural distinction:** The existing bootstrap procedure
   (`bootstrap.py`) implements frequentist resampling methods (JACKKNIFE,
   BOOT_RESIDUALS, BOOT_GAUSS_ERRS) that re-run the full iterative fit on
   each replica. The Phase 6 `BayesianBootstrap` is fundamentally different —
   it operates on NCP residuals from a single fit and uses Dirichlet-weighted
   resampling for fast uncertainty quantification without re-fitting.

3. **Configuration flag:** `userCtr.runBayesianBootstrap` (default `False`)
   controls the Phase 6 Bayesian Bootstrap. The existing `bootIC.bootstrapType`
   controls the Procedure 2 resampling. These are **independent** flags on
   **separate** configuration classes, preventing accidental cross-activation.

4. **No `bootstrap_method` flag exists:** The problem statement references a
   hypothetical `bootstrap_method='bayesian'` flag — this does not exist and
   is not needed because the two bootstrap mechanisms are architecturally
   separate (Phase 6 post-fit vs Procedure 2 re-fitting).

5. **Analysis phase compatibility:** `runAnalysisOfStoredBootstrap()`
   (Procedure 3) only processes `.npz` files from the frequentist bootstrap
   (Procedure 2). It does not attempt to interpret Bayesian Bootstrap weights.
   The Phase 6 Bayesian Bootstrap currently outputs summary statistics to
   `stdout` only — it does not write to `.npz` files that would interfere
   with Procedure 3.

**Result: ✅ PASS — Bayesian Bootstrap is cleanly separated from the
frequentist resampling pipeline.**

### C. Visualization and Plotting Verification

**File:** `vesuvio_analysis/core_functions/statistical_plugins.py`
(lines 244–523)

**Defined Plotting Functions:**

| Function | Purpose | Style Compliance | Save Path Support |
|---|---|---|---|
| `plot_outlier_scatter` (line 245) | PCA space scatter with outlier highlighting | Uses `set_thesis_style()` + `figure_factory()` ✅ | `save_path` parameter ✅ |
| `plot_cluster_ltheta` (line 296) | L vs θ scatter with DBSCAN colouring | Uses `set_thesis_style()` + `figure_factory()` ✅ | `save_path` parameter ✅ |
| `plot_bayesian_corner` (line 351) | Corner plot of parameter correlations | Uses `set_thesis_style()` ✅ but uses `plt.subplots()` directly ⚠️ | `save_path` parameter ✅ |
| `plot_posterior_kde` (line 414) | KDE posteriors vs frequentist estimates | Uses `set_thesis_style()` ✅ but uses `plt.subplots()` directly ⚠️ | `save_path` parameter ✅ |
| `plot_optimizer_residuals` (line 469) | iMinuit vs Scipy residuals comparison | Uses `set_thesis_style()` + `figure_factory(nrows=2)` ✅ | `save_path` parameter ✅ |

**Bordallo Principle Compliance:**

The "Bordallo Principle" (Experimental Data = Points/Errors;
Model = Smooth Lines) is enforced through `EXPERIMENTAL_STYLE` and
`THEORETICAL_STYLE` in `plot_style.py:80–96`. Checking usage in
`statistical_plugins.py`:

- `EXPERIMENTAL_STYLE` is **not imported** in `statistical_plugins.py`.
- `THEORETICAL_STYLE` is **not imported** in `statistical_plugins.py`.
- The plotting functions use custom scatter/line styles (appropriate for their
  purpose — they are diagnostic/statistical plots, not data-vs-model
  comparisons). This is acceptable because outlier scatter plots and cluster
  visualizations do not have an "experimental vs theoretical" distinction.

**Advisory A2: Diagnostic plots are defined but never called from the pipeline.**

The five plotting functions in `statistical_plugins.py` are **tested**
(11 tests in `test_statistical_workflow.py:278–431`) but are **never invoked**
from `_runStatisticalAnalysis()` in `run_script.py`. The Phase 6 integration
only prints summary statistics to `stdout`. To produce visual evidence of
outlier exclusion masks, cluster centroids, and bootstrap distributions,
the plots must be explicitly called with `save_path` set to
`ic.figSavePath / "stats_***.pdf"`.

**Advisory A3: Two plot functions use `plt.subplots()` instead of
`figure_factory()`.**

`plot_bayesian_corner` (line 371) and `plot_posterior_kde` (line 439) use
`plt.subplots()` directly instead of `figure_factory()`. While they correctly
call `set_thesis_style()`, they do not benefit from the standardised thesis
figure sizes enforced by `figure_factory()`. This is a minor consistency
issue, not a functional defect — the corner plot requires a dynamic grid
size that `figure_factory()` may not support directly.

**Figure Output Directory:**

No Phase 6 diagnostic plots are currently saved to
`experiments/{experiment_name}/figures/` because the plotting functions are
not called. When integrated, they should use `ic.figSavePath` (set by
`ICHelpers.completeICFromInputs` at line 107–109) to ensure consistency with
existing correction and y-space plots.

**Overwrite Risk:** Since the Phase 6 plots use distinct filenames
(e.g., `outlier_scatter.pdf`, `cluster_ltheta.pdf`), they will **not**
overwrite existing reduction or correction plots.

**Result: ⚠️ ADVISORY — Plotting infrastructure is correct but inactive.
See Remediation Plan item R2.**

---

## Detailed File-Level Audit

### `vesuvio_analysis/core_functions/statistical_plugins.py`

| Line(s) | Component | Status | Notes |
|---|---|---|---|
| 1–23 | Module docstring | ✅ | Accurate, references Rubin (1981) |
| 25–39 | Imports | ✅ | All dependencies are in `pyproject.toml` |
| 46–93 | `HardwareOutlierDetector` | ✅ | EllipticEnvelope → DBSCAN label convention correctly mapped |
| 100–165 | `PhysicsTrendClusterer` | ✅ | Noise (-1) excluded from groups ✅ |
| 172–237 | `BayesianBootstrap` | ✅ | Dirichlet(1,...,1), reproducible via seed |
| 245–293 | `plot_outlier_scatter` | ✅ | Uses `figure_factory()`, `COLORBLIND_PALETTE` |
| 296–348 | `plot_cluster_ltheta` | ✅ | Handles noise points gracefully |
| 351–411 | `plot_bayesian_corner` | ⚠️ | Uses `plt.subplots()` not `figure_factory()` |
| 414–466 | `plot_posterior_kde` | ⚠️ | Uses `plt.subplots()` not `figure_factory()` |
| 469–523 | `plot_optimizer_residuals` | ✅ | Uses `figure_factory(nrows=2)` |

### `vesuvio_analysis/core_functions/run_script.py`

| Line(s) | Component | Status | Notes |
|---|---|---|---|
| 441–549 | `_runStatisticalAnalysis()` | ✅ | Correctly gated, correct result extraction |
| 474–479 | `any_enabled` gate | ✅ | Short-circuits when all flags `False` |
| 486–491 | Result tuple handling | ✅ | Handles both 2-tuple and 3-tuple |
| 500–503 | Histogram alignment | ✅ | Trims spectra when hist/point mismatch |
| 465–472 | Lazy imports | ✅ | scikit-learn only imported when needed |

### `BaH2_500C.py`

| Line(s) | Component | Status | Notes |
|---|---|---|---|
| 272–314 | `UserScriptControls` | ✅ | All Phase 6 flags default to `False` |
| 306 | `runningTest` | ✅ | Fast-track flag for CI |
| 309 | `runOutlierDetection` | ✅ | Default `False` |
| 310 | `runPhysicsClustering` | ✅ | Default `False` |
| 311 | `runBayesianBootstrap` | ✅ | Default `False` |

### `vesuvio_analysis/core_functions/iminuit_costs.py`

| Line(s) | Component | Status | Notes |
|---|---|---|---|
| 41–126 | `NCPCostFunction` | ✅ | `_parameters` dict, `errordef`, `ndata` |
| 132–201 | `GlobalNCPCostFunction` | ✅ | Inherits `cost.Cost`, supports `CostSum` |
| 207–238 | `_build_parameters_dict` | ✅ | NaN bounds → None correctly |

### `tests/test_statistical_workflow.py`

| Line(s) | Test Class | Tests | Status |
|---|---|---|---|
| 32–105 | `TestHardwareOutlierDetector` | 3 | ✅ All pass |
| 112–187 | `TestPhysicsTrendClusterer` | 4 | ✅ All pass |
| 194–271 | `TestBayesianBootstrap` | 5 | ✅ All pass |
| 279–431 | `TestDiagnosticVisualisations` | 11 | ✅ All pass |
| 450–489 | `TestPhase6HistogramAlignment` | 2 | ✅ All pass |

**Total: 25/25 tests pass** (verified in this environment without Mantid).

---

## Advisory Items Summary

### A1: Bootstrap Error Handling Uses `print()` Instead of `logging.error()`

**File:** `vesuvio_analysis/core_functions/bootstrap_analysis.py`
**Lines:** 28–30, 63–65, 92 (bare `np.load` without try/except)

**Current Behavior:**
```python
if not(IC.bootSavePath.is_file()):
    print("Bootstrap data files not found, unable to run analysis!")
    print(f"{IC.bootSavePath.name}")
    continue
```

**Recommended:**
```python
if not IC.bootSavePath.is_file():
    logging.error(
        "Bootstrap data files not found at '%s'; unable to run analysis.",
        IC.bootSavePath,
    )
    continue
```

**Impact:** Low. Functional behavior is correct; this is a logging
best-practice issue.

### A2: Diagnostic Plots Not Invoked from Pipeline

**File:** `vesuvio_analysis/core_functions/run_script.py`, function
`_runStatisticalAnalysis()`

**Current Behavior:** The function prints summary statistics but does not
call any of the five plotting functions defined in `statistical_plugins.py`.
The plots are only exercised by unit tests.

**Recommended:** After each statistical step, invoke the corresponding
plotting function with `save_path=ic.figSavePath / "stats_<name>.pdf"`.
Example for outlier detection:

```python
if getattr(userCtr, "runOutlierDetection", False):
    detector = HardwareOutlierDetector(n_components=5, contamination=0.1)
    labels = detector.fit_predict(spectra)
    # ... existing print logic ...
    # NEW: Generate diagnostic plot
    from vesuvio_analysis.core_functions.statistical_plugins import plot_outlier_scatter
    pca_coords = detector._pca.transform(detector._scaler.transform(spectra))
    fig_dir = getattr(ic, "figSavePath", None)
    if fig_dir is not None:
        plot_outlier_scatter(pca_coords, labels,
                            save_path=fig_dir / "stats_outlier_scatter.pdf")
```

**Impact:** Medium. Without these plots, the statistical analysis produces
no visual evidence for the thesis manuscript or experimental review.

### A3: Minor Style Inconsistency in Corner/KDE Plots

**File:** `vesuvio_analysis/core_functions/statistical_plugins.py`
**Lines:** 371 (`plot_bayesian_corner`), 439 (`plot_posterior_kde`)

**Issue:** These functions use `plt.subplots()` directly instead of
`figure_factory()`. They correctly call `set_thesis_style()`, so the
overall appearance is consistent. The deviation is justified because
`plot_bayesian_corner` creates a dynamic `n × n` grid that `figure_factory()`
does not directly support.

**Impact:** Low. Acceptable deviation with clear technical justification.

---

## Remediation Plan

### R1: Upgrade Bootstrap Error Handling to `logging`

**Priority:** Low
**File:** `vesuvio_analysis/core_functions/bootstrap_analysis.py`
**Action:** Replace `print()` calls for missing/malformed `.npz` files with
`logging.error()`. Add a `try/except` around `np.load()` in `readBootData()`
to catch `FileNotFoundError`, `ValueError`, and `zipfile.BadZipFile`, with
descriptive error messages.
**Lines to modify:** 28–30, 63–65, 91–92
**Effort:** ~30 minutes

### R2: Wire Diagnostic Plots into `_runStatisticalAnalysis()`

**Priority:** Medium
**File:** `vesuvio_analysis/core_functions/run_script.py`
**Action:** After each Phase 6 step, invoke the corresponding plotting
function from `statistical_plugins.py` with `save_path` set to
`ic.figSavePath / "stats_<descriptor>.pdf"`. Guard the plot call with
a `try/except` to prevent plotting failures from aborting the analysis.
**Lines to modify:** 505–515 (outlier), 517–533 (clustering), 535–549 (bootstrap)
**Effort:** ~1 hour

### R3: Expose PCA Coordinates from `HardwareOutlierDetector`

**Priority:** Low (prerequisite for R2)
**File:** `vesuvio_analysis/core_functions/statistical_plugins.py`
**Action:** Add a `pca_coords` attribute to `HardwareOutlierDetector` that
stores the PCA-transformed coordinates after `fit_predict()`, making them
available for `plot_outlier_scatter()` without re-computing.
**Lines to modify:** 76–93 (add `self.pca_coords_ = reduced` after line 88)
**Effort:** ~15 minutes

### R4: Standardize Figure Creation in Corner/KDE Plots (Optional)

**Priority:** Low
**File:** `vesuvio_analysis/core_functions/statistical_plugins.py`
**Action:** Where feasible, use `figure_factory()` for consistent sizing.
For the corner plot, keep `plt.subplots()` but document the deviation.
**Effort:** ~30 minutes

---

## Constraint Checklist

| # | Constraint | Verified |
|---|---|---|
| 1 | Three procedural steps strictly verified (1 → 2 → 3) | ✅ Yes |
| 2 | Legacy bypass functionality mandated (all flags `False` = no Phase 6 code runs) | ✅ Yes |
| 3 | Plotting plugins audited for compliance | ✅ Yes (with advisories) |
| 4 | `.npz` contract integrity preserved | ✅ Yes |
| 5 | No hardcoded or unconditional Phase 6 execution | ✅ Yes |
| 6 | iMinuit/Scipy engine selection does not corrupt `.npz` output | ✅ Yes |
| 7 | Bayesian Bootstrap separated from frequentist bootstrap | ✅ Yes |
| 8 | Test suite passes (25/25) | ✅ Yes |

**Confidence Score: 5/5**

---

## Conclusion

The Phase 6 "Full-Stack Stats Toolchain" integration into the `vesuvio_analysis`
pipeline is **architecturally sound and fully backward-compatible**. The
implementation adheres to the established pipeline architecture documented in
`ARCHITECTURE_AUDIT.md`. All three statistical tools are correctly gated behind
independent boolean flags that default to `False`, ensuring the legacy execution
path remains mathematically and functionally identical when the flags are not
explicitly enabled.

**Key Strengths:**

1. **Clean separation:** Phase 6 operates entirely in the post-fit domain and
   never feeds back into the iterative NCP fitting or MS/GC correction loops.
2. **Flag isolation:** Each tool has its own independent flag, preventing
   accidental cross-activation.
3. **No `.npz` contract violation:** The serialization format and data arrays
   written by Procedures 1, 2, and 3 are completely unaffected by Phase 6.
4. **Comprehensive testing:** 25 unit tests cover all three classes and five
   plotting functions without any Mantid dependency.

**Areas for Improvement:**

1. The diagnostic plotting functions are defined and tested but not yet called
   from the pipeline — they need to be wired in to produce visual evidence.
2. Error handling in `bootstrap_analysis.py` should be upgraded from `print()`
   to `logging.error()` for consistency with the rest of the pipeline.

## Next Steps

1. **R2 (High Priority):** Wire diagnostic plots into `_runStatisticalAnalysis()`
   so that enabling Phase 6 flags produces saved PDF/PNG figures alongside the
   console summary. This is the most impactful improvement for thesis-readiness.
2. **R3 → R2 dependency:** Expose `pca_coords_` from `HardwareOutlierDetector`
   before wiring the outlier scatter plot.
3. **R1 (Standard Maintenance):** Replace `print()` with `logging.error()` in
   bootstrap analysis for missing `.npz` files.
4. **Phase 7 Planning:** Consider a systematic error budget module and a
   goodness-of-fit dashboard as the next statistical extension, building on the
   Phase 6 infrastructure.
5. **Integration Test:** Once R2 is complete, run a full end-to-end test with
   `runOutlierDetection=True`, `runPhysicsClustering=True`, and
   `runBayesianBootstrap=True` on a representative dataset (e.g., `BaH2_500C.py`
   with `runRoutine=True`) to validate the complete visual output pipeline.
