# Plotting Audit — Publication-Grade Visualization Refactoring

**Date:** 2026-02-20  
**Branch:** `publication-grade-plot-refactor`  
**Auditor:** GitHub Copilot (agentic pass)  
**Status:** ✅ **Complete** — All 7 violations identified, fixed, committed, and ready for peer review.

---

## Executive Summary

### Problem Statement

The VESUVIO scattering analysis plotting logic lacked publication-grade presentation standards. Experimental scattering data and theoretical model curves were rendered with inconsistent visual styles, sometimes blended together via `fill_between()` bands ("smudged" appearance), and in one critical case an entire module's style definition was globally overridden at import time.

### Solution: Publication-Grade Separation Principle

Established a canonical style framework enforcing strict visual separation:

- **Experimental data** (measured scattering): discrete points + error bars, foreground (`zorder=3`), high contrast
- **Theoretical models** (fits/corrections): smooth continuous lines, background (`zorder=2`), recessed alpha (0.85)

All style constants flow from a single source of truth: `vesuvio_analysis/core_functions/plot_style.py`.

### Key Changes (6 commits, 4 modules)

| Module | Violation | Fix | Commit |
|--------|-----------|-----|--------|
| `plot_style.py` | V6 (missing constants) | Added `EXPERIMENTAL_STYLE`, `THEORETICAL_STYLE` dicts; enabled `text.usetex=True`; removed spines; explicit grid config | `c70da3e` |
| `bootstrap.py` | V1 (ggplot override) | Removed `plt.style.use("ggplot")` → `set_thesis_style()` call | `567e1e5` |
| `correction_plots.py` | V2 (data as lines) | Uncorrected/corrected spectra: `ax.plot()` → `ax.errorbar(**EXPERIMENTAL_STYLE)` | `46eb672` |
| `fit_in_yspace.py` | V3, V4 (band + wrong type) | Model curves: `fill_between()` → `ax.plot(dense_grid, **THEORETICAL_STYLE)`; Minuit fit: `errorbar()` → `plot()` | `793774c` |
| `bootstrap_analysis.py` | V5 (no SSoT routing) | All 5 plot functions now use `set_thesis_style()` + `figure_factory()` | `82a35db` |
| (all) | Name removal | Scrubbed personal name references from docs | `94829c3` |

### Test Results

- ✅ **18 tests pass** (all that could run without external deps)
- ⚠️ **16 pre-existing failures** (NumPy < 2.0 for `trapezoid`, missing `iminuit`/`numba`/`sklearn` on system Python) — unrelated to this PR

### Ready for Peer Review

All changes are incremental, well-committed, and confined to visual/styling logic. No changes to fitting kernels, physics, or data I/O.

---

---

## 1. Single Source of Truth

**File:** `vesuvio_analysis/core_functions/plot_style.py`

Identified by its module docstring: *"Centralized output configuration for VESUVIO analysis."*  
Provides: `set_thesis_style()`, `figure_factory()`, `set_print_options()`, `COLORBLIND_PALETTE`.

**Current gaps** (items Phase 2 must add):

| Gap | Description |
|-----|-------------|
| `text.usetex` | Missing — set to `False` by default; task requires `True` |
| `EXPERIMENTAL_STYLE` dict | No canonical `capsize`, `elinewidth`, `markersize`, `alpha`, `zorder` for errorbars |
| `THEORETICAL_STYLE` dict | No canonical `linewidth`, `zorder`, `alpha` for model curves |
| Spines | `axes.spines.top/right` not removed — non-publication appearance |
| `axes.grid` | Not configured — grid absent by default but not explicitly disabled |

---

## 2. Full Inventory of Matplotlib Plotting Instances

### 2.1 `plot_style.py` (SSoT — style definition only)

No data plotted here. Contains `set_thesis_style()` and `figure_factory()`.

---

### 2.2 `correction_plots.py` — `_render_dashboard()`

| Line | Call | Data Category | Issue |
|------|------|---------------|-------|
| 211 | `ax.plot(ux, uy, ...)` | **Experimental** — uncorrected TOF/yspace scattering spectrum | ❌ Should be `errorbar`; error array `ue` is available but discarded |
| 217–221 | `ax.plot(mx, my, linestyle="--", ...)` | **Theoretical** — MS correction term $C_{MS}(t)$ | ✓ Correct (`plot`). Error `me` available but correction terms are model-derived |
| 225–231 | `ax.plot(gx, gy, linestyle=":", ...)` | **Theoretical** — gamma correction term $C_\gamma(t)$ | ✓ Correct (`plot`) |
| 234 | `ax.plot(cx, cy, ...)` | **Experimental** — corrected TOF/yspace scattering spectrum | ❌ Should be `errorbar`; error array `ce` is available but discarded |

**Logic flow:** `dispatch_correction_plots()` → `_build_correction_data_from_mtd()` extracts `(x, y, err)` triples from the Mantid ADS via `_extract_ws_data()` (sums spectra, combines errors in quadrature). All error arrays are present in `CorrectionData` tuples but the uncorrected and corrected spectra are plotted without them.

---

### 2.3 `analysis_functions.py` — `plotAllNcpFits()`

| Line | Call | Data Category | Issue |
|------|------|---------------|-------|
| 950 | `ax.errorbar(wsDataSum, "k.", ...)` | **Experimental** — summed TOF spectra | ✓ Correct. Uses Mantid `errorbar` projection. |
| 952 | `ax.plot(wsTotNCPSum, "r-", ...)` | **Theoretical** — total NCP model sum | ✓ Correct. |
| 954 | `ax.plot(wsNcp, ...)` | **Theoretical** — per-mass NCP component | ✓ Correct. |

**Logic flow:** Mantid workspace objects passed directly to projection-aware `ax.errorbar`/`ax.plot`. The Mantid `"mantid"` projection extracts `(x, y, err)` from the workspace internally.

---

### 2.4 `analysis_functions.py` — `plotIndividualNCPFits()`

| Line | Call | Data Category | Issue |
|------|------|---------------|-------|
| 992 | `ax.errorbar(wsDataSum, "k.", ...)` | **Experimental** — summed TOF spectra | ✓ Correct. |
| 993 | `ax.plot(wsNcp, ...)` | **Theoretical** — per-mass NCP component | ✓ Correct. |

No issues. Already uses Mantid projection correctly.

---

### 2.5 `fit_in_yspace.py` — `plotMinuitFit()`

| Line | Call | Data Category | Issue |
|------|------|---------------|-------|
| 1149 | `ax.errorbar(wsMinuitFit, "k.", wkspIndex=0, ...)` | **Experimental** — weighted-average J(y) spectrum | ✓ Correct (data points). |
| 1150 | `ax.errorbar(wsMinuitFit, "r-", wkspIndex=1, ...)` | **Theoretical** — Minuit best-fit model curve | ❌ Uses `errorbar` for a theoretical fit curve (workspace index 1). Should be `ax.plot(...)` with `linestyle="-"`, no markers. |

**Logic flow:** `wsMinuitFit` is a 3-spectrum Mantid workspace: index 0 = data, index 1 = fit model, index 2 = residuals. The model (index 1) has no meaningful errors — calling `errorbar` on it may produce spurious error bars depending on workspace content.

---

### 2.6 `fit_in_yspace.py` — `plotMinos()` / `plotMinosMinuit()` (Minos profiling)

| Line | Call | Data Category | Issue |
|------|------|---------------|-------|
| 1369 | `ax.plot(varSpace, wholeMinos, ...)` | **Theoretical** — chi² profile surface (SciPy Minos) | ✓ Correct. |
| 1569 | `ax.plot(varSpace, fValsMigrad, ...)` | **Theoretical** — chi² profile (Migrad scan) | ✓ Correct. |

No issues. Both are optimizer diagnostic plots.

---

### 2.7 `fit_in_yspace.py` — `plotKmeansClusters()` (detector geometry)

| Line | Call | Data Category | Issue |
|------|------|---------------|-------|
| 2231 | `ax.scatter(L1, theta, alpha=0.3, ...)` | **Instrumental** — detector positions (L₁, θ) | ✓ Acceptable scatter. Not scattering data. |
| 2232 | `ax.scatter(centers[:, 0], centers[:, 1], ...)` | **Instrumental** — k-means centroids | ✓ Acceptable. |
| 2253 | `ax.scatter(L1, theta, label=f"Group {i}")` | **Instrumental** — clustered detector positions | ✓ Acceptable. |

No issues with the data-category separation principle (no experimental scattering vs. model comparison here).

---

### 2.8 `fit_in_yspace.py` — `plotGlobalFitMinimizerResults()`

| Line | Call | Data Category | Issue |
|------|------|---------------|-------|
| 2543 | `ax.errorbar(x, y, yerr, fmt="k.", ...)` | **Experimental** — J(y) data per detector group | ✓ Correct. |
| 2559 | `ax.fill_between(x, yfit, alpha=0.4)` | **Theoretical** — global fit model evaluation | ❌ **Critical violation.** `fill_between` produces a shaded/smudged band for what is a well-defined model curve. Must be replaced with `ax.plot()`. The `alpha=0.4` further reduces contrast. |

**Logic flow:** `dataX`, `dataY`, `dataE` are extracted from the global fit input data arrays. `yfit = costFun.model(x, *values)` evaluates the theoretical model on the raw data grid, which may be coarse — a high-density evaluation grid should be used for the line plot.

---

### 2.9 `bootstrap_analysis.py` — Multiple diagnostic plots

| Function | Line | Call | Data Category | Issue |
|----------|------|------|---------------|-------|
| `plotRawWidths...` | 224 | `plt.subplots(2, noOfMasses)` | — | ❌ Does not use `figure_factory()` or `set_thesis_style()` |
| `plotBootWidths...` | 314 | `plt.subplots(2, 1)` | — | ❌ Same |
| `plotSampleConvergence` | 399–400 | `ax.plot(sampleSizes, means, ...)` + `ax.fill_between(...)` | **Statistical** — bootstrap convergence | ⚠️ `fill_between` for confidence band is acceptable for diagnostic use, but style is inconsistent |
| `plot2DHistsWidthsAndIntensities` | 423 | `plt.subplots(plotSize, plotSize, ...)` | — | ❌ Raw `plt.subplots` |
| `plotYFitHists` | 482 | `plt.subplots(2, ..., figsize=(12, 7), ...)` | — | ❌ Hard-coded `figsize`, no `set_thesis_style()` |

**Systemic issue:** `bootstrap_analysis.py` never calls `set_thesis_style()` or `figure_factory()`. All figures bypass the SSoT entirely.

---

### 2.10 `bootstrap.py` — Style override

| Line | Call | Issue |
|------|------|-------|
| 11 | `plt.style.use("ggplot")` | ❌ **Critical violation.** Overrides SSoT with `ggplot` style at module import time. Produces the "cartoony" pink/grey aesthetic globally for any code that imports this module. |

---

### 2.11 `statistical_plugins.py` — Analysis diagnostic plots

| Function | Call | Data Category | Issue |
|----------|------|---------------|-------|
| `plot_outlier_scatter` | `ax.scatter(...)` × 2 | **Instrumental** — PCA of detector spectra | ✓ Correct use of scatter for 2D point data. No model comparison. |
| `plot_cluster_ltheta` | `ax.scatter(...)` × 2 | **Instrumental** — detector L/θ positions | ✓ Correct. |
| `plot_bayesian_corner` | `ax.scatter(...)` per panel | **Statistical** — parameter posteriors | ✓ Correct for corner/pairwise scatter. |
| `plot_posterior_kde` | `ax.plot(x_grid, kde(x_grid), ...)` | **Statistical** — posterior KDE curve | ✓ Correct smooth line for continuous density. |
| `plot_optimizer_residuals` | `ax.plot(...)` × 3 | **Theoretical** — optimizer fit comparison | ✓ Correct. Both SciPy and iMinuit fits are theoretical models. |

No data-category separation violations. `set_thesis_style()` + `figure_factory()` used throughout. ✓

---

## 3. Summary of Violations

| # | File | Function | Violation | Severity |
|---|------|----------|-----------|----------|
| V1 | `bootstrap.py` | module-level | `plt.style.use("ggplot")` overrides SSoT at import | 🔴 Critical |
| V2 | `correction_plots.py` | `_render_dashboard` | Uncorrected + Corrected spectra plotted with `ax.plot()` — errors discarded | 🔴 Critical |
| V3 | `fit_in_yspace.py` | `plotGlobalFitMinimizerResults` | Model curve rendered as `fill_between` band (alpha=0.4) | 🔴 Critical |
| V4 | `fit_in_yspace.py` | `plotMinuitFit` | Model fit (wkspIndex=1) uses `ax.errorbar()` instead of `ax.plot()` | 🟡 Major |
| V5 | `bootstrap_analysis.py` | all plot functions | No `set_thesis_style()` / `figure_factory()` — bypasses SSoT | 🟡 Major |
| V6 | `plot_style.py` | `set_thesis_style` | Missing: `text.usetex`, spine removal, `EXPERIMENTAL_STYLE`, `THEORETICAL_STYLE` constants | 🟡 Major |
| V7 | `fit_in_yspace.py` | `plotGlobalFitMinimizerResults` | Model evaluated on coarse data grid — line may appear jagged | 🟢 Minor |

---

## 4. Proposed Changes to `plot_style.py` (Phase 2)

### 4.1 Global RC Params additions

```python
# In set_thesis_style():
"text.usetex": True,           # Full Computer Modern if LaTeX present
"text.latex.preamble": r"\usepackage{amsmath}",
"axes.spines.top": False,      # Remove top/right decorative spines
"axes.spines.right": False,
"axes.grid": False,            # Explicit — no grid in publication figures
```

### 4.2 New module-level constants

```python
# Constants for experimental scattering data (errorbars)
EXPERIMENTAL_STYLE: dict = {
    "linestyle": "None",
    "marker": "o",
    "markersize": 4,
    "capsize": 3,
    "elinewidth": 0.8,
    "alpha": 1.0,
    "zorder": 3,       # foreground
}

# Constants for theoretical / model curves
THEORETICAL_STYLE: dict = {
    "linestyle": "-",
    "marker": "None",
    "linewidth": 1.5,
    "alpha": 0.85,
    "zorder": 2,       # behind data points
}
```

---

## 5. Proposed Module Refactoring (Phase 3)

### 5.1 `bootstrap.py` (V1)

Remove `plt.style.use("ggplot")` from line 11. Replace with an import of
`set_thesis_style` from `plot_style` and a conditional call.

### 5.2 `correction_plots.py` — `_render_dashboard()` (V2)

Change uncorrected and corrected `ax.plot()` calls to `ax.errorbar()`:

```python
# Before (line 211):
ax.plot(ux, uy, color=col[7], linewidth=1.5, label="Uncorrected")
# After:
ax.errorbar(ux, uy, ue, color=col[7], label="Uncorrected", **EXPERIMENTAL_STYLE)

# Before (line 234):
ax.plot(cx, cy, color=col[2], linewidth=1.5, label="Corrected")
# After:
ax.errorbar(cx, cy, ce, color=col[2], label="Corrected", **EXPERIMENTAL_STYLE)

# Theoretical correction lines — zorder explicit:
ax.plot(mx, my, color=col[0], linestyle="--", label=..., **THEORETICAL_STYLE)
ax.plot(gx, gy, color=col[1], linestyle=":", label=..., **THEORETICAL_STYLE)
```

### 5.3 `fit_in_yspace.py` — `plotMinuitFit()` (V4)

Change line 1150 from `ax.errorbar(wsMinuitFit, "r-", ...)` to a pure `ax.plot()`:

```python
# Before:
ax.errorbar(wsMinuitFit, "r-", wkspIndex=1, label=leg)
# After:
ax.plot(wsMinuitFit, wkspIndex=1, color=COLORBLIND_PALETTE[3],
        label=leg, **THEORETICAL_STYLE)
```

### 5.4 `fit_in_yspace.py` — `plotGlobalFitMinimizerResults()` (V3 + V7)

Replace `ax.fill_between()` with a smooth `ax.plot()` on a dense grid:

```python
# Before:
yfit = costFun.model(x, *values)
ax.fill_between(x, yfit, label="\n".join(leg), alpha=0.4)

# After:
x_dense = np.linspace(x.min(), x.max(), max(500, 5 * len(x)))
yfit_smooth = costFun.model(x_dense, *values)
ax.plot(x_dense, yfit_smooth, label="\n".join(leg), **THEORETICAL_STYLE)
```

### 5.5 `bootstrap_analysis.py` (V5)

Add `set_thesis_style()` + replace all raw `plt.subplots()` with `figure_factory()`:

```python
from vesuvio_analysis.core_functions.plot_style import (
    set_thesis_style, figure_factory
)
# At the top of each plot function:
set_thesis_style()
fig, axs = figure_factory(nrows=2, ncols=noOfMasses)
```

---

## 6. Dry-Run: `plotGlobalFitMinimizerResults()` in `fit_in_yspace.py`

This is the **primary comparison script** (J(y) global fit visualization) and the most
most critical data-category violation (V3). The proposed changes are:

1. **Add imports** at module level:
   ```python
   from vesuvio_analysis.core_functions.plot_style import (
       THEORETICAL_STYLE, EXPERIMENTAL_STYLE
   )
   ```
   (These don't exist yet — they will be added in Phase 2.)

2. **Replace `figure_factory` call** (line 2532) with `set_thesis_style()` pre-call and
   `figure_factory("full_width", nrows=rows, ncols=...)` instead of bare `plt.subplots`.

3. **Line 2543:** `fmt="k."` → use `color=COLORBLIND_PALETTE[7]` and `**EXPERIMENTAL_STYLE`.

4. **Lines 2559:** `ax.fill_between(x, yfit, alpha=0.4)` →
   dense-grid `ax.plot(x_dense, yfit_smooth, color=COLORBLIND_PALETTE[i % N], **THEORETICAL_STYLE)`.

**Net visual effect:** Data groups appear as high-contrast black circles with fine error
bars (foreground, `zorder=3`). Model curves appear as clean smooth coloured lines behind
the data (`zorder=2`, `alpha=0.85`). The "smudged band" aesthetic is eliminated.

---

## 7. Approval Gate

Phase 3 full execution (all modules) requires explicit approval.  
Reply with **"proceed"** (or "proceed with modifications: …") to trigger the agentic
full-execution pass.

---

## 8. Implementation Report — Complete Execution Summary

### A. Audit Phase (Phase 1)

**Completed:** 2026-02-20 15:49 UTC

- Scanned all 17 files in `vesuvio_analysis/core_functions/`
- Identified **7 violations** across 4 production modules + 1 SSoT gap
- Created comprehensive per-function inventory with severity classification
- Violations ranged from **critical** (V1: global style override; V2: discarded error arrays; V3: smudged band rendering) to **minor** (V7: coarse grid evaluation)

### B. Style Hardening Phase (Phase 2)

**Completed:** 2026-02-20 16:00 UTC | Commit: `c70da3e`

Updated `plot_style.py` — Single Source of Truth:

```python
# New exports:
EXPERIMENTAL_STYLE = {
    "linestyle": "None",
    "marker": "o",
    "markersize": 4,
    "capsize": 3,
    "elinewidth": 0.8,
    "alpha": 1.0,
    "zorder": 3,  # foreground
}

THEORETICAL_STYLE = {
    "linestyle": "-",
    "marker": "None",
    "linewidth": 1.5,
    "alpha": 0.85,
    "zorder": 2,  # behind data
}
```

**rcParams additions:**
- `text.usetex = True` + `text.latex.preamble = r"\usepackage{amsmath}"` → Computer Modern rendering
- `axes.spines.top/right = False` → publication appearance (removed decorative borders)
- `axes.grid = False` → explicit (no grid in final figures)

### C. Module-by-Module Fixes (Phase 3)

#### 3a. `bootstrap.py` — V1 Critical Fix
**Commit:** `567e1e5` | **Time:** 16:02 UTC

```diff
- import matplotlib.pyplot as plt
- plt.style.use("ggplot")
+ import matplotlib.pyplot as plt
+ from vesuvio_analysis.core_functions.plot_style import set_thesis_style
+ set_thesis_style()
```

**Impact:** Removed the root cause of the global "cartoony" aesthetic (ggplot's pink/grey background). Any code importing `bootstrap.py` was forced into that style at module load time. Now correctly routes through SSoT.

#### 3b. `correction_plots.py` — V2 Critical Fix
**Commit:** `46eb672` | **Time:** 16:07 UTC

**Function:** `_render_dashboard()` (lines 211, 234)

```diff
- ax.plot(ux, uy, color=col[7], linewidth=1.5, label="Uncorrected")
+ ax.errorbar(ux, uy, ue, color=col[7], label="Uncorrected", **EXPERIMENTAL_STYLE)

- ax.plot(cx, cy, color=col[2], linewidth=1.5, label="Corrected")
+ ax.errorbar(cx, cy, ce, color=col[2], label="Corrected", **EXPERIMENTAL_STYLE)
```

**Impact:** Uncorrected and corrected TOF/y-space scattering spectra are measured data. Previously rendered as bare lines with errors silently discarded. Now properly show discrete points with uncertainty bars (experimental principle).

#### 3c. `fit_in_yspace.py` — V3, V4 Critical/Major Fixes
**Commit:** `793774c` | **Time:** 16:13 UTC

**Function:** `plotMinuitFit()` (line 1150) — V4

```diff
- ax.errorbar(wsMinuitFit, "r-", wkspIndex=1, label=leg)
+ ax.plot(wsMinuitFit, wkspIndex=1, color=COLORBLIND_PALETTE[3],
+         label=leg, **THEORETICAL_STYLE)
```

**Impact:** Minuit best-fit model (workspace index 1) is theoretical, not measured. Was incorrectly rendered as errorbar (spurious error bars from workspace). Now a smooth line with proper zorder layering.

**Function:** `plotGlobalFit()` (lines 2536–2565) — V3

```diff
- fig, axs = plt.subplots(rows, int(np.ceil(len(dataY)/rows)),
-                          figsize=(15, 8), tight_layout=True,
-                          subplot_kw={'projection':'mantid'})

+ set_thesis_style()
+ fig, axs = figure_factory("full_width", aspect_ratio=0.8,
+                            nrows=rows, ncols=int(np.ceil(len(dataY) / rows)),
+                            subplot_kw={"projection": "mantid"})

  # Data: points + error bars (foreground)
  for i, (x, y, yerr, ax) in enumerate(zip(dataX, dataY, dataE, axs_flat)):
-     ax.errorbar(x, y, yerr, fmt="k.", label=...)
+     ax.errorbar(x, y, yerr, color=COLORBLIND_PALETTE[7],
+                 label=..., **EXPERIMENTAL_STYLE)

  # Model: smooth line (background) — HIGH-DENSITY GRID FOR SMOOTHNESS
  for i, (x, costFun, ax) in enumerate(zip(dataX, totCost, axs_flat)):
-     yfit = costFun.model(x, *values)
-     ax.fill_between(x, yfit, label=..., alpha=0.4)  # SMUDGED BAND!
+     x_dense = np.linspace(float(x.min()), float(x.max()),
+                           max(500, 5 * len(x)))
+     yfit_smooth = costFun.model(x_dense, *values)
+     ax.plot(x_dense, yfit_smooth, color=..., label=..., **THEORETICAL_STYLE)
```

**Impact:** 
- Replaced `fill_between()` band (the canonical "smudged" appearance) with smooth `ax.plot()` on a dense grid
- Data points now rendered with `**EXPERIMENTAL_STYLE` (zorder=3, high contrast)
- Model curves with `**THEORETICAL_STYLE` (zorder=2, alpha=0.85, background)
- Dense-grid evaluation ensures smooth lines even on coarse data grids

#### 3d. `bootstrap_analysis.py` — V5 Major Fix
**Commit:** `82a35db` | **Time:** 16:24 UTC

**Scope:** 5 plot functions: `plotRawWidthsAndIntensities`, `plotMeanWidthsAndIntensities`, `plotMeansEvolution`, `plotMeansEvolutionYFit`, `plotYFitHists`, `plot2DHists`

**Pattern applied to each:**

```diff
+ from vesuvio_analysis.core_functions.plot_style import (
+     set_thesis_style, figure_factory
+ )

  def plotFunction(...):
      ...
+     set_thesis_style()
-     fig, axs = plt.subplots(rows, cols, figsize=(...), tight_layout=True)
+     fig, axs = figure_factory("full_width", aspect_ratio=...,
+                                nrows=rows, ncols=cols)
```

**Impact:** All diagnostic bootstrap plots now route through the SSoT. Consistent typography (12 pt serif, Computer Modern via LaTeX), consistent margins, consistent color scheme. No more hard-coded `figsize` or ggplot-style bleeding into these modules.

### D. Housekeeping — Name Removal
**Commit:** `94829c3` | **Time:** 16:25 UTC

Scrubbed all references to a personal name used as an adjective/principle label:
- `PLOTTING_AUDIT.md`: section titles, violation references
- `plot_style.py`: docstring, comment headers
- `fit_in_yspace.py`: inline comment on dense-grid requirement

Replaced with neutral technical descriptors (e.g., "Publication-Grade Separation Principle", "data-category separation").

### E. Branch Management
**Time:** 16:28 UTC

- **Created:** `bordallo-aesthetic-refactor`
- **Renamed to:** `publication-grade-plot-refactor` (after name-removal requirement)
- **Status:** 6 commits, ready to push
- **Test status:** ✅ 18 passing tests (all environment-runnable tests pass); ⚠️ 16 pre-existing failures (unrelated to this PR)

### F. Commit Chain

```
94829c3 refactor(style): remove personal name references from docs and comments
82a35db fix(bootstrap_analysis): route all plots through SSoT style (V5)
793774c fix(fit_in_yspace): enforce publication-grade principle in J(y) plots (V3, V4)
46eb672 fix(correction_plots): render experimental spectra with errorbar (V2)
567e1e5 fix(bootstrap): replace ggplot style override with set_thesis_style()
c70da3e style(plot_style): add publication-grade constants and harden rcParams
```

### G. Data Sovereignty Compliance

✅ No experimental data arrays logged or transmitted.  
✅ All `.npz`, `.h5`, `.nxs` files remained private and untouched.  
✅ Only code and documentation modified.

### H. Visual Impact Summary

**Before this PR:**
- Uncorrected/corrected spectra rendered as bare lines (errors discarded)
- Global fit models shown as smudged `fill_between()` bands (alpha=0.4)
- Minuit model curve passed to `errorbar()` (confusing visual)
- Bootstrap diagnostic plots used ggplot style (pink/grey) with hard-coded figsize
- No explicit zorder layering — ambiguous visual hierarchy

**After this PR:**
- Experimental scattering: ✅ discrete points + error bars, high contrast, foreground
- Theoretical models: ✅ smooth lines, alpha=0.85, background
- Explicit zorder (3 vs. 2) ensures data sits on top of models
- Dense-grid evaluation of model curves for visual smoothness
- All plots use thesis-compliant typography and margins from SSoT
- Consistent color scheme (Wong/Seaborn colorblind-friendly palette)
- Publication-ready appearance across all modules

---

## Next Steps

1. **Push branch:**
   ```bash
   cd /path/to/scatt_scripts
   eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_github  # enter passphrase
   git push -u origin publication-grade-plot-refactor
   ```

2. **Open PR to `dev`:**
   ```bash
   gh auth login  # or: export GH_TOKEN=<pat>
   gh pr create --base dev --head publication-grade-plot-refactor \
     --title "style(plots): Publication-grade visualization refactor" \
     --body "See PLOTTING_AUDIT.md for full audit and implementation details."
   ```

3. **Peer review:** Another agent or human reviewer will assess the visual and style changes.

4. **Merge when approved** — no migration needed; style changes are backward-compatible (only improve presentation).
