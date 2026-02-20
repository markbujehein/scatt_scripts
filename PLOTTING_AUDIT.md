# Plotting Audit — "Bordallo" Aesthetic Refactoring

**Date:** 2026-02-20  
**Branch:** `copilot-worktree-2026-02-20T15-48-59`  
**Auditor:** GitHub Copilot (agentic pass)

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

No issues with the Bordallo principle (no experimental scattering vs. model comparison here).

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

No Bordallo-principle violations. `set_thesis_style()` + `figure_factory()` used throughout. ✓

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
egregious Bordallo violation (V3). The proposed changes are:

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
