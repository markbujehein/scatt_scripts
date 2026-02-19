# Contributing to `vesuvio_analysis`

> **Audience:** MSc students, PhD students, and research software engineers
> contributing to the VESUVIO DINS analysis pipeline.  
> **Standard:** Publication-quality code for the MSc Nanoscience programme,
> University of Copenhagen.

---

## Table of Contents

1. [Quick-Start Checklist](#1-quick-start-checklist)
2. [Branch Strategy](#2-branch-strategy)
3. [Docstring Standard](#3-docstring-standard)
4. [Mandatory PR Review Checklist](#4-mandatory-pr-review-checklist)
5. [Thesis Readiness Score](#5-thesis-readiness-score)
6. [Worked Example: Flagging a Bad Commit](#6-worked-example-flagging-a-bad-commit)
7. [Self-Correcting PRs](#7-self-correcting-prs)
8. [MCP-Grounded Reviews](#8-mcp-grounded-reviews)
9. [`statistical_plugins.py` Modularity Rules](#9-statistical_pluginspy-modularity-rules)

---

## 1. Quick-Start Checklist

Before opening a PR, verify every item below:

- [ ] `python -m pytest tests/ -v` passes with 0 failures.
- [ ] No `np.trapz` in any new or modified code — use `np.trapezoid`
      (NumPy ≥ 2.0) or `_trapz_1d` from `numba_routines.py` inside `@njit`.
- [ ] No unguarded division by arrays that can contain zeros — always use
      `np.where(denominator != 0, numerator / denominator, fallback)`.
- [ ] Every new public function has a Google-style docstring with LaTeX math
      in `$...$` notation for all physical quantities.
- [ ] No use of the word **"Sieve"** anywhere — see §4-D.
- [ ] `log_manager.py` YAML key is `optimizer_agreement_check:`, not `sieve3_agreement_gate:`.

---

## 2. Branch Strategy

| Branch | Purpose | Protected |
|---|---|---|
| `main` | Production releases | ✅ yes |
| `dev` | Integration branch for all features | ✅ yes |
| `copilot/*` | Agent-driven feature branches | no |
| `feature/*` | Human-driven feature branches | no |

**All PRs must target `dev`.**  The CI gate (`ci-dev.yml`) must pass before
merge.  PRs from `dev` to `main` require additional sign-off.

---

## 3. Docstring Standard

All public functions (not prefixed `_`) must use **Google style** with
LaTeX math delimited by `$...$`:

```python
def pseudoVoigt(
    x: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
    normVoigt: bool,
) -> np.ndarray:
    """Compute a normalised pseudo-Voigt profile for each mass component.

    The pseudo-Voigt approximation (Thompson et al., 1987) is:

    $\\eta(x) = \\eta_0 \\cdot L(x, \\gamma_v) + (1 - \\eta_0) \\cdot G(x, \\sigma_v)$

    where $\\eta_0$ is the mixing parameter, $L$ is a Lorentzian, and $G$
    is a Gaussian.  The total FWHM $f$ is computed from the component FWHMs:

    $f = \\left(f_G^5 + 2.69269 f_G^4 f_L + 2.42843 f_G^3 f_L^2
               + 4.47163 f_G^2 f_L^3 + 0.07842 f_G f_L^4 + f_L^5\\right)^{1/5}$

    Args:
        x: Momentum-transfer grid, shape ``(n_masses, n_spectra, n_bins)``.
        sigma: Gaussian width parameters per mass, same shape as ``x``.
        gamma: Lorentzian width parameters per mass, same shape as ``x``.
        normVoigt: When ``True``, normalise each profile to unit area via
            trapezoidal integration.

    Returns:
        Pseudo-Voigt profiles, shape ``(n_masses, n_spectra, n_bins)``.
        Each row sums to 1 / (bin width) when ``normVoigt=True``.

    Note:
        Uses the custom physical constant ``hbar = 2.0445`` (non-SI,
        Fortran legacy convention).  Do **not** substitute the SI value
        $\\hbar = 1.0546 \\times 10^{-34}$ J·s.
    """
```

**Required sections** for physics-bearing functions:

| Section | Required when |
|---|---|
| Module-level `Note:` with LaTeX | Any function containing `J(y)`, `E_recoil`, `sigma`, `gamma`, `hbar` |
| `Args:` | Always |
| `Returns:` | Always |
| `Raises:` | When the function raises explicitly |
| `Example::` | For public API functions in `statistical_plugins.py` |

---

## 4. Mandatory PR Review Checklist

When performing a code review (human or AI-assisted), run every section below.
Report `✅ PASS`, `⚠️ WARN`, or `❌ BLOCK` for each item.

### A — Numerical Drift

| # | Check | Severity |
|---|---|---|
| A1 | **NCP kernel modified?** — If `calculateNcpSpec`, `calcGaussianResolution`, `calcLorentzianResolution`, `pseudoVoigt`, or `numericalThirdDerivative` is changed, the Numba regression test must be updated and must still pass to `atol=1e-8`. | ❌ BLOCK |
| A2 | **iMinuit–Scipy agreement maintained?** — The relative difference of chi² and all parameters must remain ≤ 1 % (`_AGREEMENT_THRESHOLD`). Raising the threshold or removing the check is prohibited. | ❌ BLOCK |
| A3 | **`np.trapz` introduced?** — `np.trapz` is deprecated in NumPy ≥ 2.0; use `np.trapezoid`. Inside Numba `@njit` blocks, use `_trapz_1d` from `numba_routines.py`. | ❌ BLOCK |
| A4 | **Unmasked division?** — Any division `a / b` where `b` can be zero without a `np.where` guard. | ⚠️ WARN (❌ BLOCK in hot path) |
| A5 | **Physical constants changed?** — Any modification to `mN=1.008`, `Ef=4906.0`, `en_to_vel=4.3737e-4`, `hbar=2.0445`. | ❌ BLOCK |

### B — Workspace Pollution (Phase 4 Lifecycle)

| # | Check | Severity |
|---|---|---|
| B1 | **Workspace created without deletion?** — Any `CreateWorkspace`, `CloneWorkspace`, or `GroupWorkspaces` without a matching `DeleteWorkspace` at teardown. | ❌ BLOCK |
| B2 | **Mantid call inside `@njit`?** — `import mantid` or `mtd[...]` inside a Numba-decorated function. | ❌ BLOCK |
| B3 | **ADS naming convention violated?** — Workspace key not `{scriptName}_{DIRECTION}_{iteration}`. | ⚠️ WARN |
| B4 | **`mtd.clear()` inside fitting loop?** — `mtd.clear()` anywhere except teardown. | ❌ BLOCK |
| B5 | **Missing `if name in mtd:` guard?** — `mtd[name]` accessed without existence check. | ⚠️ WARN |

### C — Performance Regression

| # | Check | Severity |
|---|---|---|
| C1 | **`@njit` replaced by Python loop?** — A Numba-accelerated function replaced by a Python `for` loop over spectra. | ❌ BLOCK |
| C2 | **Redundant array copy in hot path?** — `.copy()`, `np.array(...)`, or `np.asarray(...)` inside `fitNcpToArray`'s per-spectrum loop. | ⚠️ WARN |
| C3 | **Object-mode Numba fallback?** — `@jit(nopython=False)` or `@jit(forceobj=True)`. | ❌ BLOCK |
| C4 | **Mantid call in NCP cost function?** — `NCPCostFunction.__call__` or any `@njit` function calling a Mantid algorithm. | ❌ BLOCK |

### D — Documentation and Thesis Standards

| # | Check | Severity |
|---|---|---|
| D1 | **Google-style docstring with LaTeX?** — Every new physics-bearing function must have `Args:`, `Returns:`, and a `Note:` section with LaTeX in `$...$`. | ⚠️ WARN |
| D2 | **Type hints on all public functions?** — New functions not prefixed `_` without full type hints. | ⚠️ WARN |
| D3 | **`statistical_plugins.py` depends on Mantid?** — Any Mantid import or workspace access in `statistical_plugins.py`. | ❌ BLOCK |
| D4 | **`y0` missing in y-space cost function?** — Any `convolvedModel` function that does not include `y0` as the first fitting parameter after `x`. Missing `y0` introduces a systematic baseline offset in the J(y) profile that will skew thesis results. | ❌ BLOCK |
| D5 | **Banned terminology?** — The word "Sieve" in any code, comment, docstring, or YAML log key. Replace with `optimizer_agreement_check:` or "iMinuit–Scipy Numerical Agreement Check". | ❌ BLOCK |

---

## 5. Thesis Readiness Score

Every PR review must include a **Thesis Readiness Score (1–10)** at the top:

```markdown
## Thesis Readiness Score: X/10
**Verdict:** <one-line summary of the primary concern>
```

| Score | Interpretation |
|---|---|
| 9–10 | Publication-ready: all gates pass, full docstrings with LaTeX, no deprecated APIs, clean workspace lifecycle |
| 7–8 | Minor issues only: ⚠️ WARNs present but no ❌ BLOCKs |
| 5–6 | Needs revision: at least one ❌ BLOCK in category A or B |
| 3–4 | Major revision: multiple ❌ BLOCKs or physics invariant violated |
| 1–2 | Do not merge: fundamental physics error or Mantid lifecycle corrupted |

---

## 6. Worked Example: Flagging a Bad Commit

### Scenario: Missing `y0` in a new y-space model

A contributor adds a new `GAUSS1D` model to `fit_in_yspace.py` without the
`y0` baseline offset parameter:

**Bad commit:**
```python
# BEFORE — missing y0 (BLOCKS merge)
def convolvedModel(x, A, sigma):
    return signal.convolve(A * np.exp(-x**2 / (2 * sigma**2)), resDense, mode="same") * xDelta

convolvedModel._parameters = {"A": (1.0, None), "sigma": (3.0, (0.1, 20.0))}
defaultPars = {"A": 1.0, "sigma": 5.0}
```

**Review output (auto-generated):**

```
## Thesis Readiness Score: 3/10
**Verdict:** Missing `y0` baseline offset in convolvedModel — systematic bias in J(y) profile.

### ❌ BLOCK — D4: y0 missing in y-space cost function
**File:** `vesuvio_analysis/core_functions/fit_in_yspace.py:<line>`
**Problem:** `convolvedModel` does not include `y0` as the first fitting
parameter after `x`.  Without `y0`, any non-zero background in the
symmetrised J(y) workspace will be absorbed into the amplitude `A`,
biasing the width `sigma` and corrupting the mean kinetic energy estimate
in the thesis.

**Suggested fix:**
```python
# AFTER (corrected)
def convolvedModel(x, y0, A, sigma):
    return y0 + signal.convolve(
        A * np.exp(-x**2 / (2 * sigma**2)), resDense, mode="same"
    ) * xDelta

convolvedModel._parameters = {"y0": None, "A": (1.0, None), "sigma": (3.0, (0.1, 20.0))}
defaultPars = {"y0": 0.0, "A": 1.0, "sigma": 5.0}
```
Note: `y0` must be unbounded (`None`) — it can be positive or negative.
```

### Scenario: `np.trapz` introduced in a Numba function

**Bad commit:**
```python
@njit
def normalise_profile(profile, x):
    area = np.trapz(profile, x)   # ← blocked: np.trapz not in Numba
    return profile / area
```

**Review output:**

```
## Thesis Readiness Score: 4/10
**Verdict:** np.trapz is deprecated (NumPy ≥ 2.0) and unsupported inside @njit.

### ❌ BLOCK — A3: np.trapz inside @njit
**File:** `vesuvio_analysis/core_functions/numba_routines.py:<line>`
**Problem:** `np.trapz` is deprecated in NumPy ≥ 2.0 and is not available
inside Numba `@njit` functions.  This will raise a `TypingError` at JIT
compile time on NumPy ≥ 2.0.

**Suggested fix:**
```python
# AFTER (corrected) — uses _trapz_1d from numba_routines.py
@njit
def _trapz_1d(y: np.ndarray, x: np.ndarray) -> float:
    """Manual trapezoidal integration for Numba (replaces np.trapz)."""
    result = 0.0
    for i in range(len(x) - 1):
        result += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return result

@njit
def normalise_profile(profile: np.ndarray, x: np.ndarray) -> np.ndarray:
    area = _trapz_1d(profile, x)
    return profile / area
```
```

---

## 7. Self-Correcting PRs

The AI reviewer should provide a **complete fix** alongside each blocking
issue.  The structure is always:

```
### ❌ BLOCK — <Category-ID>: <issue title>
**File:** `<path>:<line>`
**Problem:** <one sentence>
**Impact on thesis:** <how this error would appear in published results>
**Suggested fix:**
<corrected code snippet>
**Test to add:**
<minimal test that would catch this regression>
```

The contributor should:
1. Apply the suggested fix.
2. Add the suggested test.
3. Re-run `python -m pytest tests/ -v` to confirm the fix.
4. Re-request review — the score must improve to ≥ 7 before merge.

---

## 8. MCP-Grounded Reviews

When the MCP servers in `vesuvio_analysis/mcp_server/` are running, the
review agent should call the following tools **before** running the static
checklist:

| Step | Tool | Purpose |
|---|---|---|
| 1 | `env_versions` | Verify iMinuit ≥ 2, Pydantic ≥ 2, Numba ≥ 0.60 |
| 2 | `check_version_compatibility` | Detect known breaking version combos |
| 3 | `ads_list_workspaces` | Confirm workspace lifecycle is clean after pipeline |
| 4 | `log_read_latest` | Check `optimizer_agreement_check.overall_gate_passed` |
| 5 | `log_grep` | Search for `np.trapz`, `sieve3`, or `Sieve` in log output |

If step 4 returns `overall_gate_passed: False`, all agreement-check items
(A2) are automatically escalated to `❌ BLOCK` regardless of diff content.

**When MCP is unavailable**, note at the top of the review:
```
> ⚠️ MCP unavailable — review based on static analysis only.
> Run `python -m vesuvio_analysis.mcp_server.environment_server` to enable
> MCP-grounded reviews.
```

---

## 9. `statistical_plugins.py` Modularity Rules

`statistical_plugins.py` must remain a **pure-Python, Mantid-free module**.

| Rule | Rationale |
|---|---|
| No `import mantid` | The module must run in CI without Mantid |
| No workspace state access | Plugins receive plain NumPy arrays only |
| Each class is independently instantiable | `HardwareOutlierDetector`, `PhysicsTrendClusterer`, `BayesianBootstrap` must work standalone |
| New plugin = new class, not new function | Keeps the public API stable when adding new statistical methods |
| Test in `test_statistical_workflow.py` | All new plugins must have at least 3 unit tests against synthetic data |
| Google-style docstring with `Example::` | Every class must demonstrate usage with a synthetic 2D array |

**Banned patterns in `statistical_plugins.py`:**

```python
# BLOCKED — Mantid dependency
from mantid.api import mtd
ws = mtd["some_workspace"]

# BLOCKED — modifying global state
import vesuvio_analysis.core_functions.analysis_functions as af
af.GLOBAL_FLAG = True

# BLOCKED — mixing fitting and statistics
from scipy.optimize import minimize   # only sklearn and numpy allowed
```

**Allowed imports in `statistical_plugins.py`:**
```python
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```
