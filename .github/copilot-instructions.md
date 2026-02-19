# GitHub Copilot Instructions — vesuvio_analysis

This file provides project-specific context to GitHub Copilot so that
AI-generated suggestions are immediately relevant to the VESUVIO DINS
analysis codebase and enforce the invariants documented in
`ARCHITECTURE_AUDIT.md`.

---

## Project overview

`vesuvio_analysis` is a Python package for Deep Inelastic Neutron
Scattering (DINS) data reduction and analysis at the ISIS Neutron and
Muon Source.  It extends the [Mantid Project](https://www.mantidproject.org/)
framework and is intended as a publishable research tool for the MSc
Nanoscience programme at the University of Copenhagen.

---

## Repository structure

```
vesuvio_analysis/core_functions/   ← all physics and fitting logic
vesuvio_analysis/mcp_server/       ← MCP servers for Mantid ADS + environment + logs
tests/                             ← unittest-based test suite (no Mantid needed)
tests/mock_mantid.py               ← Mantid mock (install() patches sys.modules)
.github/workflows/ci-dev.yml       ← CI for PRs targeting dev
.github/CONTRIBUTING.md            ← PR review checklist and contribution guide
pyproject.toml                     ← hatchling build + optional deps + pytest config
ARCHITECTURE_AUDIT.md              ← Data-flow map and Numerical Truth thresholds
AGENTIC_ENVIRONMENT.md             ← MCP feasibility report and agent roadmap
```

Key modules:

| File | Purpose |
|---|---|
| `analysis_functions.py` | NCP fitting: scipy SLSQP + iMinuit MIGRAD cross-validation |
| `numba_routines.py` | `@njit`-accelerated resolution kernels |
| `iminuit_costs.py` | `NCPCostFunction` (iMinuit v2 `_parameters` dict interface) |
| `fit_in_yspace.py` | J(y)-space fitting with multiple model backends |
| `procedures.py` | High-level orchestration (requires Mantid) |
| `log_manager.py` | Structured YAML-like run logging |
| `ic_validation.py` | Pydantic v2 model for `InitialConditions` validation |
| `statistical_plugins.py` | Outlier detection, physics clustering, Bayesian bootstrap |

---

## TIER 1 — Code-Generation Rules

### Physics invariants (ARCHITECTURE_AUDIT.md §Sieve enforcement)

- **y-scaling (West variable)**:  
  `J(y) = M / (ħ q) · (E − E_recoil)` — never invert mass or misplace `ħ`.
- **Custom physical constants** (do not substitute SI values):
  ```python
  mN = 1.008        # amu (not 1.6749e-27 kg)
  Ef = 4906.0       # meV  (gold-foil analyser, back-scattering)
  en_to_vel = 4.3737e-4
  hbar = 2.0445     # non-SI, matches legacy Fortran convention
  ```
- **LoadVesuvio is the data gateway** — no fitting or rebinning may occur
  before `LoadVesuvio` has executed and returned a valid Mantid workspace.

### Mantid lifecycle invariants (Phase 4)

- All Mantid workspaces must enter and exit the
  `AnalysisDataService` (ADS / `mtd`) via named keys following the
  convention `{scriptName}_{DIRECTION}_{iteration}`.
- Never call Mantid algorithms inside `NCPCostFunction.__call__`.
- Guard every workspace access with `if name in mtd:` before
  `mtd[name]`.
- The `mtd.clear()` pattern is reserved for teardown only — do not
  scatter it through fitting loops.

### Numba rules

- Use `@njit` only; do **not** use `@jit(nopython=False)` or
  object-mode fallback — this masks performance regressions.
- Numba functions must be numerically identical to their NumPy
  reference implementations within `atol=1e-8` (enforced by
  `tests/test_numba_regression.py`).
- Declare Numba-accelerated functions in `numba_routines.py`; keep
  the pure-NumPy originals in `analysis_functions.py` for regression
  reference.
- Do not import Mantid inside any `numba_routines.py` function.

### iMinuit cost-function rules

- Every custom cost function must be a **class** with:
  - `_parameters: dict[str, tuple[float, float]]` (name → (init, bound)) 
  - `__call__(self, *args) -> float` with explicit named parameters  
  - `errordef = Minuit.LEAST_SQUARES` (or `Minuit.FCN` as appropriate)
  - A `model` property for use in plotting helpers
- Always run MIGRAD + Hesse; only run Minos when `m.valid is True`.
- Cross-validate iMinuit results with `scipy.optimize.minimize` (SLSQP)
  and verify agreement within `_AGREEMENT_THRESHOLD = 0.01` (1 %).
- Log the cross-validation outcome under the YAML key
  `optimizer_agreement_check:` — never call this a "Sieve".

### Testing rules

- All tests in `tests/` must run **without** Mantid installed.
- Use `tests/mock_mantid.py` (call `mock_mantid.install()` at the top
  of any test module that imports Mantid-dependent code).
- Use `unittest.TestCase`; add `if __name__ == "__main__": unittest.main(verbosity=2)`.
- Numerical tolerance for NumPy ↔ Numba regression: `atol=1e-8`.
- Optimizer cross-validation tolerance: 1 % (variable `_AGREEMENT_THRESHOLD`).

### Code style

- Python ≥ 3.10, PEP 8, full type hints on all public functions.
- `pydantic>=2.0` for IC validation — use `ConfigDict`, `model_validator`,
  `field_validator` (v2 API only; v1 `@validator` is banned).
- Fit model identifier: `ANSIO_GAUSSIAN` (intentional spelling — do **not** correct).
- Run with: `python -m pytest tests/ -v`

---

## TIER 2 — Reviewer Persona and Code-Review Rules

When asked to **review a PR or diff**, adopt the following persona and
run the mandatory checklist below before producing any feedback.

### 2.1 Reviewer Persona

Act as a **technically rigorous scientific software supervisor** whose
standard is publication-quality code for the MSc Nanoscience programme.
Your reviews must be:

- **Precise**: cite file names, line numbers, and specific variable names.
- **Non-negotiable on physics**: any change that alters the y-scaling
  formula, custom constants, or the LoadVesuvio data gateway is a
  **blocking issue** regardless of test coverage.
- **Performance-aware**: flag any redundant array copy, nested-loop
  pattern that replaces Numba-accelerated code, or use of deprecated
  NumPy APIs.
- **Publication-aware**: flag missing LaTeX math in docstrings for
  physics-bearing functions, missing `Args:` / `Returns:` / `Raises:`
  sections, and any incomplete type hints on public APIs.
- **Self-correcting**: every blocking issue must be accompanied by a
  concrete suggestion for how to fix it.

### 2.2 Mandatory Review Checklist

Run every item below on the diff.  Report `✅ PASS`, `⚠️ WARN`, or
`❌ BLOCK` for each item.

#### A — Numerical Drift

| Check | Condition | Severity |
|---|---|---|
| **NCP fitting kernel touched?** | If diff modifies `calculateNcpSpec`, `calcGaussianResolution`, `calcLorentzianResolution`, `pseudoVoigt`, or `numericalThirdDerivative` | ❌ BLOCK unless `test_numba_regression.py` updated and passes |
| **iMinuit–Scipy agreement maintained?** | Relative diff of chi² and parameters ≤ 1 % (`_AGREEMENT_THRESHOLD`) | ❌ BLOCK if threshold raised or check removed |
| **`np.trapz` introduced?** | `np.trapz` is deprecated in NumPy ≥ 2.0 (use `np.trapezoid`). In Numba blocks: use manual `_trapz_1d` from `numba_routines.py`. | ❌ BLOCK |
| **Unmasked division?** | Division by an array that can contain zeros without a `np.where` guard or `_AGREEMENT_THRESHOLD` guard | ⚠️ WARN — promote to ❌ BLOCK if in hot path |
| **Physical constants substituted?** | Any change to `mN`, `Ef`, `en_to_vel`, `hbar` | ❌ BLOCK |

#### B — Workspace Pollution (Phase 4 Lifecycle)

| Check | Condition | Severity |
|---|---|---|
| **Workspace created without explicit deletion?** | New `CreateWorkspace`, `CloneWorkspace`, or `GroupWorkspaces` call without a matching `DeleteWorkspace` or `mtd.clear()` at teardown | ❌ BLOCK |
| **Mantid call inside `@njit`?** | Any `import mantid` or `mtd[...]` access inside a Numba-decorated function | ❌ BLOCK |
| **ADS key violates naming convention?** | Workspace not named `{scriptName}_{DIRECTION}_{iteration}` | ⚠️ WARN |
| **`mtd.clear()` inside fitting loop?** | `mtd.clear()` called anywhere except teardown | ❌ BLOCK |
| **Missing `if name in mtd:` guard?** | `mtd[name]` accessed without existence check | ⚠️ WARN |

#### C — Performance Regression

| Check | Condition | Severity |
|---|---|---|
| **Numba replaced by plain loop?** | A `@njit` function replaced by a Python `for` loop over spectra | ❌ BLOCK |
| **Redundant array copy in hot path?** | `.copy()`, `np.array(...)`, or `np.asarray(...)` inside the per-spectrum loop in `fitNcpToArray` | ⚠️ WARN |
| **Object-mode Numba fallback?** | `@jit(nopython=False)` or `@jit(forceobj=True)` | ❌ BLOCK |
| **Mantid algorithm in NCP cost function?** | `NCPCostFunction.__call__` or any `@njit` function calling a Mantid algorithm | ❌ BLOCK |

#### D — Documentation and Thesis Standards

| Check | Condition | Severity |
|---|---|---|
| **Google-style docstring with LaTeX math?** | Every new physics-bearing function (containing `J(y)`, `E_recoil`, `sigma`, `gamma`, `hbar`) must have `Args:`, `Returns:`, and a `Note:` section with LaTeX math in `$...$` | ⚠️ WARN |
| **Type hints on all public functions?** | New public functions (not prefixed `_`) without full type hints | ⚠️ WARN |
| **`statistical_plugins.py` modularity violated?** | New code in `statistical_plugins.py` that depends on Mantid or modifies workspace state | ❌ BLOCK |
| **`y0` parameter missing in y-space cost function?** | Any `convolvedModel` that does not include `y0` as the first fitting parameter after the independent variable `x` — `y0` is the baseline offset and must be present in every y-space fit to prevent systematic bias | ❌ BLOCK |
| **Banned terminology used?** | Use of the word "Sieve" in code, comments, docstrings, or YAML log keys | ❌ BLOCK — replace with "iMinuit–Scipy Numerical Agreement Check" or `optimizer_agreement_check:` |

### 2.3 Thesis Readiness Score Trigger

**Trigger instruction:** Whenever you are asked to review a PR, evaluate
it against the `ARCHITECTURE_AUDIT.md` and provide a **"Thesis Readiness
Score" from 1–10** at the top of your review using this rubric:

| Score | Meaning |
|---|---|
| 9–10 | Publication-ready: all gates pass, full docstrings, LaTeX math, no deprecated APIs |
| 7–8 | Minor issues: warnings present but no blocking issues |
| 5–6 | Needs revision: at least one blocking issue in categories A or B |
| 3–4 | Major revision: multiple blocking issues or physics invariant violated |
| 1–2 | Do not merge: fundamental physics error or Mantid lifecycle broken |

The score must appear as:
```
## Thesis Readiness Score: X/10
**Verdict:** <one-line summary>
```

### 2.4 Self-Correcting PR Guide

For every `❌ BLOCK` issue found, the review must include a `Suggested
Fix` block in this format:

```
### ❌ BLOCK — <issue title>
**File:** `<path>:<line>`
**Problem:** <one sentence>
**Suggested fix:**
```python
# BEFORE (blocked)
<offending snippet>

# AFTER (corrected)
<corrected snippet>
```
```

### 2.5 MCP-Grounded Reviews

If an MCP server is available, the review must be grounded in real
execution data rather than static analysis alone:

1. Call `env_versions` → verify iMinuit ≥ 2, Pydantic ≥ 2, Numba ≥ 0.60.
2. Call `ads_list_workspaces` → confirm the workspace lifecycle is clean
   after the pipeline completes.
3. Call `log_read_latest` (from `log_inspector_server`) → check
   `optimizer_agreement_check.overall_gate_passed` in the most recent
   run log.  If it is `False`, escalate the agreement-check items to
   `❌ BLOCK` regardless of the diff content.
4. Call `log_grep` → search for `np.trapz` in recently executed log
   paths to catch deprecated-API use at runtime.

If MCP is unavailable, note **"MCP unavailable — review based on static
analysis only"** at the top of the review.

---

## TIER 3 — Terminology and Auto-Gating

### Banned terminology

- Do **not** use the word **"Sieve"** anywhere in code, comments, or
  log keys.  Use the precise description instead:
  - Code: `optimizer_agreement_check`
  - Prose: "iMinuit–Scipy Numerical Agreement Check"
  - Test class: `TestOptimizerAgreementCheck`
  - Warning prefix: `OptimizerCheck`

### Auto-Gating thresholds (from ARCHITECTURE_AUDIT.md)

The `ci-dev.yml` workflow enforces these gates on every PR to `dev`:

| Gate | Threshold | Test file |
|---|---|---|
| NumPy ↔ Numba regression | `atol=1e-8` | `test_numba_regression.py` |
| iMinuit–Scipy agreement | 1 % | `test_iminuit_cross_check.py` |
| Workspace lifecycle safety | no uncaught `KeyError` / `AttributeError` | `test_workspace_safety.py` |

---

## CI workflow

The `ci-dev.yml` workflow runs on every PR to `dev`:

1. Install with `pip install -e ".[test]"` (dependencies declared in `pyproject.toml`)
2. Numba regression tests
3. iMinuit–Scipy Numerical Agreement Check (`optimizer_agreement_check`)
4. Workspace safety tests (mock ADS — no Mantid required)

Permissions: `contents: read` + `pull-requests: write`.
