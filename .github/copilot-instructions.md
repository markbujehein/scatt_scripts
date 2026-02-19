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
vesuvio_analysis/mcp_server/       ← MCP servers for Mantid ADS + environment
tests/                             ← unittest-based test suite (no Mantid needed)
tests/mock_mantid.py               ← Mantid mock (install() patches sys.modules)
.github/workflows/ci-dev.yml       ← CI for PRs targeting dev
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
