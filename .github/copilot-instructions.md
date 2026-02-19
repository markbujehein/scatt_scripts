# GitHub Copilot Instructions — vesuvio_analysis

This file provides project-specific context to GitHub Copilot so that
AI-generated suggestions are immediately relevant to the VESUVIO DINS
analysis codebase.

---

## Project overview

`vesuvio_analysis` is a Python package for Deep Inelastic Neutron
Scattering (DINS) data reduction and analysis at the ISIS Neutron and
Muon Source.  It extends the [Mantid Project](https://www.mantidproject.org/)
framework and is intended as a publishable research tool.

---

## Repository structure

```
vesuvio_analysis/core_functions/   ← all physics and fitting logic
tests/                             ← unittest-based test suite (no Mantid needed)
.github/workflows/ci-dev.yml      ← CI for PRs targeting dev
pyproject.toml                     ← hatchling build + hatch envs + pytest config
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

---

## Key conventions

### Physics
- **y-scaling**: West scaling variable  
  J(y) = M / (ℏ q) · (E − E_recoil)
- **NCP line shape**: pseudo-Voigt under the impulse approximation
- **Final state effects**: numerical third derivative of J(y)
- `Ef = 4906 meV` (gold-foil analyser), back-scattering geometry

### Code style
- Python ≥ 3.10, PEP 8, full type hints
- `pydantic>=2.0` for IC validation (`ConfigDict`, `model_validator`)
- Fit model identifier: `ANSIO_GAUSSIAN` (intentional spelling — do **not** correct)
- `iMinuit` cost functions: class with `_parameters` dict + `__call__`
- Numba routines: `@njit` only; no object-mode fallback

### Testing
- All tests in `tests/` run **without** Mantid via `tests/mock_mantid.py`
- Use `unittest.TestCase`; add `if __name__ == "__main__": unittest.main(verbosity=2)`
- Run with: `python -m pytest tests/ -v`
- Numerical tolerance for NumPy ↔ Numba regression: `atol=1e-8`
- Optimizer cross-validation tolerance: 1% (see `_AGREEMENT_THRESHOLD`)

### Terminology
- Do **not** use the word "Sieve" — describe what the check actually does
  (e.g. "iMinuit–Scipy Numerical Agreement Check", "optimizer cross-validation")
- Log YAML keys use snake_case (e.g. `optimizer_agreement_check:`)

---

## CI workflow

The `ci-dev.yml` workflow runs on every PR to `dev`:

1. Install with `pip install -e ".[test]"` (dependencies declared in `pyproject.toml`)
2. Numba regression tests
3. iMinuit–Scipy numerical agreement tests
4. Workspace safety tests (mock ADS)

Permissions: `contents: read` + `pull-requests: write` (Copilot PR annotations enabled).
