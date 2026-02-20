# Project Copilot Instructions — vesuvio_analysis (Repository-only)

This repository-only instructions file centralises agent-facing directives used for isolated, repo-scoped agent coding. It is intended to live inside `./scatt_scripts/.github/` and be the single source of truth for assistant behaviour when operating solely on this package.

---

## Purpose and Scope
- Scope: `vesuvio_analysis` package and tests under `scatt_scripts` only.
- Do not surface or reference user-private VS Code/global Copilot settings held at the project root; those are intentionally private and outside this repo-only instructions file.

## Core Operational Directives
- Follow PEP 8 and Python >= 3.10 type-hints.
- Preserve intentionally misspelled identifiers: `ANSIO_GAUSSIAN` must not be auto-corrected.
- Use `@njit`-only Numba routines; avoid object-mode fallbacks.
- Do not move or upload experimental `.npz` or data files.

## Full-Stack Statistical Check (Local + CI)
1. Local developer check sequence (run in repo root):

```bash
# create isolated env (pixi if available) or use current venv
python -m pip install -e ./scatt_scripts[ test ]
python -m pytest tests/ -q
```

2. CI recipe (summary):
- Run `pip install -e '.[test]'` as in `ci-dev.yml`.
- Execute Numba regression test with `atol=1e-8`.
- Run iMinuit–Scipy numerical agreement tests and optimizer cross-validation.

## Typst-aware Drafting Rules (for `thesis_draft` contributors)
- When producing Typst snippets, avoid exposing raw file paths to experimental data.
- Prefer short paragraphs and use Typst markup for figures and tables.
- Provide metadata (title, authors, affiliation) in frontmatter templates.

## Constants Table (canonical)
- `Ef = 4906 meV` (gold-foil analyser)
- `mN = 1.008` (use explicit values when writing physics equations)

## Testing & Terms
- Tests must be runnable without Mantid via `tests/mock_mantid.py`.
- Use `unittest.TestCase` and run with `python -m pytest tests/ -v`.

## MCP Servers (Repository guidance)
- Repository-level plugin guidance only. Do not modify user/global `mcp.json` from here.
- Recommended names and intents:
  - `playwright-researcher`: Browser automation for Mantid v6.14 docs research.
  - `zotero-library`: Local Zotero access (read-only) — path: `/home/mark/Zotero` (user-managed).
  - `thesis-files`: Whitelist `scatt_scripts` and `thesis_draft` for file-scoped agents.
  - `system-monitor`: Resource monitoring for Global Fits (advisory only).

Provide example `mcp.json` snippets in `scatt_scripts/.github/claude-skills.md` or in the repo README — do not apply to global user config automatically.

## Agent Interaction Safety
- Ask before writing to `/home/mark/.config/` or other global locations.
- Never upload private data or external datasets.

---

Created as the repo-scoped instructions for isolated agentic coding. For workspace-level (private) Copilot and VS Code settings, see the project root `.copilot/` and `.vscode/` directories (not tracked publicly if you prefer).

## Repository Invariants & Enforcement (required)

- iMinuit cost functions MUST expose an `_parameters` dict and `errordef = Minuit.LEAST_SQUARES`. Do not rely on legacy `func_code` generation.
- Use Pydantic v2 APIs (`ConfigDict`, `model_validator`, `field_validator`) for IC validation. Avoid Pydantic v1 `@validator` usage.
- Numba-accelerated code must live in `vesuvio_analysis/core_functions/numba_routines.py` and must not import Mantid or any non-NumPy/Python APIs.
- The first Mantid algorithm to produce TOF workspaces should be `LoadVesuvio` (or equivalent). Respect workspace naming: `{scriptName}_{DIRECTION}_{iteration}`.
- Maintain the SciPy ↔ iMinuit agreement check: implement `_AGREEMENT_THRESHOLD = 0.01` for chi² and parameter relative differences when both optimizers are run.
- MCP servers and tools MUST be placed under `vesuvio_analysis/mcp_server/` inside this repo (see `AGENTIC_ENVIRONMENT.md`). Agents should not attempt to start global MCP servers or modify user-level configs.

## Where to Add Rules

- If you need additional assistant directives, add them here and cite `ARCHITECTURE_AUDIT.md` for the rationale. All substantive changes must be proposed as a PR in the `scatt_scripts` repository.

---

## Annex — Project Overview, Structure & CI (merged from `copilot-instructions.md`)

### Project overview
- `vesuvio_analysis` is a Python package for Deep Inelastic Neutron Scattering (DINS) data reduction and analysis at the ISIS Neutron and Muon Source. It extends the Mantid framework and is intended as a publishable research tool.

### Repository structure (high-level)
```
vesuvio_analysis/core_functions/   ← all physics and fitting logic
tests/                             ← unittest-based test suite (no Mantid needed)
.github/workflows/ci-dev.yml        ← CI for PRs targeting dev
pyproject.toml                     ← build + test config
```

Key modules (high level): `analysis_functions.py`, `numba_routines.py`, `iminuit_costs.py`, `fit_in_yspace.py`, `procedures.py`, `log_manager.py`, `ic_validation.py`.

### Key conventions (summary)
- Physics: y-scaling (West variable), pseudo-Voigt NCP, `Ef = 4906 meV`.
- Code style: Python >= 3.10, PEP8, full type hints, Pydantic v2 for ICs.
- Numba: `@njit` only for numeric kernels; Mantid API calls must remain outside Numba code.
- Testing: tests run without Mantid via `tests/mock_mantid.py`; use `unittest.TestCase` and `pytest`.

### CI workflow (summary)
1. Install with `pip install -e ".[test]"` (dependencies in `pyproject.toml`).
2. Run Numba regression tests (`test_numba_regression.py`).
3. Run iMinuit–SciPy numerical agreement tests (`test_iminuit_cross_check.py`).
4. Run workspace safety tests (mock ADS).

Notes: CodeQL and other advanced gates can run after these numeric checks pass; see `ARCHITECTURE_AUDIT.md` for thresholds and gating logic.

---

(End of merged annex.)

