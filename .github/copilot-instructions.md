# Copilot Chat Workspace Instructions

This document exists to give the ChatGPT/VS Code Copilot‑Chat agent the context
it needs to be instantly helpful in the `scatt_scripts` repository.  Most of the
heavy‑lifting lives in `scatt_scripts/AGENTS.md`; that file is the single source
of truth for workspace‑specific environment details, invariants, and
restrictions.  Read it first and refer back frequently.

---

## 🧭 Project overview

`scatt_scripts` is a Python package implementing a high‑performance analysis
pipeline for the VESUVIO neutron spectrometer.  It extends the Mantid project
with Numba‑accelerated fitting kernels, dual‑optimizer validation, and a full
statistical workflow.  The codebase is organised as a normal Python project
(`pyproject.toml`, `tests/`) with one sibling workspace (`../thesis_draft`) for
the accompanying dissertation.

Key facts:

- Entry‑point scripts (e.g. `starch_80_RD.py`) set up nine Pydantic v2
  parameter classes and call `runScript()` defined in
  `vesuvio_analysis/core_functions/run_script.py`.
- Core computation lives under `vesuvio_analysis/core_functions/`; tight
  loops are JIT‑compiled in `numba_routines.py` and validated against standard
  NumPy paths (`test_numba_regression.py`).
- MCP servers for agent interaction live in `vesuvio_analysis/mcp_server/`.
  These must **never** modify global MCP configuration and run strictly inside
  the workspace environment.
- Experimental data (Nexus, `.npz`, etc.) is confidential and should never be
  externalised.  Caching behaviour is controlled by JSON log files under
  `experiments/` (search for `LoadVesuvio` in the README for details).

---

## ⚙️ Common commands

```bash
# install package & tests (no Mantid required)
pip install -e ".[test]"
python -m pytest tests/ -v       # run full test suite

# optional full Mantid setup:
conda create -n vesuvio python=3.11
conda activate vesuvio
conda install -c mantid mantid
pip install -e ".[dev]"
```

Pixi users may simply run `pixi run python` from the repo root – the
`pyproject.toml` already contains the necessary configuration.

Dev container helpers are in `.devcontainer/` and the CI workflows in
`.github/workflows`.

---

## 🗂 Structure & navigation

| Path | Purpose |
|------|---------|
| `vesuvio_analysis/core_functions/` | pipeline logic, cost functions, etc. |
| `vesuvio_analysis/numba_routines.py` | Numba kernels – must not import Mantid |
| `vesuvio_analysis/mcp_server/` | JSON‑RPC servers for LLM agents |
| `tests/` | pytest suite (independent of Mantid) |
| `experiments/` | runtime data and caches (git‑ignored) |

For quick exploration, run the **Explore** subagent with queries such as
"architecture overview" or "Numba regression test".

---

## 📐 Conventions & invariants

Detailed rules are enumerated in `scatt_scripts/AGENTS.md`.  Highlights:

1. **Workspace naming:** `<script>_<DIRECTION>_<iteration>`; only cleared by
   `AnalysisDataService.clear()` in teardown code.
2. **Pydantic v2 models** for initial conditions; use `ConfigDict` and
   `model_validator` (no v1 APIs).
3. **Optimizer agreement gate:** `_AGREEMENT_THRESHOLD = 0.01` in
   `analysis_functions.py` → iMinuit vs SciPy comparison.
4. **Statistical plugins** live in `statistical_plugins.py` and are guarded by
   boolean flags in `UserScriptControls`.
5. **MCP servers** must remain under `vesuvio_analysis/mcp_server/` and not
   touch global MCP config.  `mcp.json` in `.vscode/` describes the running
   endpoints.
6. **Data sovereignty:** absolutely no externalisation of `.nxs`, `.npz`, or
   `.h5` files.  Use utility `thesis-files` MCP server for controlled file
   access.

Any change to these invariants requires a PR referencing
`ARCHITECTURE_AUDIT.md` or `AGENTIC_ENVIRONMENT.md`.

---

## 🧪 Testing strategy

The test suite is run with `pytest` and covers:

- Numba regression (`test_numba_regression.py`)
- iMinuit/Scipy agreement & cost‑function interface (`test_iminuit_cross_check.py`)
- Workspace safety & naming conventions (`test_workspace_safety.py`)
- Statistical workflow plugins (`test_statistical_workflow.py`)
- Console/UX helpers and MCP availability

Legacy Mantid‑dependent tests live under `___vesuvio_analysis/tests/` and are
invoked with `python run_tests.py` when a Mantid installation is present.

During development, the CI gate (`ci-dev.yml`) runs the same suites and adds
CodeQL.

---

## 🧠 Using the agent

- **Start with `AGENTS.md`.**  It contains the dev environment tips that the
  agent must obey (Pixi, data confidentiality, invariants, etc.).
- **When exploring code,** prefer the `Explore` subagent to gather information
  quickly.  Example prompt:

  > "Run a medium thorough exploration for `analysis_functions` and summarize
  > the parameters it exposes."

- **For modifications** that touch multiple modules or require tests, sketch a
  todo list and update both code and tests accordingly.  Use the
  `manage_todo_list` tool to track progress if the task spans more than three
  steps.

- **To run shell commands,** use `run_in_terminal` with a clear explanation.
  Avoid running multiple background processes simultaneously unless you need
  them (e.g. start an MCP server then check its ports).

- **Formatting & presentation**: follow the repository’s markdown style and
  the thesis drafting rules (Typst math, SI units).  When generating textual
  content for the thesis draft, the `thesis-files` MCP tool can read/write
  under `../thesis_draft` safely.

---

## ✅ Example prompts

1. **"Help me add a new `DOUBLE_WELL` model option for y‑space fitting.  Where
should I make changes and what tests are required?"**
2. **"Why does `test_workspace_safety.py` mock the Mantid ADS, and can I
configure a similar fixture for a new algorithm?"**
3. **"Create a new MCP server that returns the number of spectra in a given
cache directory"**

---

With this file in place, Copilot Chat should have a strong starting point for
working in the repository.  Refer back to it whenever you’re uncertain about
project conventions or tooling.