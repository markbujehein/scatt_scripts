# Agentic Development Environment — VESUVIO Pipeline

> **Date:** 2026-02-19  
> **Branch:** `dev`  
> **Status:** Architectural proposal + implementation

---

## Executive Summary

This document provides the architectural specification for transforming
the VESUVIO analysis pipeline into an **AI-augmented research tool**
where LLM agents can read instrument state, validate physics invariants,
and self-correct common runtime errors — all without manual intervention.

The architecture is organised into three tiers:

| Tier | Concern | Artefact |
|---|---|---|
| 1 — Context | What the LLM *knows* about the codebase | `.github/copilot-instructions.md` |
| 2 — Tools | What the LLM can *do* with the live environment | `vesuvio_analysis/mcp_server/` |
| 3 — Automation | What the CI system *gates* automatically | `.github/workflows/ci-dev.yml` |

---

## Phase 1 — Environmental Audit

### 1.1 Codebase Structure

The pipeline entry point is a user submission script (e.g., `BaH2_500C.py`)
that defines parameter classes and dispatches to `runScript()` in
`vesuvio_analysis/run_script.py`.  The complete logical flow is documented
in `ARCHITECTURE_AUDIT.md`.

Key friction points for AI code generation (addressed by Tier 1):

| Friction Point | Risk | Tier 1 Rule |
|---|---|---|
| Custom physical constants (`mN=1.008`, `hbar=2.0445`) | AI substitutes SI values | Explicit constant table in `.github/copilot-instructions.md` |
| `ANSIO_GAUSSIAN` spelling | AI "corrects" to `ANISO_GAUSSIAN`, breaking `fitModel` dispatch | Banned-correction rule in copilot instructions |
| iMinuit v1 vs v2 API | AI generates deprecated `@cost.UnbinnedNLL` or `param_names=` syntax | `_parameters` dict requirement enforced |
| Pydantic v1 vs v2 | AI uses `@validator` (banned) | `ConfigDict` + `model_validator` requirement |
| Mantid inside `@njit` | Numba cannot JIT-compile Mantid calls | Explicit rule: no Mantid imports in `numba_routines.py` |

### 1.2 Custom Instructions Optimisation

The `.github/copilot-instructions.md` file enforces the following invariants
automatically via GitHub Copilot's workspace-level context injection:

**Numerical Truth invariants (from ARCHITECTURE_AUDIT.md):**
- y-scaling formula `J(y) = M/(ħq)·(E−E_recoil)` must not be simplified.
- `Ef = 4906 meV` (gold-foil analyser) is a fixed calibration constant.
- `LoadVesuvio` is the mandatory first step; no fitting precedes it.

**Mantid Lifecycle invariants (Phase 4):**
- Workspaces enter/exit ADS under `{scriptName}_{DIRECTION}_{iteration}` keys.
- `NCPCostFunction.__call__` must never import Mantid algorithms.
- `mtd.clear()` is reserved for teardown; it must not appear in fitting loops.

### 1.3 MCP Server Opportunity

**Feasibility Assessment: MCP + Mantid**

The Mantid C++ backend communicates with Python through Mantid's own
`MantidIPython` kernel or via a standard `import mantid` in a Mantid-aware
Python environment.  An MCP server can interface with Mantid by running
*inside* the same Mantid Python process and exporting selected state as
JSON over stdio.

Feasibility rating: **High (with caveats)**

| Capability | Feasibility | Implementation |
|---|---|---|
| Query ADS workspace names | ✅ High | `mtd.getObjectNames()` |
| Query workspace Y-data statistics | ✅ High | `ws.extractY()` → NumPy |
| Query Pixi/Conda env metadata | ✅ High | `importlib.metadata` |
| Call Mantid algorithms from the agent | ⚠️ Medium | Requires Mantid process alive |
| Real-time memory profiling | ⚠️ Medium | `tracemalloc` + MCP resource |
| Scientific paper database (Mantid Technique Papers) | ❌ Blocked | External access required |

**Caveats:**
- The MCP server must run inside the Mantid Python interpreter
  (not a separate process) because Mantid's C++ ADS is not
  accessible across process boundaries.
- In CI (no Mantid), the `mantid_ads_server.py` degrades gracefully
  and returns informative error strings.

### 1.4 Developer Environment Optimisations

For real-time performance visibility during VESUVIO runs:

**VS Code / Cursor extensions:**
- `ms-python.python` with `Python: Profile Current File` for line-level timing.
- `njpwerner.autodocstring` — enforces docstring discipline on new functions.
- `GitHub.copilot` with workspace instructions file active.

**Shell aliases (add to `.bashrc` / `.zshrc`):**
```bash
# Run the full Numba + iMinuit CI gate locally
alias vesuvio-ci="python -m pytest tests/test_numba_regression.py tests/test_iminuit_cross_check.py tests/test_workspace_safety.py -v"

# Start the Mantid ADS MCP server (requires Mantid Python)
alias vesuvio-mcp-ads="python -m vesuvio_analysis.mcp_server.mantid_ads_server"

# Start the environment MCP server
alias vesuvio-mcp-env="python -m vesuvio_analysis.mcp_server.environment_server"

# Profile a single VESUVIO run (requires Python 3.10+)
alias vesuvio-profile="python -m cProfile -s cumulative"
```

**`tracemalloc` integration** for memory profiling during bootstrap runs:
```python
import tracemalloc
tracemalloc.start()
runScript(...)
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics("lineno")[:10]:
    print(stat)
```

---

## Phase 2 — Architectural Proposal

### Tier 1 — Context (Project-Specific Rules)

File: `.github/copilot-instructions.md`

Rules are grouped into three subsections:

1. **Physics invariants** — constants, formulas, and the LoadVesuvio gateway.
2. **Mantid lifecycle invariants** — ADS naming, workspace guards, teardown.
3. **Library-specific rules** — Numba `@njit`, iMinuit `_parameters`, Pydantic v2.

See the full file for the complete rule set.

### Tier 2 — Tools (MCP Servers)

Three MCP servers are provided in `vesuvio_analysis/mcp_server/`:

#### `mantid_ads_server.py` — Mantid ADS State

Tools exposed:

| Tool | Input | Output | Use case |
|---|---|---|---|
| `ads_list_workspaces` | — | `{workspace_names, count}` | Inspect pipeline state between iterations |
| `ads_workspace_shape` | `name: str` | `{n_spectra, n_bins}` | Verify data dimensions post-LoadVesuvio |
| `ads_workspace_stats` | `name: str` | `{y_min, y_max, y_mean, y_std}` | Sanity-check Y-values after each reduction step |
| `ads_is_empty` | — | `{is_empty, count}` | Post-teardown verification |

#### `environment_server.py` — Pixi/Conda Environment Metadata

Tools exposed:

| Tool | Input | Output | Use case |
|---|---|---|---|
| `env_versions` | — | `{python_version, platform, packages}` | Diagnose import errors, version mismatches |
| `git_commit` | — | `{commit, branch}` | Trace failing run to exact code revision |
| `check_version_compatibility` | — | `{compatible, issues}` | Self-correcting agent: detect iMinuit v1, Pydantic v1, NumPy 2 + old Numba |

#### `log_inspector_server.py` — Run Log Inspector (PR Review Grounding)

Exposes the YAML-formatted run logs written by `RunLogger` as MCP tools.
Enables review agents to ground code reviews in **real execution data**.

Tools exposed:

| Tool | Input | Output | Use case |
|---|---|---|---|
| `log_list` | `root?: str` | `{count, log_files}` | Discover available run logs |
| `log_read_latest` | `root?: str` | `{content, path}` | Read the most recent run's full log |
| `log_read` | `path: str` | `{content, size_bytes}` | Read a specific log file |
| `log_grep` | `pattern: str, root?: str` | `{matches}` | Search for `np.trapz` or `overall_gate_passed` in all logs |
| `log_check_agreement` | `root?: str` | `{overall_gate_passed, chi2_gate_passed, par_gate_passed, threshold}` | Extract the `optimizer_agreement_check` block — primary review-grounding tool |

**Usage in a review session:**
```bash
# Check if the last run passed the optimizer agreement gate
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"log_check_agreement","arguments":{}}}' \
  | python -m vesuvio_analysis.mcp_server.log_inspector_server

# Search for deprecated np.trapz in all run logs
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"log_grep","arguments":{"pattern":"np\\.trapz"}}}' \
  | python -m vesuvio_analysis.mcp_server.log_inspector_server
```

**Claude Desktop configuration** (add to `~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "vesuvio-ads": {
      "command": "mantid-python",
      "args": ["-m", "vesuvio_analysis.mcp_server.mantid_ads_server"]
    },
    "vesuvio-env": {
      "command": "python",
      "args": ["-m", "vesuvio_analysis.mcp_server.environment_server"]
    },
    "vesuvio-logs": {
      "command": "python",
      "args": ["-m", "vesuvio_analysis.mcp_server.log_inspector_server"]
    }
  }
}
```

### Tier 3 — Automation (Auto-Gating PRs)

File: `.github/workflows/ci-dev.yml`

Every PR targeting `dev` must pass three gates before merge:

| Gate | Threshold | Test file | Failure action |
|---|---|---|---|
| NumPy ↔ Numba regression | `atol=1e-8` | `test_numba_regression.py` | Block merge, annotate PR |
| iMinuit–Scipy Numerical Agreement | 1 % | `test_iminuit_cross_check.py` (class `TestOptimizerAgreementCheck`) | Block merge, annotate PR |
| Mantid workspace lifecycle safety | No uncaught `KeyError`/`AttributeError` | `test_workspace_safety.py` | Block merge, annotate PR |

**Thresholds are sourced from `ARCHITECTURE_AUDIT.md`** and hardcoded in
the respective test files via `_AGREEMENT_THRESHOLD = 0.01`.  Changing
the threshold requires a PR that updates both the test file and this document.

---

## Phase 2 — Step 3: Self-Correcting Agent Roadmap

### Common runtime errors and MCP-driven fixes

#### `AttributeError: 'Minuit' object has no attribute 'migrad'` (iMinuit v1 API)

1. Agent calls `env_versions` → sees `iminuit: 1.x`.
2. Agent calls `check_version_compatibility` → gets issue string.
3. Agent proposes fix: `pip install "iminuit>=2"` and updates `pyproject.toml`.

#### `RuntimeError: y0 mismatch` (West-variable coordinate error)

1. Agent calls `ads_workspace_shape` before and after `fitInYSpaceProcedure`.
2. Agent detects shape mismatch → flags coordinate transformation error.
3. Agent checks `rebinParametersForYSpaceFit` against `y0` in the log YAML
   under `optimizer_agreement_check:` → identifies the misaligned bin edges.
4. Agent proposes fix: re-check `symmetrisationFlag` and rebin parameters.

#### `ValidationError` from `ic_validation.py` (Pydantic v2 schema)

1. Agent calls `env_versions` → sees `pydantic: 1.x`.
2. Agent calls `check_version_compatibility` → gets Pydantic issue string.
3. Agent proposes fix: `pip install "pydantic>=2"`.

#### Numba JIT cache stale after NumPy upgrade

1. Agent calls `check_version_compatibility` → detects NumPy 2.x + Numba < 0.60.
2. Agent proposes: `pip install "numba>=0.60"` and clears `__pycache__`.

### Long-term roadmap

| Phase | Goal | Mechanism |
|---|---|---|
| **Now** | Static code-generation rules | `.github/copilot-instructions.md` |
| **Now** | Environment introspection | `environment_server.py` |
| **Now** | ADS state inspection | `mantid_ads_server.py` |
| **Now** | Log-file parser MCP server | Implemented in `vesuvio_analysis/mcp_server/log_inspector_server.py`; parses `log_manager.py` YAML output → exposes `optimizer_agreement_check` results as MCP resource |
| **Near-term** | Automated chi² trend alerts | CI job that plots `chi2_gate_passed` across iterations; fails if diverging |
| **Long-term** | Interactive Mantid session server | Full MCP server running inside Mantid's embedded Python kernel, exposing `AlgorithmManager` |
| **Long-term** | Literature MCP server | Index Mantid Technique Papers + ISIS instrument papers as MCP resources |

---

## References

- MCP specification: https://modelcontextprotocol.io/
- Mantid AnalysisDataService: https://docs.mantidproject.org/nightly/concepts/AnalysisDataService.html
- iMinuit v2 documentation: https://iminuit.readthedocs.io/en/stable/
- `ARCHITECTURE_AUDIT.md` — data-flow map, Numerical Truth thresholds
