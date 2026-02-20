# Thesis AGENTS.md

## Dev environment tips
- The environment is managed by Pixi. Assume execution via `pixi run python` from the workspace root.
- Maintain absolute Data Sovereignty: NEVER externalize, upload, or transmit `.npz`, `.h5`, or `.nxs` files. Experimental TOSCA/VESUVIO data is strictly confidential.
- The `StreamManager` must respect `DataLevel` (L0-L3) serialization to prevent redundant physics recalculations.
- Ensure MCP servers and tools are placed strictly under `vesuvio_analysis/mcp_server/`.

## Testing, Statistics, and Invariants
- Any changes to fitting kernels MUST pass the Full-Stack Statistical Checks before merging to `dev`.
- **Numba Regression:** Verify 1e-8 agreement between standard NumPy and JIT-compiled paths (`@njit` only; no object-mode).
- **Optimizer Agreement:** Implement `_AGREEMENT_THRESHOLD = 0.01` for chi² and parameter relative differences between `iMinuit` and `SciPy` engines.
- **iMinuit Constraints:** Cost functions MUST expose an `_parameters` dict and `errordef = Minuit.LEAST_SQUARES`.
- **Validation:** Use Pydantic v2 APIs (`ConfigDict`, `model_validator`) for initial condition validation. Do not use Pydantic v1.
- **Constants:** Use explicit physics constants (`Ef = 4906 meV`, `mN = 1.008`).

## Drafting instructions (Typst)
- The thesis manuscript resides in `../thesis_draft` and uses **Typst**.
- When generating draft content, use Typst math syntax (e.g., `$\sigma$`). LaTeX math is reserved ONLY for Python docstrings.
- Output must be graduate-level, technically rigorous, and formatted for publication. No generic boilerplate text. All physical quantities must be in metric units (SI).

## MCP Tool execution
- `thesis-files`: Whitelisted for `scatt_scripts` and `thesis_draft` only.
- `zotero-library`: Read-only access to `/home/mark/Zotero` for citation verification.
- `playwright-researcher`: For navigating Mantid v6.14 documentation and ISIS/NIST manuals.

---

## Repository enforcement notes (summary of repo invariants)

- Location of canonical instructions: `scatt_scripts/AGENTS.md` is the single source of truth for repository-scoped agent directives.
- Numba-accelerated kernels must live in `vesuvio_analysis/core_functions/numba_routines.py` and must not import Mantid or non-pure-Python APIs.
- Keep workspace naming conventions and Mantid lifecycle invariants: `{scriptName}_{DIRECTION}_{iteration}` and only call `AnalysisDataService.clear()` in teardown code.
- MCP servers and agent tooling must be placed under `vesuvio_analysis/mcp_server/` and must not try to alter user-global MCP configuration files.

## Change policy
- Any modification to these rules must be proposed as a PR on `scatt_scripts` and reference `ARCHITECTURE_AUDIT.md` / `AGENTIC_ENVIRONMENT.md` for justification.
