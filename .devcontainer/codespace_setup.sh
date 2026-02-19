#!/usr/bin/env bash
# =============================================================================
# codespace_setup.sh — one-click VESUVIO Codespace validation
# =============================================================================
#
# Run automatically via devcontainer.json postCreateCommand, or manually:
#   bash .devcontainer/codespace_setup.sh
#
# Exit codes
# ----------
#   0 — all checks passed; environment is ready
#   1 — one or more checks failed; inspect output above the failure line
#
# =============================================================================
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
RESET="\033[0m"

_pass() { echo -e "${GREEN}  ✓ ${1}${RESET}"; }
_fail() { echo -e "${RED}  ✗ ${1}${RESET}"; exit 1; }
_warn() { echo -e "${YELLOW}  ⚠ ${1}${RESET}"; }
_header() { echo -e "\n${BOLD}── ${1} ──${RESET}"; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     VESUVIO DINS Analysis — Codespace Environment Check     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ---------------------------------------------------------------------------
# 1. Python version (from pixi env)
# ---------------------------------------------------------------------------
_header "Python"
PYTHON_VERSION=$(pixi run python --version 2>&1)
echo "  ${PYTHON_VERSION}"
if pixi run python -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'"; then
    _pass "Python version OK"
else
    _fail "Python 3.10+ is required"
fi

# ---------------------------------------------------------------------------
# 2. Verify pixi environment (pixi install already ran in postCreateCommand)
# ---------------------------------------------------------------------------
_header "Package installation"
if pixi run python -c "import vesuvio_analysis" 2>/dev/null; then
    _pass "vesuvio_analysis is importable in the pixi environment"
else
    _warn "vesuvio_analysis not yet importable — running pixi install"
    if pixi install -q; then
        _pass "pixi install succeeded"
    else
        _fail "pixi install failed"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Core scientific dependencies
# ---------------------------------------------------------------------------
_header "Scientific dependencies"
DEPS=(numpy scipy matplotlib numba iminuit jacobi sklearn pydantic yaml)
for dep in "${DEPS[@]}"; do
    if pixi run python -c "import ${dep}" 2>/dev/null; then
        VERSION=$(pixi run python -c "import ${dep}; v=getattr(${dep},'__version__','?'); print(v)" 2>/dev/null || echo "?")
        _pass "${dep} (${VERSION})"
    else
        _fail "${dep} is not importable — run: pixi install"
    fi
done

# ---------------------------------------------------------------------------
# 4. Mantid framework check (optional — not required for the unit-test suite)
# ---------------------------------------------------------------------------
_header "Mantid framework (optional)"
if pixi run python -c "from mantid.simpleapi import *" 2>/dev/null; then
    MANTID_VERSION=$(pixi run python -c "import mantid; print(mantid.__version__)" 2>/dev/null || echo "?")
    _pass "Mantid framework available (${MANTID_VERSION})"
else
    _warn "Mantid framework not found — physics unit tests will run via mock_mantid.py"
    _warn "Mantid is declared in pyproject.toml [tool.pixi.dependencies]; run: pixi install"
fi

# ---------------------------------------------------------------------------
# 5. Matplotlib headless backend
# ---------------------------------------------------------------------------
_header "Matplotlib backend"
BACKEND=$(pixi run python -c "import matplotlib; print(matplotlib.get_backend())" 2>/dev/null || echo "unknown")
echo "  Current backend: ${BACKEND}"
if [[ "${MPLBACKEND:-}" == "Agg" ]]; then
    _pass "MPLBACKEND=Agg is set (headless mode active)"
elif [[ "${BACKEND}" == "Agg" ]]; then
    _pass "Agg backend active (via matplotlibrc or import)"
else
    _warn "Backend is '${BACKEND}' — set MPLBACKEND=Agg for headless Codespace use"
fi

# ---------------------------------------------------------------------------
# 6. VESUVIO_RUNNING_TEST fast-track flag
# ---------------------------------------------------------------------------
_header "runningTest fast-track flag"
if [[ "${VESUVIO_RUNNING_TEST:-0}" == "1" ]]; then
    _pass "VESUVIO_RUNNING_TEST=1 — fast-track mode active"
    echo "    → noOfMSIterations will be capped to 0"
    echo "    → nSamples will be capped to 3 for bootstrap/jackknife"
else
    _warn "VESUVIO_RUNNING_TEST is not set to 1 — production mode"
    _warn "Set VESUVIO_RUNNING_TEST=1 to enable high-frequency code validation"
fi

# ---------------------------------------------------------------------------
# 7. Numba thread configuration
# ---------------------------------------------------------------------------
_header "Numba / BLAS thread configuration"
echo "  NUMBA_NUM_THREADS = ${NUMBA_NUM_THREADS:-<unset>}"
echo "  OMP_NUM_THREADS   = ${OMP_NUM_THREADS:-<unset>}"
echo "  MKL_NUM_THREADS   = ${MKL_NUM_THREADS:-<unset>}"
NCPU=$(pixi run python -c "import os; print(os.cpu_count())" 2>/dev/null || echo "?")
echo "  Available CPUs    = ${NCPU}"
_pass "Thread counts reported (adjust for machine type in devcontainer.json)"

# ---------------------------------------------------------------------------
# 8. CI test suite (without Mantid — uses mock_mantid.py)
# ---------------------------------------------------------------------------
_header "CI test suite"
echo "  Running: pixi run pytest tests/test_numba_regression.py tests/test_iminuit_cross_check.py tests/test_workspace_safety.py"
if pixi run python -m pytest \
        tests/test_numba_regression.py \
        tests/test_iminuit_cross_check.py \
        tests/test_workspace_safety.py \
        -v --tb=short -q 2>&1; then
    _pass "All CI gate tests passed"
else
    _fail "One or more CI gate tests failed — see output above"
fi

# ---------------------------------------------------------------------------
# 9. Port forwarding reminder
# ---------------------------------------------------------------------------
_header "Port forwarding"
echo "  Port 8501 — Streamlit dashboard  (streamlit run <script.py>)"
echo "  Port 8050 — Plotly Dash dashboard (python app.py)"
_pass "Ports 8501 and 8050 are configured for auto-forward in devcontainer.json"

# ---------------------------------------------------------------------------
# 10. Secrets guidance
# ---------------------------------------------------------------------------
_header "Codespace secrets"
echo "  The following secrets can be injected via:"
echo "  GitHub → Settings → Codespaces → Secrets"
echo ""
echo "    ORCID_TOKEN         — automated DOI registration via LogManager"
echo "    ZENODO_ACCESS_TOKEN — dataset deposition"
echo ""
if [[ -n "${ORCID_TOKEN:-}" ]]; then
    _pass "ORCID_TOKEN is set"
else
    _warn "ORCID_TOKEN not found — publication automation will be unavailable"
fi
if [[ -n "${ZENODO_ACCESS_TOKEN:-}" ]]; then
    _pass "ZENODO_ACCESS_TOKEN is set"
else
    _warn "ZENODO_ACCESS_TOKEN not found — Zenodo deposition will be unavailable"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Codespace is ready — start your VESUVIO analysis!         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
