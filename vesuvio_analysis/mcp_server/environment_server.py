"""MCP server: Pixi/Conda environment metadata inspector.

Provides a JSON-over-stdio MCP server that exposes the current Python
environment's dependency versions and platform information as MCP
resources and tools.  This enables a self-correcting LLM agent to:

1. Query the installed versions of all VESUVIO dependencies.
2. Detect version mismatches that would cause ``AttributeError`` /
   ``RuntimeError`` at runtime (e.g., iMinuit v1 vs v2 API differences,
   Pydantic v1 vs v2, Numba JIT cache invalidation).
3. Surface the git commit hash so that log entries can be traced back
   to the exact code revision.
4. Report platform information for ISIS cluster vs local debugging.

Architecture note
-----------------
This module has **no non-stdlib dependencies** other than ``importlib``
and ``subprocess``, so it can always be imported successfully —
even before the VESUVIO pip dependencies are installed.

Standalone execution
--------------------
::

    python -m vesuvio_analysis.mcp_server.environment_server

Protocol
--------
Implements the MCP ``tools/list`` → ``tools/call`` lifecycle over
JSON-RPC 2.0 / stdio transport.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Environment introspection helpers
# ---------------------------------------------------------------------------

#: Packages that are critical to VESUVIO physics correctness.
_VESUVIO_DEPS: tuple[str, ...] = (
    "numpy",
    "scipy",
    "iminuit",
    "numba",
    "pydantic",
    "pyyaml",
    "matplotlib",
    "jacobi",
    "scikit-learn",
)


def _get_package_version(package: str) -> str:
    """Return the installed version of *package*, or ``'not installed'``."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _env_versions() -> dict[str, Any]:
    """Return a dict of all VESUVIO dependency versions."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {pkg: _get_package_version(pkg) for pkg in _VESUVIO_DEPS},
    }


def _git_commit() -> dict[str, Any]:
    """Return the current git commit hash and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"commit": commit, "branch": branch}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown"}


def _check_version_compatibility() -> dict[str, Any]:
    """Check for known incompatible version combinations.

    Returns a list of issues found, empty if everything is compatible.
    """
    issues: list[str] = []

    iminuit_ver = _get_package_version("iminuit")
    if iminuit_ver != "not installed":
        major = int(iminuit_ver.split(".")[0])
        if major < 2:
            issues.append(
                f"iminuit {iminuit_ver} detected — v2+ required for "
                "'_parameters' dict interface (NCPCostFunction will fail)."
            )

    pydantic_ver = _get_package_version("pydantic")
    if pydantic_ver != "not installed":
        major = int(pydantic_ver.split(".")[0])
        if major < 2:
            issues.append(
                f"pydantic {pydantic_ver} detected — v2+ required for "
                "ConfigDict and model_validator (ic_validation.py will fail)."
            )

    numpy_ver = _get_package_version("numpy")
    numba_ver = _get_package_version("numba")
    if numpy_ver != "not installed" and numba_ver != "not installed":
        np_major, np_minor = (int(x) for x in numpy_ver.split(".")[:2])
        if np_major >= 2:
            numba_major = int(numba_ver.split(".")[0])
            if numba_major == 0:
                numba_minor = int(numba_ver.split(".")[1])
                if numba_minor < 60:
                    issues.append(
                        f"NumPy {numpy_ver} with Numba {numba_ver} — "
                        "NumPy 2.x requires Numba ≥ 0.60."
                    )

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "checked_versions": {
            "iminuit": iminuit_ver,
            "pydantic": pydantic_ver,
            "numpy": numpy_ver,
            "numba": numba_ver,
        },
    }


# ---------------------------------------------------------------------------
# MCP tool registry
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "env_versions",
        "description": (
            "Return the installed versions of all VESUVIO pipeline dependencies "
            "(numpy, scipy, iminuit, numba, pydantic, etc.) together with the "
            "Python version and platform string.  Use this to diagnose "
            "AttributeError or ImportError failures."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "git_commit",
        "description": (
            "Return the current git commit hash and branch name.  Use this to "
            "trace a failing run back to the exact code revision."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_version_compatibility",
        "description": (
            "Check for known incompatible dependency version combinations "
            "(e.g., iminuit v1, pydantic v1, NumPy 2 + old Numba).  Returns "
            "a list of issues if any are found, or an empty list if the "
            "environment is compatible."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]

_TOOL_HANDLERS: dict[str, Any] = {
    "env_versions": lambda _args: _env_versions(),
    "git_commit": lambda _args: _git_commit(),
    "check_version_compatibility": lambda _args: _check_version_compatibility(),
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 / MCP framing (shared with mantid_ads_server pattern)
# ---------------------------------------------------------------------------

def _send(obj: dict[str, Any]) -> None:
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _error_response(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "vesuvio-environment",
                    "version": "0.1.0",
                },
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": _TOOLS}}

    if method == "tools/call":
        tool_name: str = params.get("name", "")
        tool_args: dict[str, Any] = params.get("arguments", {})
        handler = _TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return _error_response(req_id, -32601, f"Unknown tool: {tool_name}")
        result = handler(tool_args)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "isError": isinstance(result, dict) and "error" in result,
            },
        }

    if method == "notifications/initialized":
        return None

    return _error_response(req_id, -32601, f"Method not found: {method}")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve() -> None:
    """Run the MCP server, reading JSON-RPC messages from stdin."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _send(_error_response(None, -32700, f"Parse error: {exc}"))
            continue
        response = _handle_request(req)
        if response is not None:
            _send(response)


if __name__ == "__main__":
    serve()
