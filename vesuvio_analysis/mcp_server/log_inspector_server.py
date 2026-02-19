"""MCP server: VESUVIO run log inspector.

Provides a JSON-over-stdio MCP server that exposes the YAML-formatted run
logs written by ``vesuvio_analysis.core_functions.log_manager.RunLogger``
as MCP tools.  This allows a review agent to ground its code review in
**real execution data** rather than static analysis alone.

Tools exposed
-------------
log_list
    List all ``.log`` files found recursively under a given directory
    (defaults to the ``experiments/`` directory).

log_read_latest
    Read the content of the most recently modified ``.log`` file under a
    given directory.

log_read
    Read a specific ``.log`` file by path.  Restricted to ``.log`` files
    under the allowlisted log root to prevent accidental exposure of
    unrelated files.

log_grep
    Search all ``.log`` files under a directory for lines matching a
    given pattern (e.g. ``np.trapz``, ``overall_gate_passed``,
    ``optimizer_agreement_check``).

log_check_agreement
    Parse the most recent log file and return the structured
    ``optimizer_agreement_check`` block, with ``overall_gate_passed``
    as a top-level key.  This is the primary tool used by the review
    agent to determine whether the iMinuit–Scipy Numerical Agreement
    Check passed on the last run.

Architecture note
-----------------
This server is **import-safe** and has no external dependencies beyond
the Python standard library.  It can therefore be imported during CI
runs (no Mantid, no pip packages) without error.

Standalone execution
--------------------
::

    python -m vesuvio_analysis.mcp_server.log_inspector_server

Protocol
--------
JSON-RPC 2.0 over stdio, following the MCP 2024-11-05 specification.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default search root (relative to repository root)
# ---------------------------------------------------------------------------

_DEFAULT_LOG_ROOT = Path(__file__).resolve().parents[2] / "experiments"


# ---------------------------------------------------------------------------
# Log file helpers
# ---------------------------------------------------------------------------

def _resolve_root(root: str | None) -> Path:
    """Resolve the search root relative to the repository root, falling back to ``experiments/``."""
    if root:
        p = Path(root)
        if p.is_absolute():
            return p
        # Interpret relative paths as relative to the repository root,
        # which is the parent of the default ``experiments/`` directory.
        repo_root = _DEFAULT_LOG_ROOT.parent
        return (repo_root / p).resolve()
    return _DEFAULT_LOG_ROOT


def _list_logs(root: str | None) -> dict[str, Any]:
    """Return all .log file paths under *root*, sorted newest-first."""
    base = _resolve_root(root)
    if not base.exists():
        return {"error": f"Directory not found: {base}", "log_files": []}
    paths = sorted(
        base.rglob("*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "root": str(base),
        "count": len(paths),
        "log_files": [str(p) for p in paths],
    }


def _read_log(path: str, allow_root: Path | None = None) -> dict[str, Any]:
    """Read a single log file and return its content as a string.

    Restricted to ``.log`` files under *allow_root* (defaults to
    ``_DEFAULT_LOG_ROOT``) to prevent accidental exposure of files
    outside the intended log directory.
    """
    p = Path(path).resolve()
    root = (allow_root or _DEFAULT_LOG_ROOT).resolve()

    # Enforce .log extension
    if p.suffix != ".log":
        return {"error": f"Rejected: only .log files may be read (got '{p.name}')."}

    # Enforce containment within the allowlisted root
    try:
        p.relative_to(root)
    except ValueError:
        return {
            "error": (
                f"Rejected: path '{p}' is outside the allowlisted log "
                f"root '{root}'."
            )
        }

    if not p.exists():
        return {"error": f"File not found: {path}"}
    try:
        content = p.read_text(encoding="utf-8")
        return {"path": str(p), "size_bytes": p.stat().st_size, "content": content}
    except OSError as exc:
        return {"error": str(exc)}


def _read_latest_log(root: str | None) -> dict[str, Any]:
    """Read the most recently modified .log file under *root*."""
    listing = _list_logs(root)
    if "error" in listing or not listing["log_files"]:
        return {"error": "No log files found.", "root": listing.get("root", "")}
    latest = listing["log_files"][0]
    resolved_root = _resolve_root(root)
    result = _read_log(latest, allow_root=resolved_root)
    if "error" not in result:
        result["is_latest"] = True
    return result


def _grep_logs(root: str | None, pattern: str) -> dict[str, Any]:
    """Search all .log files under *root* for lines matching *pattern*.

    Uses Python ``re`` (not ripgrep) so the server stays dependency-free.
    """
    base = _resolve_root(root)
    if not base.exists():
        return {"error": f"Directory not found: {base}", "matches": []}
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"Invalid regex: {exc}", "matches": []}

    matches: list[dict[str, Any]] = []
    for log_path in sorted(base.rglob("*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            if rx.search(line):
                matches.append(
                    {
                        "file": str(log_path),
                        "line": lineno,
                        "text": line.rstrip(),
                    }
                )
    return {
        "pattern": pattern,
        "root": str(base),
        "match_count": len(matches),
        "matches": matches,
    }


def _check_agreement(root: str | None) -> dict[str, Any]:
    """Parse the most recent log and extract the optimizer_agreement_check block.

    Looks for either ``optimizer_agreement_check:`` (current) or
    ``sieve3_agreement_gate:`` (legacy) in the log file and returns the
    structured key-value pairs found in that block.

    Returns
    -------
    dict with keys:
        ``found``: bool — whether the block was found.
        ``log_file``: str — path of the log file read.
        ``overall_gate_passed``: bool | None.
        ``chi2_gate_passed``: bool | None.
        ``par_gate_passed``: bool | None.
        ``threshold``: float | None.
        ``raw_block``: list[str] — raw lines of the block.
        ``legacy_key_detected``: bool — True if old ``sieve3_agreement_gate:`` key found.
    """
    latest = _read_latest_log(root)
    if "error" in latest:
        return latest

    content: str = latest.get("content", "")
    log_file: str = latest.get("path", "")

    # Support both current and legacy key names
    _CURRENT_KEY = "optimizer_agreement_check:"
    _LEGACY_KEY = "sieve3_agreement_gate:"

    legacy_detected = _LEGACY_KEY in content
    start_key = _CURRENT_KEY if _CURRENT_KEY in content else (
        _LEGACY_KEY if _LEGACY_KEY in content else None
    )

    if start_key is None:
        return {
            "found": False,
            "log_file": log_file,
            "overall_gate_passed": None,
            "chi2_gate_passed": None,
            "par_gate_passed": None,
            "threshold": None,
            "raw_block": [],
            "legacy_key_detected": legacy_detected,
            "warning": (
                "Neither 'optimizer_agreement_check:' nor 'sieve3_agreement_gate:' "
                "found in the log.  The optimizer agreement check may not have run."
            ),
        }

    # Extract the indented block that follows the key
    lines = content.splitlines()
    block_lines: list[str] = []
    in_block = False
    for line in lines:
        if line.strip().startswith(start_key.rstrip(":")):
            in_block = True
            continue
        if in_block:
            if line and not line.startswith(" ") and not line.startswith("\t"):
                break  # end of indented block
            block_lines.append(line)

    # Parse key: value pairs from the block
    parsed: dict[str, Any] = {}
    for bl in block_lines:
        m = re.match(r"\s+(\w+):\s*(.*)", bl)
        if m:
            k, v = m.group(1), m.group(2).strip()
            # Coerce booleans and floats
            if v.lower() == "true":
                parsed[k] = True
            elif v.lower() == "false":
                parsed[k] = False
            else:
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v

    result = {
        "found": True,
        "log_file": log_file,
        "overall_gate_passed": parsed.get("overall_gate_passed"),
        "chi2_gate_passed": parsed.get("chi2_gate_passed"),
        "par_gate_passed": parsed.get("par_gate_passed"),
        "threshold": parsed.get("threshold"),
        "raw_block": block_lines,
        "legacy_key_detected": legacy_detected,
    }
    if legacy_detected and start_key == _LEGACY_KEY:
        result["warning"] = (
            "Legacy key 'sieve3_agreement_gate:' detected in this log; "
            "re-run log generation with a newer version that emits "
            "'optimizer_agreement_check:'."
        )
    return result


# ---------------------------------------------------------------------------
# MCP tool registry
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "log_list",
        "description": (
            "List all VESUVIO .log files under a directory (newest first). "
            "Use this to discover which run logs are available for review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {
                    "type": "string",
                    "description": (
                        "Directory to search.  Defaults to experiments/. "
                        "Can be absolute or relative to the repository root."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "log_read_latest",
        "description": (
            "Read the most recently modified VESUVIO run log.  Use this to "
            "inspect the last run's parameters and outcomes before reviewing "
            "a PR."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "Directory to search."},
            },
            "required": [],
        },
    },
    {
        "name": "log_read",
        "description": (
            "Read a specific VESUVIO run log file by absolute path.  "
            "Restricted to .log files under the allowlisted experiments/ "
            "directory to prevent accidental exposure of unrelated files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the .log file (must be inside experiments/).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "log_grep",
        "description": (
            "Search all VESUVIO run logs for lines matching a regex pattern. "
            "Useful for finding deprecated API usage (e.g. 'np.trapz') or "
            "specific gate results (e.g. 'overall_gate_passed', "
            "'optimizer_agreement_check')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regex to search for (case-insensitive).",
                },
                "root": {"type": "string", "description": "Directory to search."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "log_check_agreement",
        "description": (
            "Parse the most recent run log and extract the structured "
            "'optimizer_agreement_check' block.  Returns 'overall_gate_passed', "
            "'chi2_gate_passed', 'par_gate_passed', and 'threshold'.  "
            "If 'overall_gate_passed' is False, escalate all A2 items to "
            "BLOCK in the PR review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "Directory to search."},
            },
            "required": [],
        },
    },
]

_TOOL_HANDLERS: dict[str, Any] = {
    "log_list": lambda args: _list_logs(args.get("root")),
    "log_read_latest": lambda args: _read_latest_log(args.get("root")),
    "log_read": lambda args: _read_log(args["path"], allow_root=_resolve_root(args.get("root"))),
    "log_grep": lambda args: _grep_logs(args.get("root"), args["pattern"]),
    "log_check_agreement": lambda args: _check_agreement(args.get("root")),
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 / MCP framing
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
                    "name": "vesuvio-log-inspector",
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
        try:
            result = handler(tool_args)
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
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
