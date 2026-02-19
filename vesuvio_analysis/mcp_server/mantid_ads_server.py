"""MCP server: Mantid AnalysisDataService (ADS) state inspector.

Provides a thin JSON-over-stdio MCP server that exposes the Mantid
``mtd`` object as a set of tools and resources.  When running inside a
real Mantid environment, this server allows an LLM agent to:

1. List all workspace names currently held in the ADS.
2. Fetch the shape (number of spectra × bins) of a named workspace.
3. Fetch basic statistics (min/max/mean of Y-values) for a workspace.
4. Check whether the ADS is empty (post-run teardown verification).

Architecture note
-----------------
This module is **import-safe**: if Mantid is not installed, all tool
handlers return a descriptive error string instead of raising.  This
allows the module to be imported during CI runs that use
``tests/mock_mantid.py`` without special-casing.

Standalone execution
--------------------
Run the server directly for use with Claude Desktop or any MCP-compatible
host::

    python -m vesuvio_analysis.mcp_server.mantid_ads_server

Protocol
--------
The server uses the JSON-RPC 2.0 framing required by the MCP spec
(https://modelcontextprotocol.io/).  Tool definitions follow the
``tools/list`` → ``tools/call`` lifecycle.
"""

from __future__ import annotations

import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Mantid ADS accessor (graceful degradation when Mantid absent)
# ---------------------------------------------------------------------------

def _get_mtd() -> Any | None:
    """Return the Mantid ADS object, or *None* if Mantid is not installed."""
    try:
        from mantid.api import mtd  # type: ignore[import]
        return mtd
    except ImportError:
        return None


def _ads_list_workspaces() -> dict[str, Any]:
    """Return a list of all workspace names currently in the ADS."""
    mtd = _get_mtd()
    if mtd is None:
        return {"error": "Mantid is not installed in this environment."}
    try:
        names: list[str] = mtd.getObjectNames()  # type: ignore[union-attr]
        return {"workspace_names": names, "count": len(names)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _ads_workspace_shape(name: str) -> dict[str, Any]:
    """Return the shape (n_spectra, n_bins) of a named workspace."""
    mtd = _get_mtd()
    if mtd is None:
        return {"error": "Mantid is not installed in this environment."}
    try:
        if name not in mtd:
            return {"error": f"Workspace '{name}' not found in ADS."}
        ws = mtd[name]
        return {
            "name": name,
            "n_spectra": ws.getNumberHistograms(),
            "n_bins": ws.blocksize(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _ads_workspace_stats(name: str) -> dict[str, Any]:
    """Return basic Y-value statistics for a named workspace."""
    import numpy as np

    mtd = _get_mtd()
    if mtd is None:
        return {"error": "Mantid is not installed in this environment."}
    try:
        if name not in mtd:
            return {"error": f"Workspace '{name}' not found in ADS."}
        ws = mtd[name]
        y = ws.extractY()
        return {
            "name": name,
            "y_min": float(np.nanmin(y)),
            "y_max": float(np.nanmax(y)),
            "y_mean": float(np.nanmean(y)),
            "y_std": float(np.nanstd(y)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _ads_is_empty() -> dict[str, Any]:
    """Return whether the ADS is currently empty."""
    mtd = _get_mtd()
    if mtd is None:
        return {"error": "Mantid is not installed in this environment."}
    try:
        names = mtd.getObjectNames()
        return {"is_empty": len(names) == 0, "count": len(names)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# MCP tool registry
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "ads_list_workspaces",
        "description": (
            "List all workspace names currently held in the Mantid "
            "AnalysisDataService (mtd).  Use this to inspect pipeline "
            "state between reduction iterations."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "ads_workspace_shape",
        "description": (
            "Return the shape (n_spectra, n_bins) of a named Mantid workspace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Workspace name in the ADS."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "ads_workspace_stats",
        "description": (
            "Return basic Y-value statistics (min, max, mean, std) for a "
            "named Mantid workspace.  Useful for sanity-checking data after "
            "each reduction step."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Workspace name in the ADS."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "ads_is_empty",
        "description": (
            "Check whether the Mantid ADS is empty.  Use after pipeline "
            "teardown to verify that all workspaces were properly cleaned up."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]

_TOOL_HANDLERS: dict[str, Any] = {
    "ads_list_workspaces": lambda _args: _ads_list_workspaces(),
    "ads_workspace_shape": lambda args: _ads_workspace_shape(args["name"]),
    "ads_workspace_stats": lambda args: _ads_workspace_stats(args["name"]),
    "ads_is_empty": lambda _args: _ads_is_empty(),
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 / MCP framing helpers
# ---------------------------------------------------------------------------

def _send(obj: dict[str, Any]) -> None:
    """Write a JSON-RPC message to stdout (MCP stdio transport)."""
    line = json.dumps(obj)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _error_response(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch a single JSON-RPC request and return the response."""
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
                    "name": "vesuvio-mantid-ads",
                    "version": "0.1.0",
                },
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": _TOOLS},
        }

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
                "isError": "error" in result,
            },
        }

    if method == "notifications/initialized":
        return None  # no response for notifications

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
