"""MCP server: Thesis files accessor.

Provides a JSON-over-stdio MCP server that exposes file operations for
whitelisted directories (scatt_scripts, thesis_draft) to enable agents to
read project files without external access.

This server enforces strict path validation to maintain data sovereignty.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Whitelisted directories (relative to workspace root)
_ALLOWED_PATHS = [
    "scatt_scripts",
    "thesis_draft",
]

# Workspace root
_WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent  # up to Mantid/

def _is_path_allowed(path: str) -> bool:
    """Check if the path is within allowed directories."""
    try:
        abs_path = _WORKSPACE_ROOT / path
        abs_path = abs_path.resolve()
        for allowed in _ALLOWED_PATHS:
            allowed_path = (_WORKSPACE_ROOT / allowed).resolve()
            if abs_path.is_relative_to(allowed_path):
                return True
        return False
    except Exception:
        return False

def _list_directory(path: str) -> list[str]:
    """List contents of a directory."""
    if not _is_path_allowed(path):
        raise ValueError(f"Access denied: {path}")
    full_path = _WORKSPACE_ROOT / path
    if not full_path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return [str(p.relative_to(_WORKSPACE_ROOT)) for p in full_path.iterdir()]

def _read_file(path: str) -> str:
    """Read contents of a file."""
    if not _is_path_allowed(path):
        raise ValueError(f"Access denied: {path}")
    full_path = _WORKSPACE_ROOT / path
    if not full_path.is_file():
        raise ValueError(f"Not a file: {path}")
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()

# MCP tool registry

_TOOLS = [
    {
        "name": "list_directory",
        "description": (
            "List the contents of a directory within the whitelisted paths "
            "(scatt_scripts, thesis_draft). Returns a list of relative paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the directory from workspace root."
                }
            },
            "required": ["path"]
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file within the whitelisted paths "
            "(scatt_scripts, thesis_draft)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file from workspace root."
                }
            },
            "required": ["path"]
        },
    },
]

_TOOL_HANDLERS = {
    "list_directory": lambda args: _list_directory(args["path"]),
    "read_file": lambda args: _read_file(args["path"]),
}

# MCP protocol

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
                    "name": "vesuvio-thesis-files",
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
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True,
                },
            }

    if method == "notifications/initialized":
        return None

    return _error_response(req_id, -32601, f"Method not found: {method}")

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