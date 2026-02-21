"""MCP server: System monitor for VESUVIO resource tracking.

Provides a JSON-over-stdio MCP server that exposes system resource information
to enable agents to monitor Numba/iMinuit resource consumption during fits.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

def _get_system_info() -> dict[str, Any]:
    """Get basic system information."""
    info = {
        "cpu_count": os.cpu_count(),
        "platform": os.uname().sysname if hasattr(os, 'uname') else 'unknown',
    }
    if _PSUTIL_AVAILABLE:
        info.update({
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent,
            },
        })
    return info

def _get_process_info(pid: int | None = None) -> dict[str, Any]:
    """Get process information."""
    if not _PSUTIL_AVAILABLE:
        return {"error": "psutil not available"}
    if pid is None:
        pid = os.getpid()
    try:
        p = psutil.Process(pid)
        return {
            "pid": p.pid,
            "name": p.name(),
            "cpu_percent": p.cpu_percent(),
            "memory_info": p.memory_info()._asdict(),
            "num_threads": p.num_threads(),
        }
    except psutil.NoSuchProcess:
        return {"error": f"No such process: {pid}"}

# MCP tool registry

_TOOLS = [
    {
        "name": "get_system_info",
        "description": (
            "Get current system resource information (CPU, memory, disk usage)."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_process_info",
        "description": (
            "Get information about a specific process by PID, or current process if not specified."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pid": {
                    "type": "integer",
                    "description": "Process ID (optional, defaults to current process)"
                }
            },
            "required": []
        },
    },
]

_TOOL_HANDLERS = {
    "get_system_info": lambda args: _get_system_info(),
    "get_process_info": lambda args: _get_process_info(args.get("pid")),
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
                    "name": "vesuvio-system-monitor",
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