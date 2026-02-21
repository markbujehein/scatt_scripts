"""MCP server: Playwright researcher for documentation navigation.

Provides a JSON-over-stdio MCP server that uses requests to fetch web pages
for Mantid v6.14 documentation and ISIS manuals.
"""

from __future__ import annotations

import json
import sys
from typing import Any

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_ALLOWED_DOMAINS = [
    "mantidproject.org",
    "isis.stfc.ac.uk",
    "docs.mantidproject.org",
    "www.isis.stfc.ac.uk",
    "github.com",
]

def _is_domain_allowed(url: str) -> bool:
    """Check if the URL domain is allowed."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc
    for allowed in _ALLOWED_DOMAINS:
        if domain.endswith(allowed):
            return True
    return False

def _fetch_webpage(url: str, query: str = "") -> str:
    """Fetch webpage content."""
    if not _REQUESTS_AVAILABLE:
        return "requests not available"
    if not _is_domain_allowed(url):
        raise ValueError(f"Domain not allowed: {url}")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    content = response.text
    if query:
        # Simple search
        if query.lower() in content.lower():
            return f"Query '{query}' found in {url}"
        else:
            return f"Query '{query}' not found in {url}"
    return content[:10000]  # Limit

# MCP tool registry

_TOOLS = [
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch content from allowed documentation websites (Mantid, ISIS)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                },
                "query": {
                    "type": "string",
                    "description": "Optional search query in the page"
                }
            },
            "required": ["url"]
        },
    },
]

_TOOL_HANDLERS = {
    "fetch_webpage": lambda args: _fetch_webpage(args["url"], args.get("query", "")),
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
                    "name": "vesuvio-playwright-researcher",
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
                    "content": [{"type": "text", "text": result}],
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