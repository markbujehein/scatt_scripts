"""Protocol conformance tests for VESUVIO MCP servers.

Validates that all six MCP servers comply with the 2025-11-25 schema:
- Correct ``protocolVersion`` in ``initialize`` response
- ``logging`` capability declared
- Standard JSON-RPC 2.0 error codes: ``-32601`` (unknown method/tool),
  ``-32602`` (missing required argument), ``-32603`` (internal error)
- ``tools/list`` returns a non-empty list
- ``additionalProperties: false`` on all tool input schemas
- Missing required argument handled with ``-32602`` (no server crash)

Runnable without Mantid or pixi::

    python -m pytest tests/test_mcp_servers.py -v
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from io import StringIO
from typing import Any
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helper: drive a server's _handle_request dispatcher
# ---------------------------------------------------------------------------

def _init_req() -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}}}


def _tools_list_req() -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


def _call_req(tool: str, args: dict[str, Any], req_id: int = 3) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": tool, "arguments": args}}


def _load_server(module_name: str) -> types.ModuleType:
    """Import (or re-import) a server module fresh."""
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------

class _MCPServerBase(unittest.TestCase):
    """Base class providing shared assertions for MCP server tests."""

    MODULE: str = ""  # override in subclass

    def setUp(self) -> None:
        self.mod = _load_server(self.MODULE)

    def _handle(self, req: dict[str, Any]) -> dict[str, Any] | None:
        return self.mod._handle_request(req)  # type: ignore[attr-defined]

    def assertInitializeOK(self) -> dict[str, Any]:
        resp = self._handle(_init_req())
        self.assertIsNotNone(resp)
        result = resp["result"]
        self.assertEqual(result["protocolVersion"], "2025-11-25",
                         "protocolVersion must be '2025-11-25'")
        self.assertIn("logging", result["capabilities"],
                      "capabilities must include 'logging'")
        return result

    def assertToolsListNonEmpty(self) -> list[dict[str, Any]]:
        resp = self._handle(_tools_list_req())
        self.assertIsNotNone(resp)
        tools = resp["result"]["tools"]
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0, "tools/list must return at least one tool")
        return tools

    def assertAllSchemasHaveAdditionalPropertiesFalse(self) -> None:
        tools = self.assertToolsListNonEmpty()
        for tool in tools:
            schema = tool.get("inputSchema", {})
            self.assertFalse(
                schema.get("additionalProperties", True),
                f"Tool '{tool['name']}' inputSchema must have additionalProperties=False",
            )

    def assertUnknownToolReturns32601(self) -> None:
        resp = self._handle(_call_req("__no_such_tool__", {}))
        self.assertIsNotNone(resp)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)

    def assertMissingRequiredArgReturns32602(self, tool: str, required_arg: str) -> None:
        resp = self._handle(_call_req(tool, {}))
        self.assertIsNotNone(resp)
        self.assertIn("error", resp,
                      f"Missing '{required_arg}' in '{tool}' must return an error response")
        self.assertEqual(resp["error"]["code"], -32602,
                         f"Expected -32602 for missing required arg '{required_arg}' in '{tool}'")


# ---------------------------------------------------------------------------
# Per-server test classes
# ---------------------------------------------------------------------------

class TestEnvironmentServer(_MCPServerBase):
    MODULE = "vesuvio_analysis.mcp_server.environment_server"

    def test_initialize(self) -> None:
        self.assertInitializeOK()

    def test_tools_list_non_empty(self) -> None:
        self.assertToolsListNonEmpty()

    def test_all_schemas_strict(self) -> None:
        self.assertAllSchemasHaveAdditionalPropertiesFalse()

    def test_unknown_tool_32601(self) -> None:
        self.assertUnknownToolReturns32601()

    def test_env_versions_call(self) -> None:
        resp = self._handle(_call_req("env_versions", {}))
        self.assertIsNotNone(resp)
        self.assertIn("result", resp)
        content_text = resp["result"]["content"][0]["text"]
        data = json.loads(content_text)
        self.assertIn("python_version", data)

    def test_check_version_compatibility_call(self) -> None:
        resp = self._handle(_call_req("check_version_compatibility", {}))
        self.assertIsNotNone(resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("compatible", data)


class TestMantidADSServer(_MCPServerBase):
    MODULE = "vesuvio_analysis.mcp_server.mantid_ads_server"

    def test_initialize(self) -> None:
        self.assertInitializeOK()

    def test_tools_list_non_empty(self) -> None:
        self.assertToolsListNonEmpty()

    def test_all_schemas_strict(self) -> None:
        self.assertAllSchemasHaveAdditionalPropertiesFalse()

    def test_unknown_tool_32601(self) -> None:
        self.assertUnknownToolReturns32601()

    def test_missing_name_arg_32602(self) -> None:
        self.assertMissingRequiredArgReturns32602("ads_workspace_shape", "name")

    def test_missing_name_arg_stats_32602(self) -> None:
        self.assertMissingRequiredArgReturns32602("ads_workspace_stats", "name")

    def test_ads_list_workspaces_no_mantid(self) -> None:
        """Without Mantid installed, ads_list_workspaces returns a graceful error."""
        resp = self._handle(_call_req("ads_list_workspaces", {}))
        self.assertIsNotNone(resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        # Either a graceful error (no Mantid) or a real result — both are valid
        self.assertTrue("error" in data or "workspace_names" in data)

    def test_ads_is_empty_no_mantid(self) -> None:
        resp = self._handle(_call_req("ads_is_empty", {}))
        self.assertIsNotNone(resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertTrue("error" in data or "is_empty" in data)


class TestLogInspectorServer(_MCPServerBase):
    MODULE = "vesuvio_analysis.mcp_server.log_inspector_server"

    def test_initialize(self) -> None:
        self.assertInitializeOK()

    def test_tools_list_non_empty(self) -> None:
        self.assertToolsListNonEmpty()

    def test_all_schemas_strict(self) -> None:
        self.assertAllSchemasHaveAdditionalPropertiesFalse()

    def test_unknown_tool_32601(self) -> None:
        self.assertUnknownToolReturns32601()

    def test_missing_pattern_32602(self) -> None:
        self.assertMissingRequiredArgReturns32602("log_grep", "pattern")

    def test_missing_path_32602(self) -> None:
        self.assertMissingRequiredArgReturns32602("log_read", "path")

    def test_log_list_returns_result(self) -> None:
        resp = self._handle(_call_req("log_list", {}))
        self.assertIsNotNone(resp)
        self.assertIn("result", resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("log_files", data)


class TestThesisFilesServer(_MCPServerBase):
    MODULE = "vesuvio_analysis.mcp_server.thesis_files_server"

    def test_initialize(self) -> None:
        self.assertInitializeOK()

    def test_tools_list_non_empty(self) -> None:
        self.assertToolsListNonEmpty()

    def test_all_schemas_strict(self) -> None:
        self.assertAllSchemasHaveAdditionalPropertiesFalse()

    def test_unknown_tool_32601(self) -> None:
        self.assertUnknownToolReturns32601()

    def test_missing_path_32602(self) -> None:
        self.assertMissingRequiredArgReturns32602("read_file", "path")

    def test_missing_path_list_32602(self) -> None:
        self.assertMissingRequiredArgReturns32602("list_directory", "path")

    def test_absolute_path_rejected(self) -> None:
        """Absolute path passed as 'path' must be rejected (access denied)."""
        resp = self._handle(_call_req("read_file", {"path": "/etc/passwd"}))
        self.assertIsNotNone(resp)
        # Should return -32603 (internal error wrapping ValueError) or isError=True
        if "error" in resp:
            self.assertEqual(resp["error"]["code"], -32603)
        else:
            self.assertTrue(resp["result"]["isError"])

    def test_path_traversal_rejected(self) -> None:
        """Path traversal must be rejected."""
        resp = self._handle(_call_req("read_file", {"path": "scatt_scripts/../../etc/passwd"}))
        self.assertIsNotNone(resp)
        if "error" in resp:
            self.assertEqual(resp["error"]["code"], -32603)
        else:
            self.assertTrue(resp["result"]["isError"])


class TestPlaywrightResearcherServer(_MCPServerBase):
    MODULE = "vesuvio_analysis.mcp_server.playwright_researcher_server"

    def test_initialize(self) -> None:
        self.assertInitializeOK()

    def test_tools_list_non_empty(self) -> None:
        self.assertToolsListNonEmpty()

    def test_all_schemas_strict(self) -> None:
        self.assertAllSchemasHaveAdditionalPropertiesFalse()

    def test_unknown_tool_32601(self) -> None:
        self.assertUnknownToolReturns32601()

    def test_missing_url_32602(self) -> None:
        self.assertMissingRequiredArgReturns32602("fetch_webpage", "url")

    def test_disallowed_domain_rejected(self) -> None:
        """Fetching an external domain not in the whitelist must be rejected."""
        resp = self._handle(_call_req("fetch_webpage", {"url": "https://evil.example.com/malware"}))
        self.assertIsNotNone(resp)
        if "error" in resp:
            self.assertEqual(resp["error"]["code"], -32603)
        else:
            self.assertTrue(resp["result"]["isError"])


class TestSystemMonitorServer(_MCPServerBase):
    MODULE = "vesuvio_analysis.mcp_server.system_monitor_server"

    def test_initialize(self) -> None:
        self.assertInitializeOK()

    def test_tools_list_non_empty(self) -> None:
        self.assertToolsListNonEmpty()

    def test_all_schemas_strict(self) -> None:
        self.assertAllSchemasHaveAdditionalPropertiesFalse()

    def test_unknown_tool_32601(self) -> None:
        self.assertUnknownToolReturns32601()

    def test_get_system_info_returns_cpu_count(self) -> None:
        resp = self._handle(_call_req("get_system_info", {}))
        self.assertIsNotNone(resp)
        data = json.loads(resp["result"]["content"][0]["text"])
        self.assertIn("cpu_count", data)


if __name__ == "__main__":
    unittest.main()
