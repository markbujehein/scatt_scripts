#!/usr/bin/env bash
# test_mcp_servers.sh — Validate all VESUVIO MCP servers against the
# 2025-11-25 schema using raw JSON-RPC 2.0 / stdio exchanges.
#
# Usage (from repo root):
#   bash vesuvio_analysis/mcp_server/test_mcp_servers.sh
#
# For interactive inspection with mcp-inspector (requires Node.js):
#   npx @modelcontextprotocol/inspector pixi run python -m vesuvio_analysis.mcp_server.environment_server
#
# Exit code: 0 = all checks passed, 1 = one or more failures.

set -euo pipefail

PASS=0
FAIL=0

# ---------------------------------------------------------------------------
# Helper: send a sequence of JSON-RPC messages to a server and capture output
# ---------------------------------------------------------------------------
rpc_call() {
    local module="$1"
    local messages="$2"
    python3 -m "$module" <<< "$messages" 2>&1 | grep -v '^$'
}

check_field() {
    local label="$1"
    local output="$2"
    local expected="$3"
    if echo "$output" | grep -qF "$expected"; then
        echo "  PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label (expected '$expected' in output)"
        echo "        Got: $(echo "$output" | head -n 10)"
        FAIL=$((FAIL + 1))
    fi
}

# ---------------------------------------------------------------------------
# Standard message sequences
# ---------------------------------------------------------------------------
INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{}}}'
INITIALIZED='{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'
TOOLS_LIST='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

run_server_checks() {
    local name="$1"
    local module="$2"
    shift 2
    local extra_calls=("$@")

    echo ""
    echo "=== $name ==="

    local seq="$INIT"$'\n'"$INITIALIZED"$'\n'"$TOOLS_LIST"
    for call in "${extra_calls[@]}"; do
        seq="$seq"$'\n'"$call"
    done

    local out
    out=$(rpc_call "$module" "$seq")

    # 1. Protocol version must be 2025-11-25
    check_field "protocolVersion=2025-11-25" "$out" '"protocolVersion": "2025-11-25"'

    # 2. Logging capability declared
    check_field "capability: logging" "$out" '"logging"'

    # 3. tools/list returns non-empty result
    check_field "tools/list result" "$out" '"tools"'
}

# ---------------------------------------------------------------------------
# Per-server checks
# ---------------------------------------------------------------------------

run_server_checks "environment_server" \
    "vesuvio_analysis.mcp_server.environment_server" \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"env_versions","arguments":{}}}' \
    '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"git_commit","arguments":{}}}' \
    '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"check_version_compatibility","arguments":{}}}'

# Verify -32602 for missing required arg
OUT_ENV=$(rpc_call "vesuvio_analysis.mcp_server.environment_server" \
    "$INIT"$'\n'"$INITIALIZED"$'\n''{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"unknown_tool","arguments":{}}}')
check_field "environment_server: -32601 for unknown tool" "$OUT_ENV" '"code": -32601'

run_server_checks "mantid_ads_server" \
    "vesuvio_analysis.mcp_server.mantid_ads_server" \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ads_list_workspaces","arguments":{}}}' \
    '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ads_is_empty","arguments":{}}}'

# Verify -32602 for missing required 'name' arg on ads_workspace_shape
OUT_ADS=$(rpc_call "vesuvio_analysis.mcp_server.mantid_ads_server" \
    "$INIT"$'\n'"$INITIALIZED"$'\n''{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"ads_workspace_shape","arguments":{}}}')
check_field "mantid_ads_server: -32602 for missing 'name'" "$OUT_ADS" '"code": -32602'

run_server_checks "log_inspector_server" \
    "vesuvio_analysis.mcp_server.log_inspector_server" \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"log_list","arguments":{}}}' \
    '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"log_check_agreement","arguments":{}}}'

# Verify -32602 for missing required 'pattern' arg on log_grep
OUT_LOG=$(rpc_call "vesuvio_analysis.mcp_server.log_inspector_server" \
    "$INIT"$'\n'"$INITIALIZED"$'\n''{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"log_grep","arguments":{}}}')
check_field "log_inspector_server: -32602 for missing 'pattern'" "$OUT_LOG" '"code": -32602'

run_server_checks "thesis_files_server" \
    "vesuvio_analysis.mcp_server.thesis_files_server"

# Verify -32602 for missing required 'path' arg
OUT_FILES=$(rpc_call "vesuvio_analysis.mcp_server.thesis_files_server" \
    "$INIT"$'\n'"$INITIALIZED"$'\n''{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"read_file","arguments":{}}}')
check_field "thesis_files_server: -32602 for missing 'path'" "$OUT_FILES" '"code": -32602'

run_server_checks "playwright_researcher_server" \
    "vesuvio_analysis.mcp_server.playwright_researcher_server"

# Verify -32602 for missing required 'url' arg
OUT_PW=$(rpc_call "vesuvio_analysis.mcp_server.playwright_researcher_server" \
    "$INIT"$'\n'"$INITIALIZED"$'\n''{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"fetch_webpage","arguments":{}}}')
check_field "playwright_researcher_server: -32602 for missing 'url'" "$OUT_PW" '"code": -32602'

run_server_checks "system_monitor_server" \
    "vesuvio_analysis.mcp_server.system_monitor_server" \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_system_info","arguments":{}}}'

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Results: $PASS passed, $FAIL failed"
echo "========================================"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
