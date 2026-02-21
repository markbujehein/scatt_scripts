# MCP Server Research Audit (2026 Updates)

This document contains comprehensive research findings on the latest (2026) implementations of specified MCP servers, focusing on current installation methods, invocation commands, environment variables, and potential conflicts with Pixi-managed Python environments.

## markitdown-mcp (Microsoft)

**Repository:** [KorigamiK/markitdown_mcp_server](https://github.com/KorigamiK/markitdown_mcp_server)  
**Latest Update:** 2026-02-05  
**Stars:** 69  

### Exact Invocation Commands
- **npx (Smithery):** `npx -y @smithery/cli install @KorigamiK/markitdown_mcp_server --client claude`
- **uvx:** `uvx --from markitdown-mcp-server markitdown-mcp` (if published)
- **Manual uv:** `uv --directory /path/to/markitdown_mcp_server run markitdown-mcp`

### Required Environment Variables
None required.

### Known Conflicts with Pixi-Managed Environments
No known conflicts. The server runs via uvx which uses isolated Python environments, separate from Pixi's management.

### Installation Instructions and Dependencies
1. **Via Smithery (Recommended):**
   ```bash
   npx -y @smithery/cli install @KorigamiK/markitdown_mcp_server --client claude
   ```

2. **Manual Installation:**
   ```bash
   git clone https://github.com/KorigamiK/markitdown_mcp_server.git
   cd markitdown_mcp_server
   uv install
   ```

**Dependencies:** 
- Python 3.8+
- markitdown library
- MCP SDK

**Claude Desktop Config Example:**
```json
{
  "mcpServers": {
    "markitdown": {
      "command": "uv",
      "args": ["--directory", "/path/to/markitdown_mcp_server", "run", "markitdown-mcp"]
    }
  }
}
```

## docling-mcp (IBM)

**Repository:** [docling-project/docling-mcp](https://github.com/docling-project/docling-mcp)  
**Latest Update:** 2026-02-20  
**Stars:** 417  

### Exact Invocation Commands
- **uvx (stdio):** `uvx --from docling-mcp docling-mcp-server --transport stdio`
- **uvx (sse):** `uvx --from docling-mcp docling-mcp-server --transport sse`
- **uvx (streamable-http):** `uvx --from docling-mcp docling-mcp-server --transport streamable-http`

### Required Environment Variables
None required.

### Known Conflicts with Pixi-Managed Environments
No known conflicts. Uses uvx with isolated Python environment.

### Installation Instructions and Dependencies
1. **Via uvx (Recommended):**
   ```bash
   uvx --from docling-mcp docling-mcp-server --transport stdio
   ```

2. **Via pip:**
   ```bash
   pip install docling-mcp
   ```

**Dependencies:**
- Python 3.9+
- docling library
- MCP SDK
- Optional: Milvus for RAG features

**Claude Desktop Config Example:**
```json
{
  "mcpServers": {
    "docling": {
      "command": "uvx",
      "args": ["--from=docling-mcp", "docling-mcp-server"]
    }
  }
}
```

## zotero-library (mcp-server-zotero)

**Repository:** [kujenga/zotero-mcp](https://github.com/kujenga/zotero-mcp)  
**Latest Update:** 2026-02-11  
**Stars:** 133  

### Exact Invocation Commands
- **uvx (Local API):** `uvx --upgrade zotero-mcp`
- **uvx (Web API):** `uvx --upgrade zotero-mcp` (with env vars)
- **Docker:** `docker run --rm -i -e ZOTERO_API_KEY=KEY -e ZOTERO_LIBRARY_ID=ID ghcr.io/kujenga/zotero-mcp:main`

### Required Environment Variables
- `ZOTERO_LOCAL=true` (for local Zotero API, default false)
- `ZOTERO_API_KEY` (required for Web API, optional for local)
- `ZOTERO_LIBRARY_ID` (required for Web API, optional for local)
- `ZOTERO_LIBRARY_TYPE` (user or group, default user)

### Known Conflicts with Pixi-Managed Environments
No known conflicts. Uses uvx or Docker for isolation.

### Installation Instructions and Dependencies
1. **Via uvx (Recommended):**
   ```bash
   uvx --upgrade zotero-mcp
   ```

2. **Via pipx:**
   ```bash
   pipx run zotero-mcp
   ```

3. **Docker:**
   ```bash
   docker pull ghcr.io/kujenga/zotero-mcp:main
   ```

**Dependencies:**
- Python 3.8+
- pyzotero
- MCP SDK
- Zotero desktop app (for local API) or Zotero account (for Web API)

**Claude Desktop Config Example (Local API):**
```json
{
  "mcpServers": {
    "zotero": {
      "command": "uvx",
      "args": ["--upgrade", "zotero-mcp"],
      "env": {
        "ZOTERO_LOCAL": "true"
      }
    }
  }
}
```

**Claude Desktop Config Example (Web API):**
```json
{
  "mcpServers": {
    "zotero": {
      "command": "uvx",
      "args": {
        "ZOTERO_API_KEY": "your-api-key",
        "ZOTERO_LIBRARY_ID": "your-user-id"
      }
    }
  }
}
```

## mcp-pandoc

**Repository:** [vivekVells/mcp-pandoc](https://github.com/vivekVells/mcp-pandoc)  
**Latest Update:** 2026-02-19  
**Stars:** 498  

### Exact Invocation Commands
- **uvx:** `uvx mcp-pandoc`

### Required Environment Variables
None required.

### Known Conflicts with Pixi-Managed Environments
Potential minor conflicts with system-level dependencies (pandoc, TeX Live). Pixi can manage Python packages but may require system installation of pandoc. No Python environment conflicts since uvx isolates execution.

### Installation Instructions and Dependencies
1. **Via uvx (Recommended):**
   ```bash
   uvx mcp-pandoc
   ```

2. **Manual:**
   ```bash
   git clone https://github.com/vivekVells/mcp-pandoc.git
   cd mcp-pandoc
   uv sync
   uv run mcp-pandoc
   ```

**Critical System Dependencies:**
- **pandoc:** Required for all conversions
  - macOS: `brew install pandoc`
  - Ubuntu/Debian: `sudo apt-get install pandoc`
  - Windows: Download from https://pandoc.org/installing.html

- **TeX Live (for PDF output):** Required only for PDF generation
  - macOS: `brew install texlive`
  - Ubuntu/Debian: `sudo apt-get install texlive-xetex`
  - Windows: Install MiKTeX or TeX Live

**Python Dependencies:**
- Python 3.8+
- pandoc Python package
- MCP SDK

**Claude Desktop Config Example:**
```json
{
  "mcpServers": {
    "mcp-pandoc": {
      "command": "uvx",
      "args": ["mcp-pandoc"]
    }
  }
}
```

## General Notes

- All servers use uvx for isolated execution, minimizing conflicts with Pixi-managed environments
- System-level tools (pandoc, TeX) may need separate installation outside Pixi
- Zotero server requires either local Zotero app or Web API credentials
- All servers support Claude Desktop integration via MCP
- Latest updates as of February 20, 2026