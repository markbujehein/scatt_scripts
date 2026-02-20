# Agent Tools Reference (repo-scoped)

Purpose: provide a minimal, repo-scoped reference for common agent tools and how to run them via the workspace Pixi environment. This file is intended for VS Code Copilot/agent workflows and does not assume Claude.

- Playwright CLI (optional): see `.claude/skills/playwright-cli/SKILL.md` for full details. To run from the Pixi env:

	```bash
	pixi run playwright-cli open https://example.com
	```

- Pixi: package manager used by this workspace. Key commands:

	- `pixi install` — resolve and install Conda + PyPI deps into `.pixi/envs/default/`.
	- `pixi run python -c "import vesuvio_analysis"` — run Python inside the Pixi env.

- Zotero / File MCP servers: configured at workspace-level in `.vscode/mcp.json` (do not modify global `~/.config` without permission).

Notes:
- Original `.claude` SKILL files remain under `.claude/skills/` if you need levelled SKILL docs. This reference keeps the repo-focused, Copilot-friendly commands concise.

