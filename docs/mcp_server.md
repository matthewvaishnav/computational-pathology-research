# Computational Pathology MCP Server

This repository now includes a small MCP server that exposes repo-scoped tools
over the standard `stdio` transport. One server implementation can be used from
various AI code editors that support the Model Context Protocol.

## Server Entry Point

Run the server directly with:

```powershell
python scripts/project_mcp_server.py
```

The server is rooted to this repository by default and exposes these tools:

- `project_overview`
- `list_project_files`
- `read_text_file`
- `search_repository`
- `run_pytest`

## Client Setup

### Codex

Register a local stdio MCP server in Codex using this command:

- `command`: `python`
- `args`: `["C:/path/to/computational-pathology-research/scripts/project_mcp_server.py"]`

If you are using a virtual environment, replace `python` with the exact
interpreter path you want Codex to launch.

### Cursor

Cursor supports project-level MCP config in `.cursor/mcp.json`. This repo now
includes a ready-to-use config:

```json
{
  "mcpServers": {
    "computational-pathology": {
      "command": "python",
      "args": ["${workspaceFolder}/scripts/project_mcp_server.py"]
    }
  }
}
```

### Windsurf

Windsurf reads user-level MCP config from
`~/.codeium/windsurf/mcp_config.json`. A starter example is included at
`mcp/windsurf.mcp_config.example.json`.

```json
{
  "mcpServers": {
    "computational-pathology": {
      "command": "python",
      "args": [
        "C:/path/to/computational-pathology-research/scripts/project_mcp_server.py"
      ]
    }
  }
}
```

## Notes

- The server writes only JSON-RPC messages to `stdout`.
- Logging and diagnostics belong on `stderr`.
- `run_pytest` is intentionally limited to repository-local paths.
