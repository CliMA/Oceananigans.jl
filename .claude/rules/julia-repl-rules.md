# Julia Execution Rules

## Prefer MCP REPL over Bash

- When an MCP REPL server is available (check for MCP Julia/REPL tools), always use it to run Julia code instead of launching Julia via the Bash tool.
- The MCP REPL is faster because it has packages already loaded and precompiled.
- Only fall back to running Julia via Bash if MCP REPL tools are not available in the current session.
