# Interactive Julia REPL for AI Agents (MCPRepl.jl)

[MCPRepl.jl](https://github.com/kahliburke/MCPRepl.jl) exposes a Julia REPL via the Model Context Protocol (MCP),
allowing AI agents to execute Julia code, run tests, and iterate quickly during development.

## Installation

If MCPRepl.jl is not already installed, add it to your global Julia environment:

```julia
using Pkg
Pkg.activate()  # Activate global environment
Pkg.add(url="https://github.com/kahliburke/MCPRepl.jl")
```

Then run the security setup (one-time):

```julia
using MCPRepl
MCPRepl.quick_setup(:lax)  # For local development (localhost only, no API key)
```

## Starting the MCP Server

Before the AI agent can use the REPL, start the server in Julia:

```julia
using MCPRepl
MCPRepl.start_proxy(port=3000)  # Recommended: persistent proxy with dashboard
# OR
MCPRepl.start!(port=3000)       # Direct REPL backend
```

The dashboard is available at `http://localhost:3000/dashboard` when using the proxy.

## Cursor Configuration

Create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "julia-repl": {
      "url": "http://localhost:3000",
      "transport": "http",
      "headers": {
        "X-MCPRepl-Target": "Oceananigans.jl"
      }
    }
  }
}
```

After creating this file, reload Cursor (Cmd+Shift+P -> "Reload Window").

## Speeding Up Development with Revise.jl

For rapid iteration, use Revise.jl alongside MCPRepl. This allows code changes to be
reflected immediately without restarting Julia:

```julia
using Revise
using MCPRepl
using Oceananigans

MCPRepl.start_proxy(port=3000)
```

With this setup:
1. The AI agent can execute code via the REPL
2. Source code edits are automatically picked up by Revise
3. No need to restart Julia or re-import packages after editing source files
4. Tests can be run interactively with immediate feedback

## Available MCP Tools

Once connected, the AI agent has access to:
- **`julia_eval`** - Execute Julia code in the REPL
- **`lsp_goto_definition`** - Navigate to symbol definitions
- **`lsp_find_references`** - Find all usages of a symbol
- **`lsp_rename`** - Rename symbols across the codebase
- **`lsp_document_symbols`** - Get file structure/outline
- **`lsp_code_actions`** - Get available quick fixes

## Typical Workflow

1. Start Julia with Revise + MCPRepl
2. Edit source files — Revise picks up changes automatically
3. Test via MCPRepl without restarting Julia
4. Iterate until complete
