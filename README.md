# Protein Binder Design MCP Server

An MCP (Model Context Protocol) server that enables LLM agents to run end-to-end protein binder design pipelines using RFdiffusion, ProteinMPNN, and ESMFold.

## Features

- **High-level, workflow-based tools** - Designed for LLM agents with <10 tools total
- **End-to-end design pipeline** - Single tool call runs RFdiffusion → ProteinMPNN → ESMFold
- **Quality metrics** - All designs include pLDDT, pTM, and interface scores
- **Interface analysis** - Analyze protein-protein interfaces
- **Hotspot suggestion** - Identify potential binding sites on targets

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/protein-design-mcp.git
cd protein-design-mcp

# Create conda environment
conda env create -f environment.yml
conda activate protein-design-mcp

# Install in development mode
pip install -e ".[dev]"
```

### Prerequisites

The following tools must be installed separately:

- **RFdiffusion** - For backbone generation
- **ProteinMPNN** - For sequence design (included in RFdiffusion)
- **ESMFold** - Installed via `fair-esm` package

### Running the Server

```bash
# Start the MCP server
python -m protein_design_mcp.server
```

## Available Tools

| Tool | Description |
|------|-------------|
| `design_binder` | Complete binder design pipeline (RFdiffusion → ProteinMPNN → ESMFold) |
| `analyze_interface` | Analyze protein-protein interface properties |
| `validate_design` | Validate a sequence with ESMFold structure prediction |
| `optimize_sequence` | Optimize a binder sequence for stability/affinity |
| `suggest_hotspots` | Suggest potential binding sites on a target |
| `get_design_status` | Check status of running design jobs |

## Example Usage

### Designing a Binder

```python
# Tool: design_binder
{
    "target_pdb": "path/to/target.pdb",
    "hotspot_residues": ["A45", "A46", "A49"],
    "num_designs": 10,
    "binder_length": 80
}
```

### Analyzing an Interface

```python
# Tool: analyze_interface
{
    "complex_pdb": "path/to/complex.pdb",
    "chain_a": "A",
    "chain_b": "B"
}
```

## Configuration

Set environment variables to configure external tool paths:

```bash
export RFDIFFUSION_PATH=/path/to/RFdiffusion
export PROTEINMPNN_PATH=/path/to/ProteinMPNN
export CACHE_DIR=~/.cache/protein-design-mcp
```

## Installing with MCP Clients

This MCP server can be used with any MCP-compatible client. Below are setup instructions for popular clients.

### Claude Desktop

1. **Install the package** in your Python environment:

```bash
# Clone and install
git clone https://github.com/jasonkim8652/protein-design-mcp.git
cd protein-design-mcp
pip install -e .
```

2. **Configure Claude Desktop** by editing the configuration file:

   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

Add the server to the `mcpServers` section:

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "python",
      "args": ["-m", "protein_design_mcp.server"],
      "cwd": "/path/to/protein-design-mcp",
      "env": {
        "RFDIFFUSION_PATH": "/path/to/RFdiffusion",
        "PROTEINMPNN_PATH": "/path/to/ProteinMPNN",
        "CACHE_DIR": "~/.cache/protein-design-mcp"
      }
    }
  }
}
```

3. **Restart Claude Desktop** to load the new server.

### Using with Conda Environment

If you're using a conda environment, specify the full path to the Python executable:

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "/path/to/conda/envs/protein-design-mcp/bin/python",
      "args": ["-m", "protein_design_mcp.server"],
      "cwd": "/path/to/protein-design-mcp",
      "env": {
        "RFDIFFUSION_PATH": "/path/to/RFdiffusion",
        "PROTEINMPNN_PATH": "/path/to/ProteinMPNN"
      }
    }
  }
}
```

To find your conda Python path:
```bash
conda activate protein-design-mcp
which python
```

### Claude Code (CLI)

For Claude Code, add the server to your project's `.mcp.json` file:

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "python",
      "args": ["-m", "protein_design_mcp.server"],
      "cwd": "/path/to/protein-design-mcp"
    }
  }
}
```

Or add it globally in `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "/path/to/conda/envs/protein-design-mcp/bin/python",
      "args": ["-m", "protein_design_mcp.server"],
      "cwd": "/path/to/protein-design-mcp"
    }
  }
}
```

### Other MCP Clients

For other MCP-compatible clients, use stdio transport with:

```bash
# Command to start the server
python -m protein_design_mcp.server
```

The server communicates via stdin/stdout using the MCP protocol.

### Verifying Installation

After configuration, you can verify the server is working by asking the LLM:

> "What protein design tools are available?"

The LLM should list the 6 available tools: `design_binder`, `analyze_interface`, `validate_design`, `optimize_sequence`, `suggest_hotspots`, and `get_design_status`.

## Development

```bash
# Run tests
pytest tests/

# Run linter
ruff check .

# Format code
black .

# Type checking
mypy src/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Server                              │
├─────────────────────────────────────────────────────────────┤
│  Tools Layer (6 high-level tools)                           │
├─────────────────────────────────────────────────────────────┤
│  Orchestration Layer (Pipeline management, caching)         │
├─────────────────────────────────────────────────────────────┤
│  Computation Layer (RFdiffusion, ProteinMPNN, ESMFold)      │
└─────────────────────────────────────────────────────────────┘
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [ESMFold](https://github.com/facebookresearch/esm)
