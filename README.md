# Protein Binder Design MCP Server

An MCP (Model Context Protocol) server that enables LLM agents to run end-to-end protein binder design pipelines using RFdiffusion, ProteinMPNN, and ESMFold.

## Features

- **High-level, workflow-based tools** - Designed for LLM agents with <10 tools total
- **End-to-end design pipeline** - Single tool call runs RFdiffusion → ProteinMPNN → ESMFold
- **Quality metrics** - All designs include pLDDT, pTM, and interface scores
- **Interface analysis** - Analyze protein-protein interfaces
- **Hotspot suggestion** - Identify potential binding sites on targets

## Quick Start

### Option 1: Docker (Recommended)

The easiest way to get started is using Docker. Everything is pre-configured - just pull and run:

```bash
# Pull the image
docker pull ghcr.io/jasonkim8652/protein-design-mcp:latest

# Create directories for models and data
mkdir -p models data cache

# Run with docker-compose (recommended)
wget https://raw.githubusercontent.com/jasonkim8652/protein-design-mcp/main/docker-compose.yml
docker-compose up
```

**First run will download model weights (~10GB)**. These are cached in the `models/` directory for future runs.

#### Docker with Claude Desktop

Configure Claude Desktop to use the Docker container:

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "docker",
      "args": [
        "compose", "run", "--rm", "-T", "protein-design", "server"
      ],
      "cwd": "/path/to/protein-design-mcp"
    }
  }
}
```

#### Docker Commands

```bash
# Start MCP server
docker-compose up

# Download models only (without starting server)
docker-compose run --rm download-models

# Interactive shell
docker-compose run --rm protein-design bash

# Build locally (instead of pulling from ghcr.io)
docker-compose build
```

#### GPU Requirements

Docker requires NVIDIA Container Toolkit for GPU support:

```bash
# Install nvidia-container-toolkit (Ubuntu/Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Option 2: Local Installation

For development or if you prefer a local installation:

```bash
# Clone the repository
git clone https://github.com/jasonkim8652/protein-design-mcp.git
cd protein-design-mcp

# Create conda environment
conda env create -f environment.yml
conda activate protein-design-mcp

# Install in development mode
pip install -e ".[dev]"
```

#### Prerequisites (Local Installation)

The following tools must be installed separately:

- **RFdiffusion** - For backbone generation
- **ProteinMPNN** - For sequence design (included in RFdiffusion)
- **ESMFold** - Installed via `fair-esm` package

#### Running the Server (Local)

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

### Real-World Example: Designing an EGFR Binder

Here's a complete workflow for designing a binder against EGFR (Epidermal Growth Factor Receptor), a key cancer drug target:

**Step 1: Find binding hotspots on EGFR**

```python
# Tool: suggest_hotspots
{
    "target": "EGFR",
    "criteria": "exposed",
    "include_literature": true
}
```

The tool automatically fetches the EGFR structure (PDB: 3VRP) and returns:

| Residues | Score | Rationale |
|----------|-------|-----------|
| A98-A100 | 0.97 | Exposed surface region (SASA: 95.8 Å²) |
| A199 | 0.95 | Ca²⁺ binding site (UniProt annotated) |
| A264 | 0.95 | Phosphotyrosine binding site |

**Step 2: Design binders targeting the hotspot**

```python
# Tool: design_binder
{
    "target_pdb": "EGFR",
    "hotspot_residues": ["A98", "A99", "A100"],
    "num_designs": 10,
    "binder_length": 80
}
```

This runs the full pipeline: RFdiffusion → ProteinMPNN → ESMFold, returning ranked designs with quality metrics (pLDDT, pTM, interface scores).

**Step 3: Validate top designs**

```python
# Tool: validate_design
{
    "sequence": "MTKLYV..."
}
```

### Basic Tool Examples

#### Designing a Binder

```python
# Tool: design_binder
{
    "target_pdb": "path/to/target.pdb",
    "hotspot_residues": ["A45", "A46", "A49"],
    "num_designs": 10,
    "binder_length": 80
}
```

#### Analyzing an Interface

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

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [ESMFold](https://github.com/facebookresearch/esm)
