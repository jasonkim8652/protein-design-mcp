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
