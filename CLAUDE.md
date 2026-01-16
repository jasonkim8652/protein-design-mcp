# CLAUDE.md - Development Guidelines for Protein Binder Design MCP Server

## Project Overview

This is an MCP (Model Context Protocol) server for protein binder design. It wraps RFdiffusion, ProteinMPNN, and ESMFold into high-level tools that enable LLM agents to run end-to-end protein design pipelines.

## Quick Start

```bash
# Set up environment
conda env create -f environment.yml
conda activate protein-design-mcp

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Start the MCP server
python -m protein_design_mcp.server
```

## Architecture

### Core Principle: Few High-Level Tools
- **Maximum 10 tools total** - LLM accuracy drops significantly with more tools
- Each tool = complete workflow, not raw API call
- Internal orchestration of multi-step pipelines

### Directory Structure
```
src/protein_design_mcp/
├── server.py           # MCP server entry point
├── tools/              # High-level MCP tools
├── pipelines/          # Wrappers for RFdiffusion, ProteinMPNN, ESMFold
├── utils/              # PDB parsing, metrics, caching
└── resources/          # MCP resources for data access
```

### Key Components

1. **Tools Layer** (`src/protein_design_mcp/tools/`)
   - `design_binder.py` - Main end-to-end binder design
   - `analyze.py` - Interface analysis
   - `validate.py` - Structure prediction validation
   - `optimize.py` - Sequence optimization

2. **Pipelines Layer** (`src/protein_design_mcp/pipelines/`)
   - Wraps external tools (RFdiffusion, ProteinMPNN, ESMFold)
   - Handles execution, error recovery, output parsing

3. **Utils Layer** (`src/protein_design_mcp/utils/`)
   - PDB file operations
   - Quality metrics (pLDDT, pTM, interface scores)
   - Result caching

## Development Guidelines

### Code Style
- Use Python 3.10+ features (type hints, dataclasses)
- Follow PEP 8 with 100 char line limit
- Use `ruff` for linting: `ruff check .`
- Use `black` for formatting: `black .`

### Type Hints
All functions must have type hints:
```python
def design_binder(
    target_pdb: str,
    hotspot_residues: list[str],
    num_designs: int = 10,
) -> DesignResult:
    ...
```

### Error Handling
- Use custom exceptions defined in `src/protein_design_mcp/exceptions.py`
- Always provide actionable error messages
- Log errors with context for debugging

```python
class InvalidPDBError(ProteinDesignError):
    """Raised when PDB file is invalid or malformed."""
    pass

# Usage
if not validate_pdb(pdb_path):
    raise InvalidPDBError(
        f"Invalid PDB file: {pdb_path}. "
        "Ensure the file contains valid ATOM records."
    )
```

### Testing
- All tools must have unit tests
- Use pytest fixtures for test data
- Mock external tools in unit tests
- Integration tests should use small test proteins

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/protein_design_mcp

# Run specific test file
pytest tests/test_tools.py -v
```

### MCP Tool Implementation Pattern
```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("protein-design-mcp")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="design_binder",
            description="Design protein binders for a target protein...",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_pdb": {"type": "string", "description": "..."},
                    "hotspot_residues": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["target_pdb", "hotspot_residues"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "design_binder":
        result = await run_design_pipeline(arguments)
        return [TextContent(type="text", text=json.dumps(result))]
```

## Common Tasks

### Adding a New Tool
1. Create tool implementation in `src/protein_design_mcp/tools/`
2. Register tool in `server.py` `list_tools()` function
3. Add handler in `call_tool()` function
4. Write tests in `tests/test_tools.py`
5. Update this document if needed

### Modifying Pipeline Steps
1. Pipeline wrappers are in `src/protein_design_mcp/pipelines/`
2. Each wrapper handles: input prep, execution, output parsing, error handling
3. Ensure idempotency where possible
4. Add retry logic for transient failures

### Running External Tools

**RFdiffusion**:
```python
# Located in pipelines/rfdiffusion.py
# Expects RFdiffusion installed and available
# GPU required for reasonable performance
```

**ProteinMPNN**:
```python
# Located in pipelines/proteinmpnn.py
# Expects ProteinMPNN model weights downloaded
```

**ESMFold**:
```python
# Located in pipelines/esmfold.py
# Uses ESM model from torch hub or local installation
# ~3GB GPU memory required
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RFDIFFUSION_PATH` | Path to RFdiffusion installation | `/opt/RFdiffusion` |
| `PROTEINMPNN_PATH` | Path to ProteinMPNN installation | `/opt/ProteinMPNN` |
| `ESMFOLD_MODEL` | ESMFold model identifier | `esmfold_v1` |
| `CACHE_DIR` | Directory for cached results | `~/.cache/protein-design-mcp` |
| `MAX_CONCURRENT_JOBS` | Maximum parallel design jobs | `4` |

## Debugging

### Enable Verbose Logging
```bash
export LOG_LEVEL=DEBUG
python -m protein_design_mcp.server
```

### Common Issues

**"CUDA out of memory"**
- Reduce batch size in pipeline configs
- Process sequences sequentially instead of parallel
- Use smaller test proteins for development

**"RFdiffusion not found"**
- Set `RFDIFFUSION_PATH` environment variable
- Ensure conda environment is activated

**"Invalid PDB format"**
- Use BioPython to validate PDB files
- Check for missing ATOM records
- Ensure chain IDs are present

## Dependencies

### Core
- `mcp` - Model Context Protocol SDK
- `torch` - PyTorch for ML models
- `biopython` - PDB file handling
- `numpy` - Numerical operations

### External Tools (must be installed separately)
- RFdiffusion - Backbone generation
- ProteinMPNN - Sequence design
- ESMFold - Structure prediction

## Git Workflow

- Main branch: `main`
- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Commit messages: Use conventional commits

```bash
# Example commits
git commit -m "feat: add design_binder tool"
git commit -m "fix: handle missing chain IDs in PDB"
git commit -m "docs: update CLAUDE.md with new tool"
```

## Performance Notes

- Single binder design (~10 designs): ~20-30 minutes on GPU
- ESMFold prediction per sequence: ~1-2 minutes
- Interface analysis: ~seconds
- Use caching to avoid redundant computation

## Contact & Resources

- PRD: See `PRD.md` for full product requirements
- MCP Docs: https://modelcontextprotocol.io/docs
- RFdiffusion: https://github.com/RosettaCommons/RFdiffusion
- ProteinMPNN: https://github.com/dauparas/ProteinMPNN
- ESMFold: https://github.com/facebookresearch/esm
