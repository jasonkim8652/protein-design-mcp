# Protein Design MCP Server

[![Smithery](https://smithery.ai/badge/protein-design-mcp)](https://smithery.ai/server/protein-design-mcp)
[![PyPI](https://img.shields.io/pypi/v/protein-design-mcp)](https://pypi.org/project/protein-design-mcp/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

An MCP (Model Context Protocol) server that enables LLM agents to run computational protein design workflows — from backbone generation and sequence design to structure prediction, stability scoring, and energy minimization.

Built on RFdiffusion, ProteinMPNN, ESMFold, AlphaFold2, ESM2, and OpenMM.

## Features

- **11 high-level tools** designed for LLM agents — no scripting or pipeline glue needed
- **End-to-end binder design** — single tool call runs RFdiffusion → ProteinMPNN → ESMFold/AlphaFold2
- **De novo backbone generation** — unconditional RFdiffusion for novel folds
- **Structure prediction** — ESMFold (fast) or AlphaFold2 (accurate) for monomers and multimers
- **Stability scoring** — ESM2 pseudo-log-likelihood with per-mutation delta scoring
- **Energy minimization** — OpenMM with AMBER14 force field and implicit solvent
- **Multi-source hotspot detection** — combines SASA, pocket detection, conservation, UniProt annotations, and PubMed literature
- **Flexible structure input** — accepts protein names, UniProt IDs, PDB IDs, or local file paths
- **Quality metrics** — pLDDT, pTM, interface scores, mpnn_score on all designs

## Quick Start

### Fastest: Auto-Setup (Recommended)

```bash
pip install protein-design-mcp
protein-design-mcp-setup          # auto-detects Docker/GPU, writes MCP config
```

That's it. The setup command:
1. Detects Docker + GPU availability
2. Pulls Docker image (if Docker found) or configures local mode
3. Auto-writes config for Claude Desktop or Claude Code
4. Model weights download lazily on first tool call

```bash
# Or specify mode explicitly:
protein-design-mcp-setup --docker    # Force Docker
protein-design-mcp-setup --local     # Force local Python
protein-design-mcp-setup --modal URL # Force Modal cloud GPU
protein-design-mcp-setup -y          # Skip confirmation
```

### Smithery (One-Click)

```bash
npx -y @smithery/cli install protein-design-mcp --client claude
```

### pip (Manual Config)

```bash
pip install protein-design-mcp             # Core (9 CPU tools)
pip install "protein-design-mcp[gpu]"      # + PyTorch + ESM (11 tools)
protein-design-mcp                         # Start server
```

Then add to your MCP client config:
```json
{"mcpServers": {"protein-design": {"command": "protein-design-mcp"}}}
```

### Option 1: Docker (Recommended)

```bash
# GPU mode (default, ~19GB image, all 11 tools)
docker pull ghcr.io/jasonkim8652/protein-design-mcp:latest
mkdir -p models data cache
wget https://raw.githubusercontent.com/jasonkim8652/protein-design-mcp/main/docker-compose.yml
docker compose up

# CPU mode (same image, no GPU required, 9 tools)
docker compose --profile cpu up

# Lite CPU image (~3-5GB, no CUDA runtime)
docker build -f Dockerfile.lite -t protein-design-mcp:lite .
docker compose --profile lite up
```

Model weights are downloaded lazily on first tool call. To pre-download, run `docker compose run --rm download-models`.

#### CPU vs GPU Mode

| | GPU Mode | CPU Mode | Lite Image |
|---|---|---|---|
| **Available tools** | All 11 tools | 9 tools (no `design_binder`, `generate_backbone`) | 9 tools |
| **Image size** | ~19 GB | ~19 GB (same image) | ~3-5 GB |
| **RFdiffusion** | Full speed (~30s/design) | Disabled | Not installed |
| **ESMFold / predict_structure** | Fast (~10s) | Slower (~2-5min) | ~2-5min |
| **ESM2 / score_stability** | Fast (~5s) | Moderate (~30s) | ~30s |
| **ProteinMPNN / optimize_sequence** | Fast (~30s) | Slower (~5-10min) | ~5-10min |
| **OpenMM / energy_minimize** | Fast | Comparable speed | Comparable |
| **AlphaFold2 (API mode)** | Works | Works (uses remote server) | Works |
| **Requirements** | NVIDIA GPU + Docker | Docker only | Docker only |

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

### Option 2: Modal (Cloud GPU — No Local GPU Required)

Deploy to your own [Modal](https://modal.com) account for serverless GPU access. Pay only for compute time (~$1.10/hr A10G), containers auto-stop after 5 min idle.

```bash
# One-time setup
pip install modal
modal setup  # links your Modal account

# Deploy (creates GPU endpoint)
git clone https://github.com/jasonkim8652/protein-design-mcp.git
cd protein-design-mcp
pip install -e .
modal deploy deploy/modal_app.py

# Test
modal run deploy/modal_app.py
```

After deploying, Modal prints your endpoint URL. Use the local MCP proxy to connect:

```bash
# Start local MCP proxy → Modal GPU
MODAL_URL=https://<your-workspace>--protein-design-tools.modal.run \
    python -m protein_design_mcp.modal_proxy
```

#### Modal with Claude Desktop

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "python",
      "args": ["-m", "protein_design_mcp.modal_proxy"],
      "env": {
        "MODAL_URL": "https://<your-workspace>--protein-design-tools.modal.run"
      }
    }
  }
}
```

All 11 tools are available (GPU on Modal). Local PDB files are automatically uploaded to Modal on each tool call.

### Option 3: Local Installation

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

- **RFdiffusion** — Backbone generation (conditional and unconditional)
- **ProteinMPNN** — Sequence design (included in RFdiffusion)
- **ESMFold** — Installed via `fair-esm` package
- **OpenMM** — For energy minimization (`conda install -c conda-forge openmm`)

#### Running the Server (Local)

```bash
# Start the MCP server
python -m protein_design_mcp.server
```

## Available Tools

### Design & Generation

| Tool | Description | Backend |
|------|-------------|---------|
| `design_binder` | End-to-end binder design pipeline | RFdiffusion → ProteinMPNN → ESMFold/AF2 |
| `generate_backbone` | De novo backbone generation (no target required) | RFdiffusion (unconditional) |
| `optimize_sequence` | Optimize sequence for stability and/or binding affinity | ProteinMPNN + ESMFold |

### Structure Prediction

| Tool | Description | Backend |
|------|-------------|---------|
| `predict_structure` | Single-chain structure prediction | ESMFold or AlphaFold2 |
| `predict_complex` | Multi-chain complex structure prediction | AlphaFold2-Multimer |
| `validate_design` | Validate sequence with structure prediction + optional RMSD to reference | ESMFold or AlphaFold2 |

### Analysis & Scoring

| Tool | Description | Backend |
|------|-------------|---------|
| `analyze_interface` | Protein-protein interface analysis (contacts, buried surface area, H-bonds) | PDB geometry |
| `suggest_hotspots` | Multi-source binding site prediction | SASA + pockets + conservation + UniProt + PubMed |
| `score_stability` | Protein stability scoring with per-mutation effects | ESM2 pseudo-log-likelihood |
| `energy_minimize` | All-atom energy minimization | OpenMM (AMBER14 + implicit solvent) |

### Job Management

| Tool | Description |
|------|-------------|
| `get_design_status` | Check status and progress of running design jobs |

## Tool Details

### `design_binder`

Runs the full binder design pipeline in a single call.

```json
{
    "target_pdb": "path/to/target.pdb",
    "hotspot_residues": ["A45", "A46", "A49"],
    "num_designs": 10,
    "binder_length": 80
}
```

Returns ranked designs with sequences, PDB structures, pLDDT, pTM, and mpnn_score.

### `generate_backbone`

Generates novel protein backbones without a target protein (unconditional RFdiffusion).

```json
{
    "length": 100,
    "num_designs": 5
}
```

### `predict_structure`

Predicts the 3D structure of a single protein sequence.

```json
{
    "sequence": "MTKLYV...",
    "predictor": "esmfold"
}
```

Returns predicted PDB, pLDDT, pTM, per-residue confidence, and PAE matrix.

### `validate_design`

Like `predict_structure`, but also computes RMSD against a reference structure.

```json
{
    "sequence": "MTKLYV...",
    "expected_structure": "path/to/reference.pdb",
    "predictor": "alphafold2"
}
```

### `predict_complex`

Predicts multi-chain complex structure using AlphaFold2-Multimer.

```json
{
    "sequences": ["BINDER_SEQ...", "TARGET_SEQ..."],
    "chain_names": ["binder", "target"]
}
```

### `score_stability`

Scores protein stability using ESM2 masked marginal probabilities. Optionally scores the effect of specific mutations.

```json
{
    "sequence": "MTKLYV...",
    "mutations": ["A42G", "L55V"],
    "reference_sequence": "MTKLAV..."
}
```

Returns per-residue scores, mutation effects (delta log-likelihood), and overall stability score.

### `energy_minimize`

Performs all-atom energy minimization using OpenMM.

```json
{
    "pdb_path": "path/to/structure.pdb",
    "num_steps": 500,
    "solvent": "implicit"
}
```

Returns minimized PDB, initial/final energy, energy change, and RMSD from input.

### `suggest_hotspots`

Multi-source hotspot prediction. Accepts protein names, UniProt IDs, PDB IDs, or file paths — the structure is fetched automatically.

```json
{
    "target": "EGFR",
    "criteria": "druggable",
    "include_literature": true
}
```

Criteria options:
- `"exposed"` — Surface-accessible residues (SASA-based)
- `"druggable"` — Druggable pockets (volume + hydrophobicity scoring)
- `"conserved"` — Evolutionarily conserved residues (BLAST-based)

Returns hotspot residues with evidence from SASA, pocket geometry, UniProt annotations, conservation scores, and PubMed literature.

### `analyze_interface`

Analyzes protein-protein interfaces from a complex PDB.

```json
{
    "complex_pdb": "path/to/complex.pdb",
    "chain_a": "A",
    "chain_b": "B",
    "distance_cutoff": 8.0
}
```

Returns interface residues, buried surface area, hydrogen bonds, salt bridges, and hydrophobic contacts.

### `optimize_sequence`

Optimizes a binder sequence for stability, binding affinity, or both.

```json
{
    "current_sequence": "MTKLYV...",
    "target_pdb": "path/to/target.pdb",
    "optimization_target": "both",
    "fixed_positions": [1, 5, 10]
}
```

## Example Workflow: EGFR Binder Design

**Step 1: Identify binding hotspots**

```json
// suggest_hotspots
{
    "target": "EGFR",
    "criteria": "exposed",
    "include_literature": true
}
```

**Step 2: Design binders**

```json
// design_binder
{
    "target_pdb": "EGFR",
    "hotspot_residues": ["A98", "A99", "A100"],
    "num_designs": 10,
    "binder_length": 80
}
```

**Step 3: Score stability of top designs**

```json
// score_stability
{
    "sequence": "MTKLYV..."
}
```

**Step 4: Energy minimize the best structure**

```json
// energy_minimize
{
    "pdb_path": "path/to/best_design.pdb",
    "num_steps": 1000
}
```

**Step 5: Predict binder-target complex**

```json
// predict_complex
{
    "sequences": ["BINDER_SEQ...", "EGFR_SEQ..."],
    "chain_names": ["binder", "target"]
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVICE` | `"auto"` (detect GPU), `"cuda"`, or `"cpu"` | `auto` |
| `RFDIFFUSION_PATH` | Path to RFdiffusion installation | `/opt/RFdiffusion` |
| `PROTEINMPNN_PATH` | Path to ProteinMPNN installation | `/opt/ProteinMPNN` |
| `COLABFOLD_PATH` | Path to `colabfold_batch` executable | `/opt/localcolabfold/.../colabfold_batch` |
| `COLABFOLD_BACKEND` | `"api"` (uses MMseqs2 server) or `"local"` | `api` |
| `CACHE_DIR` | Cache directory for results | `~/.cache/protein-design-mcp` |
| `TORCH_HOME` | ESM model weights directory | (PyTorch default) |
| `SKIP_MODEL_DOWNLOAD` | Skip eager model download in Docker | `true` |

**Note**: The Docker image uses API backend by default, which sends MSA queries to ColabFold's public server. This avoids downloading ~2TB of local databases. For high-throughput or offline use, set `COLABFOLD_BACKEND=local`.

## Installing with MCP Clients

### Claude Desktop

1. **Install the package**:

```bash
git clone https://github.com/jasonkim8652/protein-design-mcp.git
cd protein-design-mcp
pip install -e .
```

2. **Configure Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

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

3. **Restart Claude Desktop** to load the server.

### Claude Code (CLI)

Add to your project's `.mcp.json`:

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

### Using with Conda

Specify the full path to the conda Python executable:

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

Use stdio transport:

```bash
python -m protein_design_mcp.server
```

The server communicates via stdin/stdout using the MCP protocol.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP Server (11 tools)                      │
├──────────────────────────────────────────────────────────────┤
│  Design         design_binder, generate_backbone,            │
│                 optimize_sequence                             │
│  Prediction     predict_structure, predict_complex,           │
│                 validate_design                               │
│  Analysis       analyze_interface, suggest_hotspots,          │
│                 score_stability, energy_minimize              │
│  Management     get_design_status                             │
├──────────────────────────────────────────────────────────────┤
│  Pipelines (async)                                            │
│  ├─ RFdiffusion    (conditional + unconditional backbone)    │
│  ├─ ProteinMPNN    (sequence design, fixed positions)        │
│  ├─ ESMFold        (fast structure prediction)               │
│  ├─ AlphaFold2     (accurate monomer + multimer via ColabFold)│
│  ├─ ESM2 Scorer    (stability, mutation effects)             │
│  └─ OpenMM         (AMBER14 energy minimization)             │
├──────────────────────────────────────────────────────────────┤
│  Utilities                                                    │
│  ├─ Structure fetching (RCSB, AlphaFold DB, UniProt)         │
│  ├─ SASA + druggable pocket detection                        │
│  ├─ Conservation scoring (BLAST-based)                       │
│  ├─ UniProt annotation (binding sites, active sites)         │
│  ├─ PubMed literature mining                                 │
│  └─ Job queue, caching, metrics                              │
└──────────────────────────────────────────────────────────────┘
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

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [ESMFold / ESM2](https://github.com/facebookresearch/esm)
- [ColabFold](https://github.com/sokrypton/ColabFold) - Fast AlphaFold2 predictions with MMseqs2
- [AlphaFold2](https://github.com/deepmind/alphafold)
- [OpenMM](https://github.com/openmm/openmm) - Molecular dynamics and energy minimization
