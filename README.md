# Protein Design MCP Server

[![Smithery](https://smithery.ai/badge/protein-design-mcp)](https://smithery.ai/server/protein-design-mcp)
[![PyPI](https://img.shields.io/pypi/v/protein-design-mcp)](https://pypi.org/project/protein-design-mcp/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

An [MCP](https://modelcontextprotocol.io) server that gives LLM agents access to computational protein design tools. Ask your LLM to design binders, predict structures, score stability, or minimize energy — it calls the right tool automatically.

Built on RFdiffusion, ProteinMPNN, ESMFold, AlphaFold2, ESM2, and OpenMM.

## Installation

Choose the method that fits your situation. Listed from simplest to most customizable.

---

### 1. Auto-Setup (Recommended)

One command. Detects your environment, pulls Docker if available, writes MCP client config.

```bash
pip install protein-design-mcp
protein-design-mcp-setup
```

What it does:
- Checks for Docker and NVIDIA GPU
- Pulls the Docker image (or falls back to local Python mode)
- Writes config for Claude Desktop or Claude Code automatically
- Model weights download lazily on first tool call

Options:
```bash
protein-design-mcp-setup --docker    # Force Docker mode
protein-design-mcp-setup --local     # Force local Python mode
protein-design-mcp-setup --modal URL # Use Modal cloud GPU
protein-design-mcp-setup -y          # Skip confirmation prompt
```

---

### 2. Smithery

If you use [Smithery](https://smithery.ai):

```bash
npx -y @smithery/cli install protein-design-mcp --client claude
```

---

### 3. pip + Manual Config

```bash
pip install protein-design-mcp             # Core (9 CPU tools)
pip install "protein-design-mcp[gpu]"      # + PyTorch + ESM (all 11 tools)
```

Add to your MCP client config:

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):
```json
{
  "mcpServers": {
    "protein-design": {
      "command": "protein-design-mcp"
    }
  }
}
```

**Claude Code** (`.mcp.json` in your project root):
```json
{
  "mcpServers": {
    "protein-design": {
      "command": "protein-design-mcp"
    }
  }
}
```

Restart your client after editing config.

---

### 4. Docker

For isolated, reproducible environments. All dependencies and model weights are handled inside the container.

```bash
# GPU mode (all 11 tools, ~19GB image)
docker pull ghcr.io/jasonkim8652/protein-design-mcp:latest

# CPU mode (9 tools, same image, no GPU required)
DEVICE=cpu docker pull ghcr.io/jasonkim8652/protein-design-mcp:latest

# Lite CPU image (9 tools, ~3-5GB, no CUDA runtime)
docker build -f Dockerfile.lite -t protein-design-mcp:lite .
```

Docker MCP config:
```json
{
  "mcpServers": {
    "protein-design": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm", "--gpus", "all",
        "-e", "SKIP_MODEL_DOWNLOAD=true",
        "-v", "protein-design-models:/models",
        "ghcr.io/jasonkim8652/protein-design-mcp:latest"
      ]
    }
  }
}
```

For CPU mode, remove `"--gpus", "all"` and add `"-e", "DEVICE=cpu"`.

Docker GPU requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

### 5. Modal (Cloud GPU)

No local GPU? Deploy to your own [Modal](https://modal.com) account. Serverless GPU on demand, billed per-second (~$1.10/hr A10G). Containers auto-stop after 5 min idle.

```bash
pip install modal
modal setup                          # One-time: link your Modal account

git clone https://github.com/jasonkim8652/protein-design-mcp.git
cd protein-design-mcp
pip install -e .
modal deploy deploy/modal_app.py     # Deploy GPU endpoint
```

After deploying, Modal prints your endpoint URL. Connect via the local proxy:

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

All 11 tools available. Local PDB files are automatically sent to Modal.

---

### 6. From Source (Development)

```bash
git clone https://github.com/jasonkim8652/protein-design-mcp.git
cd protein-design-mcp
pip install -e ".[gpu,dev]"
python -m protein_design_mcp.server
```

For full GPU pipeline, install [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) and [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) separately and set `RFDIFFUSION_PATH` / `PROTEINMPNN_PATH`.

---

## CPU vs GPU

| | GPU | CPU |
|---|---|---|
| **Tools available** | All 11 | 9 (no `design_binder`, `generate_backbone`) |
| **RFdiffusion** | ~30s/design | Disabled |
| **ESMFold** | ~10s | ~2-5min |
| **ESM2** | ~5s | ~30s |
| **ProteinMPNN** | ~30s | ~5-10min |
| **OpenMM** | Fast | Comparable |
| **AlphaFold2 (API)** | Works | Works |

GPU is auto-detected. To force CPU mode, set `DEVICE=cpu`.

## Available Tools

### Design & Generation

#### `design_binder` (GPU only)

End-to-end binder design: RFdiffusion (backbone) -> ProteinMPNN (sequence) -> ESMFold (validation).

```json
{
  "target_pdb": "path/to/target.pdb",
  "hotspot_residues": ["A45", "A46", "A49"],
  "num_designs": 10,
  "binder_length": 80
}
```

Returns ranked designs with sequences, PDB structures, pLDDT, pTM, and mpnn_score.

#### `generate_backbone` (GPU only)

De novo backbone generation using unconditional RFdiffusion. No target protein required.

```json
{"length": 100, "num_designs": 5}
```

#### `optimize_sequence`

Redesign a protein sequence for improved stability and/or binding affinity using ProteinMPNN.

```json
{
  "current_sequence": "MTKLYV...",
  "target_pdb": "path/to/target.pdb",
  "optimization_target": "both",
  "fixed_positions": [1, 5, 10]
}
```

### Structure Prediction

#### `predict_structure`

Single-chain structure prediction via ESMFold (fast) or AlphaFold2 (accurate).

```json
{"sequence": "MTKLYV...", "predictor": "esmfold"}
```

Returns PDB file, mean pLDDT, pTM, per-residue confidence.

#### `predict_complex`

Multi-chain complex structure prediction using AlphaFold2-Multimer.

```json
{
  "sequences": ["BINDER_SEQ...", "TARGET_SEQ..."],
  "chain_names": ["binder", "target"]
}
```

Returns predicted complex PDB with pLDDT, pTM/ipTM, and PAE matrix.

#### `validate_design`

Predict structure of a designed sequence and optionally compute RMSD against a reference.

```json
{
  "sequence": "MTKLYV...",
  "expected_structure": "path/to/reference.pdb",
  "predictor": "esmfold"
}
```

### Analysis & Scoring

#### `analyze_interface`

Analyze protein-protein interface: contacts, buried surface area, hydrogen bonds, salt bridges.

```json
{"complex_pdb": "path/to/complex.pdb", "chain_a": "A", "chain_b": "B"}
```

#### `suggest_hotspots`

Predict binding hotspots from multiple sources. Accepts protein names, UniProt IDs, PDB IDs, or file paths.

```json
{"target": "EGFR", "criteria": "druggable", "include_literature": true}
```

Criteria: `"exposed"` (SASA), `"druggable"` (pocket geometry), `"conserved"` (evolution).

#### `score_stability`

Protein stability scoring via ESM2 pseudo-log-likelihood. Optionally score individual mutations.

```json
{
  "sequence": "MTKLYV...",
  "mutations": ["A42G", "L55V"]
}
```

Returns overall stability score and per-mutation delta log-likelihood (stabilizing/destabilizing).

#### `energy_minimize`

All-atom energy minimization with OpenMM (AMBER14 + implicit solvent).

```json
{"pdb_path": "path/to/structure.pdb", "num_steps": 500, "solvent": "implicit"}
```

Returns minimized PDB, energy change, and RMSD from input.

### Utility

#### `get_design_status`

Check progress of long-running design jobs.

```json
{"job_id": "abc123"}
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVICE` | `"auto"`, `"cuda"`, or `"cpu"` | `auto` |
| `RFDIFFUSION_PATH` | Path to RFdiffusion installation | `/opt/RFdiffusion` |
| `PROTEINMPNN_PATH` | Path to ProteinMPNN installation | `/opt/ProteinMPNN` |
| `COLABFOLD_BACKEND` | `"api"` (remote MSA) or `"local"` (local DB) | `api` |
| `CACHE_DIR` | Cache directory | `~/.cache/protein-design-mcp` |
| `TORCH_HOME` | ESM model weights directory | (PyTorch default) |
| `SKIP_MODEL_DOWNLOAD` | Skip eager weight download in Docker | `true` |

## Architecture

```
MCP Server (stdio)
 |
 +-- Design tools
 |    +-- design_binder      RFdiffusion -> ProteinMPNN -> ESMFold
 |    +-- generate_backbone   RFdiffusion (unconditional)
 |    +-- optimize_sequence   ProteinMPNN + ESMFold
 |
 +-- Prediction tools
 |    +-- predict_structure   ESMFold or AlphaFold2
 |    +-- predict_complex     AlphaFold2-Multimer (ColabFold)
 |    +-- validate_design     Structure prediction + RMSD
 |
 +-- Analysis tools
 |    +-- analyze_interface   PDB geometry analysis
 |    +-- suggest_hotspots    SASA + pockets + UniProt + PubMed
 |    +-- score_stability     ESM2 pseudo-log-likelihood
 |    +-- energy_minimize     OpenMM (AMBER14)
 |
 +-- Utilities
      +-- Structure fetching (RCSB, AlphaFold DB, UniProt)
      +-- Conservation scoring, job queue, caching
```

## Development

```bash
git clone https://github.com/jasonkim8652/protein-design-mcp.git
cd protein-design-mcp
pip install -e ".[gpu,dev]"

pytest tests/           # Run tests
ruff check .            # Lint
black .                 # Format
mypy src/               # Type check
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## References

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) - Protein backbone generation
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) - Sequence design
- [ESMFold / ESM2](https://github.com/facebookresearch/esm) - Structure prediction and stability scoring
- [ColabFold](https://github.com/sokrypton/ColabFold) - Fast AlphaFold2 with MMseqs2
- [OpenMM](https://github.com/openmm/openmm) - Molecular dynamics and energy minimization
