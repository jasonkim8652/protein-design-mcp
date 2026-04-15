# Protein Design MCP Server

[![PyPI](https://img.shields.io/pypi/v/protein-design-mcp)](https://pypi.org/project/protein-design-mcp/)
[![Docker Hub](https://img.shields.io/docker/v/jeonghyeonkim8652/protein-design-mcp?label=docker%20hub&logo=docker)](https://hub.docker.com/r/jeonghyeonkim8652/protein-design-mcp)
[![GHCR](https://img.shields.io/badge/ghcr.io-protein--design--mcp-blue?logo=github)](https://github.com/jasonkim8652/protein-design-mcp/pkgs/container/protein-design-mcp)
[![Smithery](https://smithery.ai/badge/protein-design-mcp)](https://smithery.ai/server/protein-design-mcp)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

An [MCP](https://modelcontextprotocol.io) server that gives LLM agents access to computational protein design tools. Ask your LLM to design binders, generate de novo folds, predict structures, score interfaces, or relax with Rosetta — it calls the right tool automatically.

**19 tools total**, spanning generative design, structure prediction, physics-based scoring, and analysis. Built on RFdiffusion, ProteinMPNN, ESMFold, AlphaFold2, **Boltz-2**, **PyRosetta**, ESM2, and OpenMM.

| Distribution | Tools out-of-the-box | Extras |
|---|---|---|
| `pip install "protein-design-mcp[gpu]"` | 13 core tools | `[rosetta]` (license required), `[boltz]` (isolated venv) |
| `docker pull jeonghyeonkim8652/protein-design-mcp` | 13 core tools (GPU), 10 (CPU) | PyRosetta / Boltz not bundled (license + torch conflict) |

The 6 non-bundled tools (`rosetta_*` x4, `predict_*_boltz` x2) install cleanly via pip extras — see [Optional Tools](#optional-tools-pyrosetta--boltz-2).

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
pip install protein-design-mcp                      # Core CPU (10 tools)
pip install "protein-design-mcp[gpu]"               # + PyTorch + ESM (13 tools)
pip install "protein-design-mcp[gpu,rosetta]"       # + PyRosetta (17 tools) *
pip install "protein-design-mcp[gpu,rosetta,boltz]" # + Boltz-2 (all 19 tools) **
```

\* PyRosetta requires a [free academic license](https://www.pyrosetta.org/downloads). The `[rosetta]` extra installs `pyrosetta-installer` which fetches the wheel after you accept the license.

\** Boltz needs `torch>=2.2` which conflicts with RFdiffusion's `torch==2.0.1`. Install in an isolated venv, not alongside `[gpu]`.

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

Isolated, reproducible environments with all computational backends pre-installed. **Primary registry: [Docker Hub](https://hub.docker.com/r/jeonghyeonkim8652/protein-design-mcp).**

#### Pull

```bash
# Latest release (GPU image, ~12GB, bundles RFdiffusion + ProteinMPNN + ESMFold + ColabFold + ESM2 + OpenMM)
docker pull jeonghyeonkim8652/protein-design-mcp:latest
docker pull jeonghyeonkim8652/protein-design-mcp:1.0.0   # pin a specific version
```

GHCR mirror (equivalent):
```bash
docker pull ghcr.io/jasonkim8652/protein-design-mcp:latest
```

Tools included in the image: **13 of 19**. The 6 license/conflict-gated tools (`rosetta_*`, `predict_*_boltz`) are not bundled — install via pip extras instead. See [Optional Tools](#optional-tools-pyrosetta--boltz-2).

#### Run (GPU)

Model weights download lazily on first use and persist in a named volume so subsequent runs are instant:

```bash
docker volume create protein-design-models

docker run --rm -i \
  --gpus all \
  -v protein-design-models:/models \
  -v $(pwd):/data \
  jeonghyeonkim8652/protein-design-mcp:latest
```

GPU mode requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The image stdin/stdout is the MCP protocol — you normally don't run it directly, your MCP client does (see below).

#### Run (CPU)

No GPU — works with the same image:

```bash
docker run --rm -i \
  -e DEVICE=cpu \
  -v protein-design-models:/models \
  -v $(pwd):/data \
  jeonghyeonkim8652/protein-design-mcp:latest
```

CPU mode disables `design_binder`, `design_fold`, `generate_backbone` (RFdiffusion is GPU-only) → 10 tools available.

#### MCP client config

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, or `%APPDATA%/Claude/claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm", "--gpus", "all",
        "-e", "SKIP_MODEL_DOWNLOAD=true",
        "-v", "protein-design-models:/models",
        "-v", "/absolute/path/to/your/pdbs:/data",
        "jeonghyeonkim8652/protein-design-mcp:latest"
      ]
    }
  }
}
```

**Claude Code** (`.mcp.json` in your project root):

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm", "--gpus", "all",
        "-v", "protein-design-models:/models",
        "-v", "${workspaceFolder}:/data",
        "jeonghyeonkim8652/protein-design-mcp:latest"
      ]
    }
  }
}
```

For **CPU-only hosts**, drop `"--gpus", "all"` and add `"-e", "DEVICE=cpu"`.

Restart your MCP client after editing config.

#### Tags available on Docker Hub

| Tag | Purpose |
|---|---|
| `latest` | Tracks the most recent release on `main` |
| `1.0.0`, `1.0`, `1` | Semver-pinned (recommended for production) |
| `<sha>` | Exact commit SHA (immutable) |

Check the full tag list at [hub.docker.com/r/jeonghyeonkim8652/protein-design-mcp/tags](https://hub.docker.com/r/jeonghyeonkim8652/protein-design-mcp/tags).

#### Build locally

```bash
git clone https://github.com/jasonkim8652/protein-design-mcp.git
cd protein-design-mcp
docker build -t protein-design-mcp:dev .                  # GPU image
docker build -f Dockerfile.lite -t protein-design-mcp:lite .  # CPU-only, ~3-5GB
```

The GPU build needs ~30 GB free disk and ~20 minutes.

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

All 19 tools available. Local PDB files are automatically sent to Modal.

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
| **Tools available** | All 19 | 14 (no `design_binder`, `design_fold`, `generate_backbone`, `predict_structure_boltz`, `predict_affinity_boltz`) |
| **RFdiffusion** | ~30s/design | Disabled |
| **Boltz-2** | ~10-30s | Disabled |
| **ESMFold** | ~10s | ~2-5min |
| **ESM2** | ~5s | ~30s |
| **ProteinMPNN** | ~30s | ~5-10min |
| **PyRosetta** | Fast | Comparable |
| **OpenMM** | Fast | Comparable |
| **AlphaFold2 (API)** | Works | Works |

GPU is auto-detected. To force CPU mode, set `DEVICE=cpu`.

## Available Tools

Tools marked **(optional)** are not bundled in the Docker image. See [Optional Tools](#optional-tools-pyrosetta--boltz-2) for install.

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

#### `design_fold` (GPU only)

End-to-end de novo fold design: RFdiffusion (unconditional backbone) → ProteinMPNN (sequence) → AlphaFold2 (validation, falls back to ESMFold). Returns ranked designs filtered by pLDDT/pTM.

```json
{"length": 120, "num_designs": 10, "num_sequences_per_backbone": 4}
```

#### `design_sequence`

Design sequences for a given backbone using ProteinMPNN. Unlike `optimize_sequence` (which refines an existing sequence), this designs from scratch given only a backbone PDB — the correct tool after `generate_backbone`. Optionally validates each design with ESMFold.

```json
{
  "backbone_pdb": "path/to/backbone.pdb",
  "num_sequences": 8,
  "sampling_temp": 0.1,
  "fixed_positions": [1, 5, 10],
  "validate": true
}
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

#### `predict_structure_boltz` (GPU only, **optional**)

Single-chain structure prediction with [Boltz-2](https://github.com/jwohlwend/boltz) — a fast, high-accuracy open model competitive with AF2.

```json
{"sequence": "MTKLYV...", "model": "boltz2", "num_samples": 1}
```

Returns predicted PDB, mean pLDDT, pTM.

#### `predict_affinity_boltz` (GPU only, **optional**)

Multi-chain complex + binding affinity prediction with Boltz-2. Returns affinity score alongside the predicted complex structure and confidence metrics.

```json
{"sequences": ["BINDER_SEQ...", "TARGET_SEQ..."], "model": "boltz2"}
```

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

### Rosetta (Physics-Based Design & Scoring) — **optional**

PyRosetta-backed tools for physics-based scoring, relaxation, and fixed-backbone design. All use `ref2015` by default. **Not bundled in Docker** — install via `pip install "protein-design-mcp[rosetta]"` after accepting the PyRosetta license.

#### `rosetta_score`

Score a structure with a Rosetta energy function. Returns total score, per-residue energies, and component breakdown.

```json
{"pdb_path": "path/to/structure.pdb", "score_function": "ref2015"}
```

#### `rosetta_relax`

FastRelax protocol to find a low-energy conformation. Returns relaxed PDB, energy before/after, and CA-RMSD from input.

```json
{"pdb_path": "path/to/structure.pdb", "nstruct": 1}
```

#### `rosetta_interface_score`

Interface analysis via `InterfaceAnalyzerMover`: binding energy (dG_separated), buried surface area (dSASA), interface hydrogen bonds, packstat.

```json
{"pdb_path": "path/to/complex.pdb", "chains": "A_B"}
```

#### `rosetta_design`

Fixed-backbone redesign pipeline: score → PackRotamers → MinMover → score. Returns designed PDB, mutation list, and energy delta. Composite tool — in benchmark mode, call `rosetta_score` / `rosetta_relax` individually instead.

```json
{"pdb_path": "path/to/input.pdb", "chains": "A_B", "fixed_positions": [12, 14, 18]}
```

### Utility

#### `get_design_status`

Check progress of long-running design jobs.

```json
{"job_id": "abc123"}
```

## Optional Tools: PyRosetta + Boltz-2

These 6 tools (`rosetta_score`, `rosetta_relax`, `rosetta_interface_score`, `rosetta_design`, `predict_structure_boltz`, `predict_affinity_boltz`) are **not included in the Docker image** because:

- **PyRosetta** requires a Rosetta license (free for academics, paid for commercial) and cannot be legally redistributed in a container.
- **Boltz-2** needs `torch>=2.2`, while RFdiffusion's dependency chain (e3nn, dgl) pins `torch==2.0.1`. Both cannot coexist in one venv.

### Installing PyRosetta tools

1. Register at [pyrosetta.org/downloads](https://www.pyrosetta.org/downloads) and accept the license.
2. Install alongside the MCP server:
   ```bash
   pip install "protein-design-mcp[gpu,rosetta]"
   python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
   ```
3. Verify: `python -c "import pyrosetta; print(pyrosetta.__version__)"`

The 4 `rosetta_*` tools become available immediately.

### Installing Boltz-2 tools

Create a separate virtualenv (isolated from the RFdiffusion torch stack):

```bash
python -m venv ~/.venvs/protein-design-boltz
source ~/.venvs/protein-design-boltz/bin/activate
pip install "protein-design-mcp[boltz]"
```

Then point your MCP client at this venv's `protein-design-mcp` binary (or run two MCP servers — one for RFdiffusion/Docker tools, one for Boltz).

### Configuring two MCP servers side-by-side

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "--gpus", "all",
               "-v", "protein-design-models:/models",
               "jeonghyeonkim8652/protein-design-mcp:latest"]
    },
    "protein-design-boltz": {
      "command": "/home/you/.venvs/protein-design-boltz/bin/protein-design-mcp"
    }
  }
}
```

Your LLM will see all 19 tools through the two servers and call whichever is appropriate.

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
 |    +-- design_binder         RFdiffusion -> ProteinMPNN -> ESMFold
 |    +-- design_fold           RFdiffusion -> ProteinMPNN -> AlphaFold2
 |    +-- generate_backbone     RFdiffusion (unconditional)
 |    +-- design_sequence       ProteinMPNN (+ optional ESMFold validation)
 |    +-- optimize_sequence     ProteinMPNN + ESMFold
 |
 +-- Structure prediction
 |    +-- predict_structure       ESMFold or AlphaFold2
 |    +-- predict_complex         AlphaFold2-Multimer (ColabFold)
 |    +-- predict_structure_boltz Boltz-2 (monomer)
 |    +-- predict_affinity_boltz  Boltz-2 (complex + affinity)
 |    +-- validate_design         Structure prediction + RMSD
 |
 +-- Rosetta (PyRosetta)
 |    +-- rosetta_score           ref2015 energy scoring
 |    +-- rosetta_relax           FastRelax
 |    +-- rosetta_interface_score InterfaceAnalyzerMover
 |    +-- rosetta_design          PackRotamers + MinMover
 |
 +-- Analysis tools
 |    +-- analyze_interface   PDB geometry analysis
 |    +-- suggest_hotspots    SASA + pockets + UniProt + PubMed
 |    +-- score_stability     ESM2 pseudo-log-likelihood
 |    +-- energy_minimize     OpenMM (AMBER14)
 |
 +-- Utilities
      +-- get_design_status  Job queue polling
      +-- Structure fetching (RCSB, AlphaFold DB, UniProt)
      +-- Conservation scoring, caching
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
- [Boltz](https://github.com/jwohlwend/boltz) - Open structure and affinity prediction
- [PyRosetta](https://www.pyrosetta.org/) - Physics-based protein modeling
- [OpenMM](https://github.com/openmm/openmm) - Molecular dynamics and energy minimization
