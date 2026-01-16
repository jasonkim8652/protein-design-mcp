# Product Requirements Document: Protein Binder Design MCP Server

## Executive Summary

This document outlines the requirements for building an MCP (Model Context Protocol) server that enables LLM agents to run end-to-end protein binder design pipelines. The server wraps RFdiffusion, ProteinMPNN, and ESMFold into high-level, purpose-driven tools that facilitate computational protein design workflows.

## Problem Statement

Protein binder design is a complex multi-step process that requires:
1. Generating novel protein backbones that can bind to target proteins (RFdiffusion)
2. Designing amino acid sequences for the generated backbones (ProteinMPNN)
3. Validating that designed sequences fold correctly (ESMFold)

Current approaches require researchers to manually orchestrate these tools, understand their APIs, and handle complex parameter configurations. This creates barriers for:
- Non-expert users trying to leverage AI-driven protein design
- LLM agents attempting to automate protein design workflows
- Rapid prototyping and iteration on design hypotheses

## Goals

### Primary Goals
1. **Enable LLM-driven protein design**: Allow AI agents to autonomously execute complete binder design workflows
2. **Simplify complex pipelines**: Abstract away tool-specific details into intuitive, high-level operations
3. **Maintain scientific rigor**: Ensure generated designs follow best practices and include quality metrics

### Success Metrics
- LLM agents can successfully complete binder design tasks with >90% accuracy in tool selection
- Time to first design reduced by 80% compared to manual workflows
- All designs include validation metrics (pLDDT, pTM, interface metrics)

## Key Design Principles

### 1. Few, High-Level Tools
Research shows LLM tool selection accuracy drops significantly beyond 30 tools. This server MUST maintain fewer than 10 tools total.

| Principle | Rationale |
|-----------|-----------|
| Maximum 10 tools | Ensures reliable LLM tool selection |
| Each tool = complete workflow | Users think in workflows, not API calls |
| Sensible defaults | Minimize required parameters |

### 2. Workflow-Based Design
Each tool represents a complete user workflow, not a raw API call:
- **Bad**: `run_rfdiffusion(params...)` → `run_proteinmpnn(params...)` → `run_esmfold(params...)`
- **Good**: `design_binder(target_pdb, hotspot_residues)` → returns validated designs

### 3. Internal Orchestration
Complex multi-step pipelines should be handled internally:
- RFdiffusion → ProteinMPNN → ESMFold validation runs as single tool call
- Retry logic and error handling managed internally
- Progress reporting via MCP notifications

## Functional Requirements

### Core Tools

#### 1. `design_binder`
**Purpose**: Complete end-to-end binder design pipeline

**Inputs**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target_pdb` | string | Yes | Path to target protein PDB file |
| `hotspot_residues` | string[] | Yes | Residues on target for binder interface (e.g., ["A45", "A46", "A49"]) |
| `num_designs` | int | No | Number of designs to generate (default: 10) |
| `binder_length` | int | No | Length of binder in residues (default: 80) |

**Outputs**:
```json
{
  "designs": [
    {
      "id": "design_001",
      "sequence": "MKLLVVF...",
      "structure_pdb": "/path/to/structure.pdb",
      "metrics": {
        "plddt": 85.2,
        "ptm": 0.78,
        "interface_plddt": 82.1,
        "binding_energy": -12.5
      }
    }
  ],
  "summary": {
    "total_generated": 10,
    "passed_filters": 7,
    "best_design_id": "design_003"
  }
}
```

**Internal Pipeline**:
1. Run RFdiffusion to generate backbone structures
2. Run ProteinMPNN to design sequences (8 sequences per backbone)
3. Run ESMFold to predict structures of designed sequences
4. Calculate metrics and filter designs
5. Return ranked results

#### 2. `analyze_interface`
**Purpose**: Analyze protein-protein interface properties

**Inputs**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `complex_pdb` | string | Yes | Path to protein complex PDB |
| `chain_a` | string | Yes | Chain ID of first protein |
| `chain_b` | string | Yes | Chain ID of second protein |

**Outputs**:
```json
{
  "interface_residues": {
    "chain_a": ["45", "46", "49", "52"],
    "chain_b": ["12", "15", "16"]
  },
  "buried_surface_area": 1250.5,
  "hydrogen_bonds": 8,
  "salt_bridges": 2,
  "hydrophobic_contacts": 15,
  "shape_complementarity": 0.72
}
```

#### 3. `validate_design`
**Purpose**: Validate a designed protein sequence

**Inputs**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sequence` | string | Yes | Amino acid sequence |
| `expected_structure` | string | No | Path to expected structure PDB for comparison |

**Outputs**:
```json
{
  "predicted_structure_pdb": "/path/to/prediction.pdb",
  "plddt": 87.3,
  "ptm": 0.82,
  "pae_matrix": [[...], ...],
  "rmsd_to_expected": 1.2,  // if expected_structure provided
  "secondary_structure": "HHHHHHHCCCCEEEEEE..."
}
```

#### 4. `optimize_sequence`
**Purpose**: Optimize an existing binder sequence for improved properties

**Inputs**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `current_sequence` | string | Yes | Starting sequence |
| `target_pdb` | string | Yes | Target protein PDB |
| `optimization_target` | string | No | What to optimize: "stability", "affinity", "both" (default: "both") |
| `fixed_positions` | int[] | No | Positions to keep fixed |

**Outputs**:
```json
{
  "optimized_sequence": "MKVVLLF...",
  "mutations": ["A5V", "L12M", "G45A"],
  "predicted_improvement": {
    "stability_delta": "+2.1 kcal/mol",
    "affinity_delta": "-0.8 kcal/mol"
  },
  "metrics": {
    "plddt": 89.1,
    "interface_score": -15.2
  }
}
```

#### 5. `suggest_hotspots`
**Purpose**: Analyze a target protein and suggest potential binding hotspots

**Inputs**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target_pdb` | string | Yes | Path to target protein PDB |
| `chain_id` | string | No | Specific chain to analyze (default: first chain) |
| `criteria` | string | No | "druggable", "exposed", "conserved" (default: "exposed") |

**Outputs**:
```json
{
  "suggested_hotspots": [
    {
      "residues": ["A45", "A46", "A49"],
      "score": 0.92,
      "rationale": "Large exposed hydrophobic patch"
    },
    {
      "residues": ["A120", "A123", "A124"],
      "score": 0.85,
      "rationale": "Known functional site"
    }
  ],
  "surface_analysis": {
    "total_surface_area": 15230.5,
    "hydrophobic_patches": 3,
    "charged_regions": 5
  }
}
```

#### 6. `get_design_status`
**Purpose**: Check status of running design jobs (for long-running operations)

**Inputs**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | string | Yes | Job ID from design_binder call |

**Outputs**:
```json
{
  "status": "running",  // "queued", "running", "completed", "failed"
  "progress": {
    "current_step": "proteinmpnn",
    "designs_completed": 5,
    "total_designs": 10
  },
  "estimated_time_remaining": "5 minutes"
}
```

### Resources (Read-Only Data)

#### 1. `protein://structures/{pdb_id}`
Access to PDB structures by ID

#### 2. `protein://designs/{job_id}/{design_id}`
Access to generated design files

## Non-Functional Requirements

### Performance
- Single binder design (10 designs) should complete in <30 minutes on GPU
- ESMFold predictions should complete in <2 minutes per sequence
- Server should handle concurrent requests via job queue

### Reliability
- All operations should be idempotent where possible
- Failed jobs should be recoverable/resumable
- Results should be cached to avoid redundant computation

### Usability
- Clear error messages with actionable suggestions
- Progress reporting for long-running operations
- Example prompts and workflows in documentation

### Security
- Input validation for all PDB files
- Sandboxed execution of computational tools
- Rate limiting for resource-intensive operations

## Technical Architecture

### Technology Stack
- **MCP Framework**: Python MCP SDK
- **Core Tools**:
  - RFdiffusion (structure generation)
  - ProteinMPNN (sequence design)
  - ESMFold (structure prediction)
- **Supporting Libraries**:
  - BioPython (PDB parsing)
  - PyRosetta (optional, for interface analysis)
  - NumPy/SciPy (numerical operations)

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Server                              │
├─────────────────────────────────────────────────────────────┤
│  Tools Layer                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │design_binder │ │analyze_iface │ │validate_seq  │  ...    │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Pipeline Manager (RFdiffusion→MPNN→ESMFold)         │  │
│  │  Job Queue & Status Tracking                          │  │
│  │  Result Caching                                       │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Computation Layer                                           │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │RFdiffusion │ │ProteinMPNN │ │  ESMFold   │              │
│  └────────────┘ └────────────┘ └────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure
```
protein-design-mcp/
├── src/
│   └── protein_design_mcp/
│       ├── __init__.py
│       ├── server.py              # MCP server entry point
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── design_binder.py   # Main binder design tool
│       │   ├── analyze.py         # Interface analysis tools
│       │   ├── validate.py        # Validation tools
│       │   └── optimize.py        # Sequence optimization
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── rfdiffusion.py     # RFdiffusion wrapper
│       │   ├── proteinmpnn.py     # ProteinMPNN wrapper
│       │   └── esmfold.py         # ESMFold wrapper
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── pdb.py             # PDB file utilities
│       │   ├── metrics.py         # Quality metrics
│       │   └── cache.py           # Result caching
│       └── resources/
│           ├── __init__.py
│           └── structures.py      # PDB structure access
├── tests/
│   ├── __init__.py
│   ├── test_tools.py
│   ├── test_pipelines.py
│   └── fixtures/
│       └── test_pdbs/
├── examples/
│   ├── basic_binder_design.py
│   └── batch_design.py
├── docs/
│   └── usage.md
├── environment.yml
├── pyproject.toml
├── CLAUDE.md
├── README.md
└── LICENSE
```

## Implementation Phases

### Phase 1: Foundation
- Set up project structure and MCP server skeleton
- Implement `validate_design` tool with ESMFold integration
- Basic PDB parsing utilities
- Unit tests for core functionality

### Phase 2: Core Pipeline
- Implement RFdiffusion wrapper
- Implement ProteinMPNN wrapper
- Implement `design_binder` tool with full pipeline
- Job queue and status tracking

### Phase 3: Analysis Tools
- Implement `analyze_interface` tool
- Implement `suggest_hotspots` tool
- Add quality metrics calculations

### Phase 4: Optimization
- Implement `optimize_sequence` tool
- Add result caching
- Performance optimization

### Phase 5: Polish
- Comprehensive documentation
- Example workflows
- Integration tests
- Error handling improvements

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU memory limitations | Can't process large targets | Implement chunking, provide clear size limits |
| Long computation times | Poor user experience | Job queue, progress reporting, caching |
| Model availability | External dependencies | Containerized deployment, fallback options |
| LLM tool confusion | Incorrect tool selection | Clear tool descriptions, examples, <10 tools |

## Appendix

### Example LLM Interactions

**Simple Binder Design**:
```
User: Design a binder for the protein in target.pdb, targeting residues 45-50 on chain A

LLM: I'll use the design_binder tool to create binders targeting those residues.
[Calls design_binder with target_pdb="target.pdb", hotspot_residues=["A45","A46","A47","A48","A49","A50"]]
```

**Interface Analysis**:
```
User: Analyze the interface between chains A and B in my complex

LLM: I'll analyze the protein-protein interface for you.
[Calls analyze_interface with complex_pdb="complex.pdb", chain_a="A", chain_b="B"]
```

### References
- [MCP Specification](https://modelcontextprotocol.io/docs)
- [RFdiffusion Paper](https://www.nature.com/articles/s41586-023-06415-8)
- [ProteinMPNN Paper](https://www.science.org/doi/10.1126/science.add2187)
- [ESMFold Paper](https://www.science.org/doi/10.1126/science.ade2574)
