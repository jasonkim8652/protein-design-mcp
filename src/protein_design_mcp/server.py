"""
Protein Binder Design MCP Server

Main entry point for the MCP server that exposes protein design tools.
"""

import asyncio
import json
import logging
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server instance
server = Server("protein-design-mcp")

# Device detection: "auto" checks for CUDA availability, "cpu" forces CPU mode
_DEVICE_ENV = os.environ.get("DEVICE", "auto").lower()
if _DEVICE_ENV == "auto":
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"
else:
    DEVICE = _DEVICE_ENV

# Tools that require GPU (RFdiffusion dependency)
GPU_ONLY_TOOLS = {"design_binder", "generate_backbone"}

logger.info(f"Device mode: {DEVICE} (GPU-only tools {'enabled' if DEVICE != 'cpu' else 'disabled'})")


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    Tool(
        name="design_binder",
        description=(
            "Design protein binders for a target protein. Runs complete pipeline: "
            "RFdiffusion (backbone generation) → ProteinMPNN (sequence design) → "
            "ESMFold (structure validation). Returns ranked designs with quality metrics."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "target_pdb": {
                    "type": "string",
                    "description": "Path to target protein PDB file",
                },
                "hotspot_residues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Residues on target for binder interface, e.g., ['A45', 'A46', 'A49']"
                    ),
                },
                "num_designs": {
                    "type": "integer",
                    "description": "Number of designs to generate (default: 10)",
                    "default": 10,
                },
                "binder_length": {
                    "type": "integer",
                    "description": "Length of binder in residues (default: 80)",
                    "default": 80,
                },
            },
            "required": ["target_pdb", "hotspot_residues"],
        },
    ),
    Tool(
        name="analyze_interface",
        description=(
            "Analyze protein-protein interface properties including buried surface area, "
            "hydrogen bonds, salt bridges, and shape complementarity."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "complex_pdb": {
                    "type": "string",
                    "description": "Path to protein complex PDB file",
                },
                "chain_a": {
                    "type": "string",
                    "description": "Chain ID of first protein",
                },
                "chain_b": {
                    "type": "string",
                    "description": "Chain ID of second protein",
                },
            },
            "required": ["complex_pdb", "chain_a", "chain_b"],
        },
    ),
    Tool(
        name="validate_design",
        description=(
            "Validate a designed protein sequence by predicting its structure with ESMFold "
            "or AlphaFold2 and calculating quality metrics (pLDDT, pTM)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence to validate",
                },
                "expected_structure": {
                    "type": "string",
                    "description": "Optional path to expected structure PDB for RMSD comparison",
                },
                "predictor": {
                    "type": "string",
                    "enum": ["esmfold", "alphafold2"],
                    "default": "esmfold",
                    "description": (
                        "Structure predictor to use. ESMFold is faster, AlphaFold2 may be more accurate."
                    ),
                },
            },
            "required": ["sequence"],
        },
    ),
    Tool(
        name="optimize_sequence",
        description=(
            "Optimize an existing binder sequence for improved stability and/or binding affinity."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "current_sequence": {
                    "type": "string",
                    "description": "Starting amino acid sequence",
                },
                "target_pdb": {
                    "type": "string",
                    "description": "Path to target protein PDB",
                },
                "optimization_target": {
                    "type": "string",
                    "enum": ["stability", "affinity", "both"],
                    "description": "What to optimize (default: both)",
                    "default": "both",
                },
                "fixed_positions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Positions to keep fixed (1-indexed)",
                },
            },
            "required": ["current_sequence", "target_pdb"],
        },
    ),
    Tool(
        name="suggest_hotspots",
        description=(
            "Analyze a target protein and suggest potential binding hotspots. "
            "Can fetch structures automatically - just provide a protein name like 'EGFR', "
            "a UniProt ID like 'P00533', a PDB ID like '1IVO', or a local PDB file path. "
            "Integrates UniProt annotations, conservation, and literature for evidence-based suggestions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Target protein - can be a protein name (e.g., 'EGFR'), "
                        "UniProt ID (e.g., 'P00533'), PDB ID (e.g., '1IVO'), "
                        "or path to a local PDB file"
                    ),
                },
                "chain_id": {
                    "type": "string",
                    "description": "Specific chain to analyze (default: first chain)",
                },
                "criteria": {
                    "type": "string",
                    "enum": ["druggable", "exposed", "conserved"],
                    "description": "Hotspot selection criteria (default: exposed)",
                    "default": "exposed",
                },
                "include_literature": {
                    "type": "boolean",
                    "description": "Search PubMed for known binding partners (default: false)",
                    "default": False,
                },
            },
            "required": ["target"],
        },
    ),
    Tool(
        name="get_design_status",
        description="Check status of running design jobs for long-running operations.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID from design_binder call",
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="predict_complex",
        description=(
            "Predict the structure of a protein complex using AlphaFold2-Multimer. "
            "Use this to validate binder-target complexes and assess interface quality."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sequences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of amino acid sequences, one per chain. "
                        "E.g., [binder_sequence, target_sequence]"
                    ),
                },
                "chain_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional chain identifiers (default: A, B, C, ...)",
                },
            },
            "required": ["sequences"],
        },
    ),
    Tool(
        name="predict_structure",
        description=(
            "Predict the 3D structure of a single protein chain using ESMFold or AlphaFold2. "
            "Returns predicted PDB file path, mean pLDDT, pTM, and per-residue pLDDT scores."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence to predict structure for",
                },
                "predictor": {
                    "type": "string",
                    "enum": ["esmfold", "alphafold2"],
                    "default": "esmfold",
                    "description": (
                        "Structure predictor to use. ESMFold is faster, "
                        "AlphaFold2 may be more accurate."
                    ),
                },
            },
            "required": ["sequence"],
        },
    ),
    Tool(
        name="score_stability",
        description=(
            "Score protein stability using ESM2 pseudo-log-likelihood. "
            "Higher scores indicate more thermodynamically favorable sequences. "
            "Optionally compute per-mutation delta log-likelihood to assess the effect "
            "of point mutations on stability."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Amino acid sequence to score",
                },
                "mutations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of mutations in 'X42Y' format "
                        "(e.g., ['A42G', 'L55V']) for delta scoring"
                    ),
                },
                "reference_sequence": {
                    "type": "string",
                    "description": (
                        "Optional wild-type sequence for mutation scoring. "
                        "Inferred from mutations if not provided."
                    ),
                },
            },
            "required": ["sequence"],
        },
    ),
    Tool(
        name="energy_minimize",
        description=(
            "Energy-minimize a protein structure using OpenMM with AMBER14 force field "
            "and optional implicit solvent (GBn2). Returns minimized PDB, energy change, "
            "and RMSD from initial structure."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pdb_path": {
                    "type": "string",
                    "description": "Path to input PDB file to minimize",
                },
                "force_field": {
                    "type": "string",
                    "default": "amber14-all.xml",
                    "description": "OpenMM force field XML file",
                },
                "num_steps": {
                    "type": "integer",
                    "default": 500,
                    "description": "Maximum minimization iterations",
                },
                "solvent": {
                    "type": "string",
                    "enum": ["implicit", "none"],
                    "default": "implicit",
                    "description": "Solvent model: implicit (GBn2) or none (vacuum)",
                },
            },
            "required": ["pdb_path"],
        },
    ),
    Tool(
        name="generate_backbone",
        description=(
            "Generate de novo protein backbones using RFdiffusion unconditional generation. "
            "No target protein required. Generates novel foldable backbones of a specified length."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "length": {
                    "type": "integer",
                    "description": "Backbone length in residues",
                },
                "num_designs": {
                    "type": "integer",
                    "description": "Number of designs to generate (default: 10)",
                    "default": 10,
                },
            },
            "required": ["length"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available tools, filtering out GPU-only tools in CPU mode."""
    if DEVICE == "cpu":
        return [t for t in TOOLS if t.name not in GPU_ONLY_TOOLS]
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    # Block GPU-only tools in CPU mode
    if DEVICE == "cpu" and name in GPU_ONLY_TOOLS:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"Tool '{name}' requires GPU (RFdiffusion). "
                        f"Current device: cpu. Set DEVICE=cuda or use the GPU Docker image.",
                        "tool": name,
                    },
                    indent=2,
                ),
            )
        ]

    try:
        if name == "design_binder":
            result = await handle_design_binder(arguments)
        elif name == "analyze_interface":
            result = await handle_analyze_interface(arguments)
        elif name == "validate_design":
            result = await handle_validate_design(arguments)
        elif name == "optimize_sequence":
            result = await handle_optimize_sequence(arguments)
        elif name == "suggest_hotspots":
            result = await handle_suggest_hotspots(arguments)
        elif name == "get_design_status":
            result = await handle_get_design_status(arguments)
        elif name == "predict_complex":
            result = await handle_predict_complex(arguments)
        elif name == "predict_structure":
            result = await handle_predict_structure(arguments)
        elif name == "score_stability":
            result = await handle_score_stability(arguments)
        elif name == "energy_minimize":
            result = await handle_energy_minimize(arguments)
        elif name == "generate_backbone":
            result = await handle_generate_backbone(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        # Use compact JSON for large responses to reduce stdio overhead
        text = json.dumps(result, indent=2)
        if len(text) > 1_000_000:
            text = json.dumps(result, separators=(",", ":"))
        return [TextContent(type="text", text=text)]

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": str(e), "tool": name}, indent=2),
            )
        ]


# =============================================================================
# Tool Handlers (to be implemented)
# =============================================================================


async def handle_design_binder(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle design_binder tool call."""
    from protein_design_mcp.tools.design_binder import design_binder

    target_pdb = arguments.get("target_pdb")
    if not target_pdb:
        return {"error": "target_pdb is required"}

    hotspot_residues = arguments.get("hotspot_residues")
    if not hotspot_residues:
        return {"error": "hotspot_residues is required"}

    result = await design_binder(
        target_pdb=target_pdb,
        hotspot_residues=hotspot_residues,
        num_designs=arguments.get("num_designs", 10),
        binder_length=arguments.get("binder_length", 80),
    )
    return result


async def handle_analyze_interface(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle analyze_interface tool call."""
    from protein_design_mcp.tools.analyze import analyze_interface

    complex_pdb = arguments.get("complex_pdb")
    if not complex_pdb:
        return {"error": "complex_pdb is required"}

    chain_a = arguments.get("chain_a")
    if not chain_a:
        return {"error": "chain_a is required"}

    chain_b = arguments.get("chain_b")
    if not chain_b:
        return {"error": "chain_b is required"}

    result = await analyze_interface(
        complex_pdb=complex_pdb,
        chain_a=chain_a,
        chain_b=chain_b,
        distance_cutoff=arguments.get("distance_cutoff", 8.0),
    )
    return result


async def handle_validate_design(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle validate_design tool call."""
    from protein_design_mcp.tools.validate import validate_design

    sequence = arguments.get("sequence")
    if not sequence:
        return {"error": "sequence is required"}

    result = await validate_design(
        sequence=sequence,
        expected_structure=arguments.get("expected_structure"),
        predictor=arguments.get("predictor", "esmfold"),
    )
    return result


async def handle_optimize_sequence(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle optimize_sequence tool call."""
    from protein_design_mcp.tools.optimize import optimize_sequence

    current_sequence = arguments.get("current_sequence")
    if not current_sequence:
        return {"error": "current_sequence is required"}

    target_pdb = arguments.get("target_pdb")
    if not target_pdb:
        return {"error": "target_pdb is required"}

    result = await optimize_sequence(
        current_sequence=current_sequence,
        target_pdb=target_pdb,
        optimization_target=arguments.get("optimization_target", "both"),
        fixed_positions=arguments.get("fixed_positions"),
    )
    return result


async def handle_suggest_hotspots(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle suggest_hotspots tool call."""
    from protein_design_mcp.tools.hotspots import suggest_hotspots

    target = arguments.get("target")
    if not target:
        return {"error": "target is required"}

    result = await suggest_hotspots(
        target=target,
        chain_id=arguments.get("chain_id"),
        criteria=arguments.get("criteria", "exposed"),
        include_literature=arguments.get("include_literature", False),
    )
    return result


async def handle_get_design_status(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle get_design_status tool call."""
    from protein_design_mcp.tools.status import get_design_status

    job_id = arguments.get("job_id")
    if not job_id:
        return {"error": "job_id is required"}

    result = await get_design_status(job_id=job_id)
    return result


async def handle_predict_complex(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle predict_complex tool call using AlphaFold2-Multimer."""
    from protein_design_mcp.pipelines.alphafold2 import AlphaFold2Runner

    sequences = arguments.get("sequences")
    if not sequences:
        return {"error": "sequences is required"}

    if len(sequences) < 2:
        return {"error": "At least 2 sequences are required for complex prediction"}

    runner = AlphaFold2Runner()
    result = await runner.predict_complex(
        sequences=sequences,
        chain_names=arguments.get("chain_names"),
    )

    import tempfile

    # Write PDB to file instead of embedding inline (avoids multi-MB responses)
    pdb_file = tempfile.NamedTemporaryFile(
        suffix=".pdb", prefix="complex_", delete=False, mode="w"
    )
    pdb_file.write(result.pdb_string)
    pdb_file.close()

    response = {
        "predicted_structure_pdb": pdb_file.name,
        "plddt": result.plddt,
        "ptm": result.ptm,
        "plddt_per_residue": result.plddt_per_residue.tolist(),
        "sequences": sequences,
        "num_chains": len(sequences),
    }

    # Write PAE matrix to file if available (N x N can be huge)
    if result.pae_matrix is not None:
        pae_path = pdb_file.name.replace(".pdb", "_pae.json")
        with open(pae_path, "w") as pf:
            json.dump(result.pae_matrix.tolist(), pf)
        response["pae_matrix_path"] = pae_path

    return response


async def handle_predict_structure(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle predict_structure tool call."""
    from protein_design_mcp.tools.predict_structure import predict_structure

    sequence = arguments.get("sequence")
    if not sequence:
        return {"error": "sequence is required"}

    result = await predict_structure(
        sequence=sequence,
        predictor=arguments.get("predictor", "esmfold"),
    )
    return result


async def handle_score_stability(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle score_stability tool call."""
    from protein_design_mcp.tools.score_stability import score_stability

    sequence = arguments.get("sequence")
    if not sequence:
        return {"error": "sequence is required"}

    result = await score_stability(
        sequence=sequence,
        mutations=arguments.get("mutations"),
        reference_sequence=arguments.get("reference_sequence"),
    )
    return result


async def handle_energy_minimize(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle energy_minimize tool call."""
    from protein_design_mcp.tools.energy_minimize import energy_minimize

    pdb_path = arguments.get("pdb_path")
    if not pdb_path:
        return {"error": "pdb_path is required"}

    result = await energy_minimize(
        pdb_path=pdb_path,
        force_field=arguments.get("force_field", "amber14-all.xml"),
        num_steps=arguments.get("num_steps", 500),
        solvent=arguments.get("solvent", "implicit"),
    )
    return result


async def handle_generate_backbone(arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle generate_backbone tool call for unconditional RFdiffusion."""
    from protein_design_mcp.pipelines.rfdiffusion import run_unconditional

    length = arguments.get("length")
    if not length:
        return {"error": "length is required"}

    result = await run_unconditional(
        length=length,
        num_designs=arguments.get("num_designs", 10),
    )
    return result


# =============================================================================
# Resources
# =============================================================================


@server.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    """Return list of available resource templates."""
    return [
        ResourceTemplate(
            uriTemplate="protein://structures/{pdb_id}",
            name="PDB Structure",
            description="Access PDB structures by ID",
        ),
        ResourceTemplate(
            uriTemplate="protein://designs/{job_id}/{design_id}",
            name="Design Result",
            description="Access generated design files",
        ),
    ]


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
