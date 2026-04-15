"""
High-level MCP tools for protein design.

Core tools exposed via MCP:
- design_binder: End-to-end binder design pipeline (RFdiffusion + ProteinMPNN + ESMFold)
- design_fold: End-to-end de novo fold design (RFdiffusion + ProteinMPNN + AlphaFold2)
- design_sequence: ProteinMPNN sequence design for a given backbone
- analyze_interface: Protein-protein interface analysis
- validate_design: Structure prediction and validation
- optimize_sequence: Sequence optimization (ProteinMPNN)
- suggest_hotspots: Binding site prediction

Optional tools (imported lazily by server handlers — require extra deps):
- rosetta_score / rosetta_relax / rosetta_interface_score / rosetta_design
  → `pip install "protein-design-mcp[rosetta]"` + valid PyRosetta license
- predict_structure_boltz / predict_affinity_boltz
  → `pip install "protein-design-mcp[boltz]"` (needs torch>=2.2)
"""

from protein_design_mcp.tools.design_binder import design_binder
from protein_design_mcp.tools.design_fold import design_fold
from protein_design_mcp.tools.design_sequence import design_sequence
from protein_design_mcp.tools.analyze import analyze_interface
from protein_design_mcp.tools.validate import validate_design
from protein_design_mcp.tools.optimize import optimize_sequence
from protein_design_mcp.tools.hotspots import suggest_hotspots

__all__ = [
    "design_binder",
    "design_fold",
    "design_sequence",
    "analyze_interface",
    "validate_design",
    "optimize_sequence",
    "suggest_hotspots",
]
