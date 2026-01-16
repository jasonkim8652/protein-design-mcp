"""
MCP Resources for protein data access.

This module provides resource handlers for:
- protein://structures/{pdb_id} - Access PDB structures
- protein://designs/{job_id}/{design_id} - Access generated designs
"""

from protein_design_mcp.resources.structures import get_structure, list_structures

__all__ = [
    "get_structure",
    "list_structures",
]
