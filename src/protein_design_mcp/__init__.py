"""
Protein Binder Design MCP Server

An MCP server that enables LLM agents to run end-to-end protein binder design pipelines
using RFdiffusion, ProteinMPNN, and ESMFold.
"""

__version__ = "2.4.1"
__author__ = "Protein Design MCP Team"

from protein_design_mcp.server import main

__all__ = ["main", "__version__"]
