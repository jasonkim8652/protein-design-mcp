"""
Custom exceptions for protein-design-mcp.

All exceptions inherit from ProteinDesignError for easy catching.
"""


class ProteinDesignError(Exception):
    """Base exception for protein design errors."""

    pass


class InvalidPDBError(ProteinDesignError):
    """Raised when PDB file is invalid or malformed."""

    pass


class PipelineError(ProteinDesignError):
    """Raised when a pipeline step fails."""

    pass


class RFdiffusionError(PipelineError):
    """Raised when RFdiffusion execution fails."""

    pass


class ProteinMPNNError(PipelineError):
    """Raised when ProteinMPNN execution fails."""

    pass


class ESMFoldError(PipelineError):
    """Raised when ESMFold prediction fails."""

    pass


class ValidationError(ProteinDesignError):
    """Raised when validation fails."""

    pass


class ResourceNotFoundError(ProteinDesignError):
    """Raised when a requested resource is not found."""

    pass
