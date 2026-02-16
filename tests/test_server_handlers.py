"""Tests for server handler wiring -- 4 stub handlers dispatching to tool functions.

Tests that:
1. Each handler calls the correct tool function with proper arguments
2. Required argument validation returns {"error": ...} when args are missing
3. Optional arguments get correct defaults
"""

from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# handle_design_binder
# ---------------------------------------------------------------------------


class TestHandleDesignBinder:
    """Tests for handle_design_binder dispatching to tools.design_binder.design_binder."""

    async def test_calls_design_binder_with_required_args(self):
        """Handler should import and call design_binder with correct arguments."""
        from protein_design_mcp.server import handle_design_binder

        mock_result = {"designs": [], "summary": {"total_generated": 0}}

        with patch(
            "protein_design_mcp.tools.design_binder.design_binder",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            result = await handle_design_binder(
                {"target_pdb": "/tmp/target.pdb", "hotspot_residues": ["A45", "A46"]}
            )

            mock_fn.assert_awaited_once_with(
                target_pdb="/tmp/target.pdb",
                hotspot_residues=["A45", "A46"],
                num_designs=10,
                binder_length=80,
            )
            assert result == mock_result

    async def test_passes_optional_args(self):
        """Handler should forward optional num_designs and binder_length."""
        from protein_design_mcp.server import handle_design_binder

        with patch(
            "protein_design_mcp.tools.design_binder.design_binder",
            new_callable=AsyncMock,
            return_value={"designs": []},
        ) as mock_fn:
            await handle_design_binder(
                {
                    "target_pdb": "/tmp/t.pdb",
                    "hotspot_residues": ["B10"],
                    "num_designs": 5,
                    "binder_length": 60,
                }
            )

            mock_fn.assert_awaited_once_with(
                target_pdb="/tmp/t.pdb",
                hotspot_residues=["B10"],
                num_designs=5,
                binder_length=60,
            )

    async def test_error_when_target_pdb_missing(self):
        """Handler should return error dict when target_pdb is missing."""
        from protein_design_mcp.server import handle_design_binder

        result = await handle_design_binder({"hotspot_residues": ["A1"]})
        assert "error" in result
        assert "target_pdb" in result["error"]

    async def test_error_when_hotspot_residues_missing(self):
        """Handler should return error dict when hotspot_residues is missing."""
        from protein_design_mcp.server import handle_design_binder

        result = await handle_design_binder({"target_pdb": "/tmp/t.pdb"})
        assert "error" in result
        assert "hotspot_residues" in result["error"]

    async def test_error_when_both_required_missing(self):
        """Handler should return error dict when called with empty arguments."""
        from protein_design_mcp.server import handle_design_binder

        result = await handle_design_binder({})
        assert "error" in result


# ---------------------------------------------------------------------------
# handle_analyze_interface
# ---------------------------------------------------------------------------


class TestHandleAnalyzeInterface:
    """Tests for handle_analyze_interface dispatching to tools.analyze.analyze_interface."""

    async def test_calls_analyze_interface_with_required_args(self):
        """Handler should call analyze_interface with correct arguments."""
        from protein_design_mcp.server import handle_analyze_interface

        mock_result = {"buried_surface_area": 1200.0, "hydrogen_bonds": 5}

        with patch(
            "protein_design_mcp.tools.analyze.analyze_interface",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            result = await handle_analyze_interface(
                {"complex_pdb": "/tmp/complex.pdb", "chain_a": "A", "chain_b": "B"}
            )

            mock_fn.assert_awaited_once_with(
                complex_pdb="/tmp/complex.pdb",
                chain_a="A",
                chain_b="B",
                distance_cutoff=8.0,
            )
            assert result == mock_result

    async def test_passes_optional_distance_cutoff(self):
        """Handler should forward custom distance_cutoff."""
        from protein_design_mcp.server import handle_analyze_interface

        with patch(
            "protein_design_mcp.tools.analyze.analyze_interface",
            new_callable=AsyncMock,
            return_value={},
        ) as mock_fn:
            await handle_analyze_interface(
                {
                    "complex_pdb": "/tmp/c.pdb",
                    "chain_a": "A",
                    "chain_b": "C",
                    "distance_cutoff": 10.0,
                }
            )

            mock_fn.assert_awaited_once_with(
                complex_pdb="/tmp/c.pdb",
                chain_a="A",
                chain_b="C",
                distance_cutoff=10.0,
            )

    async def test_error_when_complex_pdb_missing(self):
        """Handler should return error dict when complex_pdb is missing."""
        from protein_design_mcp.server import handle_analyze_interface

        result = await handle_analyze_interface({"chain_a": "A", "chain_b": "B"})
        assert "error" in result
        assert "complex_pdb" in result["error"]

    async def test_error_when_chain_a_missing(self):
        """Handler should return error dict when chain_a is missing."""
        from protein_design_mcp.server import handle_analyze_interface

        result = await handle_analyze_interface(
            {"complex_pdb": "/tmp/c.pdb", "chain_b": "B"}
        )
        assert "error" in result
        assert "chain_a" in result["error"]

    async def test_error_when_chain_b_missing(self):
        """Handler should return error dict when chain_b is missing."""
        from protein_design_mcp.server import handle_analyze_interface

        result = await handle_analyze_interface(
            {"complex_pdb": "/tmp/c.pdb", "chain_a": "A"}
        )
        assert "error" in result
        assert "chain_b" in result["error"]


# ---------------------------------------------------------------------------
# handle_optimize_sequence
# ---------------------------------------------------------------------------


class TestHandleOptimizeSequence:
    """Tests for handle_optimize_sequence dispatching to tools.optimize.optimize_sequence."""

    async def test_calls_optimize_sequence_with_required_args(self):
        """Handler should call optimize_sequence with correct arguments."""
        from protein_design_mcp.server import handle_optimize_sequence

        mock_result = {"optimized_sequence": "MKVGA", "mutations": []}

        with patch(
            "protein_design_mcp.tools.optimize.optimize_sequence",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            result = await handle_optimize_sequence(
                {"current_sequence": "MKVGA", "target_pdb": "/tmp/target.pdb"}
            )

            mock_fn.assert_awaited_once_with(
                current_sequence="MKVGA",
                target_pdb="/tmp/target.pdb",
                optimization_target="both",
                fixed_positions=None,
            )
            assert result == mock_result

    async def test_passes_optional_args(self):
        """Handler should forward optional optimization_target and fixed_positions."""
        from protein_design_mcp.server import handle_optimize_sequence

        with patch(
            "protein_design_mcp.tools.optimize.optimize_sequence",
            new_callable=AsyncMock,
            return_value={},
        ) as mock_fn:
            await handle_optimize_sequence(
                {
                    "current_sequence": "ACDE",
                    "target_pdb": "/tmp/t.pdb",
                    "optimization_target": "stability",
                    "fixed_positions": [1, 3, 5],
                }
            )

            mock_fn.assert_awaited_once_with(
                current_sequence="ACDE",
                target_pdb="/tmp/t.pdb",
                optimization_target="stability",
                fixed_positions=[1, 3, 5],
            )

    async def test_error_when_current_sequence_missing(self):
        """Handler should return error dict when current_sequence is missing."""
        from protein_design_mcp.server import handle_optimize_sequence

        result = await handle_optimize_sequence({"target_pdb": "/tmp/t.pdb"})
        assert "error" in result
        assert "current_sequence" in result["error"]

    async def test_error_when_target_pdb_missing(self):
        """Handler should return error dict when target_pdb is missing."""
        from protein_design_mcp.server import handle_optimize_sequence

        result = await handle_optimize_sequence({"current_sequence": "MKVGA"})
        assert "error" in result
        assert "target_pdb" in result["error"]

    async def test_error_when_both_required_missing(self):
        """Handler should return error dict when called with empty arguments."""
        from protein_design_mcp.server import handle_optimize_sequence

        result = await handle_optimize_sequence({})
        assert "error" in result


# ---------------------------------------------------------------------------
# handle_get_design_status
# ---------------------------------------------------------------------------


class TestHandleGetDesignStatus:
    """Tests for handle_get_design_status dispatching to tools.status.get_design_status."""

    async def test_calls_get_design_status_with_job_id(self):
        """Handler should call get_design_status with the job_id."""
        from protein_design_mcp.server import handle_get_design_status

        mock_result = {"status": "running", "job_id": "abc123"}

        with patch(
            "protein_design_mcp.tools.status.get_design_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_fn:
            result = await handle_get_design_status({"job_id": "abc123"})

            mock_fn.assert_awaited_once_with(job_id="abc123")
            assert result == mock_result

    async def test_error_when_job_id_missing(self):
        """Handler should return error dict when job_id is missing."""
        from protein_design_mcp.server import handle_get_design_status

        result = await handle_get_design_status({})
        assert "error" in result
        assert "job_id" in result["error"]

    async def test_error_when_job_id_is_none(self):
        """Handler should return error dict when job_id is explicitly None."""
        from protein_design_mcp.server import handle_get_design_status

        result = await handle_get_design_status({"job_id": None})
        assert "error" in result

    async def test_error_when_job_id_is_empty_string(self):
        """Handler should return error dict when job_id is empty string."""
        from protein_design_mcp.server import handle_get_design_status

        result = await handle_get_design_status({"job_id": ""})
        assert "error" in result
