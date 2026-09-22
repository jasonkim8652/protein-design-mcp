import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.alphafold3 import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()

SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDN"


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_alphafold3")


def _base_params(**over):
    params = {
        "chains": [{"sequence": SEQ, "unpaired_msa": None, "paired_msa": None, "copies": 1}],
        "seeds": [1],
        "num_recycles": 10,
        "num_diffusion_samples": 5,
        "max_template_date": "2021-09-30",
        "resolve_msa_overlaps": True,
        "flash_attention_implementation": "triton",
        "save_embeddings": False,
        "save_distogram": False,
        "buckets": [256, 512],
        "conformer_max_iterations": None,
    }
    params.update(over)
    return params


def test_manifest_loads_and_is_not_composite():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_documents_dispatch_and_msa_policy():
    doc = _manifest().doc
    assert "## How this tool is dispatched" in doc
    assert "## MSA is optional and NEVER built by this tool" in doc


def test_build_args_msa_free_chain_becomes_empty_strings():
    args = build_args(_manifest(), _base_params())
    job = json.loads(args[0])
    assert job["chains"][0]["unpaired_msa"] == ""
    assert job["chains"][0]["paired_msa"] == ""


def test_build_args_reads_real_msa_files(tmp_path):
    unpaired = tmp_path / "unpaired.a3m"
    unpaired.write_text(">query\nMKT\n")
    paired = tmp_path / "paired.a3m"
    paired.write_text(">query\nMKT\n")

    args = build_args(
        _manifest(),
        _base_params(
            chains=[
                {
                    "sequence": SEQ,
                    "unpaired_msa": str(unpaired),
                    "paired_msa": str(paired),
                    "copies": 1,
                }
            ]
        ),
    )
    job = json.loads(args[0])
    assert job["chains"][0]["unpaired_msa"] == ">query\nMKT\n"
    assert job["chains"][0]["paired_msa"] == ">query\nMKT\n"


def test_build_args_rejects_one_null_one_path(tmp_path):
    unpaired = tmp_path / "unpaired.a3m"
    unpaired.write_text(">query\nMKT\n")
    with pytest.raises(ValueError, match="BOTH null or BOTH a path"):
        build_args(
            _manifest(),
            _base_params(
                chains=[
                    {
                        "sequence": SEQ,
                        "unpaired_msa": str(unpaired),
                        "paired_msa": None,
                        "copies": 1,
                    }
                ]
            ),
        )


def test_build_args_rejects_nonexistent_msa_path(tmp_path):
    # A nonexistent path INSIDE the working tree (tmp_path), not some
    # arbitrary absolute path -- stat() on a path outside the working tree
    # hits an unrelated sandbox PermissionError in this environment (the
    # documented pre-existing baseline quirk WAVE-COMMON warns not to
    # chase), which would mask the FileNotFoundError-shaped behavior this
    # test actually means to exercise.
    missing = tmp_path / "does_not_exist.a3m"
    with pytest.raises(ValueError, match="does not resolve to an existing file"):
        build_args(
            _manifest(),
            _base_params(
                chains=[
                    {
                        "sequence": SEQ,
                        "unpaired_msa": str(missing),
                        "paired_msa": str(missing),
                        "copies": 1,
                    }
                ]
            ),
        )


def test_build_args_rejects_missing_msa_keys():
    with pytest.raises(ValueError, match="unpaired_msa"):
        build_args(
            _manifest(),
            _base_params(chains=[{"sequence": SEQ, "copies": 1}]),
        )


def test_build_args_rejects_invalid_sequence_characters():
    with pytest.raises(ValueError, match="sequence"):
        build_args(
            _manifest(),
            _base_params(
                chains=[{"sequence": "mkt123", "unpaired_msa": None, "paired_msa": None}]
            ),
        )


def test_build_args_passes_through_scalar_job_fields():
    args = build_args(_manifest(), _base_params(num_recycles=3, num_diffusion_samples=2))
    job = json.loads(args[0])
    assert job["num_recycles"] == 3
    assert job["num_diffusion_samples"] == 2
    assert job["seeds"] == [1]
    assert job["buckets"] == [256, 512]


def test_parse_output_extracts_summary_metrics(tmp_path):
    summary_path = tmp_path / "job_summary_confidences.json"
    summary_path.write_text(
        json.dumps(
            {
                "ranking_score": 0.87,
                "ptm": 0.9,
                "iptm": 0.8,
                "fraction_disordered": 0.1,
                "has_clash": False,
            }
        )
    )
    result = parse_output(
        _manifest(),
        CompletedRun(
            returncode=0,
            stdout="",
            stderr="",
            workdir=tmp_path,
            outputs={
                "summary_confidences_json": str(summary_path),
                "per_sample_cifs": [str(tmp_path / "a.cif"), str(tmp_path / "b.cif")],
            },
        ),
    )
    assert result["ranking_score"] == pytest.approx(0.87)
    assert result["ptm"] == pytest.approx(0.9)
    assert result["iptm"] == pytest.approx(0.8)
    assert result["has_clash"] is False
    assert result["num_samples"] == 2


def test_parse_output_raises_when_summary_missing():
    with pytest.raises(ValueError, match="summary_confidences_json"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={}),
        )


# --- Corner cases required by CLAUDE.md's TDD workflow ---------------------


def test_parse_output_zero_valued_metrics_survive(tmp_path):
    """ptm/iptm = 0.0 and has_clash = False must survive -- 0/False are
    valid values, not "missing"."""
    summary_path = tmp_path / "job_summary_confidences.json"
    summary_path.write_text(
        json.dumps(
            {"ranking_score": 0.0, "ptm": 0.0, "iptm": 0.0, "fraction_disordered": 0.0, "has_clash": False}
        )
    )
    result = parse_output(
        _manifest(),
        CompletedRun(
            returncode=0, stdout="", stderr="", workdir=tmp_path,
            outputs={"summary_confidences_json": str(summary_path), "per_sample_cifs": []},
        ),
    )
    assert result["ranking_score"] == 0.0
    assert result["ptm"] == 0.0
    assert result["has_clash"] is False
    assert result["num_samples"] == 0


def test_parse_output_single_sample_cif_not_a_list(tmp_path):
    """When only one sample is produced, run.outputs['per_sample_cifs'] may
    be a bare string rather than a one-item list (see results.collect_outputs'
    'multiple' semantics) -- num_samples must still be 1, not len(str)."""
    summary_path = tmp_path / "job_summary_confidences.json"
    summary_path.write_text(json.dumps({"ranking_score": 0.5}))
    result = parse_output(
        _manifest(),
        CompletedRun(
            returncode=0, stdout="", stderr="", workdir=tmp_path,
            outputs={
                "summary_confidences_json": str(summary_path),
                "per_sample_cifs": str(tmp_path / "only.cif"),
            },
        ),
    )
    assert result["num_samples"] == 1


def test_build_args_single_chain_boundary():
    """Boundary: exactly one chain, minItems=1, must work."""
    args = build_args(_manifest(), _base_params())
    job = json.loads(args[0])
    assert len(job["chains"]) == 1


def test_validation_rejects_empty_chains_array():
    with pytest.raises(ToolInputError, match="chains"):
        validate_and_fill(_manifest(), {"chains": []})


def test_validation_fills_defaults():
    params = validate_and_fill(
        _manifest(),
        {"chains": [{"sequence": SEQ, "unpaired_msa": None, "paired_msa": None}]},
    )
    assert params["seeds"] == [1]
    assert params["num_recycles"] == 10
    assert params["num_diffusion_samples"] == 5
    assert params["max_template_date"] == "2021-09-30"
    assert params["conformer_max_iterations"] is None
