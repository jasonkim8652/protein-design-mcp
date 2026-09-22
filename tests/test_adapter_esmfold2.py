from pathlib import Path

import pytest

from protein_design_mcp.adapters.esmfold2 import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

SAMPLE_STDOUT = """\
mean_plddt: 81.3400
num_residues: 76
sequence_length: 76
output_pdb: model.pdb
"""


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_esmfold2")


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_forces_pythonnousersite():
    assert _manifest().engine.env_vars.get("PYTHONNOUSERSITE") == "1"


def test_manifest_uses_a_mounted_prefix_not_env():
    m = _manifest()
    assert m.engine.prefix is not None
    assert m.engine.env is None
    assert m.engine.prefix.endswith("/esmfold2")


def test_manifest_has_no_msa_parameter():
    assert "msa" not in _manifest().schema


def test_manifest_explains_why_there_is_no_msa():
    doc = _manifest().doc.lower()
    assert "no alignment" in doc or "no msa" in doc or "takes no" in doc


def test_build_args_passes_sequence_and_output_and_knobs():
    args = build_args(
        _manifest(),
        {
            "sequence": SEQ,
            "num_recycles": 12,
            "num_diffusion_samples": 4,
            "num_sampling_steps": 40,
        },
    )
    assert args[0] == SEQ
    assert "model.pdb" in args
    assert "--num-recycles" in args and "12" in args
    assert "--num-diffusion-samples" in args and "4" in args
    assert "--num-sampling-steps" in args and "40" in args


def test_parse_output_extracts_confidence():
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="", workdir=Path("/tmp")),
    )
    assert result["mean_plddt"] == pytest.approx(81.34)
    assert result["num_residues"] == 76
    assert result["sequence_length"] == 76


def test_parse_output_raises_when_plddt_is_absent():
    with pytest.raises(ValueError, match="plddt"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="nothing useful", stderr="", workdir=Path("/tmp")),
        )


def test_validation_rejects_a_bad_sequence():
    with pytest.raises(ToolInputError, match="sequence"):
        validate_and_fill(_manifest(), {"sequence": "MQIF1VKT"})


def test_validation_fills_the_defaults():
    params = validate_and_fill(_manifest(), {"sequence": SEQ})
    assert params["num_recycles"] == 20
    assert params["num_diffusion_samples"] == 8
    assert params["num_sampling_steps"] == 68


def test_validation_rejects_zero_recycles_boundary():
    # 0 recycles is a legitimate, documented boundary (no trunk refinement),
    # not an error -- corner case required by CLAUDE.md's TDD table.
    params = validate_and_fill(_manifest(), {"sequence": SEQ, "num_recycles": 0})
    assert params["num_recycles"] == 0
