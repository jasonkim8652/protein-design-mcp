import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.promera import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

CHAINS = {
    "A1": {
        "type": "protein",
        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "entity_id": 1,
    }
}

SAMPLE_CONF = {
    "complex_plddt": 0.6474,
    "complex_ptm": 0.0946,
    "chain_plddt": {"A1": 0.6474},
    "ptm": {"A1": 0.0946},
    "iCS": {},
    "msa_depth": {"A1": 1},
    "msa_path": {"A1": None},
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_promera")


def _completed_run_with_conf(tmp_path: Path, conf: dict) -> CompletedRun:
    conf_path = tmp_path / "confidence.json"
    conf_path.write_text(json.dumps(conf))
    cif_path = tmp_path / "structure.cif"
    cif_path.write_text("data_target\n#\n")
    return CompletedRun(
        returncode=0,
        stdout="[12:00:00 rank0] Done batch (1 target(s))\n",
        stderr="",
        workdir=tmp_path,
        outputs={"confidence_json": str(conf_path), "structure_cif": str(cif_path)},
    )


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.composite is False
    assert m.requires.gpu is True


def test_manifest_uses_a_mounted_prefix():
    m = _manifest()
    assert m.engine.prefix is not None
    assert m.engine.prefix.endswith("/promera")


def test_manifest_sets_promera_weights_env_var():
    assert "PROMERA_WEIGHTS" in _manifest().engine.env_vars


def test_manifest_distinguishes_itself_from_ipsae():
    doc = _manifest().doc
    assert "run_ipsae" in doc


def test_manifest_excludes_design_task():
    doc = (_manifest().doc + _manifest().summary).lower()
    assert "design" not in doc.split("## excluded")[-1] if "## excluded" in doc.lower() else True


def test_build_args_serializes_chains_and_null_msa():
    args = build_args(
        _manifest(),
        {
            "chains": CHAINS,
            "msa": None,
            "recycling_steps": 4,
            "diffusion_samples": 5,
            "diffusion_steps": 200,
            "num_seeds": 1,
        },
    )
    assert "--schema" in args
    schema_arg = args[args.index("--schema") + 1]
    assert json.loads(schema_arg) == CHAINS
    assert "--msa" in args
    msa_arg = args[args.index("--msa") + 1]
    assert json.loads(msa_arg) is None
    assert "--recycling-steps" in args and "4" in args


def test_build_args_rejects_msa_keys_not_matching_chains():
    with pytest.raises(ValueError, match="chain"):
        build_args(
            _manifest(),
            {
                "chains": CHAINS,
                "msa": {"B1": None},
                "recycling_steps": 4,
                "diffusion_samples": 5,
                "diffusion_steps": 200,
                "num_seeds": 1,
            },
        )


def test_build_args_accepts_msa_dict_covering_every_chain():
    args = build_args(
        _manifest(),
        {
            "chains": CHAINS,
            "msa": {"A1": "/tmp/a1.a3m"},
            "recycling_steps": 4,
            "diffusion_samples": 5,
            "diffusion_steps": 200,
            "num_seeds": 1,
        },
    )
    msa_arg = args[args.index("--msa") + 1]
    assert json.loads(msa_arg) == {"A1": "/tmp/a1.a3m"}


def test_parse_output_returns_confidence_and_notes_no_ics():
    result = parse_output(_manifest(), _completed_run_with_conf(Path("/tmp"), SAMPLE_CONF))
    assert result["complex_plddt"] == pytest.approx(0.6474)
    assert result["chain_plddt"] == {"A1": 0.6474}


def test_parse_output_raises_when_confidence_json_is_missing():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="confidence_json"):
        parse_output(_manifest(), run)


def test_validation_requires_msa_key_present():
    with pytest.raises(ToolInputError, match="msa"):
        validate_and_fill(_manifest(), {"chains": CHAINS})


def test_validation_accepts_explicit_null_msa():
    params = validate_and_fill(_manifest(), {"chains": CHAINS, "msa": None})
    assert params["msa"] is None


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"chains": CHAINS, "msa": None})
    assert params["recycling_steps"] == 4
    assert params["diffusion_samples"] == 5
    assert params["diffusion_steps"] == 200
    assert params["num_seeds"] == 1
