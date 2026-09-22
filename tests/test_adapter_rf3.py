import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.rf3 import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

CHAINS = [
    {"chain_id": "A", "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"}
]

SAMPLE_CONF = {
    "ptm": 0.2824,
    "iptm": 0.1495,
    "has_clash": False,
    "ranking_score": 0.1761,
    "overall_plddt": 0.6421,
    "overall_pae": 23.11,
    "overall_pde": 9.94,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_rf3")


def _completed_run_with_conf(tmp_path: Path, conf: dict) -> CompletedRun:
    conf_path = tmp_path / "summary_confidences.json"
    conf_path.write_text(json.dumps(conf))
    cif_path = tmp_path / "structure.cif"
    cif_path.write_text("data_target\n#\n")
    return CompletedRun(
        returncode=0, stdout="Outputs for target written\n", stderr="",
        workdir=tmp_path,
        outputs={"summary_confidences_json": str(conf_path), "structure_cif": str(cif_path)},
    )


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.requires.gpu is True


def test_manifest_uses_a_mounted_prefix():
    assert _manifest().engine.prefix.endswith("/foundry")


def test_manifest_states_recycles_floor_of_two():
    spec = _manifest().schema["n_recycles"]
    assert spec["minimum"] == 2


def test_manifest_states_no_auto_msa():
    doc = _manifest().doc
    assert "never" in doc.lower() and "alignment" in doc.lower()


def test_build_args_serializes_chains_and_msa():
    args = build_args(
        _manifest(),
        {
            "chains": CHAINS,
            "msa": None,
            "n_recycles": 10,
            "diffusion_batch_size": 5,
            "num_steps": 50,
        },
    )
    assert "--chains" in args
    chains_arg = args[args.index("--chains") + 1]
    assert json.loads(chains_arg) == CHAINS
    assert "--msa" in args
    assert json.loads(args[args.index("--msa") + 1]) is None
    assert "--n-recycles" in args and "10" in args


def test_build_args_rejects_duplicate_chain_ids():
    with pytest.raises(ValueError, match="chain_id"):
        build_args(
            _manifest(),
            {
                "chains": [
                    {"chain_id": "A", "sequence": "ACDE"},
                    {"chain_id": "A", "sequence": "FGHI"},
                ],
                "msa": None,
                "n_recycles": 10,
                "diffusion_batch_size": 5,
                "num_steps": 50,
            },
        )


def test_build_args_rejects_msa_keys_not_matching_chain_ids():
    with pytest.raises(ValueError, match="chain_id"):
        build_args(
            _manifest(),
            {
                "chains": CHAINS,
                "msa": {"B": "/tmp/b.a3m"},
                "n_recycles": 10,
                "diffusion_batch_size": 5,
                "num_steps": 50,
            },
        )


def test_parse_output_extracts_summary_confidences():
    result = parse_output(_manifest(), _completed_run_with_conf(Path("/tmp"), SAMPLE_CONF))
    assert result["ptm"] == pytest.approx(0.2824)
    assert result["iptm"] == pytest.approx(0.1495)
    assert result["has_clash"] is False


def test_parse_output_raises_when_summary_json_is_missing():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="summary_confidences_json"):
        parse_output(_manifest(), run)


def test_validation_rejects_n_recycles_of_one():
    with pytest.raises(ToolInputError, match="n_recycles"):
        validate_and_fill(
            _manifest(), {"chains": CHAINS, "msa": None, "n_recycles": 1}
        )


def test_validation_requires_msa_key_present():
    with pytest.raises(ToolInputError, match="msa"):
        validate_and_fill(_manifest(), {"chains": CHAINS})


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"chains": CHAINS, "msa": None})
    assert params["n_recycles"] == 10
    assert params["diffusion_batch_size"] == 5
    assert params["num_steps"] == 50
