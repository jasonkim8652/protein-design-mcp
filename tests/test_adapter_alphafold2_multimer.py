import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.alphafold2_multimer import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

SEQS = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDA",
    "MASSQTNSAGGGKKD",
]

SAMPLE_SCORES = {
    "plddt": [80.1, 79.4],
    "ptm": 0.282,
    "iptm": 0.0213,
    "max_pae": 25.3,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_alphafold2_multimer")


def _completed_run(tmp_path: Path, scores: dict) -> CompletedRun:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    pdb = results / "test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
    pdb.write_text("ATOM\n")
    scores_path = results / "test_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json"
    scores_path.write_text(json.dumps(scores))
    pae_path = results / "test_predicted_aligned_error_v1.json"
    pae_path.write_text(json.dumps([[0.0]]))
    return CompletedRun(
        returncode=0, stdout="Done\n", stderr="", workdir=tmp_path,
        outputs={
            "best_model_pdb": str(pdb),
            "scores_json": str(scores_path),
            "pae_json": str(pae_path),
        },
    )


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "structure_prediction"
    assert m.requires.gpu is True


def test_manifest_uses_a_mounted_prefix():
    assert _manifest().engine.prefix.endswith("/colabfold")


def test_manifest_forbids_remote_msa_modes_in_the_doc():
    doc = _manifest().doc.lower()
    assert "mmseqs2_uniref_env" in doc
    assert "single_sequence" in doc


def test_manifest_has_no_msa_mode_parameter():
    # msa-mode is derived from `msa`, never a separate exposed enum -- the
    # wrapper enforces the remote-server restriction, not just the docs.
    assert "msa_mode" not in _manifest().schema
    assert "model_type" not in _manifest().schema


def test_build_args_null_msa_uses_single_sequence_mode():
    args = build_args(
        _manifest(),
        {
            "sequences": SEQS, "msa": None, "num_recycle": 3, "num_models": 5,
            "num_seeds": 1, "random_seed": 0, "num_ensemble": 1,
            "pair_mode": "unpaired_paired", "pair_strategy": "greedy",
            "use_dropout": False, "rank": "auto", "stop_at_score": 100.0,
        },
    )
    assert "--msa-mode" in args
    assert args[args.index("--msa-mode") + 1] == "single_sequence"
    seq_arg = args[args.index("--sequences") + 1]
    assert json.loads(seq_arg) == SEQS


def test_build_args_rejects_remote_msa_mode_even_if_smuggled_in():
    # There is no msa_mode parameter (see test above), so the only way this
    # can be reached is a malformed params dict; the wrapper's own argv
    # construction must never be able to emit an mmseqs2_* mode.
    args = build_args(
        _manifest(),
        {
            "sequences": SEQS, "msa": "/tmp/complex.a3m", "num_recycle": 3,
            "num_models": 5, "num_seeds": 1, "random_seed": 0,
            "num_ensemble": 1, "pair_mode": "unpaired_paired",
            "pair_strategy": "greedy", "use_dropout": False, "rank": "auto",
            "stop_at_score": 100.0,
        },
    )
    assert "mmseqs2" not in " ".join(args)
    assert "--msa-path" in args
    assert args[args.index("--msa-path") + 1] == "/tmp/complex.a3m"


def test_parse_output_extracts_scores(tmp_path):
    result = parse_output(_manifest(), _completed_run(tmp_path, SAMPLE_SCORES))
    assert result["ptm"] == pytest.approx(0.282)
    assert result["iptm"] == pytest.approx(0.0213)


def test_parse_output_raises_when_scores_missing():
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={})
    with pytest.raises(ValueError, match="scores_json"):
        parse_output(_manifest(), run)


def test_validation_requires_msa_key_present():
    with pytest.raises(ToolInputError, match="msa"):
        validate_and_fill(_manifest(), {"sequences": SEQS})


def test_validation_fills_defaults():
    params = validate_and_fill(_manifest(), {"sequences": SEQS, "msa": None})
    assert params["num_recycle"] == 3
    assert params["num_models"] == 5
    assert params["pair_mode"] == "unpaired_paired"


def test_validation_rejects_empty_sequence_list():
    with pytest.raises(ToolInputError):
        validate_and_fill(_manifest(), {"sequences": [], "msa": None})
