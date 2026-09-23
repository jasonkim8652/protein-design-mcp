import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.rfdiffusion3_binder import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

BASE_PARAMS = {
    "target_pdb": "/tmp/target.pdb",
    "contig": "50-50,/0,A1-150",
    "select_hotspots": "A62,A89,A121",
    "length": None,
    "redesign_motif_sidechains": False,
    "partial_t": None,
    "diffusion_batch_size": 8,
    "num_timesteps": 200,
    "step_scale": 1.5,
    "seed": None,
}


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_rfdiffusion3_binder")


def _completed_run(tmp_path: Path, metrics: dict, num_structures: int = 1) -> CompletedRun:
    structures = []
    metadata = []
    for i in range(num_structures):
        cif = tmp_path / f"input_job_0_model_{i}.cif.gz"
        cif.write_bytes(b"\x1f\x8b")
        structures.append(str(cif))
        meta_path = tmp_path / f"input_job_0_model_{i}.json"
        meta_path.write_text(json.dumps({
            "metrics": metrics,
            "diffused_index_map": {"A1": "B1"},
            "ckpt_path": "/home/jk661/.foundry/checkpoints/rfd3_latest.ckpt",
            "seed": None,
        }))
        metadata.append(str(meta_path))
    return CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"structure_cif": structures, "metadata_json": metadata},
    )


def test_manifest_loads_and_requires_gpu():
    m = _manifest()
    assert m.category == "binder_generation"
    assert m.requires.gpu is True


def test_build_args_serializes_job_and_engine_sections():
    args = build_args(_manifest(), BASE_PARAMS)
    payload = json.loads(args[0])
    assert payload["job"]["input"] == "/tmp/target.pdb"
    assert payload["job"]["contig"] == "50-50,/0,A1-150"
    assert payload["job"]["select_hotspots"] == "A62,A89,A121"
    assert payload["engine"]["diffusion_batch_size"] == 8
    assert payload["engine"]["seed"] is None


def test_build_args_omits_select_hotspots_when_none():
    params = dict(BASE_PARAMS, select_hotspots=None)
    args = build_args(_manifest(), params)
    payload = json.loads(args[0])
    assert "select_hotspots" not in payload["job"]


def test_build_args_omits_length_and_partial_t_when_none():
    args = build_args(_manifest(), BASE_PARAMS)
    payload = json.loads(args[0])
    assert "length" not in payload["job"]
    assert "partial_t" not in payload["job"]


def test_build_args_includes_partial_t_when_set():
    params = dict(BASE_PARAMS, partial_t=5.0)
    args = build_args(_manifest(), params)
    payload = json.loads(args[0])
    assert payload["job"]["partial_t"] == 5.0


def test_parse_output_reads_rank0_metrics(tmp_path):
    metrics = {"n_clashing.interresidue_clashes_w_sidechain": 0, "radius_of_gyration": 12.3}
    run = _completed_run(tmp_path, metrics, num_structures=3)
    result = parse_output(_manifest(), run)
    assert result["num_structures"] == 3
    assert result["metrics"] == metrics
    assert result["diffused_index_map"] == {"A1": "B1"}
    assert result["ckpt_path"].endswith("rfd3_latest.ckpt")


def test_parse_output_raises_when_metadata_missing(tmp_path):
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    with pytest.raises(ValueError, match="metadata_json"):
        parse_output(_manifest(), run)


def test_validation_rejects_select_hotspots_with_space():
    with pytest.raises(ToolInputError, match="select_hotspots"):
        validate_and_fill(
            _manifest(),
            {
                "target_pdb": "/tmp/target.pdb",
                "contig": "50-50,/0,A1-150",
                "select_hotspots": "A62, A89",
            },
        )


def test_validation_rejects_contig_with_rfdiffusion1_style_slash_break_and_spaces():
    with pytest.raises(ToolInputError, match="contig"):
        validate_and_fill(
            _manifest(),
            {"target_pdb": "/tmp/target.pdb", "contig": "A1-150/0 50-50"},
        )


def test_validation_accepts_wellformed_contig_and_hotspots():
    params = validate_and_fill(
        _manifest(),
        {
            "target_pdb": "/tmp/target.pdb",
            "contig": "50-50,/0,A1-150",
            "select_hotspots": "A62,A89,A121",
        },
    )
    assert params["select_hotspots"] == "A62,A89,A121"


def test_validation_defaults_select_hotspots_to_none():
    params = validate_and_fill(
        _manifest(),
        {"target_pdb": "/tmp/target.pdb", "contig": "50-50,/0,A1-150"},
    )
    assert params.get("select_hotspots") is None


def test_validation_fills_diffusion_defaults():
    params = validate_and_fill(
        _manifest(),
        {"target_pdb": "/tmp/target.pdb", "contig": "50-50,/0,A1-150"},
    )
    assert params["diffusion_batch_size"] == 8
    assert params["num_timesteps"] == 200
    assert params["step_scale"] == 1.5


# --- the co-generated sequence must reach the caller -------------------------


def test_parse_output_returns_the_sequence_rfdiffusion3_generated(tmp_path):
    """RFdiffusion3 co-generates a real sequence, not a poly-alanine
    placeholder -- confirmed by reading a run's output, where the designed
    chain A came back with 18 distinct residue types. It was buried in a
    gzipped CIF that the result never mentioned, so a caller could not tell it
    existed and the predictable next move was to run run_mpnn for one --
    discarding a sequence the model had already produced, and (without
    chains_to_design) redesigning the target along with it.
    """
    import gzip, json
    from pathlib import Path
    from protein_design_mcp.adapters.rfdiffusion3_binder import parse_output
    from protein_design_mcp.dispatch.env import CompletedRun

    cif = tmp_path / "m0.cif.gz"
    rows = [
        ("ATOM", "N", "N", ".", "GLY", "A", "1", "1"),
        ("ATOM", "C", "CA", ".", "GLY", "A", "1", "1"),
        ("ATOM", "C", "CA", ".", "TRP", "A", "1", "2"),
        ("ATOM", "C", "CA", ".", "LYS", "B", "1", "1"),
    ]
    header = ["group_PDB", "type_symbol", "label_atom_id", "label_alt_id",
              "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id"]
    body = "loop_\n" + "".join(f"_atom_site.{h}\n" for h in header)
    body += "".join(" ".join(r) + "\n" for r in rows)
    gzip.open(cif, "wt").write("data_x\n" + body + "#\n")

    meta = tmp_path / "m0.json"
    meta.write_text(json.dumps({"metrics": {}, "ckpt_path": "c", "diffused_index_map": {}}))

    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path,
                       outputs={"metadata_json": [str(meta)], "structure_cif": [str(cif)]})
    out = parse_output(None, run)
    assert out["sequences"] == [{"A": "GW", "B": "K"}], out.get("sequences")


def test_parse_output_survives_a_cif_it_cannot_read(tmp_path):
    """A sequence is a bonus on top of the structure. Failing the whole call
    because one CIF did not parse would throw away a finished GPU run."""
    import json
    from protein_design_mcp.adapters.rfdiffusion3_binder import parse_output
    from protein_design_mcp.dispatch.env import CompletedRun

    bad = tmp_path / "bad.cif.gz"
    bad.write_bytes(b"not gzip at all")
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"metrics": {}, "ckpt_path": "c", "diffused_index_map": {}}))
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path,
                       outputs={"metadata_json": [str(meta)], "structure_cif": [str(bad)]})
    out = parse_output(None, run)
    assert out["num_structures"] == 1
    assert out["sequences"] == [None]
