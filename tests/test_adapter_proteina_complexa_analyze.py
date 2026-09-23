from pathlib import Path

import pytest
import yaml

from protein_design_mcp.adapters.proteina_complexa_analyze import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import parse_manifest
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()
MANIFEST_PATH = MANIFEST_DIR / "run_proteina_complexa_analyze.yaml"


def _manifest():
    """See test_adapter_proteina_complexa_generate.py's _manifest() docstring
    for why this parses the file directly rather than via the directory-wide
    loaders."""
    return parse_manifest(yaml.safe_load(MANIFEST_PATH.read_text()))


def _staged_paths(tmp_path, names):
    """Simulate what engine.stage's stage_inputs already did to
    params["structure_paths"] by the time build_args sees it: each entry
    lives at workdir/structure_paths/<basename>."""
    stage_dir = tmp_path / "structure_paths"
    stage_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = stage_dir / name
        p.write_text("ATOM\n")
        paths.append(str(p))
    return paths


def _base_params(tmp_path, **overrides):
    args = {
        "structure_paths": ["/tmp/a.pdb", "/tmp/b.pdb"],
        "sequences": ["MKT", "MKV"],
        **overrides,
    }
    return validate_and_fill(_manifest(), args)


# --- manifest shape ---


def test_manifest_loads_and_is_cpu_run_analysis():
    m = _manifest()
    assert m.category == "run_analysis"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_uses_prefix_not_env():
    engine = _manifest().engine
    assert engine.prefix == "/home/jk661/.conda/envs/proteina_complexa"
    assert engine.env is None
    assert engine.entry == ("complexa", "analyze")
    assert engine.repo == "proteinfoundation"


def test_manifest_stages_structure_paths():
    assert _manifest().engine.stage == ("structure_paths",)


def test_manifest_sets_complexa_init():
    """complexa refuses every step but init/demo/download/validate/status
    without COMPLEXA_INIT set -- verified live, 2026-09-22 (see
    run_proteina_complexa_generate's identical check)."""
    assert _manifest().engine.env_vars.get("COMPLEXA_INIT") == "uv"


def test_manifest_sets_foldseek_and_mmseqs_exec():
    env_vars = _manifest().engine.env_vars
    assert env_vars.get("FOLDSEEK_EXEC") == "/home/jk661/.local/bin/foldseek"
    assert env_vars.get("MMSEQS_EXEC") == "/usr/local/bin/mmseqs"


def test_structure_paths_and_sequences_are_required():
    schema = _manifest().schema
    assert schema["structure_paths"]["required"] is True
    assert schema["sequences"]["required"] is True


def test_optional_metrics_are_not_required():
    schema = _manifest().schema
    for key in (
        "designability_scrmsd_ca",
        "codesignability_scrmsd_ca",
        "codesignability_scrmsd_all_atom",
        "interface_hbonds_tmol",
    ):
        assert schema[key].get("required", False) is False


def test_doc_documents_the_verified_finding():
    doc = _manifest().doc
    assert "Verified: diversity needs only structures" in doc
    assert "Not verified live" in doc


def test_doc_names_the_producing_tool_for_each_optional_metric():
    doc = _manifest().doc
    assert "run_mpnn" in doc
    assert "run_esmfold2" in doc


# --- validation corner cases ---


def test_validation_rejects_missing_structure_paths():
    with pytest.raises(ToolInputError, match="structure_paths"):
        validate_and_fill(_manifest(), {"sequences": ["MKT"]})


def test_validation_rejects_missing_sequences():
    with pytest.raises(ToolInputError, match="sequences"):
        validate_and_fill(_manifest(), {"structure_paths": ["/tmp/a.pdb"]})


def test_validation_rejects_empty_structure_paths():
    with pytest.raises(ToolInputError):
        validate_and_fill(
            _manifest(), {"structure_paths": [], "sequences": []}
        )


def test_validation_accepts_single_design():
    params = validate_and_fill(
        _manifest(), {"structure_paths": ["/tmp/a.pdb"], "sequences": ["MKT"]}
    )
    assert len(params["structure_paths"]) == 1


def test_validation_fills_defaults(tmp_path):
    params = _base_params(tmp_path)
    assert params["result_type"] == "protein_binder"
    assert params["compute_foldseek_diversity"] is True
    assert params["mmseqs_min_seq_id"] == 0.1
    assert params["analysis_modes"] == ["binder", "monomer"]
    assert "designability_scrmsd_ca" not in params


def test_validation_accepts_zero_mmseqs_min_seq_id(tmp_path):
    """0.0 is a legitimate (if extreme) value, not falsy-for-'unset'."""
    params = _base_params(tmp_path, mmseqs_min_seq_id=0.0)
    assert params["mmseqs_min_seq_id"] == 0.0


# --- build_args: CSV materialization ---


def test_build_args_writes_binder_results_csv_into_recovered_workdir(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb", "design_1.pdb"])
    params = validate_and_fill(
        _manifest(), {"structure_paths": staged, "sequences": ["MKT", "MKV"]}
    )
    build_args(_manifest(), params)
    csv_path = tmp_path / "analyze_results" / "binder_results_analyze_0.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().splitlines()
    assert len(rows) == 3  # header + 2 designs


def test_build_args_csv_has_pdb_path_and_sequence_columns(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb", "design_1.pdb"])
    params = validate_and_fill(
        _manifest(), {"structure_paths": staged, "sequences": ["MKT", "MKV"]}
    )
    build_args(_manifest(), params)
    csv_path = tmp_path / "analyze_results" / "binder_results_analyze_0.csv"
    header = csv_path.read_text().splitlines()[0]
    assert "pdb_path" in header
    assert "sequence" in header
    assert "run_name" in header
    assert "task_name" in header


def test_build_args_csv_includes_optional_metric_columns_only_when_supplied(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb", "design_1.pdb"])
    params = validate_and_fill(
        _manifest(),
        {
            "structure_paths": staged,
            "sequences": ["MKT", "MKV"],
            "designability_scrmsd_ca": [1.2, 2.3],
        },
    )
    build_args(_manifest(), params)
    csv_path = tmp_path / "analyze_results" / "binder_results_analyze_0.csv"
    header = csv_path.read_text().splitlines()[0]
    assert "_res_scRMSD_ca_external" in header
    assert "_res_co_scRMSD_ca_external" not in header
    assert "generated_n_interface_hbonds_tmol" not in header


def test_build_args_csv_preserves_zero_valued_optional_metric(tmp_path):
    """A hydrogen-bond count of 0 is real data, not 'omit this design'."""
    staged = _staged_paths(tmp_path, ["design_0.pdb", "design_1.pdb"])
    params = validate_and_fill(
        _manifest(),
        {
            "structure_paths": staged,
            "sequences": ["MKT", "MKV"],
            "interface_hbonds_tmol": [0, 3],
        },
    )
    build_args(_manifest(), params)
    csv_path = tmp_path / "analyze_results" / "binder_results_analyze_0.csv"
    import csv as csv_module

    with csv_path.open() as handle:
        rows = list(csv_module.DictReader(handle))
    assert rows[0]["generated_n_interface_hbonds_tmol"] == "0"
    assert rows[1]["generated_n_interface_hbonds_tmol"] == "3"


def test_build_args_rejects_sequences_length_mismatch(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb", "design_1.pdb"])
    params = validate_and_fill(
        _manifest(), {"structure_paths": staged, "sequences": ["MKT"]}
    )
    with pytest.raises(ValueError, match="sequences"):
        build_args(_manifest(), params)


def test_build_args_rejects_optional_metric_length_mismatch(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb", "design_1.pdb"])
    params = validate_and_fill(
        _manifest(),
        {
            "structure_paths": staged,
            "sequences": ["MKT", "MKV"],
            "designability_scrmsd_ca": [1.2],
        },
    )
    with pytest.raises(ValueError, match="designability_scrmsd_ca"):
        build_args(_manifest(), params)


def test_build_args_handles_single_design(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb"])
    params = validate_and_fill(
        _manifest(), {"structure_paths": staged, "sequences": ["MKT"]}
    )
    build_args(_manifest(), params)
    csv_path = tmp_path / "analyze_results" / "binder_results_analyze_0.csv"
    rows = csv_path.read_text().splitlines()
    assert len(rows) == 2  # header + 1 design


# --- build_args: argv ---


def test_build_args_uses_fixed_analyze_config(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb"])
    params = validate_and_fill(
        _manifest(), {"structure_paths": staged, "sequences": ["MKT"]}
    )
    args = build_args(_manifest(), params)
    assert args[0] == "/home/jk661/projects/proteina-complexa/configs/analyze.yaml"
    assert args[-1] == "--verbose"


def test_build_args_points_results_dir_at_materialized_csv_dir(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb"])
    params = validate_and_fill(
        _manifest(), {"structure_paths": staged, "sequences": ["MKT"]}
    )
    args = build_args(_manifest(), params)
    assert "++results_dir=analyze_results" in args
    assert "++base_config_name=analyze" in args


def test_build_args_omits_threshold_overrides_when_unset(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb"])
    params = validate_and_fill(
        _manifest(), {"structure_paths": staged, "sequences": ["MKT"]}
    )
    args = build_args(_manifest(), params)
    assert not any("success_thresholds" in a for a in args)
    assert not any("designability_thresholds" in a for a in args)


def test_build_args_renders_nested_success_thresholds(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb"])
    params = validate_and_fill(
        _manifest(),
        {
            "structure_paths": staged,
            "sequences": ["MKT"],
            "success_thresholds": {"scRMSD": {"threshold": 2.0, "op": "<"}},
        },
    )
    args = build_args(_manifest(), params)
    joined = " ".join(args)
    assert "++aggregation.success_thresholds=" in joined
    assert "scRMSD" in joined
    assert "threshold:2.0" in joined


def test_build_args_renders_analysis_modes_list(tmp_path):
    staged = _staged_paths(tmp_path, ["design_0.pdb"])
    params = validate_and_fill(
        _manifest(),
        {"structure_paths": staged, "sequences": ["MKT"], "analysis_modes": ["binder"]},
    )
    args = build_args(_manifest(), params)
    assert '++aggregation.analysis_modes=["binder"]' in args


def test_build_args_raises_on_empty_structure_paths_defensively(tmp_path):
    """Belt-and-suspenders: validation already enforces minItems=1, but
    build_args must never divide/index into an empty list either."""
    with pytest.raises(ValueError):
        build_args(_manifest(), {"structure_paths": [], "sequences": []})


# --- parse_output ---


def _write_diversity_csv(path: Path, column: str, value: tuple) -> None:
    path.write_text(f"run_name,task_name,{column}\nmcp_run,mcp_analyze_set,\"{value}\"\n")


def test_parse_output_reads_foldseek_and_mmseqs_diversity(tmp_path):
    foldseek_csv = tmp_path / "res_div_foldseek_binder_all_generated.csv"
    _write_diversity_csv(
        foldseek_csv, "_res_diversity_foldseek_binder_all_generated", (0.5, 1, 2)
    )
    mmseqs_csv = tmp_path / "res_div_mmseqs_all_generated.csv"
    _write_diversity_csv(mmseqs_csv, "_res_diversity_mmseqs_all_generated", (1.0, 2, 2))
    combined_csv = tmp_path / "RAW_protein_binder_results_analyze_combined.csv"
    combined_csv.write_text("pdb_path\n/tmp/a.pdb\n/tmp/b.pdb\n")

    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={
            "diversity_foldseek_csv": [str(foldseek_csv)],
            "diversity_mmseqs_csv": [str(mmseqs_csv)],
            "combined_results_csv": [str(combined_csv)],
        },
    )
    result = parse_output(_manifest(), run)
    assert result["foldseek_diversity"] == {"score": 0.5, "num_clusters": 1, "num_samples": 2}
    assert result["mmseqs_diversity"] == {"score": 1.0, "num_clusters": 2, "num_samples": 2}
    assert result["num_designs"] == 2


def test_parse_output_picks_all_generated_row_among_several(tmp_path):
    """When success-threshold subsets ALSO produced diversity CSVs (because
    optional metrics were supplied), the driver must still report the
    FULL-set ("all_generated") row, not an arbitrary one."""
    successful_csv = tmp_path / "res_div_foldseek_binder_successful_self.csv"
    _write_diversity_csv(
        successful_csv, "_res_diversity_foldseek_binder_successful_self", (0.9, 1, 1)
    )
    all_csv = tmp_path / "res_div_foldseek_binder_all_generated.csv"
    _write_diversity_csv(
        all_csv, "_res_diversity_foldseek_binder_all_generated", (0.4, 2, 5)
    )
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"diversity_foldseek_csv": [str(successful_csv), str(all_csv)]},
    )
    result = parse_output(_manifest(), run)
    assert result["foldseek_diversity"]["num_samples"] == 5


def test_parse_output_handles_single_structure_diversity_result(tmp_path):
    """diversity_foldseek returns (1.0, 1, 1) for exactly one valid
    structure (see compute_diversity.py) -- must parse cleanly, not crash."""
    foldseek_csv = tmp_path / "res_div_foldseek_binder_all_generated.csv"
    _write_diversity_csv(
        foldseek_csv, "_res_diversity_foldseek_binder_all_generated", (1.0, 1, 1)
    )
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"diversity_foldseek_csv": [str(foldseek_csv)]},
    )
    result = parse_output(_manifest(), run)
    assert result["foldseek_diversity"] == {"score": 1.0, "num_clusters": 1, "num_samples": 1}


def test_parse_output_handles_none_diversity_cell(tmp_path):
    """compute_foldseek_diversity appends None for a group with zero valid
    structures -- must degrade to a None result, not crash on ast.literal_eval."""
    foldseek_csv = tmp_path / "res_div_foldseek_binder_all_generated.csv"
    foldseek_csv.write_text(
        "run_name,_res_diversity_foldseek_binder_all_generated\nmcp_run,\n"
    )
    run = CompletedRun(
        returncode=0, stdout="", stderr="", workdir=tmp_path,
        outputs={"diversity_foldseek_csv": [str(foldseek_csv)]},
    )
    result = parse_output(_manifest(), run)
    assert result["foldseek_diversity"] is None


def test_parse_output_handles_missing_diversity_outputs_entirely(tmp_path):
    """Both diversity computations disabled -- must not raise, both fields
    None."""
    run = CompletedRun(returncode=0, stdout="", stderr="", workdir=tmp_path, outputs={})
    result = parse_output(_manifest(), run)
    assert result["foldseek_diversity"] is None
    assert result["mmseqs_diversity"] is None
    assert result["num_designs"] is None
