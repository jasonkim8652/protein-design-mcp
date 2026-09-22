import pytest

from protein_design_mcp.manifest.schema import (
    Manifest,
    ManifestError,
    parse_manifest,
)

MINIMAL = {
    "name": "run_prodigy",
    "category": "scoring",
    "engine": {"repo": "prodigy", "env": "scoring", "entry": ["prodigy"]},
    "summary": "Estimate binding free energy for a protein-protein complex.",
    "doc": "## What this is\nPRODIGY.\n",
    "schema": {
        "complex_pdb": {
            "type": "string",
            "pattern": r"\.(pdb|cif)$",
            "required": True,
            "description": "Path to the complex structure.",
            "example": "complex.pdb",
        }
    },
}


def test_parses_minimal_manifest():
    m = parse_manifest(MINIMAL)
    assert m.name == "run_prodigy"
    assert m.category == "scoring"
    assert m.engine.env == "scoring"
    assert m.engine.entry == ("prodigy",)
    assert m.composite is False
    assert m.requires.gpu is False
    assert m.requires.license_gated is False


def test_composite_flag_is_read():
    data = {**MINIMAL, "composite": True}
    assert parse_manifest(data).composite is True


def test_requires_block_is_read():
    data = {
        **MINIMAL,
        "requires": {"gpu": True, "weights": "ckpts/x.ckpt", "license_gated": True},
    }
    m = parse_manifest(data)
    assert m.requires.gpu is True
    assert m.requires.weights == "ckpts/x.ckpt"
    assert m.requires.license_gated is True


def test_rejects_bad_tool_name():
    data = {**MINIMAL, "name": "designBinder"}
    with pytest.raises(ManifestError, match="name"):
        parse_manifest(data)


def test_rejects_missing_summary():
    data = {k: v for k, v in MINIMAL.items() if k != "summary"}
    with pytest.raises(ManifestError, match="summary"):
        parse_manifest(data)


def test_rejects_unknown_category():
    data = {**MINIMAL, "category": "miscellaneous"}
    with pytest.raises(ManifestError, match="category"):
        parse_manifest(data)


def test_manifest_is_frozen():
    m = parse_manifest(MINIMAL)
    with pytest.raises(Exception):
        m.name = "other"


def test_rejects_whitespace_only_summary():
    data = {**MINIMAL, "summary": "   "}
    with pytest.raises(ManifestError, match="summary"):
        parse_manifest(data)


def test_accepts_empty_schema():
    data = {**MINIMAL, "schema": {}}
    m = parse_manifest(data)
    assert m.schema == {}


def test_rejects_missing_schema():
    data = {k: v for k, v in MINIMAL.items() if k != "schema"}
    with pytest.raises(ManifestError, match="schema"):
        parse_manifest(data)


def test_rejects_a_schema_entry_that_is_not_a_mapping():
    """Regression for FIX 6: schema: {p: "string"} used to parse cleanly
    and only blow up later, deep inside ToolRegistry.tools(), with an
    opaque AttributeError that takes down tools/list for every tool."""
    data = {**MINIMAL, "schema": {"p": "string"}}
    with pytest.raises(ManifestError, match="p"):
        parse_manifest(data)


def test_rejects_minimum_without_a_type():
    """A typeless numeric spec lets a bool pass as 1/0 in validation.py's
    range check, since bool is an int subclass."""
    data = {**MINIMAL, "schema": {"n": {"minimum": 0}}}
    with pytest.raises(ManifestError, match="n"):
        parse_manifest(data)


def test_rejects_maximum_without_a_type():
    data = {**MINIMAL, "schema": {"n": {"maximum": 10}}}
    with pytest.raises(ManifestError, match="n"):
        parse_manifest(data)


def test_accepts_minimum_with_a_type():
    data = {**MINIMAL, "schema": {"n": {"type": "integer", "minimum": 0}}}
    m = parse_manifest(data)
    assert m.schema["n"]["minimum"] == 0


def test_outputs_default_to_empty():
    assert parse_manifest(MINIMAL).outputs == ()


def test_outputs_are_parsed():
    data = {
        **MINIMAL,
        "outputs": [
            {"name": "minimized_pdb", "pattern": "minimized.pdb",
             "description": "The relaxed structure."},
        ],
    }
    (out,) = parse_manifest(data).outputs
    assert out.name == "minimized_pdb"
    assert out.pattern == "minimized.pdb"
    assert out.description == "The relaxed structure."


def test_output_without_name_is_rejected():
    data = {**MINIMAL, "outputs": [{"pattern": "x.pdb"}]}
    with pytest.raises(ManifestError, match="name"):
        parse_manifest(data)


def test_output_without_pattern_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x"}]}
    with pytest.raises(ManifestError, match="pattern"):
        parse_manifest(data)


def test_absolute_output_pattern_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x", "pattern": "/etc/passwd"}]}
    with pytest.raises(ManifestError, match="relative"):
        parse_manifest(data)


def test_output_pattern_escaping_the_workdir_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x", "pattern": "../escape.pdb"}]}
    with pytest.raises(ManifestError, match="relative"):
        parse_manifest(data)


def test_duplicate_output_names_are_rejected():
    data = {
        **MINIMAL,
        "outputs": [{"name": "x", "pattern": "a.pdb"},
                    {"name": "x", "pattern": "b.pdb"}],
    }
    with pytest.raises(ManifestError, match="duplicate"):
        parse_manifest(data)


def test_timeout_defaults_to_one_hour():
    assert parse_manifest(MINIMAL).timeout_s == 3600


def test_output_multiple_defaults_to_false():
    data = {
        **MINIMAL,
        "outputs": [{"name": "minimized_pdb", "pattern": "minimized.pdb"}],
    }
    (out,) = parse_manifest(data).outputs
    assert out.multiple is False


def test_output_multiple_is_parsed():
    data = {
        **MINIMAL,
        "outputs": [
            {"name": "designs_fasta", "pattern": "seqs/*.fa", "multiple": True},
        ],
    }
    (out,) = parse_manifest(data).outputs
    assert out.multiple is True


def test_output_name_with_path_traversal_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "../evil", "pattern": "x.pdb"}]}
    with pytest.raises(ManifestError, match="name"):
        parse_manifest(data)


def test_output_name_with_absolute_component_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "/abs", "pattern": "x.pdb"}]}
    with pytest.raises(ManifestError, match="name"):
        parse_manifest(data)


def test_output_name_empty_string_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "", "pattern": "x.pdb"}]}
    with pytest.raises(ManifestError, match="name"):
        parse_manifest(data)


def test_ordinary_output_names_still_parse():
    data = {
        **MINIMAL,
        "outputs": [
            {"name": "minimized_pdb", "pattern": "minimized.pdb"},
            {"name": "designs_fasta", "pattern": "seqs/*.fa", "multiple": True},
        ],
    }
    names = [out.name for out in parse_manifest(data).outputs]
    assert names == ["minimized_pdb", "designs_fasta"]


def test_timeout_is_parsed():
    assert parse_manifest({**MINIMAL, "timeout_s": 120}).timeout_s == 120


def test_nonpositive_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": 0})


def test_boolean_true_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": True})


def test_boolean_false_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": False})


def test_float_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": 1.9})


def test_engine_stage_defaults_to_empty():
    assert parse_manifest(MINIMAL).engine.stage == ()


def _with_path_param():
    """MINIMAL's own 'complex_pdb' param has no 'format', so staging tests
    need a schema entry that actually declares 'format: path'."""
    return {
        **MINIMAL,
        "schema": {
            **MINIMAL["schema"],
            "structure": {
                "type": "string",
                "format": "path",
                "pattern": r"\.pdb$",
                "required": True,
                "description": "Structure file.",
                "example": "s.pdb",
            },
        },
    }


def test_engine_stage_names_a_real_path_parameter():
    data = _with_path_param()
    data["engine"] = {**data["engine"], "stage": ["structure"]}
    m = parse_manifest(data)
    assert m.engine.stage == ("structure",)


def test_engine_stage_rejects_an_unknown_parameter_name():
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "stage": ["nonexistent_param"]}}
    with pytest.raises(ManifestError, match="nonexistent_param"):
        parse_manifest(data)


def test_engine_stage_rejects_a_non_path_parameter():
    """MINIMAL's 'complex_pdb' schema entry has no 'format: path' — staging
    it must be refused rather than silently doing nothing useful."""
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "stage": ["complex_pdb"]}}
    with pytest.raises(ManifestError, match="format: path"):
        parse_manifest(data)


def test_engine_stage_rejects_a_duplicate_name():
    data = _with_path_param()
    data["engine"] = {**data["engine"], "stage": ["structure", "structure"]}
    with pytest.raises(ManifestError, match="stage"):
        parse_manifest(data)


def test_engine_stage_must_be_a_list_of_strings():
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "stage": "structure"}}
    with pytest.raises(ManifestError, match="stage"):
        parse_manifest(data)


def _with_array_of_path_param():
    """An array-of-path schema entry (items.format == 'path'), the shape
    run_boltzgen_fold/design_fold/analyze need to stage many caller-supplied
    files (a prior tool's whole outputs list) into one shared directory."""
    return {
        **MINIMAL,
        "schema": {
            **MINIMAL["schema"],
            "generated_files": {
                "type": "array",
                "items": {"type": "string", "format": "path"},
                "required": True,
                "description": "Files to stage together.",
                "example": ["a.cif", "a.npz"],
            },
        },
    }


def test_engine_stage_accepts_an_array_of_path_parameter():
    data = _with_array_of_path_param()
    data["engine"] = {**data["engine"], "stage": ["generated_files"]}
    m = parse_manifest(data)
    assert m.engine.stage == ("generated_files",)


def test_engine_stage_rejects_an_array_whose_items_are_not_format_path():
    """An array param without items.format == 'path' is exactly as
    unstageable as a scalar without format: path."""
    data = {
        **MINIMAL,
        "schema": {
            **MINIMAL["schema"],
            "plain_list": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Not paths.",
            },
        },
    }
    data["engine"] = {**data["engine"], "stage": ["plain_list"]}
    with pytest.raises(ManifestError, match="format: path"):
        parse_manifest(data)


def test_integer_timeout_still_accepted():
    m = parse_manifest({**MINIMAL, "timeout_s": 120})
    assert m.timeout_s == 120


# --- Task 2: EngineSpec.prefix -----------------------------------------------


def test_env_only_engine_has_no_prefix():
    m = parse_manifest(MINIMAL)
    assert m.engine.env == "scoring"
    assert m.engine.prefix is None


def test_prefix_only_engine_is_parsed(tmp_path):
    data = {
        **MINIMAL,
        "engine": {"repo": "boltz", "prefix": str(tmp_path), "entry": ["boltz"]},
    }
    m = parse_manifest(data)
    assert m.engine.prefix == str(tmp_path)
    assert m.engine.env is None


def test_declaring_both_env_and_prefix_is_a_load_error(tmp_path):
    data = {
        **MINIMAL,
        "engine": {
            "repo": "boltz",
            "env": "scoring",
            "prefix": str(tmp_path),
            "entry": ["boltz"],
        },
    }
    with pytest.raises(ManifestError, match=r"run_prodigy.*both") as exc:
        parse_manifest(data)
    assert "prefix" in str(exc.value)


def test_declaring_neither_env_nor_prefix_is_a_load_error():
    data = {**MINIMAL, "engine": {"repo": "boltz", "entry": ["boltz"]}}
    with pytest.raises(ManifestError, match=r"run_prodigy.*neither"):
        parse_manifest(data)


def test_prefix_must_be_an_absolute_path():
    data = {
        **MINIMAL,
        "engine": {"repo": "boltz", "prefix": "relative/path", "entry": ["boltz"]},
    }
    with pytest.raises(ManifestError, match="absolute"):
        parse_manifest(data)


def test_prefix_must_not_contain_dotdot(tmp_path):
    data = {
        **MINIMAL,
        "engine": {
            "repo": "boltz",
            "prefix": str(tmp_path / ".." / "boltz"),
            "entry": ["boltz"],
        },
    }
    with pytest.raises(ManifestError, match=r"\.\."):
        parse_manifest(data)


# --- Task 2: EngineSpec.mounts ------------------------------------------------


def test_mounts_defaults_to_empty():
    assert parse_manifest(MINIMAL).engine.mounts == ()


def test_mounts_empty_list_is_valid_not_an_error():
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "mounts": []}}
    assert parse_manifest(data).engine.mounts == ()


def test_mounts_accepts_existing_absolute_paths(tmp_path):
    mount_dir = tmp_path / "src"
    mount_dir.mkdir()
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "mounts": [str(mount_dir)]}}
    m = parse_manifest(data)
    assert m.engine.mounts == (str(mount_dir),)


def test_mounts_rejects_a_relative_path():
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "mounts": ["relative/path"]}}
    with pytest.raises(ManifestError, match="absolute"):
        parse_manifest(data)


def test_mounts_rejects_a_dotdot_path(tmp_path):
    mount_dir = tmp_path / "src"
    mount_dir.mkdir()
    data = {
        **MINIMAL,
        "engine": {
            **MINIMAL["engine"],
            "mounts": [str(mount_dir / ".." / "src")],
        },
    }
    with pytest.raises(ManifestError, match=r"\.\."):
        parse_manifest(data)


def test_mounts_rejects_a_nonexistent_path(tmp_path):
    missing = tmp_path / "does_not_exist"
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "mounts": [str(missing)]}}
    with pytest.raises(ManifestError, match="does not exist"):
        parse_manifest(data)


def test_mounts_must_be_a_list_of_strings():
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "mounts": "not-a-list"}}
    with pytest.raises(ManifestError, match="mounts"):
        parse_manifest(data)


# --- Task 2: EngineSpec.env_vars ----------------------------------------------


def test_env_vars_defaults_to_empty():
    assert parse_manifest(MINIMAL).engine.env_vars == {}


def test_env_vars_empty_mapping_is_valid():
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "env_vars": {}}}
    assert parse_manifest(data).engine.env_vars == {}


def test_env_vars_are_read():
    data = {
        **MINIMAL,
        "engine": {
            **MINIMAL["engine"],
            "env_vars": {"PYTHONNOUSERSITE": "1", "HF_HOME": "/scratch/hf"},
        },
    }
    m = parse_manifest(data)
    assert m.engine.env_vars == {"PYTHONNOUSERSITE": "1", "HF_HOME": "/scratch/hf"}


def test_env_vars_empty_string_value_is_kept_not_dropped():
    """'' is falsy in Python — this codebase has been bitten by
    falsy-versus-absent before, so an explicit '' value must survive
    parsing rather than being treated as if the key were never set."""
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "env_vars": {"FOO": ""}}}
    m = parse_manifest(data)
    assert m.engine.env_vars == {"FOO": ""}
    assert "FOO" in m.engine.env_vars


def test_env_vars_must_be_a_mapping():
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "env_vars": ["FOO=1"]}}
    with pytest.raises(ManifestError, match="env_vars"):
        parse_manifest(data)


def test_env_vars_rejects_a_non_string_value():
    data = {**MINIMAL, "engine": {**MINIMAL["engine"], "env_vars": {"FOO": 1}}}
    with pytest.raises(ManifestError, match="env_vars"):
        parse_manifest(data)


# --- stage_subdir: explicit relative placement for a staged name ----------


def test_stage_subdir_defaults_to_empty():
    assert parse_manifest(MINIMAL).engine.stage_subdir == {}


def test_stage_subdir_is_read_for_a_staged_name():
    data = _with_path_param()
    data["engine"] = {
        **data["engine"],
        "stage": ["structure"],
        "stage_subdir": {"structure": "design_dir/refold_cif"},
    }
    m = parse_manifest(data)
    assert m.engine.stage_subdir == {"structure": "design_dir/refold_cif"}


def test_stage_subdir_rejects_a_name_not_in_stage():
    data = _with_path_param()
    data["engine"] = {
        **data["engine"],
        "stage": ["structure"],
        "stage_subdir": {"other_param": "design_dir"},
    }
    with pytest.raises(ManifestError, match="stage_subdir"):
        parse_manifest(data)


def test_stage_subdir_rejects_an_absolute_path():
    data = _with_path_param()
    data["engine"] = {
        **data["engine"],
        "stage": ["structure"],
        "stage_subdir": {"structure": "/etc/passwd"},
    }
    with pytest.raises(ManifestError, match="stage_subdir"):
        parse_manifest(data)


def test_stage_subdir_rejects_a_dotdot_path():
    data = _with_path_param()
    data["engine"] = {
        **data["engine"],
        "stage": ["structure"],
        "stage_subdir": {"structure": "../escape"},
    }
    with pytest.raises(ManifestError, match="stage_subdir"):
        parse_manifest(data)


def test_stage_subdir_must_be_a_mapping():
    data = _with_path_param()
    data["engine"] = {
        **data["engine"],
        "stage": ["structure"],
        "stage_subdir": ["structure"],
    }
    with pytest.raises(ManifestError, match="stage_subdir"):
        parse_manifest(data)
