import pytest

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

MANIFEST_DIR = manifest_dir()


def _manifest():
    return next(m for m in load_manifests(MANIFEST_DIR) if m.name == "run_mmseqs_search")


def test_manifest_loads_and_is_the_msa_category():
    m = _manifest()
    assert m.category == "msa"
    assert m.composite is False


def test_manifest_requires_gpu():
    """Confirmed live (task-3-report.md): CPU-mode search against these
    GPU-padded databases is not practical on this host."""
    assert _manifest().requires.gpu is True


def test_manifest_declares_the_four_outputs():
    names = {o.name for o in _manifest().outputs}
    assert names == {"unpaired_a3m", "paired_a3m", "templates_a3m", "search_summary"}
    for out in _manifest().outputs:
        assert out.multiple is False  # exactly one file each, always


def test_engine_uses_env_not_prefix():
    """This engine needs no Python package beyond the standard library, so
    it dispatches via `env` (like the four CPU tools) rather than `prefix`
    (the host-mounted-conda-env mechanism built for GPU ML engines with
    heavy, pre-installed dependency trees) -- see the manifest's own
    engine.mounts comment for the reasoning."""
    engine = _manifest().engine
    assert engine.env == "mmseqs"
    assert engine.prefix is None


def test_engine_mounts_the_binary_and_the_database_root():
    assert set(_manifest().engine.mounts) == {
        "/usr/local/bin/mmseqs",
        "/opt/alphafold3_data/mmseqs_db",
    }


def test_sequence_is_required():
    schema = _manifest().schema
    assert schema["sequence"]["required"] is True


def test_unpaired_databases_defaults_to_all_three_with_minitems_one():
    spec = _manifest().schema["unpaired_databases"]
    assert spec["default"] == ["uniref90", "mgnify", "small_bfd"]
    assert spec["minItems"] == 1
    assert set(spec["items"]["enum"]) == {"uniref90", "mgnify", "small_bfd"}


def test_pair_and_search_templates_default_true():
    schema = _manifest().schema
    assert schema["pair"]["default"] is True
    assert schema["search_templates"]["default"] is True


def test_every_mmseqs_knob_the_brief_named_is_a_parameter():
    """sensitivity, e-value, max sequences, coverage, identity thresholds,
    iterations, which databases to search -- the brief's own enumerated
    list of knobs that must never be hidden behind a fixed default."""
    schema = _manifest().schema
    for key in (
        "sensitivity",
        "e_value",
        "max_sequences",
        "coverage",
        "coverage_mode",
        "min_seq_id",
        "num_iterations",
        "unpaired_databases",
        "threads",
        "use_gpu",
        "template_e_value",
        "max_template_hits",
    ):
        assert key in schema, f"{key} is not exposed as a schema parameter"
        assert schema[key]["description"], f"{key} has no description"
        assert "default" in schema[key], f"{key} has no documented default"


# --- corner case: a query whose sequence contains characters MMseqs2's own
# createdb does not reject, but this tool does (verified live -- see the
# manifest's own doc and task-3-report.md: mmseqs createdb silently accepts
# digits, lowercase, spaces and '*' without complaint). ------------------


def test_sequence_rejects_digits():
    with pytest.raises(ToolInputError, match="sequence"):
        validate_and_fill(_manifest(), {"sequence": "ACD123EFG"})


def test_sequence_rejects_lowercase():
    with pytest.raises(ToolInputError):
        validate_and_fill(_manifest(), {"sequence": "acdefg"})


def test_sequence_rejects_embedded_whitespace():
    with pytest.raises(ToolInputError):
        validate_and_fill(_manifest(), {"sequence": "AC DEFG"})


def test_sequence_rejects_a_fasta_header_character():
    """A literal '>' would corrupt the FASTA record this tool's wrapper
    script writes internally (`>query\\n{sequence}\\n`)."""
    with pytest.raises(ToolInputError):
        validate_and_fill(_manifest(), {"sequence": "AC>DEFG"})


def test_sequence_accepts_standard_amino_acids_and_ambiguity_codes():
    params = validate_and_fill(_manifest(), {"sequence": "ACDEFGHIKLMNPQRSTVWYXBZJUO"})
    assert params["sequence"] == "ACDEFGHIKLMNPQRSTVWYXBZJUO"


def test_sequence_rejects_empty_string():
    with pytest.raises(ToolInputError):
        validate_and_fill(_manifest(), {"sequence": ""})
