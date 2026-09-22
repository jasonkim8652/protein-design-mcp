"""``app._resolve_path_params`` must resolve every ``format: path`` schema
entry to an absolute path -- including array items, not just scalars (see
run_boltzgen_fold/design_fold/analyze's ``generated_files`` parameter,
which is an array of ``format: path`` items so ``engine.stage`` can copy
every one of them, from ``manifest.schema._validate_stage``'s array-of-path
support)."""

from pathlib import Path

from protein_design_mcp.app import _resolve_path_params
from protein_design_mcp.manifest.schema import EngineSpec, Manifest


def _manifest(schema: dict) -> Manifest:
    return Manifest(
        name="run_x",
        category="run_analysis",
        engine=EngineSpec(repo="x", entry=("x",), env="x"),
        summary="s",
        doc="d",
        schema=schema,
    )


SCALAR_SCHEMA = {
    "structure": {"type": "string", "format": "path", "required": True},
}

ARRAY_SCHEMA = {
    "generated_files": {
        "type": "array",
        "items": {"type": "string", "format": "path"},
        "required": True,
    },
}


def test_resolves_a_scalar_path_param(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_path_params(_manifest(SCALAR_SCHEMA), {"structure": "model.pdb"})
    assert resolved["structure"] == str(tmp_path / "model.pdb")


def test_resolves_every_item_of_an_array_path_param(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_path_params(
        _manifest(ARRAY_SCHEMA), {"generated_files": ["a.cif", "sub/b.npz"]}
    )
    assert resolved["generated_files"] == [
        str(tmp_path / "a.cif"),
        str(tmp_path / "sub" / "b.npz"),
    ]


def test_an_already_absolute_array_item_is_left_pointing_at_the_same_file(tmp_path):
    absolute = str(tmp_path / "already_absolute.cif")
    resolved = _resolve_path_params(
        _manifest(ARRAY_SCHEMA), {"generated_files": [absolute]}
    )
    assert resolved["generated_files"] == [absolute]


def test_an_empty_array_path_param_resolves_to_an_empty_list():
    resolved = _resolve_path_params(_manifest(ARRAY_SCHEMA), {"generated_files": []})
    assert resolved["generated_files"] == []


def test_a_missing_array_path_param_is_left_untouched():
    resolved = _resolve_path_params(_manifest(ARRAY_SCHEMA), {})
    assert "generated_files" not in resolved


def test_a_non_list_non_string_value_for_an_array_param_is_left_untouched():
    """Defensive: validation should already have rejected this by the time
    _resolve_path_params runs, but resolution must not crash on it."""
    resolved = _resolve_path_params(_manifest(ARRAY_SCHEMA), {"generated_files": None})
    assert resolved["generated_files"] is None
