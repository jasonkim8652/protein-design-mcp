"""``describe_tool``'s category enum must be the registry's own category set.

They were two hand-maintained lists of the same thing and one drifted:
``manifest/schema.py`` gained ``target_analysis`` when run_interface_residues
and run_epitope_scan were added, ``meta_tools.py`` did not. Both tools were
registered and appeared in ``tools/list``, but
``describe_tool(category='target_analysis')`` was refused as an invalid value.

That failure is specifically bad. ``target_analysis`` exists because four
binder generators require hotspots and nothing produced them; a model that
follows the documented discovery path to find out what can produce them is
told the category does not exist, while the tools sit in the registry.

Found by driving the real server from the harness -- every category the harness
asked about answered except this one.
"""

from __future__ import annotations

from protein_design_mcp.app import build_registry, manifest_dir
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.manifest.schema import CATEGORIES
from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST


def _enum() -> list[str]:
    return DESCRIBE_TOOL_MANIFEST.schema["category"]["enum"]


def test_the_enum_is_exactly_the_registrys_category_set():
    """Derived, not duplicated -- so it cannot drift again."""
    assert set(_enum()) == set(CATEGORIES)


def test_target_analysis_is_offered():
    """The specific regression: registered tools in an unreachable category."""
    assert "target_analysis" in _enum()


def test_every_category_a_manifest_actually_uses_is_offered():
    """The end the enum exists to serve: any category a shipped tool declares
    must be reachable through describe_tool."""
    used = {m.category for m in load_manifests(manifest_dir())}
    missing = used - set(_enum())
    assert not missing, f"tools ship in categories describe_tool will not accept: {sorted(missing)}"


def test_the_summary_lists_the_same_categories_as_the_enum():
    """The summary is what a model reads before it ever sees the schema. A
    category missing there is invisible even though the call would succeed."""
    summary = DESCRIBE_TOOL_MANIFEST.summary
    missing = [c for c in _enum() if c not in summary]
    assert not missing, f"describe_tool's summary does not mention {missing}"


def test_every_offered_category_resolves_to_at_least_one_tool_or_is_meta():
    """An offered category that matches nothing sends a model down a dead end.
    `meta` is exempt: its tools are registered outside the manifest registry."""
    registry = build_registry(device="cuda")
    populated = {m.category for m in registry.manifests()} if hasattr(registry, "manifests") else set()
    if not populated:  # registry shape differs; fall back to the manifests on disk
        populated = {m.category for m in load_manifests(manifest_dir())}
    for category in _enum():
        if category == "meta":
            continue
        assert category in populated, f"describe_tool offers {category!r} but no tool is in it"
