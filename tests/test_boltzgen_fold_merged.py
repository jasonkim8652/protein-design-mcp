"""One refolding tool with a stated mode, not two tools differing by a flag.

`run_boltzgen_fold` and `run_boltzgen_design_fold` had identical parameters,
identical outputs, the same engine entry and the same adapter flags. The only
difference was `--steps folding` versus `--steps design_folding`: refold the
design WITH its target, or ALONE.

Two names for one step is not the atomistic surface this server is for -- it is
a caller having to learn twice. And the distinction is precisely the one
docs/TOOLS.md already says must be a parameter:

    `chains` is never inferred. Whether a prediction runs with the target
    present or on the binder alone is the caller's decision and one of the
    most consequential it makes -- the same binder predicted alone and in
    complex are different experiments.

So it is `with_target`, required and with no default, because a decision that
consequential must be stated rather than inherited.
"""

from __future__ import annotations

import pytest

from protein_design_mcp.adapters.boltzgen_fold import build_args
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill


@pytest.fixture(scope="module")
def manifests():
    return {m.name: m for m in load_manifests(manifest_dir())}


@pytest.fixture
def manifest(manifests):
    return manifests["run_boltzgen_fold"]


def _params(manifest, **overrides):
    base = {"design_spec": "/w/design_spec.yaml",
            "generated_files": ["/w/a.cif", "/w/a.npz"],
            "with_target": True}
    base.update(overrides)
    return validate_and_fill(manifest, base)


def test_the_separate_design_fold_tool_is_gone(manifests):
    assert "run_boltzgen_design_fold" not in manifests, (
        "it differed from run_boltzgen_fold by one --steps value and nothing else"
    )


def test_with_target_has_no_default(manifest):
    """Defaulting it would make the more consequential of the two experiments
    the silent one."""
    assert manifest.schema["with_target"].get("default") is None
    assert manifest.schema["with_target"].get("required") is True


def test_omitting_the_mode_is_refused(manifest):
    with pytest.raises(ToolInputError, match="with_target"):
        validate_and_fill(manifest, {"design_spec": "/w/s.yaml",
                                     "generated_files": ["/w/a.cif"]})


def test_with_target_true_refolds_in_complex(manifest):
    args = build_args(manifest, _params(manifest, with_target=True))
    assert args[args.index("--steps") + 1] == "folding"


def test_with_target_false_refolds_the_design_alone(manifest):
    args = build_args(manifest, _params(manifest, with_target=False))
    assert args[args.index("--steps") + 1] == "design_folding"


def test_exactly_one_steps_flag_is_passed(manifest):
    args = build_args(manifest, _params(manifest, with_target=True))
    assert args.count("--steps") == 1


def test_the_docs_say_which_question_each_mode_answers(manifest):
    text = f"{manifest.doc}\n{manifest.schema['with_target'].get('description', '')}"
    assert "interface" in text.lower(), "with_target=true is an interface estimate"
    assert "alone" in text.lower(), "with_target=false is a self-consistency check"
