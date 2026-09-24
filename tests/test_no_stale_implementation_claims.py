"""No manifest may call an implemented tool unimplemented.

Twelve places said things like "`run_boltz` (not yet implemented)". Each was
true when written and stopped being true without the note changing, and the
reader is a model choosing between tools: told its alternative does not exist,
it does not go looking. One round planned run_boltzgen_fold to fold an
RFdiffusion3 backbone -- a pairing that cannot work -- while the note in front
of it said the general-purpose predictors were unavailable.

Written as a test rather than a one-time cleanup because the claim decays: any
tool named here is by definition one that exists, so the sentence can only ever
become false as the surface grows.
"""

from __future__ import annotations

import re

import pytest

from protein_design_mcp.app import manifest_dir
from protein_design_mcp.manifest.loader import load_manifests

#: "(not yet implemented)", "is not implemented", "does not exist yet", and the
#: near neighbours a future note is likely to use.
_UNIMPLEMENTED = re.compile(
    r"(not yet implemented|not implemented|does not exist yet|"
    r"is planned|will be added|coming soon)", re.I)


@pytest.fixture(scope="module")
def manifests():
    return {m.name: m for m in load_manifests(manifest_dir())}


def test_no_manifest_calls_an_existing_tool_unimplemented(manifests):
    offences = []
    for manifest in manifests.values():
        text = f"{manifest.summary}\n{manifest.doc}\n" + "\n".join(
            (o.description or "") for o in (manifest.outputs or ()))
        for sentence in re.split(r"(?<=[.;\n])\s+", text):
            if not _UNIMPLEMENTED.search(sentence):
                continue
            named = [n for n in re.findall(r"run_[a-z0-9_]+", sentence)
                     if n in manifests]
            for name in named:
                offences.append(f"{manifest.name}: {name} -- {sentence.strip()[:120]}")
    assert not offences, "these tools exist:\n  " + "\n  ".join(offences)


def test_the_check_reads_the_text_a_model_actually_sees(manifests):
    """summary and doc both reach the model -- describe_tool returns the doc,
    tools/list returns the summary -- so a stale claim in either misleads."""
    m = manifests["run_boltz"]
    assert m.summary.strip() and m.doc.strip()
