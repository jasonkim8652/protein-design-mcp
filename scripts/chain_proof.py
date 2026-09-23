#!/usr/bin/env python
"""Verify that one tool's OUTPUT actually works as the next tool's INPUT.

``live_proof.py`` checks each tool in isolation: it runs, and its result
carries the keys the manifest promises. Everything below passed that check and
was still broken:

* ``run_genie3_binder`` emits its design as ``UNK`` residues with CA atoms
  only. ProDy does not classify that as protein, so ``run_mpnn`` never sees the
  chain it is supposed to design.
* ``run_mpnn`` had no way to name a chain, so given a generator's two-chain
  complex it redesigned the target too -- a 504-residue target plus an
  80-residue binder came back as one 585-residue sequence that is neither.
* Generators disagree about which chain holds the design (A, B, A, B, B), so a
  convention learned from one is wrong for the next.

None of that is visible from a single tool's result. It is only visible when
the handoff is actually made, which is what this script does.

Each CHAIN below is a real sequence of MCP calls through the same
``ServerApp.call_tool`` path a client uses, with assertions about the HANDOFF
rather than about either tool alone. A chain may start from a fixture produced
by an earlier live run -- regenerating a backbone to test what a sequence
designer does with it spends GPU on the part that is not under test.

Usage::

    python scripts/chain_proof.py                 # every chain
    python scripts/chain_proof.py --chain mpnn_preserves_target
    python scripts/chain_proof.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "UNK": "X",
}


def chain_sequences(path: str | Path) -> dict[str, str]:
    """``{chain: one-letter sequence}`` from a PDB, first model only."""
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        key = (chain, line[22:27])
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(chain, []).append(THREE_TO_ONE.get(line[17:20].strip(), "?"))
    return {c: "".join(v) for c, v in chains.items()}


def is_placeholder(sequence: str) -> bool:
    """A chain carrying no designed sequence: UNK, poly-Ala, poly-Gly."""
    return len(set(sequence)) <= 1


@dataclass
class Step:
    """One tool call in a chain.

    ``arguments`` may be a callable taking the results collected so far, which
    is how a step consumes what the previous one produced -- the handoff this
    script exists to exercise.
    """

    tool: str
    arguments: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]
    expect_error: bool = False
    #: Hand the payload to the chain's ``check`` whether or not the call
    #: errored. Some handoffs are acceptable EITHER way -- a refusal and a
    #: usable result can both be honest -- and what the chain actually asserts
    #: is that the outcome is not a silent wrong answer. Treating the error as
    #: a chain failure here would hide the very case the chain exists to judge.
    tolerate_error: bool = False


@dataclass
class Chain:
    name: str
    why: str
    steps: list[Step]
    #: Raises AssertionError when the handoff is wrong. Receives
    #: ``{step_index: payload}``.
    check: Callable[[dict[int, Any]], None]
    needs: list[str] = field(default_factory=list)


# --- fixtures produced by earlier live runs ---------------------------------
#
# Paths rather than regenerated inputs: the chains below test what a CONSUMER
# does with a producer's output, and re-running a 15-minute generator to
# obtain a file that already exists would spend GPU on the part not under test.
# Each is checked for existence first and its chain skipped (not failed) when
# absent, so a fresh checkout reports "not run" rather than a false failure.

WORKSPACE = Path("/home/jk661/projects/proteinmem-mcp/mcp_workspace")
TWO_CHAIN_COMPLEX = (
    WORKSPACE / "pdmcp-results/pdmcp-27d99138cd53/structures/out/"
    "design_0-atomized-bb-False.pdb"
)
GENIE3_UNK_COMPLEX = (
    WORKSPACE / "pdmcp-results/pdmcp-fbba68ef7c35/binders/output/target/pdbs/target_0.pdb"
)

#: A short two-chain complex folded fresh by this chain. ipSAE scores the
#: INTERFACE between a chain pair, so a monomer gives it nothing to score --
#: and on a monomer it does not say so, it dies with
#: `cannot access local variable 'n0res_byres_all'`. The pair has to be real.
TWO_CHAIN_FOLD_INPUT = [
    {"sequence": "GSHMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "msa": None, "copies": 1},
    {"sequence": "MEKAIKELLDTLKQLLEEYNVSEEEAKKLLEKLKEL", "msa": None, "copies": 1},
]


def _check_mpnn_preserves_target(results: dict[int, Any]) -> None:
    before = chain_sequences(TWO_CHAIN_COMPLEX)
    target_in, design_in = before["A"], before["B"]

    designs = results[0].get("designs") or []
    assert designs, f"run_mpnn returned no designs: {results[0]}"

    for entry in designs:
        seq = entry.get("sequence", "")
        assert ":" in seq, (
            "run_mpnn returned a single unsplit sequence. Before chains_to_design "
            f"existed this was one {len(target_in) + len(design_in)}-residue string "
            "that was neither the target nor the binder."
        )
        kept, _, designed = seq.partition(":")
        assert kept == target_in, (
            "the target chain was NOT preserved: run_mpnn redesigned the chain it "
            "was told to leave alone"
        )
        assert len(designed) == len(design_in), (
            f"the designed chain changed length: {len(design_in)} -> {len(designed)}"
        )
        assert designed != design_in, "the design chain was not redesigned at all"
        assert not is_placeholder(designed), (
            f"run_mpnn returned another placeholder chain: {designed[:40]}"
        )


def _check_genie3_unk_is_refused_or_named(results: dict[int, Any]) -> None:
    """genie3's design is UNK/CA-only, which ProDy does not see as protein.

    Either outcome is acceptable as long as it is HONEST: a clear error, or a
    result that does not silently claim to have designed the binder. What must
    not happen is run_mpnn returning a sequence for the target alone while the
    caller believes it designed the binder.
    """
    before = chain_sequences(GENIE3_UNK_COMPLEX)
    design_len = len(before["A"])   # genie3 puts its design in chain A
    target_len = len(before["B"])

    payload = results[0]
    if isinstance(payload, dict) and payload.get("error"):
        return  # refused outright: honest

    designs = payload.get("designs") or []
    assert designs, f"neither an error nor any designs: {payload}"
    for entry in designs:
        seq = entry.get("sequence", "")
        total = len(seq.replace(":", ""))
        assert total != target_len, (
            "run_mpnn silently designed the TARGET only and returned it as a "
            f"design ({total} residues = the target's length). The binder chain "
            "is UNK/CA-only and was skipped, with nothing in the result saying so."
        )
        assert total >= design_len, (
            f"the returned sequence ({total}) is shorter than the design chain "
            f"({design_len}), so the design cannot be in it"
        )


def _ipsae_args_from_boltz(results: dict[int, Any]) -> dict[str, Any]:
    """Take the PAE and structure run_boltz just wrote.

    The paths come straight from boltz's own result rather than being copied:
    ipSAE does not take the confidence summary as an argument, it derives that
    path from the PAE path, so a PAE moved away from its sibling fails on a
    file the caller never named.
    """
    outputs = results[0].get("outputs") or {}
    pae = outputs.get("pae_npz") or []
    structures = outputs.get("structures") or []
    assert pae and structures, f"run_boltz returned no PAE or structure: {list(outputs)}"
    first = pae[0] if isinstance(pae, list) else pae
    model = structures[0] if isinstance(structures, list) else structures
    return {"pae_file": first, "structure": model}


def _check_ipsae_read_the_boltz_pae(results: dict[int, Any]) -> None:
    payload = results[1]
    assert not payload.get("error"), payload["error"][:300]
    assert "ipsae" in payload, (
        f"run_ipsae returned no ipsae score for a Boltz PAE: {list(payload)}"
    )
    assert payload.get("chain_pair"), "no chain pair was scored"



def _ipsae_args_from_af3(results: dict[int, Any]) -> dict[str, Any]:
    """AlphaFold 3's PAE lives in `confidences_json`, not in the summary.

    `summary_confidences_json` holds scalars; ipSAE needs the matrix, and the
    two files sit side by side with almost the same name. Picking the wrong one
    is a KeyError deep in the engine.
    """
    outputs = results[0].get("outputs") or {}
    conf = outputs.get("confidences_json")
    model = outputs.get("model_cif")
    assert conf and model, f"run_alphafold3 returned no confidences or model: {list(outputs)}"
    return {
        "pae_file": conf[0] if isinstance(conf, list) else conf,
        "structure": model[0] if isinstance(model, list) else model,
    }


def _check_ipsae_scored_an_interface(results: dict[int, Any]) -> None:
    payload = results[1]
    assert not payload.get("error"), str(payload["error"])[:300]
    assert "ipsae" in payload, f"no ipsae score: {list(payload)}"
    assert payload.get("chain_pair"), "no chain pair was scored"


def _fold_the_designed_chain(results: dict[int, Any]) -> dict[str, Any]:
    """Take run_mpnn's design and fold it alone.

    run_mpnn joins the chains it returned with ':'. A folding tool takes ONE
    sequence, so passing the joined string folds the target and the binder as
    a single run-on chain -- silently, because ':' is not an amino acid the
    validator rejects but is not a chain break the folder honours either. The
    caller has to split it, and which half is the design depends on the chain
    order the generator used.
    """
    designs = results[0].get("designs") or []
    assert designs, f"run_mpnn returned no designs: {results[0]}"
    sequence = designs[0]["sequence"]
    assert ":" in sequence, "expected a two-chain design from a two-chain input"
    designed = sequence.split(":")[-1]   # chains_to_design='B' -> the last part
    return {"sequence": designed}


def _check_the_fold_is_of_the_design_alone(results: dict[int, Any]) -> None:
    designs = results[0].get("designs") or []
    designed = designs[0]["sequence"].split(":")[-1]
    payload = results[1]
    assert not payload.get("error"), str(payload["error"])[:300]
    length = payload.get("sequence_length") or payload.get("num_residues")
    assert length == len(designed), (
        f"run_esmfold2 folded {length} residues but the design is "
        f"{len(designed)} -- the joined ':' sequence was passed through"
    )


CHAINS: list[Chain] = [
    Chain(
        name="mpnn_preserves_target",
        why=(
            "A generator hands back target + design in one file. The sequence "
            "designer must change only the design. Before chains_to_design this "
            "returned one 585-residue sequence that was neither."
        ),
        needs=[str(TWO_CHAIN_COMPLEX)],
        steps=[
            Step("run_mpnn", {
                "backbone_pdb": str(TWO_CHAIN_COMPLEX),
                "chains_to_design": "B",
                "num_sequences": 2,
                "model_type": "soluble",
                "seed": 37,
            }),
        ],
        check=_check_mpnn_preserves_target,
    ),
    Chain(
        name="genie3_unk_handoff_is_honest",
        why=(
            "run_genie3_binder emits UNK/CA-only, which ProDy does not classify "
            "as protein. The handoff must fail loudly or return something the "
            "caller can tell apart from a real design -- not the target's own "
            "sequence presented as a binder."
        ),
        needs=[str(GENIE3_UNK_COMPLEX)],
        steps=[
            Step("run_mpnn", {
                "backbone_pdb": str(GENIE3_UNK_COMPLEX),
                "chains_to_design": "A",
                "num_sequences": 1,
                "model_type": "soluble",
                "seed": 1,
            }, tolerate_error=True),
        ],
        check=_check_genie3_unk_is_refused_or_named,
    ),
    Chain(
        name="boltz_pae_reaches_ipsae",
        why=(
            "run_boltz's own output says its pae_npz is 'for run_ipsae', but "
            "the parameter was named pae_json and rejected anything but .json "
            "-- so the one chain the manifests advertised was refused before "
            "the engine saw it. ipSAE's own CLI help lists .npz as supported."
        ),
        steps=[
            Step("run_boltz", {"chains": TWO_CHAIN_FOLD_INPUT, "diffusion_samples": 1,
                               "recycling_steps": 1, "sampling_steps": 25}),
            Step("run_ipsae", _ipsae_args_from_boltz),
        ],
        check=_check_ipsae_read_the_boltz_pae,
    ),
    Chain(
        name="af3_pae_reaches_ipsae",
        why=(
            "run_ipsae is described as comparable ACROSS predictors, but only "
            "two of eleven declared a PAE output. AlphaFold 3 carries one in "
            "confidences_json and never said so, so the tool looked "
            "incompatible with the predictor most likely to be trusted."
        ),
        steps=[
            Step("run_alphafold3", {"chains": [
                {"sequence": "GSHMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
                 "unpaired_msa": None, "paired_msa": None, "copies": 1},
                {"sequence": "MEKAIKELLDTLKQLLEEYNVSEEEAKKLLEKLKEL",
                 "unpaired_msa": None, "paired_msa": None, "copies": 1},
            ]}),
            Step("run_ipsae", _ipsae_args_from_af3),
        ],
        check=_check_ipsae_scored_an_interface,
    ),
    Chain(
        name="mpnn_sequence_folds_alone",
        why=(
            "run_mpnn returns its chains joined with ':'. A folding tool takes "
            "one sequence, so handing the joined string over folds target and "
            "binder as one run-on chain -- and nothing refuses it, because ':' "
            "is neither a rejected character nor an honoured chain break."
        ),
        needs=[str(TWO_CHAIN_COMPLEX)],
        steps=[
            Step("run_mpnn", {
                "backbone_pdb": str(TWO_CHAIN_COMPLEX),
                "chains_to_design": "B", "num_sequences": 1,
                "model_type": "soluble", "seed": 11,
            }),
            Step("run_esmfold2", _fold_the_designed_chain),
        ],
        check=_check_the_fold_is_of_the_design_alone,
    ),
]


async def run_chain(app: Any, chain: Chain) -> tuple[str, str]:
    from mcp.types import TextContent

    results: dict[int, Any] = {}
    for index, step in enumerate(chain.steps):
        arguments = step.arguments(results) if callable(step.arguments) else step.arguments
        raw = await app.call_tool(step.tool, arguments)

        is_error = getattr(raw, "isError", False)
        content = getattr(raw, "content", raw)
        text = ""
        for item in content if isinstance(content, list) else [content]:
            if isinstance(item, TextContent):
                text = item.text
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"error": text}

        if is_error and not step.expect_error and not step.tolerate_error:
            return "FAIL", f"step {index} ({step.tool}) errored: {str(payload)[:300]}"
        if step.expect_error and not is_error:
            return "FAIL", f"step {index} ({step.tool}) was expected to fail and did not"
        results[index] = payload

    try:
        chain.check(results)
    except AssertionError as exc:
        return "FAIL", str(exc)
    return "PASS", ""


async def main_async(selected: list[str] | None) -> int:
    from protein_design_mcp.app import ServerApp, build_registry
    from protein_design_mcp.server import DEVICE

    app = ServerApp(build_registry(device=DEVICE))
    chains = [c for c in CHAINS if not selected or c.name in selected]

    failures = 0
    for chain in chains:
        missing = [p for p in chain.needs if not Path(p).exists()]
        if missing:
            print(f"SKIP  {chain.name}: fixture not present ({missing[0]})")
            continue
        started = time.time()
        status, detail = await run_chain(app, chain)
        mark = "PASS " if status == "PASS" else "FAIL "
        print(f"{mark} {chain.name}  ({time.time() - started:.1f}s)")
        if status != "PASS":
            failures += 1
            print(f"       why: {chain.why}")
            print(f"       {detail}")
    print(f"\n{len(chains)} chain(s), {failures} failed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chain", action="append", dest="chains")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for chain in CHAINS:
            print(f"{chain.name}\n    {chain.why}")
        return 0
    return asyncio.run(main_async(args.chains))


if __name__ == "__main__":
    raise SystemExit(main())
