#!/usr/bin/env python
"""Audit every manifest's prose against the surface it describes.

The documentation defects found so far were each caught by a run, and each
belonged to a class the whole surface could have been checked for at once:

    "(not yet implemented)" about tools that exist        26 places
    `pae_json` after the parameter became `pae_file`      a rename's leftovers
    `run_boltzgen_design_fold` after it was merged away   a removed tool
    "works on any predictor's PAE" when three of eleven   an overclaim
    design_spec required and produced by nothing          no stated provenance

The reader is a model choosing between tools and filling their arguments, so a
name that no longer exists is worse than an omission: it sends the model to ask
for something and spend a step on the refusal.

Each check below is mechanical -- it compares prose against the registry rather
than judging writing. What it cannot check is whether a description is USEFUL;
that stays a human's job.

Usage::

    python scripts/doc_audit.py                # every finding
    python scripts/doc_audit.py --check names  # one class
    python scripts/doc_audit.py --list         # the classes
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

TOOL_NAME = re.compile(r"\brun_[a-z0-9_]+\b")
BACKTICKED = re.compile(r"`([a-z_][a-z0-9_]*)`")

#: Machine-specific paths have no business in text a model reads: it will copy
#: one into an argument, and the file exists on one host.
HOST_PATH = re.compile(r"(?<![\w/])/(?:home|Users|file_server)/[A-Za-z0-9._-]+/")

#: An absolute claim next to a capability is the shape an overclaim takes.
#: "works on any predictor's PAE" was true of three predictors out of eleven.
OVERCLAIM = re.compile(
    r"\b(any|every|all)\s+(?:\w+\s+){0,2}"
    r"(predictor|generator|tool|structure|format|sequence|model)s?\b", re.I)

#: Phrasing that tells the caller a required input is theirs to produce.
SELF_SUPPLIED = re.compile(
    r"(you write|you provide|you supply|caller-authored|author it|"
    r"write this yourself|supply your own|your own \w+ file|"
    r"your own structure|any structure)", re.I)


def manifest_text(manifest) -> str:
    parts = [manifest.summary or "", manifest.doc or ""]
    for spec in (manifest.schema or {}).values():
        if isinstance(spec, dict):
            parts.append(str(spec.get("description") or ""))
    for output in (manifest.outputs or ()):
        parts.append(output.description or "")
    return "\n".join(parts)


def check_names(manifests) -> list[str]:
    """Every run_* name in prose must be a tool that exists.

    Three shapes look like a tool name and are not, each one a false report the
    first version produced: an UPSTREAM SCRIPT (`run_alphafold.py`, which is
    AlphaFold 3's own entry point), a FAMILY GLOB (`run_boltzgen_*`, which the
    tool-name pattern truncates to `run_boltzgen_`), and a PARAMETER that
    happens to start with run_ (`run_clustering`, on run_boltzgen_analyze).
    """
    known = set(manifests)
    found = []
    for name, manifest in sorted(manifests.items()):
        text = manifest_text(manifest)
        own_params = set(manifest.schema or {})
        for mentioned in sorted(set(TOOL_NAME.findall(text))):
            if mentioned in known or mentioned in own_params:
                continue
            if re.search(rf"{re.escape(mentioned)}(?:\.py|\*)", text):
                continue
            if mentioned.endswith("_"):
                continue
            found.append(f"{name}: names {mentioned}, which is not a tool")
    return found


def check_parameters(manifests) -> list[str]:
    """A backticked word this tool does not have, that is ANOTHER TOOL'S
    parameter and not a word of ordinary English.

    The first version flagged `msa`, `length`, `num_designs` and thirty more:
    generic words that are parameters somewhere, mentioned here in prose that
    was not about a parameter at all. Only a word that is a parameter NOWHERE
    on the surface but here, or one that used to be this tool's own and was
    renamed, is worth reporting -- so this now checks the narrow case that
    actually bit: a name left behind by a rename (`pae_json` after the
    parameter became `pae_file`).
    """
    found = []
    for name, manifest in sorted(manifests.items()):
        own = set(manifest.schema or {})
        own_outputs = {o.name for o in (manifest.outputs or ())}
        text = manifest_text(manifest)
        for word in sorted(set(BACKTICKED.findall(text))):
            if word in own or word in own_outputs:
                continue
            # "this tool's `x`" / "`x` is required" is prose ABOUT this tool's
            # own parameter, so a word used that way and absent from the schema
            # is a leftover.
            claims_ownership = re.search(
                rf"(this tool's|its own|the)\s+`{re.escape(word)}`\s+"
                rf"(parameter|argument|is required|must)", text)
            if claims_ownership:
                found.append(
                    f"{name}: prose calls `{word}` this tool's parameter, and the "
                    "schema has no such field -- a rename's leftover")
    return found


def check_host_paths(manifests) -> list[str]:
    found = []
    for name, manifest in sorted(manifests.items()):
        for hit in sorted(set(HOST_PATH.findall(manifest_text(manifest)))):
            found.append(f"{name}: prose carries a host path under /{hit.strip('/')}/")
    return found


def check_overclaims(manifests) -> list[str]:
    found = []
    for name, manifest in sorted(manifests.items()):
        for hit in OVERCLAIM.finditer(manifest_text(manifest)):
            found.append(f"{name}: \"{hit.group(0)}\" -- verify it holds for all of them")
    return found


def check_provenance(manifests) -> list[str]:
    """Every required path parameter says where a caller gets one."""
    found = []
    for name, manifest in sorted(manifests.items()):
        for param, spec in (manifest.schema or {}).items():
            if not isinstance(spec, dict) or not spec.get("required"):
                continue
            is_path = spec.get("format") == "path" or (
                isinstance(spec.get("items"), dict)
                and spec["items"].get("format") == "path")
            if not is_path:
                continue
            description = str(spec.get("description") or "")
            if TOOL_NAME.search(description) or SELF_SUPPLIED.search(description):
                continue
            found.append(
                f"{name}.{param}: a required path whose description names no "
                "producing tool and does not say the caller writes it")
    return found



def check_descriptions(manifests) -> list[str]:
    """Every parameter and every output carries a description a model can use.

    This is the text the model actually reads to fill an argument -- it arrives
    as the `inputSchema` description over MCP, and `proteinmem-mcp`'s validator
    reads the same field before a call. A parameter without one gives the model
    a name and a type and nothing about what moving it does.

    `describe_it` is deliberately weak: it checks that text EXISTS and is more
    than a restatement of the name, not that it is good. Whether a description
    says what changes when the parameter moves, and over what range, stays a
    reader's judgement -- but a missing one, or `num_designs: "number of
    designs"`, is decidable here.
    """
    found = []
    for name, manifest in sorted(manifests.items()):
        for param, spec in sorted((manifest.schema or {}).items()):
            if not isinstance(spec, dict):
                continue
            text = " ".join(str(spec.get("description") or "").split())
            if not text:
                found.append(f"{name}.{param}: no description at all")
                continue
            words = text.rstrip(".").lower().split()
            if len(words) < 4:
                found.append(f"{name}.{param}: description is {len(words)} word(s) -- \"{text}\"")
                continue
            if " ".join(words) == param.replace("_", " "):
                found.append(f"{name}.{param}: description only restates the name")
        for output in (manifest.outputs or ()):
            if not (output.description or "").strip():
                found.append(f"{name}.outputs.{output.name}: no description at all")
    return found


CHECKS = {
    "names": (check_names, "a run_* name in prose that is not a tool"),
    "parameters": (check_parameters, "a backticked parameter this tool does not have"),
    "host-paths": (check_host_paths, "a machine-specific path in text a model reads"),
    # Advisory: most hits are ordinary English ("every predicted structure"),
    # and an absolute claim is only wrong when it is not true of all of them.
    # Reported for a reader to judge, never counted as a failure.
    "overclaims": (check_overclaims, "an absolute claim about a capability (ADVISORY)"),
    "provenance": (check_provenance, "a required path with no stated source"),
    "descriptions": (check_descriptions, "a parameter or output with no usable description"),
}


def main(argv: "list[str] | None" = None) -> int:
    from protein_design_mcp.app import manifest_dir
    from protein_design_mcp.manifest.loader import load_manifests

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="append", choices=sorted(CHECKS))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for key, (_, why) in sorted(CHECKS.items()):
            print(f"{key:12} {why}")
        return 0

    manifests = {m.name: m for m in load_manifests(manifest_dir())}
    selected = args.check or sorted(CHECKS)
    total = 0
    for key in selected:
        run, why = CHECKS[key]
        findings = run(manifests)
        if key != "overclaims":
            total += len(findings)
        print(f"== {key} ({why}): {len(findings)}")
        for finding in findings:
            print(f"   {finding}")
    print(f"\n{len(manifests)} manifests, {total} finding(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
