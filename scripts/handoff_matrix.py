#!/usr/bin/env python
"""Check every producer -> consumer pair for a format mismatch, statically.

Three broken handoffs were found one at a time, each by running a workflow and
watching it die at a different step:

    run_boltz.pae_npz (.npz)          -> run_ipsae.pae_file   pattern \\.json$
    run_rfdiffusion3_binder (.cif.gz) -> run_mpnn.backbone_pdb pattern \\.(pdb|cif)$
    run_genie3_binder (UNK/CA-only)   -> run_mpnn             invisible to ProDy

Finding them that way costs a GPU run each and only covers the pairs some model
happened to plan. Every one of them was decidable from the manifests alone: a
declared output's file extension against the declared pattern of the parameter
it is meant to feed.

So this walks the whole surface at once. For each output whose description
NAMES a consumer tool ("for run_ipsae", "feed this to run_mpnn"), it finds the
consumer's path parameters and reports whether the extension the producer
writes can satisfy any of them.

A name-based link is deliberate: the manifests already say who consumes what,
in prose a model reads, and a separate machine-readable graph would be a second
place for the same fact to drift. What this turns into a test is that the prose
and the schema agree.

Usage::

    python scripts/handoff_matrix.py            # report every mismatch
    python scripts/handoff_matrix.py --all      # every pair, matching or not
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

#: Extensions a pattern is tested against. Real ones only: these are what the
#: engines in this server actually write.
CANDIDATE_SUFFIXES = (".pdb", ".cif", ".cif.gz", ".pdb.gz", ".json", ".npz",
                      ".a3m", ".fa", ".fasta", ".csv", ".txt", ".pkl", ".trb",
                      ".yaml", ".yml", ".npy", ".pt")

_TOOL_MENTION = re.compile(r"\brun_[a-z0-9_]+\b")

#: A MENTION is not always a handoff. `run_rfdiffusion2.metadata_trb` says
#: "same role as run_rfdiffusion_binder's metadata_trb output ... Not parsed by
#: this tool" -- a comparison, and reading it as a handoff reported a mismatch
#: that does not exist. Requiring feeding language instead cut 21 checks to 1,
#: which is the opposite failure: most real handoffs are described in prose
#: that never says "feed". So keep the broad match and subtract the sentences
#: that explicitly disclaim one.
_NOT_A_HANDOFF = re.compile(
    r"(not parsed|same role as|unlike|rather than|instead of|as opposed to|"
    r"is not (?:read|used|consumed))", re.I)


def _sentences_naming(description: str, tool: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;])\s+", description or "") if tool in s]


def suffixes_for(pattern: str | None, name: str) -> set[str]:
    """Which of the real extensions this parameter's pattern admits."""
    if not pattern:
        # No pattern is no constraint: everything is admitted, which is its own
        # kind of finding but not a mismatch.
        return set(CANDIDATE_SUFFIXES)
    return {s for s in CANDIDATE_SUFFIXES if re.search(pattern, f"/tmp/example{s}")}


def output_suffixes(pattern: str | None) -> set[str]:
    """The extensions a declared output glob can produce."""
    if not pattern:
        return set()
    found = {s for s in CANDIDATE_SUFFIXES if pattern.endswith(s)}
    if found:
        # ".cif.gz" also ends with ".gz"-less ".cif"? No -- but "*.cif.gz"
        # matches only .cif.gz, so keep the longest match and drop its prefixes.
        longest = max(found, key=len)
        return {longest}
    return set()


def path_parameters(manifest) -> dict[str, str | None]:
    """Every parameter that takes a path, including ARRAYS of them.

    `run_boltzgen_filter.metrics_files` is `type: array` with
    `items: {format: path}`, and looking only at scalars reported its producer
    as having nowhere to go -- a mismatch that did not exist, from a checker
    blind to half the shapes a path arrives in.
    """
    found: dict[str, str | None] = {}
    for name, spec in (manifest.schema or {}).items():
        if not isinstance(spec, dict):
            continue
        if spec.get("format") == "path":
            found[name] = spec.get("pattern")
            continue
        items = spec.get("items")
        if spec.get("type") == "array" and isinstance(items, dict) \
                and items.get("format") == "path":
            found[name] = items.get("pattern")
    return found


def main(argv: "list[str] | None" = None) -> int:
    from protein_design_mcp.app import manifest_dir
    from protein_design_mcp.manifest.loader import load_manifests

    # argv is a parameter so the suite can call this directly; parse_args()
    # with no argument reads pytest's own command line and exits.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    manifests = {m.name: m for m in load_manifests(manifest_dir())}
    problems: list[str] = []
    checked = 0

    for producer in manifests.values():
        for output in (producer.outputs or ()):
            produced = output_suffixes(output.pattern)
            if not produced:
                continue
            description = output.description or ""
            named = set()
            for candidate in _TOOL_MENTION.findall(description):
                if candidate not in manifests or candidate == producer.name:
                    continue
                context = " ".join(_sentences_naming(description, candidate))
                if _NOT_A_HANDOFF.search(context):
                    continue
                named.add(candidate)
            for consumer_name in sorted(named):
                consumer = manifests[consumer_name]
                params = path_parameters(consumer)
                if not params:
                    continue
                checked += 1
                accepting = {p: suffixes_for(pat, p) for p, pat in params.items()}
                ok = {p for p, allowed in accepting.items() if produced & allowed}
                line = (f"{producer.name}.{output.name} ({', '.join(sorted(produced))}) "
                        f"-> {consumer_name}")
                if ok:
                    if args.all:
                        print(f"OK    {line}: accepted by {', '.join(sorted(ok))}")
                    continue
                problems.append(
                    f"{line}\n"
                    f"      no path parameter accepts it. "
                    + "; ".join(f"{p} admits {sorted(a) or 'nothing'}"
                                for p, a in sorted(accepting.items()))
                )

    # A required path that NO declared output can satisfy cannot be filled from
    # a workflow at all. `design_spec` is required by all six BoltzGen tools and
    # produced by none of them -- it is authored by the caller -- and a model
    # planning run_boltzgen_fold with no way to know that fails on it every
    # time. Either the parameter says where it comes from, or the tool cannot be
    # reached from a plan.
    producible: set[str] = set()
    for producer in manifests.values():
        for output in (producer.outputs or ()):
            producible |= output_suffixes(output.pattern)

    unreachable: list[str] = []
    for consumer in manifests.values():
        for name, spec in (consumer.schema or {}).items():
            if not isinstance(spec, dict) or not spec.get("required"):
                continue
            pattern = path_parameters(consumer).get(name)
            if name not in path_parameters(consumer):
                continue
            allowed = suffixes_for(pattern, name)
            if allowed & producible:
                continue
            described = (spec.get("description") or "").lower()
            if any(k in described for k in ("you write", "author", "supply your own",
                                            "caller-authored", "you provide",
                                            "write this yourself")):
                continue
            unreachable.append(
                f"{consumer.name}.{name} requires {sorted(allowed) or 'an unmatched shape'} "
                "and no tool here produces one; say where it comes from or the tool "
                "cannot be reached from a planned workflow"
            )

    for problem in problems:
        print(f"MISMATCH {problem}")
    for problem in unreachable:
        print(f"UNREACHABLE {problem}")
    problems = problems + unreachable
    print(f"\n{checked} named handoff(s) checked, {len(problems)} mismatch(es)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
