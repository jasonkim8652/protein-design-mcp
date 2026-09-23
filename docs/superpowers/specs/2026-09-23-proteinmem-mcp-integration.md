# Integrating with proteinmem-mcp

Written 2026-09-23, once the destination for this work became explicit: these tools are
meant to land in **RomeroLab/proteinmem-mcp**, so its tool-aware debate and multi-agent
harness can drive them.

Recorded now because it changes how "done" should be judged here — and because the
integration is not the drop-in it first appears.

## The plumbing genuinely is drop-in

`proteinmem-mcp` consumes MCP servers the standard way:

```
configs/mcp_docker.json     "mcpServers": { "protein-design": { "command": "docker", "args": [...] } }
proteinmem_mcp/tool_aware_debate.py
                            MCPToolClient.call_tool(name, arguments)
                            validate_tool_arguments(schema)   <- reads each tool's inputSchema
```

It launches a server as a container over stdio, discovers its tools, and **validates
arguments against the tool's own schema** before calling. That last point matters: the
parameter policy this project adopted — expose every knob, and make each description say
what the parameter does, what changes when it moves, and a sensible range — feeds exactly
that validator and the model reading it. `scripts/container_run.py` already emits the
command shape that config wants.

Two mechanical changes: point the `protein-design` entry at the new image, and replace
`--gpus all` with `--device nvidia.com/gpu=7`, which is how this project makes the
single-GPU constraint structural rather than conventional.

## The real work is that this is a REPLACEMENT

The server already registered there as `protein-design` is **the v1.0.0 server this
project rewrote**. Its tools are `design_binder`, `analyze_interface`,
`predict_structure`, `predict_complex`, `score_stability` — the composites that were
deliberately removed.

Reference counts in that repo:

| Old tool | Files referencing it |
|---|---|
| `design_binder` | 30 |
| `score_stability` | 14 |
| `analyze_interface` | 11 |
| `predict_complex` | 6 |
| `predict_structure` | 4 |

Most are artifacts and test fixtures; the live logic concentrates in
`debate_protein_design.py`, `proteinmem_mcp/binder_workflow.py` and
`run_proteinmem_binder.py`.

`design_binder` is the tool whose chain-order assumption returned the **target** as its
own design (spec §1) — the defect that motivated this entire rewrite. Anything in that
harness still calling it is calling something known to be wrong.

## Why the multi-agent framing improves rather than suffers

A harness that calls `design_binder` is one where an agent says "make me a binder" in a
single call and trusts the result. There is nothing to deliberate about.

Decomposing into 41 atomic tools creates the decisions a debate can actually be about:

- which generator (7 in `binder_generation`, with genuinely different conditioning
  grammars across the RFdiffusion generations alone)
- MSA or not, and from which producer — a first-class parameter with no default, because
  running MSA-free and running with an alignment are different experiments
- which of 9 structure predictors to verify with, and on what axis they differ
- where hotspots come from: measured from a complex (`run_interface_residues`) or
  predicted from an unbound target (`run_epitope_scan`), with the evidence exposed so an
  agent can argue about the ranking rather than inherit it
- multimer or binder-alone, which is never inferred from input shape

Each is a real choice with a defensible answer on either side. That is the substrate a
debate needs.

## Sequencing

1. Finish this branch (AF3's containerisation is the last open item).
2. Swap the `mcp_docker.json` entry; switch `--gpus all` to the pinned device.
3. Rewrite the call sites that use the removed composites as compositions of atomic
   tools. Scope this deliberately — the 30-file count is mostly inert.
4. Update the debate prompts so the agents know the new surface *and its selection
   axes*. A tool list without the axes is a worse prompt than the old one: more options,
   no basis for choosing between them.

## Open question, not to be assumed

`RomeroLab/proteinmem-mcp` is not this user's personal repository. A local branch
`fix/harness-robustness` exists from earlier work, and **forking and opening a PR was
deliberately deferred pending the user's decision**. Push route (fork versus direct) and
permissions must be settled before anything leaves this machine. Until then, local
branches only.

---

## Decisions taken 2026-09-23

Settled by the user; recorded so they are not relitigated.

**1. No composite tool is ever exposed.** The MCP tool surface is the atomistic `run_*`
set and nothing else. This rules out the compatibility-shim option that would have
reproduced `design_binder`'s behaviour in `proteinmem` so existing call sites kept
working — that would have resurrected the orchestrator one layer up, which is exactly
what §3.1 forbids. The harness may still contain workflow code a human or script drives;
what it must not do is hand a model a single call that hides the steps.

**2. GPU is selected by environment variable.** `--device nvidia.com/gpu=${...:-0}`
rather than a hardcoded 7. Index 7 is this machine's constraint, not a property of the
software, and a shared repository should not carry it. The structural guarantee — the
container sees exactly one GPU, so an engine cannot reach another index — is preserved;
only which index moves.

**3. AlphaFold 3 is an optional mount.** If the 8 GB venv mount is absent, that one tool
is excluded from the registry and the other 40 work normally. The loader already reports
per-tool exclusions with reasons (commit `032b210`), so this needs no new mechanism and
fails visibly rather than silently.

**4. Order of publication: server, then image, then harness PR.** The harness PR points
at an image tag, so that tag must exist first. Pushing the harness first would produce a
PR referencing something unbuildable.

## Consequence of decision 1 — the real shape of the work

`binder_workflow.py`, `debate_protein_design.py` and `run_proteinmem_binder.py` currently
reach for `design_binder`, which does generation, sequence design and folding in one
call. Replacing it means the harness composes:

```
run_epitope_scan | run_interface_residues        -> hotspots
  -> run_rfdiffusion3_binder | run_boltzgen_design | run_genie3_binder | ...
  -> run_mpnn | run_boltzgen_inverse_fold
  -> run_esmfold2 | run_chai1 | run_boltz | ...   (with msa stated, never inherited)
  -> run_ipsae | run_prodigy | run_rosetta_interface
```

Each arrow is a decision with a defensible answer either way. That is the substrate the
debate needs, and it is why this is an improvement rather than a migration cost.

**Keep the target-sequence guard.** `binder_workflow.py:99` refuses a design identical to
or containing the target chain — added as defence against `design_binder`'s chain-order
bug. The tool is gone, but the bug class is not: `run_mpnn`'s first FASTA record is its
own input, and any composition can make the same mistake. The guard is cheap and now
protects a pipeline the model assembles itself.

**5. Same repository, branch and tags — not a fork.** `v1.0.0` is already tagged, so
both versions are preserved without duplicating history: the tag fixes that point
permanently and its published image stays on Docker Hub untouched. A fork would add no
preservation the tag does not already give, while splitting issues, CI and any future
fix across two places, and losing the fact that this *is* the next version of the same
project.

Docker Hub was briefly thought to constrain this. It does not: an image is built and
pushed from a local checkout on any branch, with whatever tag is chosen. Only Docker
Hub's own automated-build service couples tags to branches, and this project does not
use it.

The case for a fork would be two versions developed in parallel. That does not apply
here — v1's `design_binder` returns the target as its own design (spec §1), which is the
defect this rewrite exists to fix. v1 is being replaced, not maintained.

Plan: tag `v2.0.0`, publish the matching image tag, and state the v1 → v2 breaking
change at the top of the README so anyone arriving at the repo sees which version they
want before they read further.

**6. No merge to `main` — the tag is on the branch.** Corrects this section's original
plan, which said to merge first. A tag names a commit; it does not need that commit to
be on `main`, and an image is built from a local checkout on any branch. Merging is a
separate decision that can be taken later without changing anything `v2.0.0` points at.
The README's breaking-change section is what tells an arriving reader which version they
want, and it does not depend on which branch is checked out by default.

---

## Release defects found while preparing the tag (2026-09-23)

None of these were visible from the source or the host test suite. Each was found by
deriving the deployment and then actually running it.

| # | Defect | Why it mattered | Caught by |
|---|---|---|---|
| 1 | README still documented v1's 19 tools and `design_binder` | the release's front page would describe removed tools | reading it before tagging |
| 2 | image `CMD` ran `scripts/live_proof.py` | `docker run -i <image>` returned proof output, not an MCP session — the harness could not connect at all | reading the Dockerfile against `mcp_docker.json`'s launch shape |
| 3 | `pyproject` said `1.0.0`; `Server()` passed no version, so `serverInfo` reported `1.30.0` — the MCP SDK's version | a client could not tell v1's surface from v2's | live handshake probe |
| 4 | `container_run.py` emitted `-it` | `cannot attach stdin to a TTY-enabled container` — the generated command was unusable as the MCP invocation it exists to produce | piping into it |
| 5 | `DEVICE=auto` reported `cpu` inside a correctly GPU-pinned container | **27 of 39 tools silently excluded**; the published image would have served 14 | live handshake probe with mounts |

Defect 5 is the instructive one. v1 detected CUDA with `import torch;
torch.cuda.is_available()`, which was sound when the server shared one environment with
its engines. v2's `server` environment deliberately shares no dependencies with any
engine, so it has no torch, so the `except ImportError: "cpu"` fallback fired every
time. The architecture change invalidated the probe, and nothing failed loudly — the
server started, listed tools, and answered calls. It just answered with a third of them.

The replacement asks the question the server actually has: not "can I run CUDA" (it
never does; the engines do) but "is a GPU attached to this container", which is a
numbered `/dev/nvidia<N>` node and needs no dependency at all.

Verified after the fixes, live against the built image over a real MCP handshake:
**41 tools with mounts, 7 without**, `serverInfo` reporting `2.0.0`.
