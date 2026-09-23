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
