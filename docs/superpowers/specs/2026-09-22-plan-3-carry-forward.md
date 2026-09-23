# Carry-forward into plan 3

Written 2026-09-22 at the close of the env-substrate plan
(`docs/superpowers/plans/2026-09-21-env-substrate-and-cpu-engines.md`). Plan 2 shipped
the environment substrate and the first four CPU adapters — `run_prodigy`,
`run_ipsae`, `run_openmm_minimize`, `run_mpnn` — all four proven live inside the
multi-environment Docker image. Plan 3 adds the GPU engines.

The full final review (449 lines, verdict MERGE, 0 blockers, 17 non-blockers) is
committed at `docs/superpowers/reviews/2026-09-22-plan-2-final-review.md`. This file
records only what plan 3 must act on.

## The framing that decides priority

**Plan 3 and plan 4 add ~25 adapters by copying these four.** A flaw that is cosmetic
in one adapter is structural at 29. Everything in the next section is cheap now and
expensive after the copying starts.

## Gate items — fix before the first plan-3 adapter is written

### G1. One malformed manifest removes EVERY tool

**Controller-reproduced, not taken on report.** Injecting a single unmarked
`run_ghost_tool` reference into `run_mpnn.yaml`'s summary produced:

```
list_tools: ['describe_tool']
call_tool('run_prodigy', ...) -> {"error": "unknown tool: 'run_prodigy'"}
```

`run_prodigy` is a separate, valid, live-proven manifest. One typo in an unrelated YAML
takes down the entire tool surface. The failure is loud in the server log and
**completely invisible to the model**, which simply sees a server that offers one tool.

`ToolRegistry` already supports per-tool exclusion with reasons — the startup exclusion
table uses it. Loading should route a malformed manifest into that same mechanism
instead of aborting the whole load. At 29 manifests maintained by more than one person,
the current behaviour is an availability cliff with a one-character trigger.

### G2. `app.py:258-275` discards `run.outputs` when `parse_output` raises

When an adapter's parser throws, the collected output paths are dropped with it. Only 2
of the 4 adapters name the paths in their own exception message. Plan 3's engines run
for tens of minutes to hours; a regex hiccup on a 90-minute sampler makes completed,
collected files unreachable. Preserve `run.outputs` on the error path before any
long-running adapter lands.

### G3. `pdbfixer` is installed but never used

**Controller-verified.** `Dockerfile.envs:86` installs `pdbfixer=1.11` into the `md`
env and line 87 asserts it imports. `scripts/engines/openmm_minimize.py` never imports
it — `Dockerfile.envs:83` already says so in a comment.

Consequence: any PDB carrying HETATM records, waters, or missing heavy atoms fails,
while `run_openmm_minimize`'s documented input requirement is only "a PDB file". This is
directive #2 (a model must be able to call the tool correctly from its documentation)
and it is the **template wrapper script** every later engine script is modelled on.
Either wire pdbfixer in or state the real precondition in the manifest and doc.

### G4. `run_prodigy.yaml` omits `timeout_s`

The other three manifests declare it; `run_prodigy` inherits the 3600s default for an
engine that finishes in milliseconds. `run_prodigy` is the most-copied manifest in the
repo, so as written it teaches new contributors to omit the very field C11 was carried
forward to introduce. Declare it.

## Model-facing defects (directive #2) — found by the controller, NOT by the review

The final review did not surface either of these. Both are confirmed by direct probe
against the running server, and both live in the meta-tool whose entire job is teaching
a model to call the others.

### M1. `describe_tool` violates the `isError` contract

```
call_tool('run_nonexistent')          -> isError=True
describe_tool(name='run_nonexistent') -> isError=False, payload {"error": "unknown tool: ..."}
```

Every `describe_tool` failure mode is affected, including the empty-category case. A
model branches on the protocol-level `isError` flag before reading the body, so a
failure here is indistinguishable from a success and the error object can be carried
into the next call as if it were tool metadata. Fix before `describe_tool` becomes the
primary discovery surface for 29 tools.

### M2. Four of the six advertised `category` values return errors

The enum offers `generation`, `monomer_generation`, `sequence_design`, `cofolding`,
`scoring`, `meta`. Only `sequence_design` and `scoring` return tools; the rest return
`{"error": "no available tools in category ..."}`.

Three are **transient** — plan 3 populates `generation`, `monomer_generation` and
`cofolding`. **`meta` is permanent**: `describe_tool` is the only meta tool and it
excludes *itself* from its own category listing, so a model asking what meta tools
exist is told there are none. The plan-2 carry-forward recorded only the `meta` case
and framed it as cosmetic; it is four times wider than that, and the `meta` half is a
real self-exclusion bug rather than an empty category.

## Lower priority, recorded so they are not rediscovered

- **No concurrency bound on dispatch.** The CUDA-OOM message gives actively wrong
  advice when the real cause is two concurrent GPU dispatches rather than one oversized
  job. This gets worse the moment plan 3's GPU engines land — it belongs near the top
  of plan 3, not here, if GPU concurrency is possible at all.
- **Nothing reaps preserved workdirs or `results_dir()`.** Workdirs are deliberately
  preserved on failure (the PRESERVE contract). Nothing ever deletes them. Hour-long
  GPU jobs writing structure ensembles will fill the disk.
- **`results_dir()` is a fixed `/tmp/pdmcp-results` created with `exist_ok=True`** —
  open to local co-tenancy symlink capture. Distinct from C9 and not reachable over
  HTTP; it is a local-multi-user concern.
- **`run_openmm_minimize`'s `iterations` field always echoes the requested cap**, so it
  reports the ceiling rather than the iterations actually run. A model reading it
  cannot tell whether minimisation converged.

## Checked negatives — do not re-investigate

The review probed these rather than reading them, and they held:

- **Containment** (`results.py`): `_CheckedMatch` is constructed in exactly one place,
  there is one `shutil.copy2`, and both `multiple` modes exit through the same return.
  File-symlink, dir-symlink-via-glob-component, symlinked-scratch-root and both arity
  modes were all exercised. The `multiple=False` asymmetry found earlier in plan 2 is
  gone. `..` and absolute patterns are rejected at parse time.
- **Process lifecycle** (`dispatch/env.py`): `start_new_session` makes pgid == pid; the
  grandchild-sentinel and `os.kill(pid, 0)` → `ProcessLookupError` tests are real
  proofs rather than inferences; `CancelledError` propagates.
- **C11 is genuinely superseded** — `run()`'s `timeout` is a required keyword-only
  argument and no dispatch-layer constant survives. (G4 is about a manifest omitting
  the field, not about the mechanism.)
- **C9 escalation**: no new HTTP-reachable read primitive was found. Staging is fenced
  by the `\.(pdb|cif)$` patterns and the output globs. **C9 itself remains open and
  unresolved** — the HTTP transport still has no authentication while every tool takes
  a caller-supplied path. It needs a decision before anyone runs `--host 0.0.0.0`.
- **The orchestrator is blocked.** `import protein_design_mcp.tools` raises
  `ModuleNotFoundError`; `tools()`, `resolve()` and `by_category()` all read one
  `_available` map; `describe_tool` has no private path. Controller-verified separately
  across `resolve()`, `call_tool()` and `describe_tool()` for `design_binder`,
  `suggest_hotspots`, `analyze_interface`, `optimize_binder` and `validate_design`,
  with `run_prodigy` in the same probe to prove the probe discriminates.

## One trap from this plan worth carrying

**A blocking result that arrives as the wrong exception type proves nothing.** While
verifying that composite tools are unreachable, the controller's first probe used
`app.registry` — the attribute is `_registry`. Every composite came back
`AttributeError` and *looked* blocked, when the probe was simply broken. A second probe
reported `describe_tool` leaking composite names; it was matching the error message
echoing the caller's own input. Require the *right* failure for the *right* reason, and
put a known-good case in the same probe so a broken probe cannot masquerade as a pass.

## M4 — licence information must not reach the model (user directive, 2026-09-22)

The user's instruction: do not put per-engine licence information in anything the model
sees. Licences are a deployment decision for whoever runs the server, not a criterion a
model should weigh when choosing a tool — a model that reasons about licensing will
either avoid a tool it was given, or assert a licence claim it has no standing to make.

Audited and fixed: exactly one leak existed. `run_prodigy.yaml` described
`run_rosetta_interface` as depending on "a license-gated PyRosetta install", which
`describe_tool` served verbatim. Reworded to availability — "not available in every
deployment of this server. When it is not listed, use PRODIGY." — which is what the
model can actually act on. Manifest `summary`/`description` fields were otherwise clean.

**Still to do:** a loader rule rejecting licence vocabulary in model-facing fields
(`summary`, `description`, `docs/tools/*.md`), in the same shape as the existing
cross-reference marker rule. Without it nothing stops a new manifest reintroducing it,
and 29 more manifests are coming. Licence data stays in this spec family and in
`docs/TOOL_LIST.md`, which are operator-facing.
