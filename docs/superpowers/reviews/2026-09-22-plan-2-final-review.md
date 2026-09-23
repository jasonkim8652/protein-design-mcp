# Final whole-branch review — plan 2 (env substrate + four CPU engines)

Range `38eb4b9..82bdef0`, 21 commits, 60 files, branch `dev`, nothing pushed.

**Verdict: MERGE.** 0 merge blockers, 17 non-blockers.

The three things this review was pointed at hardest — the containment choke point, the
process lifecycle, and whether the orchestrator is actually unreachable — each held up
under direct probing, not just under reading. The non-blockers below are almost all of
the "cheap now, expensive at 29" kind, and I have ordered them by how much plan 3
multiplies them rather than by severity in isolation.

Every finding is labelled CONFIRMED (verified by reading the tree and/or running code)
or PLAUSIBLE (reasoned, not executed). Items already recorded in
`docs/superpowers/specs/2026-09-21-plan-2-carry-forward.md` are not repeated.

---

## Part 1 — What I verified as sound

These are stated because they were the assignment, and because a negative result that
was actually tested is worth more than silence.

### 1.1 The containment choke point holds, symmetrically — CONFIRMED

`src/protein_design_mcp/results.py`. I verified the structural invariant by reading and
then attacked it by running code.

- `_CheckedMatch` is constructed in exactly one place, `results.py:99`.
- `shutil.copy2` appears exactly once in the module, `results.py:138`, inside
  `_copy_checked_match`, whose only parameter type is `_CheckedMatch`.
- `_checked_matches` (`results.py:131`) runs **every** surviving match through
  `_check_containment` on a single `return` shared by both `multiple` modes. The
  `multiple=False` ambiguity branch (`results.py:120-130`) also checks each match before
  naming it in the error. There is no path from a glob match to a copy that bypasses the
  check.

The asymmetry an earlier round found — `multiple=False` permitting a symlink escape that
`multiple=True` refused — is gone. I probed all four shapes against the real module:

| Case | Result |
|---|---|
| file symlink → outside, `multiple=False` | `OutputPathEscapeError` |
| file symlink → outside, `multiple=True` | `OutputPathEscapeError` |
| directory symlink traversed by a `*/` glob component | `OutputPathEscapeError` |
| workdir itself reached through a symlink (symlinked scratch root) | collected normally, no bare `ValueError` |

Manifest-level `..` and absolute patterns are rejected at parse time
(`manifest/schema.py:237-241`); I confirmed `../x`, `/abs/x` and `a/../../x` all raise
`ManifestError`. The two-pass design (match-and-check everything, then copy) means a
later spec's escape leaves nothing copied, and `tests/test_results.py:316` pins that.

### 1.2 Process lifecycle is correct — CONFIRMED

`src/protein_design_mcp/dispatch/env.py`.

- `start_new_session=True` (`env.py:106`) makes the child a session and process-group
  leader, so `process.pid == pgid` and `os.killpg(process.pid, ...)` is well-formed.
- Orphan grandchildren: `tests/test_env_dispatcher.py:145` spawns a non-detached
  grandchild that writes a sentinel after the timeout, and asserts the sentinel never
  appears. That is a real test of the group kill, not an inference.
- Cancellation: `tests/test_env_dispatcher.py:173` reads the child's own pid from a file
  and asserts `os.kill(pid, 0)` raises `ProcessLookupError` after `task.cancel()`. The
  `except BaseException` block (`env.py:133-141`) kills and reaps, and re-raises, so
  `CancelledError` propagates rather than being swallowed. The
  `contextlib.suppress(asyncio.CancelledError)` around `await process.wait()` is
  followed by `raise` of the original exception, so it is not the usual antipattern.
- Zombies: SIGKILL is followed by `await process.wait()` in both the timeout and the
  `BaseException` paths; the cancellation test would fail on a zombie only if asyncio
  had not reaped, and it passes.
- A child that ignores SIGTERM is a non-issue: SIGKILL is sent directly, never SIGTERM.
- Concurrent dispatches are isolated by construction — each run gets
  `/tmp/pdmcp-<uuid4[:12]>` created with `exist_ok=False` (`env.py:61-64`), and
  collected results are namespaced by that same directory name as `run_id`.

### 1.3 C11 is genuinely superseded, not shadowed — CONFIRMED

`EnvDispatcher.run`'s `timeout` is keyword-only with **no default** (`env.py:85`), so
every caller must supply one; `app.py:254` supplies `manifest.timeout_s`. `grep` over
`src/` finds no surviving process-wide `DEFAULT_TIMEOUT_S` in the dispatch layer — the
only one left is `manifest/schema.py:36`, which is the *manifest field's* default and is
per-tool overridable. The per-manifest timeout replaced the constant; it does not sit
alongside it unused.

### 1.4 The orchestrator is unreachable — CONFIRMED

- `import protein_design_mcp.tools` raises `ModuleNotFoundError`. The PEP 420 namespace
  trap named in the carry-forward was actually closed, not just git-clean.
- `ToolRegistry.__init__` partitions into `_available` / `_reasons` once
  (`registry.py:66-71`); `tools()` reads `_available`, `resolve()` reads `_available`
  first and only consults `_reasons` to produce a refusal (`registry.py:109-115`), and
  `by_category()` reads `_available`. Composites cannot appear in any of the three.
- `describe_tool` has no private path: `_describe_one` goes through `registry.resolve()`
  (`meta_tools.py:73`) and `_describe_category` through `registry.by_category()`
  (`meta_tools.py:115`). A composite name returns the exclusion reason plus the available
  list, never the manifest body.
- `app.ADAPTERS` (`app.py:37-42`) has four entries; an unmapped manifest returns an
  error rather than dispatching.
- `pipelines/` and `resources/` are still on disk but I verified neither is imported by
  `protein_design_mcp.server` at import time.

### 1.5 The staging PRESERVE contract is coherent — CONFIRMED

- `_validate_stage` (`manifest/schema.py:141-159`) requires each `stage` entry to name a
  real schema parameter that is `format: path`, checked at load, and `_parse_engine`
  rejects duplicates.
- Failure: `app.py:244-250` catches `OSError` from `stage_inputs` and re-raises an
  `EngineError` using `run()`'s own preserve-and-name wording, so the workdir is not
  silently orphaned. `tests/test_server_wiring.py:508` pins it.
- Success: `run()` removes the workdir including the staged copies, same as any run.
- Concurrency: staging targets `workdir/<param_name>/<basename>` under a per-run uuid
  directory, so two concurrent `run_ipsae` calls with inputs of the same basename cannot
  collide. `tests/test_staging.py:35` pins the two-parameter case.
- A `stage` param naming a path outside the workdir is the *normal* case, not an error —
  `_resolve_path_params` has already made it absolute, and staging exists precisely to
  pull it in. Nothing about it escapes the workdir.

### 1.6 C9: I found no new HTTP-reachable escalation

I specifically looked for a way the new code turns the unauthenticated HTTP transport
into a file-read or file-write primitive, since that was called out as the one thing
worth reporting. I did not find one, and I want that stated as a checked negative rather
than left ambiguous:

- `stage_inputs` does copy a caller-named file into the workdir, which is the closest
  thing to an arbitrary read. But `run_ipsae.structure` is constrained by
  `pattern: '\.(pdb|cif)$'`, and the collecting glob is `structure/*.txt`, so the staged
  file can never be collected back out and returned. The same holds for
  `run_mpnn.backbone_pdb` (`seqs/*.fa`) and `run_openmm_minimize.input_pdb`
  (`minimized.pdb`).
- No adapter echoes input file content into its payload. The engine-failure path returns
  `stderr[-2000:]`, which is engine diagnostics, not file content.

The one genuinely new filesystem exposure is local, not network, and is N7 below.

---

## Part 2 — Non-blockers, ordered by what plan 3 multiplies

### N1. One malformed manifest removes *every* tool — CONFIRMED

`src/protein_design_mcp/manifest/loader.py:141-151`, `src/protein_design_mcp/app.py:68-83`

`load_manifests` raises on the first bad file, and the cross-manifest checks
(`_check_unique`, `_check_sibling_docs`, `_check_doc_references`) raise for the whole
set. `build_registry` catches `ManifestError` and serves an **empty** registry.

I reproduced this. I copied the four shipped manifests to a temp directory, changed one
line of `run_mpnn.yaml`'s doc to mention a tool that does not exist, pointed
`PROTEIN_MCP_MANIFEST_DIR` at it, and got:

```
tools after ONE bad manifest: []
list_tools: ['describe_tool']
call run_prodigy: {"error": "unknown tool: 'run_prodigy'"}   isError=True
```

Failure scenario at 29 manifests: a contributor lands `run_chai1` and, in the same PR,
renames `run_esmfold2` to `run_esmfold`. `run_prodigy.yaml`'s doc still says
`run_esmfold2`. The server starts, logs one ERROR line nobody is watching, and every
tool disappears. The model sees a server with one tool (`describe_tool`) that answers
"unknown tool" for everything, with no indication why.

The answer to "loud at load or silent at call time" is therefore: loud at load, but with
a total blast radius and a diagnostic the model never sees. The registry already has a
per-tool exclusion mechanism with reasons (`registry.excluded()`), and it is the right
home for this: load each file independently, exclude the ones that fail with their own
`ManifestError` as the reason, and demote the cross-manifest checks to exclusions of the
offending manifest rather than an abort of the set. Cheap at 4 files, structural at 29.

Not a blocker: all four shipped manifests are valid and `tests/test_server_wiring.py:125`
(`test_real_manifests_all_load`) pins that.

### N2. A parse failure discards outputs that were already collected — CONFIRMED

`src/protein_design_mcp/app.py:258-275`

`parse_output` runs *after* `dispatcher.run` has already collected declared outputs into
`results_dir()` and removed the workdir. If `parse_output` raises, the generic
`except Exception` handler returns only `f"adapter for {name} failed: {exc}"`. `run` is
in scope but `run.outputs` is never consulted, so the collected file paths reach the
caller only if the individual adapter happened to interpolate them into its own
exception message.

Two of four do (`adapters/mpnn.py:128-131` includes `{fasta_paths}`,
`adapters/ipsae.py:154-157` includes the per-file errors). Two do not
(`adapters/openmm_minimize.py:38-42`, `adapters/prodigy.py:55-58` — though prodigy
declares no outputs, so it is vacuous there). The convention is therefore 50/50 and the
next 25 adapters copy whichever they open first.

Failure scenario: a plan-3 diffusion sampler runs for 90 minutes, writes 200 backbone
PDBs which are collected to `results_dir()`, then prints a log line that breaks the
adapter's regex. The model gets `adapter for run_rfdiffusion failed: ...` and the 200
files are on disk with no path anyone can name. The fix belongs in `app.py`, once —
append `run.outputs` to the error payload when a run completed — not in 29 adapter
docstrings.

### N3. The `md` environment installs pdbfixer; the engine script never uses it — CONFIRMED

`Dockerfile.envs:86`, `scripts/engines/openmm_minimize.py:12-13,29-32`

The image builds `md` with `openmm=8.6.1 pdbfixer=1.11` and even verifies the import at
build time. The script imports only `openmm` and goes straight from
`PDBFile(args.input_pdb)` to `Modeller(...).addHydrogens(forcefield)`.

Failure scenario: `run_openmm_minimize(input_pdb="1abc.pdb")` on anything downloaded
from the RCSB. The first `HOH`, `SO4`, `HETATM` ligand, or missing heavy atom produces
`ValueError: No template found for residue ...`, a non-zero exit, and a raw OpenMM
traceback tail as the tool error. The manifest's "What you must supply" says only "A PDB
file. Multi-chain inputs are relaxed as one system." (`run_openmm_minimize.yaml:44-45`,
and the same text in `docs/tools/run_openmm_minimize.md:31-32`) — nothing warns that the
input must be a complete, ligand-free, water-free polymer.

This is directly on the user's directive #2: the documentation as written does not let a
model call this tool correctly for the input it would most naturally reach for. The
dependency to fix it is already installed. This is also the first `scripts/engines/*.py`
wrapper, i.e. the template for every engine that needs one.

### N4. `iterations` reports the requested cap, never the count performed — CONFIRMED

`scripts/engines/openmm_minimize.py:57`

```python
print(f"iterations: {args.max_iterations}")
```

`simulation.minimizeEnergy(maxIterations=...)` returns nothing and OpenMM exposes no
iteration count, so this field is definitionally equal to an input the caller supplied.
`adapters/openmm_minimize.py:51` parses it and `run_openmm_minimize.yaml:48-50` documents
it as a returned value.

Failure scenario: a model calls with the default `max_iterations: 500`, gets
`iterations: 500` back, concludes minimisation hit the cap without converging, and
re-runs at `max_iterations: 10000` — when in fact the structure converged in 40 steps.
The honest fix is to drop the field or rename it `max_iterations_requested`. One line,
and it is in the file that 25 more engine wrappers will be copied from.

### N5. No bound on concurrent dispatches — CONFIRMED (absence); impact PLAUSIBLE

`src/protein_design_mcp/app.py:251`, `src/protein_design_mcp/dispatch/env.py:80`

`call_tool` awaits `dispatcher.run` with nothing limiting how many are in flight. The MCP
SDK dispatches each `tools/call` as its own task, so a model that fires four
`run_mpnn` calls gets four `micromamba run` subprocesses at once.

For plan 2's four CPU engines this is merely wasteful. For plan 3's GPU samplers it is a
correctness problem: two concurrent `run_rfdiffusion` on one card produce a CUDA OOM,
which `env.py:148-154` then translates into "reduce the number of samples, shorten the
input, or use a smaller model variant" — advice that is actively wrong, because the real
remedy is to not run them concurrently. A per-`engine.env` (or per-device) semaphore in
`EnvDispatcher` is the natural place and costs a handful of lines now.

### N6. Nothing ever reaps preserved workdirs or collected results — CONFIRMED

`src/protein_design_mcp/dispatch/env.py:181`, `src/protein_design_mcp/results.py:142-147`

`shutil.rmtree` runs only on the fully successful path. Every start failure, non-zero
exit, timeout, output-collection failure and staging failure leaves `/tmp/pdmcp-<hex>/`
behind permanently, by design and with no TTL. Separately, every successful run's
collected outputs land in `/tmp/pdmcp-results/<run_id>/` and are never removed either —
also by design, since results must outlive the workdir, but with no ceiling.

Failure scenario: a long-lived container where a model iterates a design campaign —
a few thousand `run_mpnn`/`run_openmm_minimize` calls with a 10% failure rate fills the
container's writable layer with orphaned scratch directories and a permanent copy of
every structure ever produced. The recovery is `docker restart`, which also destroys
every result path the model is still holding.

### N7. `results_dir()` is a fixed, guessable name under `/tmp` — CONFIRMED (code); exploit requires local co-tenancy

`src/protein_design_mcp/results.py:142-147`, `src/protein_design_mcp/results.py:199`

Workdirs are safe: `pdmcp-<uuid4>` created with `exist_ok=False`, unguessable and
unpre-plantable. The results directory is not: it is the constant
`Path(gettempdir()) / "pdmcp-results"`, created with `mkdir(parents=True, exist_ok=True)`,
which follows an existing symlink.

Failure scenario, on a shared host rather than inside the container: a local user creates
`/tmp/pdmcp-results` as a symlink to a directory they own before the server first runs.
Every collected output is then written into attacker-controlled space, and because
`adapters/ipsae.py:146` and `adapters/mpnn.py:112` read those files back *after* the copy,
the attacker has a window to substitute content that the adapter then parses and returns
to the model as engine output.

This is a different vector from C9 and does not make the HTTP transport worse. It is
cheap to close (`PROTEIN_MCP_RESULTS_DIR` already exists; default it under the scratch
root, or `mkdir` with `exist_ok=False` on a per-process subdirectory).

### N8. The template manifest omits `timeout_s` — CONFIRMED

`src/protein_design_mcp/manifests/run_prodigy.yaml`

`run_prodigy.yaml` has no `timeout_s`, so it inherits `DEFAULT_TIMEOUT_S = 3600` for an
engine the manifest's own summary describes as running "in milliseconds". The three
manifests written *in* this plan all set it (300 / 1800 / 1800). The oldest and most
copied manifest is the one that teaches "omit this field". Since C11 was carried forward
specifically to get per-manifest timeouts, having the exemplar not use the field is the
wrong lesson to propagate 25 times. Set `timeout_s: 60` on `run_prodigy` — or make the
field required.

### N9. An engine `entry` hardcodes the container's filesystem layout — CONFIRMED

`src/protein_design_mcp/manifests/run_openmm_minimize.yaml:9`

```yaml
entry: ["python", "/app/scripts/engines/openmm_minimize.py"]
```

`/app` is a Dockerfile.envs artifact (`WORKDIR /app`). Running the server outside the
image against a host conda env named `md` gives `FileNotFoundError` from the Python
interpreter, not from the dispatcher, so it surfaces as a non-zero exit with a traceback
rather than the dispatcher's "could not start engine" message. This is the first manifest
to reference a repo-shipped wrapper script and every future one that needs a wrapper will
copy the absolute path. A `${PROTEIN_MCP_ENGINE_SCRIPTS}` substitution, or resolving
wrapper scripts via `importlib.resources` the way `manifest_dir()` already does, fixes the
class rather than the instance.

### N10. Adapter divergence on partial success — CONFIRMED

Across `src/protein_design_mcp/adapters/`:

| | prodigy | ipsae | openmm | mpnn |
|---|---|---|---|---|
| output source | stdout | collected file | stdout | collected file |
| primary field missing | raise | raise | raise | raise |
| secondary field missing | `None` + `logger.warning` | row skipped entirely | `None` | `None` |
| `del manifest` | yes | yes | **no** | yes |
| error names the output paths | n/a | yes | no | yes |

The primary-field rule is consistent and correct. The secondary-field rule is not:
`prodigy.py:60-70` degrades `dissociation_constant_M` and `intermolecular_contacts` to
`None`, whereas `ipsae.py:101-109` treats *any* non-numeric required cell as a reason to
skip the whole row, so a real ipSAE run that emits a placeholder in `pDockQ` yields
"ipSAE produced no chain-pair row" rather than a result with one null field.

Failure scenario: a future ipSAE version prints `NA` in `pDockQ` for a single-chain-pair
complex. The tool returns an error and loses the `ipsae` value it successfully computed.
Pick one rule — I would take prodigy's, since a model can act on a null field but not on
an error — write it into the adapter template, and note it in the plan-3 brief.

`del manifest` in three of four is cosmetic but is exactly the kind of thing that becomes
noise at 29 files; either all adapters do it or none should.

### N11. `_check_sibling_docs` still counts composites — CONFIRMED

`src/protein_design_mcp/manifest/loader.py:124-138`. C8's first half (verify every
`run_*` mention resolves) landed in this plan; the second half did not. Inert today
because no composite manifest ships. It becomes live the moment one does, forcing a
sibling-comparison section into docs about a tool the model can never call.

### N12. `_has_marker_for_mention` matches per name, not per occurrence — CONFIRMED

`src/protein_design_mcp/manifest/loader.py:61-69`. The regex searches the whole
paragraph for `<name> (not yet implemented)`. A paragraph that mentions `run_chai1` twice
— once marked, once bare — passes. Commit `9785cb5` narrowed this from "anywhere in the
text" to "anywhere in the paragraph", which is most of the value; the remainder is the
duplicate-mention case. Low.

### N13. The copy2 invariant is held by docstring, not by a test — CONFIRMED

`src/protein_design_mcp/results.py:50-68` describes the `_CheckedMatch` /
`_copy_checked_match` invariant in careful detail, and the invariant does currently hold.
Nothing enforces it: `grep` finds no test or lint rule asserting that `results.py`
contains exactly one `shutil.copy2`, or that `_CheckedMatch` is constructed in exactly
one place. The module's own comment predicts the failure mode ("adding a new branch to
`collect_outputs` ... cannot skip the check without also having to invent its own copy
call"), which is true but is a claim about contributor behaviour. A four-line AST or
source-grep test would make it a claim about the code.

### N14. Hardlinks defeat `_check_containment` — CONFIRMED (by probe); design limit, not a defect

I hardlinked a file from outside the workdir to `workdir/a.txt` and collected it with
pattern `*.txt`. It was copied out with its content intact, because `Path.resolve()`
cannot distinguish a hardlink from the original file — no filesystem API can. This is not
fixable at this layer and is not a threat under the intended model (the engine runs
inside the container as the same user and could simply read the file itself). Worth one
sentence in the `_check_containment` docstring, which currently implies symlinks are the
general case it defends against rather than the only one it can.

### N15. A staged input can be collected back out as a result — CONFIRMED (as an absent check)

`run_ipsae` is safe: the stage directory is `structure/`, the input must match
`\.(pdb|cif)$`, and the output glob is `structure/*.txt`. Nothing enforces that
relationship. A plan-3 manifest that writes `stage: ["input_pdb"]` with
`outputs: [{pattern: "input_pdb/*.pdb"}]` — a natural thing to write, since the stage
directory is where that engine's outputs land — would collect the caller's own input back
and present it under `outputs` as an engine product. A loader rule that an output pattern
may not have a stage directory as its first path component would close it at load time.

### N16. Advertised resource templates have no handler — CONFIRMED; pre-existing

`src/protein_design_mcp/server.py:71-85` registers `list_resource_templates` advertising
`protein://structures/{pdb_id}` and `protein://designs/{job_id}/{design_id}`, and there is
no `read_resource` handler anywhere in the module. A client that follows the advertised
template gets a protocol-level error. This predates the plan, but the plan deleted the
tools that produced the `job_id`s in the second template, so it is now doubly dead.

### N17. `README.md` documents a tool surface that no longer exists — CONFIRMED; pre-existing

`README.md:11,67,149,247,268,286,537,557` still describe "19 tools", `design_binder`,
`design_fold`, `generate_backbone`, `predict_structure_boltz`, and an ASCII diagram of the
RFdiffusion→ProteinMPNN→ESMFold composite. All of those are gone as of `df08252`. This is
the text `pyproject.toml` ships as the PyPI long description (visible in the checked-in
`PKG-INFO`). Human-facing rather than model-facing, so it is not directive #2, but it is
the first thing anyone reads and it is now wholly wrong. Plans 3 and 4 will make it worse
before they make it better; a one-paragraph "this README describes the pre-refresh
surface; see `docs/tools/`" note would cost nothing.

---

## Part 3 — Directive #2: can a model call these four from the documentation alone?

Read as the model: yes for three, with one caveat and one gap.

- **Units are stated in field names**, consistently: `binding_affinity_kcal_per_mol`,
  `dissociation_constant_M`, `initial_potential_energy_kj_mol`,
  `energy_change_kj_mol`. Parameter units are in descriptions ("PAE cutoff in
  Angstroms", "Temperature in Celsius"). This is genuinely good and worth keeping as the
  convention.
- **Required inputs are unambiguous**: the generated tables mark required, default,
  constraint and description per parameter, and every parameter carries an `example`.
- **`run_prodigy`, `run_mpnn`** are callable from their docs alone.
- **`run_openmm_minimize`** is callable but its "What you must supply" is wrong by
  omission — see N3.
- **`run_ipsae`** is callable in form but not in practice: `pae_json` is described only
  as "PAE matrix JSON produced by a structure predictor", and ipSAE accepts several
  mutually incompatible predictor formats. A model has no way to know which. This is
  inherent to plan 2 (no cofolding tool ships yet to produce one), so I am not counting it
  as a finding — but when plan 3 lands `run_chai1`/`run_boltz2`, the `pae_json`
  description must name which of their outputs to pass.
- The docs do not state each tool's `timeout_s`. A model choosing `num_sequences: 128`
  for `run_mpnn` on CPU has no way to know it has 1800s. Minor; the doc generator could
  emit it for free.

---

## Recommended order for the plan-3 brief

Fix before any adapter is copied: **N1** (loader blast radius), **N2** (outputs lost on
parse failure), **N8** (template manifest sets `timeout_s`), **N10** (pick one
partial-success rule and write it into the template).

Fix while the openmm wrapper is still the only one: **N3** (pdbfixer), **N4**
(`iterations`), **N9** (hardcoded `/app`).

Fix before the first GPU engine: **N5** (concurrency bound — the misleading OOM advice
makes this worse than it looks), **N6** (scratch/results reaping).

Everything else is cleanup.
