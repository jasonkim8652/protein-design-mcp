# Carry-forward into plan 2

Written 2026-09-21 at the close of the manifest-core plan
(`docs/superpowers/plans/2026-09-21-manifest-core.md`). Plan 2 covers spec §5.1/§5.3
— the pixi multi-environment build and Dockerfile — and the remaining ~28 engine
adapters. This file records what the first plan deliberately left behind, so none of
it has to be rediscovered.

## Deliberately deferred from the spec

Recorded in the plan's own self-review; not gaps.

- MCP Resource documentation layer (spec §4.3 layer 3). Layers 1 and 2 shipped.
- `PROFILE` env var (spec §6) for exposing a 9-tool core subset.
- Weights and licence **detection**. The filter exists in `ToolRegistry` and is live;
  nothing populates `available_weights` or `licensed`, so any manifest declaring
  `requires.weights` is excluded. Startup now logs the exclusion table, so this is
  visible rather than silent — but it must be implemented before the first
  weights-gated engine lands.
- `max_residues` enforcement. Parsed into `Manifest`, enforced nowhere.
- `get_job_status`.
- `run_mpnn`. `dauparas/LigandMPNN` is not checked out on this machine; only the
  `ligandmpnn_env` conda environment exists.
- `pixi.toml` / `pixi.lock` and the multi-environment Dockerfile.

## Unreachable code removed in plan 2, Task 9

`src/protein_design_mcp/tools/` — 17 modules, 2061 lines — is gone (commit
`df08252`). Its `__init__.py` eagerly imported seven removed-tool modules, so it was
an import hazard, not merely dead weight.

`tools/status.py` was NOT deleted with the rest. It moved to
`src/protein_design_mcp/job_status.py`: `get_design_status` is a pure query over
`utils/job_queue.py`, not orchestration, and it is the implementation the planned
`get_job_status` meta-tool (spec §3.2) should adopt rather than rewrite.

**Caveat for whoever implements `get_job_status`:** `_estimate_time_remaining` in that
module hardcodes per-step timings for `rfdiffusion`, `proteinmpnn` and `esmfold` — the
OLD composite pipeline's steps, which no longer exist. Its 6 tests in
`tests/test_job_queue.py::TestGetDesignStatus` pass, but they pin stale behaviour.
Replace the estimator when you wire the tool up; do not trust its numbers.

Six test files went with the package: `test_design_binder.py`,
`test_validate_design.py`, `test_optimize.py`, `test_hotspots.py`, `test_tools.py`,
`test_analyze.py`. They carried 26 of the suite's 41 known failures (hotspots 17,
optimize 7, design_binder 1, analyze 1, validate_design 0, tools 0), so the baseline
is now **15**.

`tools/hotspots.py` was reviewed for salvage before deletion and nothing was kept: it
is orchestration glue over `utils/` (sasa, uniprot, conservation, pubmed,
fetch_structure), where the real analysis lives untouched. What it added on top was
hardcoded-weight scoring heuristics and blanket `except Exception` swallowing, neither
of which fits the manifest-driven adapter pattern. Likewise `tools/analyze.py`:
`analyze_interface`'s hand-rolled distance-cutoff heuristics are superseded by
`run_prodigy` (shipped) and `run_rosetta_interface` (planned), per spec §8.

Recoverable from git at `7a45f13:src/protein_design_mcp/tools/` if any of this proves
wrong.

## Findings the final review carried forward

Not merge blockers; each is cheap now and expensive at 29 manifests.

| Ref | Finding | Why it matters at scale |
|---|---|---|
| C7 | Two code paths derive a `Tool` from a `Manifest`: `registry._json_schema_for` and a hand-inlined copy in `app.list_tools` for `describe_tool`. They agree only by coincidence. | A contributor adding a meta-tool copies the inline block instead of the helper. |
| C8 | Cross-manifest doc references are unverified. `run_prodigy`'s doc names `run_rosetta_interface` and `run_ipsae` as "(not yet implemented)". When those land, the doc actively misleads and **no test fails**. Also, `_check_sibling_docs` counts composite manifests toward category size. | Add a loader rule that every `run_[a-z0-9_]+` mentioned in a doc resolves to a known manifest, and exclude composites from the sibling count. |
| C9 | The HTTP transport has no authentication and every tool takes a caller-supplied filesystem path. Default bind is `127.0.0.1` and the `--host` help now warns, but an operator following the GPU-host deployment has an arbitrary-file-read-shaped primitive exposed. | Needs a real answer before anyone runs `--host 0.0.0.0`. |
| C11 | One process-wide `DEFAULT_TIMEOUT_S`, read at import. PRODIGY takes milliseconds; BoltzGen takes hours. | A per-manifest `timeout_s` is one field now, 28 edits later. |
| — | No `outputs:` declaration in the manifest schema. PRODIGY parses stdout, so the template teaches stdout parsing; most of the 28 write files. | The first file-writing adapter will invent a convention and 27 will copy it. Design `outputs:` **with** that first engine, not before. |

## Deferred minors still open

Triaged by the final review. Fix-soon items are worth folding into plan 2's first
task:

- Duplicate-name and sibling-doc errors name the TOOL, not the FILE; nothing enforces
  `filename == tool name`. Compounds with schema-entry validation at 29 manifests.
- `ToolRegistry`'s exclusion messages are written for a human operator ("Set
  DEVICE=cuda") but are rendered verbatim to a model that cannot set env vars.
- `dispatch/env.py` buffers stdout/stderr fully via `communicate()`. A diffusion
  sampler printing per-step progress for an hour is the exact bad profile. Cap the
  retained tail rather than streaming.
- `test_the_meta_tool_manifest_is_itself_valid` is narrow; fixing C7 closes it for
  free.
- `describe_tool(category="meta")` returns an error although the enum offers "meta".
  Drop it from the enum or special-case it.

Accepted as-is, recorded so nobody re-litigates them: `_require` error file context
(the loader wraps it), `test_manifest_is_frozen` catching bare `Exception`,
`ToolNotAvailable(KeyError)` base class, non-string dict keys colliding under
`to_jsonable`, `RecursionError` on cyclic structures, OOM substring possibly
mislabelling a host-RAM OOM, the double-catch in `env.py`'s exception handling, and
`log_level="info"` being hardcoded.

One that needs a comment rather than a change: `to_jsonable`'s `int` branch precedes
the numpy branch, and is safe **only** because no numpy integer type subclasses
Python `int` — unlike `np.float64`, which does subclass `float`. That rationale is
load-bearing and undocumented, and is exactly what a future "tidy-up" of branch
order would break.

## Three traps this plan fell into, worth naming

**A test can confirm a file is current without confirming it is correct.** The
doc-staleness test compares committed output against freshly generated output. When
the generator emitted a broken Markdown table, both sides were broken identically and
the test passed. Structural assertions (here, that every table row has the same
unescaped-cell count as the header) catch what equality checks cannot.

**A fix that improves a test's reliability can destroy its validity.** While making
the HTTP transport test non-flaky, an inline copy of the production server setup
replaced the call to `run_server`. Every test still passed, and the guard no longer
guarded anything. When a test is the only protection for a specific bug, re-verify
after every change that it still fails when that bug is reintroduced — in the
production code, not in the test's own copy.

**A deletion list is only as good as the dependency scan behind it.** This document
originally named five test files to delete with `tools/` and stated they accounted for
25 of the 41 failures. The count was right; the list was not. Two further files —
`tests/test_analyze.py` and `tests/test_job_queue.py` — imported the package and were
never checked for, because the list was assembled from "tests named after removed
tools" rather than from `grep -rn "protein_design_mcp\.tools" tests/`. It surfaced
only at execution time, and only because the implementer's brief required a safety
grep and forbade working around a hit. Derive deletion lists from the importer graph,
never from naming conventions.

**A clean `git ls-files` does not prove a deleted package is unimportable.** After the
deletion commit, `src/protein_design_mcp/tools/` still existed on disk holding nothing
but a git-ignored `__pycache__`. Under PEP 420 a bare directory on the path is a
namespace package, so `import protein_design_mcp.tools` still succeeded, returning
`_NamespacePath([...])`. Git tracked nothing there and every grep was clean, yet the
working tree masked the removal — any reference the grep had missed would have
imported fine locally and failed only on a fresh clone, i.e. exactly where nobody was
looking. Confirm a removal by attempting the import and requiring `ModuleNotFoundError`.
