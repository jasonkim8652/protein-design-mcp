# SDD ledger — plan: docs/superpowers/plans/2026-09-21-env-substrate-and-cpu-engines.md

Spec: docs/superpowers/specs/2026-09-21-atomistic-tool-refresh-design.md (§5.1, §5.2, §5.3)
Carry-forward: docs/superpowers/specs/2026-09-21-plan-2-carry-forward.md
Branch: dev
BASE at start: 38eb4b98103a368add0738cf320865b3998f368b
Isolation: dedicated clone `protein-design-mcp-dev`; the user's primary checkout
`~/projects/protein-design-mcp` holds uncommitted work and is never touched.
Prior context: plan 1 (manifest core) is merged into this branch; the dispatch
contract was proven live — real PRODIGY through the real SDK handler returned
-11.2 kcal/mol on 1BRS inside `protein-design-mcp:envs-proto`.

## Pre-flight conflict scan

### Cross-task pairs (shared file or interface)

| Pair | Produced → Consumed | Finding |
|---|---|---|
| 1 → 2,3,5,6 | `OutputSpec`, `Manifest.outputs`, `Manifest.timeout_s` | consistent; Task 2 constructs `OutputSpec(name=, pattern=)` positionally-by-keyword matching Task 1's field order |
| 2 → 3,5 | `collect_outputs`, `CompletedRun.outputs`, `run(..., outputs=)` | consistent; Task 3's `_RecordingDispatcher` mirrors the new signature exactly |
| 3 → 5 | successful payload gains an `outputs` key | consistent; Task 5's live case expects `outputs` in `expect_keys` |
| 4,5,6 → 7 | adapters registered in `ADAPTERS`; `CASES` must cover every tool | consistent; Task 7's coverage test is what enforces it |
| 4,5,6 → each other | each regenerates `docs/tools/` | expected; the plan already notes that running them out of order makes the staleness test fail until regeneration. That is the guard working. |
| 4 → 8 | Task 4 un-marks `run_ipsae` in run_prodigy.yaml; Task 8 adds the loader rule | **checked the actual file**: both markers sit on the SAME LINE as their mention (`run_prodigy.yaml:32` for run_rosetta_interface, `:37` for run_ipsae). Task 8's per-line rule therefore keeps accepting `run_rosetta_interface` (still marked, lands in plan 4) after Task 4 un-marks only `run_ipsae`. No conflict. |
| 5 → 7 | `env: md` declared before Task 7 creates it | **see ruling 3** |
| 8 → app.py | `_json_schema_for` renamed public | Task 8 is the only task touching that name; no earlier task imports it |
| 9 → (repo) | deletes `tools/` + 5 test files | arithmetic checks out: 41 − (7 test_optimize + 17 test_hotspots + 1 test_design_binder) = 16, which is what the plan predicts |

### Intra-task self-consistency

| Task | Finding |
|---|---|
| 1 | tests reuse the existing `MINIMAL` fixture in tests/test_manifest_schema.py — it exists; `_parse_outputs` uses `Path`, and the plan says to add the import |
| 2 | `CompletedRun` gains a mutable default, and the plan correctly specifies `field(default_factory=dict)` plus the `field` import |
| 3 | `_RecordingDispatcher` duck-types `EnvDispatcher.run`; app calls it by keyword, so the stub's signature matches |
| 4 | ipSAE row regex skips the `Chn1` header row explicitly; the "no row" test uses text with no numeric row at all |
| 5 | the engine script is shipped in-package and invoked by absolute path `/app/scripts/engines/...`, which Task 7's `COPY scripts/` makes true. Same ordering as ruling 3. |
| 6 | **see rulings 1 and 2** |
| 7 | `test_every_case_declares_expected_result_keys` requires non-empty `expect_keys`; the plan also allows reducing the ipSAE case to a failure path — **see ruling 4** |
| 8 | the doc-reference regex would match a manifest's own name; the plan handles it with `mentioned == manifest.name` |
| 9 | pure deletion; the plan requires a grep proving nothing imports the package before deleting |

### Rulings

**Ruling 1 (Task 6): the `entry` for `run_mpnn` is unverified and the implementer must confirm it.**
The plan specifies `entry: ["python", "-m", "ligandmpnn.run"]`. I checked PyPI: `ligandmpnn` 0.1.2 is "a pip installable version of LigandMPNN with pre-trained models included" and hard-pins `torch==2.2.1` and `biopython==1.79` (which vindicates giving it its own environment), but the metadata does not expose the module or console-script name. The implementer must verify the real invocation against the installed package and correct the manifest's `entry` if it differs. The manifest is the single place that needs to change.
*Cost if wrong:* Task 7's live case for `run_mpnn` fails and reveals it there instead. Cheap, and caught inside this plan.

**Ruling 2 (Task 6): keep BOTH the declared FASTA output and the stdout parser; let Task 7 settle which is real.**
The manifest declares `outputs: [{name: designs_fasta, pattern: "seqs/*.fa"}]` while `parse_output` reads `run.stdout`. That is not necessarily contradictory — the engine may do both — but nothing in Task 6 proves it. Keeping both is right: the FASTA file is a genuine artifact worth returning to the caller regardless. If Task 7's live run shows the engine writes only to the file and does not echo to stdout, `parse_output` must be changed to read the collected file path from `run.outputs["designs_fasta"]`. I am recording this as an EXPECTED Task 7 discovery rather than a Task 6 defect.
*Cost if wrong:* if the engine writes neither, Task 2's `collect_outputs` raises `FileNotFoundError` naming `designs_fasta` and the workdir is preserved — a diagnosable failure, not a silent one.

**Ruling 3 (Tasks 5 and 6 precede Task 7): registering a tool whose environment does not exist yet is acceptable and must not be flagged as a defect.**
`run_openmm_minimize` declares `env: md` and `run_mpnn` declares `env: mpnn`; Task 7 is what creates both environments and copies `scripts/` into the image. Between those tasks the tools are registered but undispatchable. Nothing breaks: `ToolRegistry` does not validate environment existence, no unit test dispatches them, and the live proof that would catch it IS Task 7. Ordering them the other way would mean writing the Dockerfile against manifests that do not exist.
*Cost if wrong:* a reviewer flags it as a gap and I spend a round explaining. Recorded here so that does not happen.

**Ruling 4 (Task 7): a reduced ipSAE case still declares `expect_keys`.**
The plan permits reducing `run_ipsae`'s live case to its failure path if no valid PAE fixture can be produced without a co-folding tool (a plan 3 dependency), and separately requires every case to declare non-empty `expect_keys`. A failure-path case satisfies both by declaring `expect_keys: ["error"]` and asserting `isError` is True. The implementer must state plainly in its report that the success path is unproven until plan 3, rather than fabricating a passing result.
*Cost if wrong:* `run_ipsae` ships with its success path unexercised. Visible in the report and in plan 3's first task.

## Progress

### Task 1 review (spec ✅, quality: changes requested)

Reviewer probed the path-escape logic live against every case I named and found it
SOUND: `'..'`, `'*/../../etc/passwd'`, `'foo//../bar'` all rejected via
`Path(pattern).parts`; `'./foo.pdb'` correctly accepted; no input reaches a glob
outside the scratch directory. Also confirmed `OutputSpec` is itself frozen, that
validation order leaves earlier failures taking precedence, and that
`_validate_schema_entries` and empty-`schema: {}` acceptance are untouched.

Two Important findings in `_parse_timeout`, both confirmed live, both the exact
footguns the brief warned about:
  - `timeout_s: true` -> `int(True) == 1` -> silently a ONE-SECOND timeout
  - `timeout_s: 1.9` -> `int(1.9) == 1` -> silently truncated
These go to the fix loop.

Task 1: minor (deferred): an empty-string `pattern` is reported as "is missing
  required key 'pattern'" although the key WAS supplied (schema.py `_parse_outputs`).
  Confusing for a manifest author.
Task 1: minor (deferred): a bare `'.'` pattern is accepted because `Path('.').parts`
  is empty. Not an escape — it resolves to the scratch dir itself — but nonsensical
  as a declared output.

Task 1: fix round 1/5 (2 addressed, 0 open; commits 1cb929d..e556766)
  Strict `isinstance(data, int)` policy chosen; bool check sits BEFORE the int check
  (re-reviewer read the control flow, not just the outcome — there is no int()
  coercion at all now). Controller verified live: True/False -> "got bool",
  1.9 -> "got float", 120 -> accepted as 120. String "120" -> "got str".
Task 1: minor (deferred): the `timeout_s: false` test is vacuous — the pre-fix code
  also raised (with "must be positive"), so its ManifestError match passes for the
  wrong reason. The sibling `timeout_s: true` test IS non-vacuous and covers the same
  branch, so the coverage is real.
Task 1: complete (commits 38eb4b9..e556766, review clean, 28/28 schema tests;
  suite 41)

### Task 2 review (spec ✅, quality: changes requested)

Process-lifecycle regression check CLEAN: `collect_outputs` is called strictly after
the `try/except BaseException` block that owns killpg/cancellation has exited with
returncode 0, in its own try. Both lifecycle tests unmodified in the diff and still
passing. `results_dir()` reads the env at call time. `CompletedRun.outputs` defaults
correctly at all four pre-existing construction sites.

**Ruling 5 (Task 2, Important + the glob design question): namespace copies by spec
name, and make an ambiguous glob an ERROR unless the spec opts into `multiple`.**
Two things, one root cause — the collector assumes one spec means one file at a
predictable name.
(a) The reviewer found a real bug: `target = destination / source.name` uses only the
basename, so two specs matching the same basename in different subdirectories (e.g.
`designs/out.pdb` and `scores/out.pdb`) silently overwrite each other, and BOTH
returned paths then point at the second file's content. No test covers it because
every test uses one spec. Fix by namespacing the destination with the spec name.
(b) On the design question I put to the reviewer: silently taking `sorted(matches)[0]`
is not defensible in a mechanism whose whole purpose is to make outputs declared
rather than incidental. It is the same failure class as a missing file — the one this
task exists to catch — except it returns a plausible wrong answer instead of raising.
So: an ambiguous match raises, naming every match, unless the spec sets
`multiple: true`, in which case `collect_outputs` returns a list for that name and the
manifest author has stated intent explicitly.
This is not hypothetical. Task 6 declares `outputs: [{name: designs_fasta, pattern:
"seqs/*.fa"}]` for an inverse-folding engine that plausibly writes several files; it
must set `multiple: true`, and I will carry that into its dispatch.
*Cost if wrong:* one extra manifest field, and any engine whose glob is loose now
fails loudly at first run instead of returning an arbitrary file. That is the trade I
want — a noisy failure beats a quiet wrong answer, and the noise arrives during
implementation rather than in someone's results.

**Scope decision:** folding Minor "only `FileNotFoundError` is caught" into the same
round. `shutil.copy2` can raise other `OSError` subtypes (disk full, permission,
`IsADirectoryError` when a pattern matches a directory), and those currently escape as
raw exceptions instead of the `EngineError` that tells the caller the workdir was
preserved. It is the same code path the round is already editing.

Task 2: minor (deferred): `copy2` rather than `move`, though the source is deleted
  seconds later — wasted I/O, not a bug.
Task 2: minor (deferred): `copy2` follows symlinks; the pattern validation constrains
  the pattern string, not a symlink an engine places inside its own workdir. Low risk
  under today's trust model.
Task 2: minor (deferred): the results directory is never cleaned up. A real leak once
  28 engines run repeatedly; out of this task's scope, belongs with plan 3 or 4.

### Task 2 — CONTROLLER-CONFIRMED FINDING while the re-review was running

I probed the question I had put to the re-reviewer and answered it definitively:
**`OutputSpec.name` has NO parse-time validation, and ruling 5's fix promoted it to a
path component.** Measured:
  name='../evil' -> written to <results>/evil/ instead of <results>/run/ — escapes the
                    run_id isolation (stayed under the results root only by luck)
  name='/abs'    -> PermissionError writing to '/abs' — an ATTEMPTED escape to the
                    filesystem root, blocked only by permissions. `Path('/tmp/x') / '/abs'`
                    returns '/abs'; an absolute component RESETS the path. With a
                    writable absolute name this would have succeeded.
  name='a/b'     -> nested dirs created, stays inside
  parse_manifest({... 'outputs':[{'name':'../evil', ...}]}) -> ACCEPTED, no error.
Not remotely exploitable — a manifest is trusted repo content, not caller input — but
it is a new attack surface that ruling 5's own fix created, and 28 more manifests are
coming. `pattern` is validated against exactly this; `name` must be too.
Folding into Task 2's next fix round alongside whatever the re-review returns.

### Task 2 re-review of fix round 1 (findings 1-3 addressed; two NEW gaps)

Re-reviewer independently confirmed my `spec.name` traversal finding, and found a
second bug I MISSED: within a single `multiple=True` spec, two matches sharing a
basename across subdirectories still collide inside that spec's own directory. Both
copies land on the same path and the returned list holds two identical strings
pointing at the second file's content. My own probe used `a.fa`/`b.fa` — different
basenames — so it never hit this.

**Ruling 6 (Task 2, round 2): preserve each match's workdir-relative path inside the
spec directory; validate `OutputSpec.name` as strictly as `pattern`; and give the
ambiguity error its own type.**
Three items, two of them the same root cause — using only `source.name` throws away
the structure that makes a match unique.
(a) Intra-spec collision: copy to `spec_dest / source.relative_to(workdir)` rather
than `spec_dest / source.name`. A path relative to the workdir is unique by
construction, so collision becomes impossible rather than unlikely. This matters
concretely for `run_mpnn`, whose engine may write per-chain subdirectories with
repeating filenames — half the designs would vanish silently.
(b) `OutputSpec.name` validation: ruling 5's fix promoted `name` from a dict key to a
path component without giving it `pattern`'s checks. Measured by both me and the
re-reviewer: `../evil` escapes the run directory, and an absolute name replaces the
whole path because `Path('/tmp/x') / '/abs'` returns `/abs`. Validate at parse time.
(c) `AmbiguousOutputError(OSError)`: the re-reviewer would change the
`FileNotFoundError` reuse and I agree — "too many matched" is not "not found", it
misleads a reader and anyone writing `except FileNotFoundError`, and subclassing
`OSError` means `env.py`'s already-widened handler needs no change.
*Cost if wrong:* (a) result paths gain subdirectory structure, so a consumer reading
`outputs[name]` as a flat filename would need updating — nothing does today. (b) a
manifest using an exotic output name now fails at parse; that is the intent. (c) one
new exported name in a module 28 engines touch.

Task 2: fix round 1/5 (3 addressed; commits 0483627..9350848) — namespace by spec
  name, `multiple` flag, ambiguous-glob rejection, widened OSError.
Task 2: fix round 2/5 (3 addressed; commits 9350848..ec5a1f9) — intra-spec collision
  via source.relative_to(workdir), OUTPUT_NAME_RE allowlist, AmbiguousOutputError.
Task 2: fix round 3/5 (1 addressed; commits ec5a1f9..1bb847d) — resolve both sides
  before comparing; OutputPathEscapeError; second call site with the same exposure
  also fixed. Implementer honestly reported it could NOT reproduce a bare ValueError
  on this platform rather than shipping a test that cannot fail.

**CONTROLLER-CONFIRMED: round 3's escape guard covers only ONE of the two paths.**
The round-3 report claims "a genuine escape (symlink inside workdir pointing outside)
is refused via OutputPathEscapeError, not silently copied". I probed it and the claim
is true only for `multiple=True`:
    multiple=False (THE DEFAULT) -> symlink escape ALLOWED, file contents copied out
    multiple=True               -> refused with OutputPathEscapeError
Cause: `relative_to` was only ever needed on the multiple branch, so the guard landed
there. The single-valued branch still builds `spec_dest / source.name` with no check —
and that is the branch every simple engine uses.
This is the third consecutive round where a fix covered one path and missed its
neighbour: round 1 fixed inter-spec collision and missed intra-spec; round 3 fixed the
multiple branch and missed the default. Escalating per the skill — rounds 4-5 get a
fresh implementer one model tier up.

Task 2: fix round 4/5 (1 addressed; commits 1bb847d..a36e866) — fresh implementer on
  a higher tier, per the escalation rule. It RESTRUCTURED rather than patching the
  second branch: `_check_containment()` is now the only constructor of a frozen
  `_CheckedMatch` token, that token is the sole parameter type of
  `_copy_checked_match()`, and that function holds the module's only `shutil.copy2`
  call. `collect_outputs` became two passes — check every match, then copy — and
  `multiple` survives only as `copied if spec.multiple else copied[0]`. The duplicate
  branch was deleted, not patched, so a future branch cannot skip the check.
  CONTROLLER VERIFIED the full matrix directly:
    multiple=False symlink-escape -> OutputPathEscapeError
    multiple=False normal         -> copied
    multiple=True  symlink-escape -> OutputPathEscapeError
    multiple=True  normal         -> copied
  and confirmed by grep that `results.py:138` is the single real copy2 call (the other
  occurrence at :56 is the docstring stating the invariant). env.py untouched; both
  lifecycle tests pass.
Task 2: minor (deferred): the single-valued copy target moved from
  `spec_dest / source.name` to the workdir-relative path — the deliberate price of
  collapsing both branches onto one path. Basename is always preserved and nothing
  today reads a flat filename, but `designs/out.pdb` now lands at
  `<spec>/designs/out.pdb`.
Task 2: minor (deferred): an in-workdir symlink is still dereferenced into a real copy
  by copy2 — safe now that escapes are refused before any copy.
Task 2: minor (deferred): pre-existing ruff F401, unused `import os` in
  tests/test_results.py, inherited from the brief's verbatim file.
Task 2: complete (commits e556766..a36e866, 4 fix rounds, 73 tests in the three
  touched files; suite 41)

### Task 3 review (spec ✅, quality Approved; two plan-mandated items for me to rule on)

Reviewer confirmed: `_RecordingDispatcher`'s signature matches `EnvDispatcher.run`
exactly including the keyword-only `outputs` (not a loose superset stub, so a
signature regression would be caught); `manifest.outputs` reaches the single dispatcher
call site for every tool and `outputs=()` is behaviourally identical to omitting it;
error paths untouched; both new tests are non-vacuous (the timeout test asserts 45 vs
the old 3600 global; the outputs test would raise TypeError pre-fix).

**Ruling 7 (Task 3, item 1): the conditional `outputs` key stands — REJECTING the
reviewer's recommendation, because its premise does not hold here.**
The reviewer argued `if run.outputs:` keys off the run rather than the tool, so a
model would see the key appear and vanish per invocation, and recommended emitting the
key (even as `{}`) whenever `manifest.outputs` is non-empty. The general principle is
right and matches how we resolved the `default`/`example` ambiguity earlier. But Task
2's `collect_outputs` RAISES on zero matches for every spec, `multiple` or not. So a
successful run of a tool that declares outputs always yields one entry per spec, and
`run.outputs` is truthy exactly when `manifest.outputs` is non-empty. The two
conditions are equivalent in practice and the vanishing-key scenario cannot occur.
*Cost if wrong:* if a future change ever lets `collect_outputs` return empty on
success, the key would silently disappear. Mitigated by the fact that such a change
would have to delete the zero-match raise, which has its own tests.

**Ruling 8 (Task 3, item 2): guard the payload merge — an adapter returning `outputs`
must fail loudly, not be silently overwritten.**
`payload = {**parse_output(...), "outputs": run.outputs}` lets dispatcher data win
unconditionally. No adapter does this today (`prodigy.parse_output` returns four
unrelated keys), but 28 more adapters are coming and the collision is silent data
loss — the adapter's own field vanishes with no warning at any layer. Raise instead,
so the adapter author sees it the first time they run their own tool rather than
discovering a missing field in results later.
*Cost if wrong:* an adapter that legitimately wants a field called `outputs` must
rename it. That is the correct trade — the name is taken by the dispatcher contract.

Task 3: fix round 1/5 (1 addressed; commits d70f23b..dc89284) — collision check runs
  BEFORE the merge, raises ValueError caught by the generic adapter-failure wrapper so
  it surfaces as isError=True naming the tool and the reserved key.

**Ruling 9 (Task 3): the `outputs` collision check fires UNCONDITIONALLY — accepting
the implementation over the narrower rule I had floated.**
I had asked whether the check should fire only when the dispatcher actually has
outputs to merge, on the reasoning that a tool declaring none has no conflict. The
implementation reserves the key unconditionally and the re-reviewer judged that
correct. I agree, and the reason is worth recording: a conditional rule creates action
at a distance. An adapter could use an `outputs` field happily for months, and then
break the day someone adds an `outputs:` block to that tool's manifest — a change in
one file silently invalidating a field in another. Unconditional reservation is a
stable contract and collapses to one sentence for 28 future adapter authors:
"`outputs` is reserved."
*Cost if wrong:* a stdout-parsing adapter that wanted the name `outputs` must pick
another. Trivial, and the error says so at first run.

Task 3: complete (commits a36e866..dc89284, review clean, 17/17 wiring tests)
  CONTROLLER VERIFIED the suite directly: exactly 41 failures, sorted FAILED list
  byte-identical to baseline. (The re-reviewer wrote "~41-42"; the real number is 41.)

### Task 4 review (spec ✅, quality: changes requested — 1 Critical, 4 Important)

Reviewer confirmed clean: both file params carry `format: path`; `parse_output` raises
rather than returning None; `run_prodigy.yaml`'s edit is accurate with its other
sibling still correctly marked; docs regenerated and the pipe-bearing pattern column
renders escaped; multi-character chain IDs handled.

**Ruling 10 (Task 4, CRITICAL — my fixture was fabricated): parse ipSAE output by
COLUMN NAME, not position, and replace the fixture with the real table shape.**
I fetched the actual DunbrackLab/IPSAE source and confirmed the reviewer's premise.
The real header is:
  Chn1 Chn2  PAE Dist  Type   ipSAE  ipSAE_d0chn  ipSAE_d0dom  ipTM_af  ipTM_d0chn
  pDockQ  pDockQ2  LIS  n0res n0chn n0dom d0res d0chn d0dom nres1 nres2 dist1 dist2 Model
The plan's fixture was `Chn1 Chn2 ipSAE ipTM_af pDockQ LIS` — invented, not observed.
Correcting the reviewer on the symptom: it predicted silent mislabelling, but the
regex requires `[\d.]+` in the 3rd-5th positions and the real 5th column is `Type`
("asym"), so the match FAILS and the adapter raises "no chain-pair row". Loud, not
silent — but it means `run_ipsae` would never have worked against the real tool.
This is the second time a fabricated fixture has bitten: PRODIGY's was corrected from
`[+]` to `[++]`, where the regexes happened not to depend on it. Here the whole column
structure was wrong. Standing rule for plans 3 and 4: an adapter's fixture must come
from the engine's own source or a real run, never from recollection.
*Cost if wrong:* header-name parsing is slightly more code than a positional regex,
and it breaks if upstream renames a column — which is the correct failure, because a
renamed column means the number means something else.

Task 4: fix round 1/5 (5 addressed; commits 0bc6461..36fe5cb)
  Ruling 10 implemented: `_find_header` scans for the first line whose whitespace
  tokens contain all required column names EXACTLY, builds a name->index map, and
  reads only the needed columns. Fixture replaced with the real 24-column table.
  Re-reviewer probed robustness on every axis: header found structurally so leading
  blank lines and indentation are tolerated; no header at all raises naming the
  missing columns rather than an IndexError; a short row is skipped cleanly; unused
  non-numeric columns (PAE, Dist, Type) are never converted; exact-match lookup
  distinguishes `ipSAE` from `ipSAE_d0chn`/`ipSAE_d0dom`.
  CONTROLLER VERIFIED: rebuilt a row from the upstream format string and parsed it —
  chain_pair A_B, ipsae 0.7213, iptm_af 0.69, pdockq 0.5412, all correct.
Task 4: complete (commits dc89284..36fe5cb, review clean, 17/17; suite 41)
  `scoring` is now a two-member category, so the sibling-doc rule fires for the first
  time; both manifests carry the heading.

Pre-Task-5 check (applying ruling 10 BEFORE the mistake rather than after): verified
every OpenMM symbol the plan's engine script imports actually exists. openmm 8.6 in
the BindCraft env has app.PDBFile / ForceField / Modeller / Simulation / HBonds /
NoCutoff and openmm.LangevinMiddleIntegrator — all present. No fabricated API here.

### Task 5 review (spec ✅, quality Approved — no correctness bugs)

Reviewer confirmed: the output filename cannot be absolutized because it is injected
in `build_args` and never passes through `params`/`schema`; `parse_output` returns no
`outputs` key so the reserved-name guard is satisfied; the doc names both siblings
accurately and states it says nothing about whether two chains bind; hydrogen addition
IS disclosed under "What this is", so the atom-count change is on the record.
MY CONCERN WAS WRONG: I worried scientific notation could break the energy regexes.
`f"{value:.4f}"` never renders exponential regardless of magnitude, so `-?[\d.]+` is
safe. Recording it because a future reviewer should not re-raise it.

Task 5: minor (deferred): `minimized.pdb` is hardcoded in both the manifest's
  `outputs[0].pattern` and the adapter's `OUTPUT_NAME`. I had flagged this as a
  template defect worth fixing before 28 engines copy it; the reviewer supplied the
  nuance that changes my mind — deriving the filename from the manifest works ONLY
  for single-file non-glob outputs, and engines using `*.pdb` with `multiple: true`
  cannot hand a glob to a script as an output path. So the rule does not generalise,
  the benefit is one engine's duplicated literal, and divergence fails loudly via
  collect_outputs' zero-match raise. Deferring.
Task 5: minor (deferred): `_ITER_RE` returns None on no match while missing energies
  raise. Defensible because the script emits all four lines atomically, but the
  asymmetry is untested.
Task 5: CARRY TO TASK 7 (live run): `timeout_s: 1800` against a declared
  `max_iterations` ceiling of 10000. The script uses `nonbondedMethod=NoCutoff` with
  no periodic box, so nonbonded cost is O(N^2) per step and 10000 iterations on CPU
  for a protein-sized system could exceed 30 minutes. The default is 500 so this is a
  boundary question, not a live defect — confirm empirically when OpenMM actually runs.
Task 5: complete (commits 36fe5cb..5b2c923, review clean, 7/7; suite 41)

Task 6: complete (commits 5b2c923..b96154f, review clean, 10/10; suite 41)
  Ruling A applied (`multiple: true` on designs_fasta, pinned by a test).
  Ruling B RESOLVED with real evidence: the implementer downloaded the actual
  ligandmpnn==0.1.2 sdist and read entry_points.txt (three console scripts all ->
  ligandmpnn.run:main) plus run.py's `if __name__ == "__main__"` guard, which is what
  makes `python -m ligandmpnn.run` equivalent. The brief's entry was genuine; left
  unchanged.
  THIRD fabricated-fixture catch: the brief's NATIVE FASTA record carried an invented
  `overall_confidence=` that the real format does not emit. Behaviour was unaffected
  (detection is `id=`-presence only) but the fixture was corrected anyway — a false
  fixture teaches the next reader something untrue.
  CONTROLLER VERIFIED with records rebuilt in the real run.py header format: the input
  sequence is NOT returned as a design, num_designs is right, the design carries its
  id, and multiple is True. This is the same bug class as the `design_binder` defect
  that returned the target chain as its own binder.
  Reviewer also confirmed from source: the enum and CHECKPOINT_FOR keys match exactly
  so a validated value cannot KeyError; `cwd=str(workdir)` makes `--out_folder .` land
  in the scratch dir matching `seqs/*.fa`; `_records` handles CRLF and multi-line
  sequences.
Task 6: minor (deferred): `_records` yields an empty `sequence` for a header with no
  body rather than rejecting it. Unreachable from the real engine, which always writes
  a sequence line.
Task 6: minor (deferred): the doc states seed-based reproducibility only in the
  parameter description, not in the doc body a model skims.

PRE-TASK-7 NOTE — preflight ruling 2 is still open and Task 7 settles it.
The reviewer found run.py writes designs to `base_folder + "/seqs/" + name + ".fa"`.
Whether it ALSO echoes FASTA to stdout is still unverified, and `mpnn.parse_output`
currently reads `run.stdout`. If the live run shows it does not echo, parse_output
must read the collected file via `run.outputs["designs_fasta"]` instead.

### Task 7 live run — three unknowns settled, one CRITICAL defect exposed

SETTLED (preflight ruling 2): ligandmpnn NEVER echoes FASTA to stdout. Verified by
source and by two live runs. The adapter was wrong; Task 7 fixed it to read
`run.outputs["designs_fasta"]` (a list, since multiple: true) and rewrote its unit
tests. Live payload shows 2 real designs parsed.
SETTLED (Task 5 carry-forward): OpenMM at the 10000-iteration ceiling on a
maximally-clashing 500-residue system, forced CPU platform, took 180.9s — roughly 10x
headroom under timeout_s: 1800. The declared ceiling is safe.
Incidental corrections the implementer made and reported: pdbfixer==1.11 does not
exist on PyPI (1.12.0 does) so it came from conda-forge; ligandmpnn's torch==2.2.1
pin dragged ~2.8GB of CUDA runtime until CPU-only torch was installed first (final
image 3.63GB, no CUDA); ligandmpnn needs setuptools<81; the mpnn env needed an
LD_LIBRARY_PATH activate.d hook for a libstdc++/ICU ABI mismatch; and the brief's own
coverage test would have failed as written because `describe_tool` is not in
`build_registry().tools()`.

**Ruling 11 (CRITICAL — `run_ipsae` cannot work as built; Task 4's adapter parses a
stream the engine never writes to).**
The Task 7 implementer reported that ipsae==1.0.1 writes three files and prints
nothing. I verified it against upstream source myself: the table is emitted with
`OUT.write("\nChn1 Chn2  PAE Dist ...")` into `path_stem + ".txt"`, never `print()`.
So Task 4's stdout parser — including the header-name rewrite I ordered in ruling 10 —
was applied to the wrong stream. `run_ipsae` has never been able to work.
It is worse than swapping the source. ipSAE writes its outputs NEXT TO THE INPUT
STRUCTURE, and `format: path` absolutizes that input, so the files land outside the
scratch directory where a relative `outputs:` pattern cannot see them and where
`collect_outputs`' containment check would refuse them anyway.
An adapter cannot fix this alone: `build_args(manifest, params)` never receives the
workdir, which `dispatcher.run` creates afterwards. This is exactly spec §5.2 step 1
— "write inputs to a scratch directory" — which plan 1's final review recorded as
unimplemented and which was harmless only while every engine read stdout.
Remedy: a declarative manifest field naming the path parameters the dispatcher must
stage into the workdir before running, so an engine that writes beside its input
writes inside the scratch directory. Sending it to the Task 7 implementer, which has
the built image and can verify live.
*Cost if wrong:* a new manifest field that 28 engines may mostly not need. The
alternative — per-adapter staging — would require reshaping the adapter interface to
hand every adapter a workdir, which is a larger change for the same benefit.

Task 7: fix round 1/5 (ruling 11; commits f2df037..3ffdbc9) — declarative
  `EngineSpec.stage` staging; run_ipsae reads its results file; live success path
  reached.
Task 7: fix round 2/5 (3 addressed; commits 3ffdbc9..02e119d) — staging-span workdir
  contract now PRESERVE, matching run()'s wording; setuptools pinned ==79.0.1 (was a
  `<81` range, violating the exact-pin constraint); run_ipsae doc names results_txt.
  The implementer caught its own regression along the way (a doc rewording line-broke
  a phrase a test asserted on, 42 -> fixed -> 41) by re-running the suite.
  CONTROLLER VERIFIED: a staging failure returns isError True with
  "could not stage input(s) for run_ipsae: ... Working directory preserved for
  diagnosis: /tmp/pdmcp-4101", and the workdir count rose by exactly one.
  Also re-ran the container myself: all five live cases isError False with real values
  (prodigy -11.2, openmm energy change -1248779.7, mpnn 2 designs, ipsae 0.056897),
  ALL CHECKS PASSED.
Task 7: minor (deferred): no parse-time cross-check between a `stage` name and an
  `outputs` spec name, or a directory the engine creates at the workdir root.
Task 7: minor (deferred): output paths in the payload expose the param-name staging
  subdirectory.
Task 7: minor (deferred): the without-`stage` test does not independently assert
  `run()` received `workdir=None`.
Task 7: ACCEPTED FOR PLAN 3: for multi-gigabyte staged inputs prefer a HARDLINK with
  copy fallback over a symlink — a hardlink resolves as an ordinary path inside the
  workdir and cannot trip `OutputPathEscapeError`, whereas a symlinked stage input
  could if an output glob ever matched it.
Task 7: complete (commits b96154f..02e119d, 2 fix rounds; 436 host tests pass,
  suite 41; four engines proven live in one image)

Task 8: fix round 1/5 (4 items; commits 1f9bf9c..9a55956) — summary+doc scanning,
  per-paragraph scope, fenced-code exemption, ServerApp.list_tools() assertion.
Task 8: fix round 2/5 (1 item; commits 9a55956..9785cb5) — CONTROLLER CAUGHT that
  FIX 1 had not actually taken effect. My repro injected an unmarked `run_ghost_tool`
  into run_prodigy's SUMMARY, confirmed via yaml.safe_load that it landed there, and
  load_manifests still ACCEPTED it. The implementer had added markers to the real
  manifest and believed the check was live.
  Root cause was worse than my hypothesis: the marker lookup was PARAGRAPH-WIDE
  (`marker in paragraph.lower()`), so one marked mention cleared every other mention
  in the same paragraph. The rule was loose in general, not just on summary. Fixed by
  associating a marker with its specific mention.
  CONTROLLER VERIFIED all four cases after the fix: baseline loads; an unmarked
  summary reference is rejected naming the field and the tool; a marked summary
  reference loads; and a paragraph mixing a marked and an unmarked mention rejects
  only the unmarked one.
Task 8: minor (deferred): the error message does not distinguish "wrong marker
  format" from "unknown tool".
Task 8: minor (deferred): `describe_tool` and `get_job_status` are invisible to the
  `run_*` mention regex.
Task 8: complete (commits 02e119d..9785cb5, 2 fix rounds, 442 host tests; suite 41)

Task 9: BLOCKED by implementer — correctly. Its safety grep found two files outside
  the plan's 5-file deletion list importing from tools/: tests/test_analyze.py
  (protein_design_mcp.tools.analyze) and tests/test_job_queue.py::TestGetDesignStatus
  (protein_design_mcp.tools.status). 32 passing tests across the two. It declined to
  make the call itself because it changes stated plan scope. That was the right call:
  the brief forbade working around a hit.
Task 9: CONTROLLER CONFIRMED the blocker and found the defect is MINE. The carry-
  forward doc (docs/superpowers/specs/2026-09-21-plan-2-carry-forward.md lines 32-36)
  listed 5 test files to delete with tools/ and never scanned the rest of tests/ for
  importers. Measured per-file failure counts to replace the doc's arithmetic:
  hotspots 17, optimize 7, proteinmpnn 6, rfdiffusion 4, pdb_utils 2, sasa 1, esmfold
  1, design_binder 1, analyze 1, alphafold2 1 = 41. The doc's "25 of 41" for the five
  files was CORRECT (17+7+1+0+0); the error was the dependency scan, not the counts.
Ruling 12 (Task 9): SPLIT the deletion — the two blockers are opposite in kind.
  (a) tools/analyze.py + tests/test_analyze.py are DELETED with the rest.
      analyze_interface is composite glue: 170 lines of hand-rolled distance-cutoff
      heuristics (_count_hydrogen_bonds, _count_salt_bridges,
      _count_hydrophobic_contacts) over utils/pdb.py. Spec §8 designates its
      replacements as run_rosetta_interface + run_prodigy, and run_prodigy already
      ships, computing binding affinity from a real contact model. Keeping both is the
      C7 "two code paths agreeing only by coincidence" failure, and the model cannot
      reach analyze_interface anyway — it is unregistered. Deleting it serves the
      user's directive #3 (composite tools unusable) rather than regressing coverage.
  (b) tools/status.py is RELOCATED to src/protein_design_mcp/job_status.py, NOT
      deleted. get_design_status is a pure job-queue query over utils/job_queue.py —
      not orchestration — and it is the implementation the planned get_job_status
      meta-tool (spec §3.2; carry-forward deferred list) will adopt. Its 6 tests in
      TestGetDesignStatus move with it: 3 import sites and 3 patch targets change
      from protein_design_mcp.tools.status to protein_design_mcp.job_status.
      tests/test_job_queue.py contributes 0 of the 41 known failures and stays.
  Net: tools/ is removed ENTIRELY, so the eagerly-importing __init__.py hazard is
  gone — the actual point of the task. 6 test files delete (the plan's 5 + analyze),
  carrying 26 of the 41 failures. REVISED EXPECTED BASELINE: 15 failures, not the
  16 the plan states. Zero currently-passing tests may break.
  Cost if wrong: analyze.py's geometric interface summary is recoverable from git at
  7a45f13:src/protein_design_mcp/tools/analyze.py. Cheap to undo.
Ruling 12 note: _estimate_time_remaining in status.py hardcodes rfdiffusion/
  proteinmpnn/esmfold step timings from the OLD composite pipeline. It is stale for
  the new architecture but is covered by passing tests; left as-is and flagged for
  whoever implements get_job_status. Do NOT rewrite it in this task.
Task 9: re-dispatched with task-9-brief-revised.md carrying Ruling 12. Implementer
  returned DONE, commit df08252 (base 9785cb5), 15 failed / 406 passed.
Task 9: CONTROLLER VERIFIED INDEPENDENTLY, not taken on report. (a) tools/ has 0
  git-tracked files; (b) `diff <(git show 9785cb5:src/.../tools/status.py)
  src/protein_design_mcp/job_status.py` is empty — byte-identical, and git records it
  as a 100% rename with 0 changed lines, so the no-refactor constraint held;
  (c) the reference grep over src/ tests/ pyproject.toml returns 0 hits;
  (d) TestGetDesignStatus 6/6 pass after the move; (e) full suite FAILED list is
  proteinmpnn 6, rfdiffusion 4, pdb_utils 2, sasa 1, esmfold 1, alphafold2 1 = 15,
  matching Ruling 12's revised prediction file-for-file. Zero regressions.
Task 9: CONTROLLER FOUND AND FIXED a residue the implementer's clean grep masked.
  `src/protein_design_mcp/tools/` still existed on disk holding only a git-ignored
  __pycache__. Under PEP 420 a bare directory on the path is a NAMESPACE PACKAGE, so
  `import protein_design_mcp.tools` still SUCCEEDED (verified: it returned
  _NamespacePath([...tools])). The commit was correct — git tracked nothing there —
  but the working tree masked the removal, and any missed reference would have
  imported cleanly for a local reviewer while failing on a fresh clone. Removed the
  directory (git-ignored scratch only, working tree still clean); the import now
  raises ModuleNotFoundError. Worth remembering: a clean `git ls-files` on a deleted
  package does not prove the package is unimportable.
Task 9: complete (commit df08252, 0 fix rounds, 406 host tests; suite baseline now 15)
PLAN 2 STATUS: all 9 tasks complete. Next: whole-branch final review on the most
  capable model, then the finishing-a-development-branch menu. Pending controller
  edit staged outside the repo at scratchpad/carry-forward-correction.md — the
  carry-forward spec still names the 5-file deletion list and the stale "25 of 41",
  and still points at tools/hotspots.py for salvage. Apply after the final review so
  the reviewer sees the unedited state (a dangling-reference catch there is a useful
  signal on review quality).
CONTROLLER SMOKE TEST after Task 9 (before the final review returned). Booted
  ServerApp(build_registry(device='cpu')) on the host and probed the three paths a
  model can actually reach. Two probes of my own were WRONG before they were right —
  recorded because the corrections are the point:
  (i) I first used `app.registry`; the attribute is `_registry`, so every composite
      came back "AttributeError" and looked blocked when my probe was simply broken.
  (ii) My leak detector flagged describe_tool as leaking composite names. It was
      matching the error message ECHOING THE CALLER'S OWN INPUT ("unknown tool:
      'design_binder'"), not leaked metadata. No leak exists.
  Do not trust a blocking result that comes back as the WRONG exception type.
DIRECTIVE #3 VERIFIED (composites unreachable), all three paths, device=cpu:
  registry.resolve() raises ToolNotAvailable("unknown tool: ...") for design_binder,
  suggest_hotspots, analyze_interface, optimize_binder, validate_design;
  call_tool() returns isError=True with the same message; describe_tool(name=...)
  returns no metadata for any of them. run_prodigy resolves, so the probe discriminates.
  list_tools() = 5: describe_tool, run_ipsae, run_mpnn, run_openmm_minimize, run_prodigy.
FINDING A (CONFIRMED, controller, directive #2): `describe_tool` VIOLATES THE isError
  CONTRACT. call_tool('run_nonexistent') returns isError=True. But
  describe_tool(name='run_nonexistent') returns **isError=False** carrying
  {"error": "unknown tool: ..."} in the payload. Same for every describe_tool failure
  mode, including the empty-category case. A model branches on the protocol-level
  isError flag; here a failure is indistinguishable from success at that level, and the
  error object can be propagated as if it were tool metadata. This is the meta-tool
  whose entire job is teaching a model to call the others correctly, so the cost lands
  exactly where it hurts. Not a correctness blocker for the engines, but it should be
  fixed before 25 more adapters make describe_tool the primary discovery surface.
FINDING B (CONFIRMED, controller, directive #2): 4 of the 6 categories the `category`
  enum ADVERTISES return an error. Enum = generation, monomer_generation,
  sequence_design, cofolding, scoring, meta. Only sequence_design and scoring return
  tools; generation, monomer_generation, cofolding and meta return
  {"error": "no available tools in category ..."} (with isError=False, per Finding A).
  The carry-forward recorded only the `meta` case and framed it as cosmetic. It is 4x
  wider than recorded. Three of the four are TRANSIENT — plans 3/4 populate generation,
  monomer_generation and cofolding. `meta` is PERMANENT: describe_tool is itself the
  only meta tool and it excludes ITSELF from its own category listing, so a model
  asking what meta tools exist is told there are none. Supersedes the carry-forward's
  narrower note.
FINAL WHOLE-BRANCH REVIEW (opus, 38eb4b9..82bdef0, 21 commits / 60 files / 313KB):
  verdict MERGE, 0 blockers, 17 non-blockers. Written to final-review-plan-2.md.
  It probed rather than read on the three areas I pointed it at hardest, and all three
  held: containment (both `multiple` modes through one return; the plan-2 symlink
  asymmetry is gone), process lifecycle (pgid==pid, grandchild sentinel, CancelledError
  propagates), orchestrator blocking. It also confirmed C11 is genuinely superseded and
  reported C9 as a CHECKED NEGATIVE (no new HTTP-reachable read primitive; staging is
  fenced by the `\.(pdb|cif)$` patterns) — C9 itself remains open.
CONTROLLER RE-VERIFIED the review's three most consequential claims rather than taking
  them. All three CONFIRMED:
  (1) One malformed manifest removes EVERY tool. I injected an unmarked run_ghost_tool
      reference into run_mpnn.yaml's summary: list_tools collapsed to ['describe_tool']
      and the unrelated, valid, live-proven run_prodigy returned "unknown tool". Loud in
      the log, invisible to the model. Manifest restored; tree clean; 5 tools back.
  (2) pdbfixer=1.11 is installed at Dockerfile.envs:86 and asserted to import at :87,
      but scripts/engines/openmm_minimize.py never imports it — Dockerfile.envs:83
      already admits this in a comment. Any HETATM/water/missing-atom PDB fails while
      the doc promises only "a PDB file".
  (3) run_prodigy.yaml is the ONLY one of the four omitting timeout_s. The other three
      declare it. The most-copied manifest teaches omitting it.
REVIEW GAP WORTH RECORDING: the final review did NOT surface Finding A (describe_tool
  returns isError=False carrying an error payload) or Finding B (4 of 6 advertised
  category values error, with `meta` permanently self-excluding). My review prompt
  asked explicitly about model usability as directive #2, and it found the pdbfixer
  doc mismatch under that heading but not the meta-tool's own contract violation. A
  reviewer pointed at "can a model use this" looked at the DOCS and not at the
  RESPONSE ENVELOPE. Worth aiming a future review prompt at the envelope explicitly.
HANDOFF: docs/superpowers/specs/2026-09-22-plan-3-carry-forward.md written — G1-G4
  gate items, M1-M2 model-facing defects, lower-priority items, checked negatives, and
  the wrong-exception-type trap. Plan 2 is COMPLETE.
