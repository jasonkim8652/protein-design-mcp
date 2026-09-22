# Live proof: run_prodigy dispatch contract, end-to-end, in Docker

Date: 2026-09-21/22. Repo: `/home/jk661/projects/protein-design-mcp-dev`, branch `dev`.
Docker 29.6.2, host has 2.3T free.

## 1. What was built: `Dockerfile.envs`

A new file, `Dockerfile.envs`, next to (not replacing) the existing
`Dockerfile`, `Dockerfile.full`, `Dockerfile.lite`, `Dockerfile.colabfold`,
`Dockerfile.patch`. None of those were touched (`git diff --stat` against
them is empty).

**Base image:** `mambaorg/micromamba:2.9.0-ubuntu22.04` — pinned to an exact
micromamba version (2.9.0) and OS (Ubuntu 22.04 / jammy), not `latest`.
Chosen because it puts the `micromamba` binary at `/usr/local/bin/micromamba`
unconditionally on `PATH`, which is the entire reason to pick this base:
`EnvDispatcher`'s default `runner` is the literal string `"micromamba"`
(`dispatch/env.py`), resolved via `PATH` at subprocess-exec time by whatever
process runs the server — so it must be on `PATH` for the *server* process,
not just reachable via some activation script.

**Two micromamba environments, matching spec §5.1 exactly:**

- `scoring` — `python=3.11`, `pip`, and `prodigy-prot` installed via `pip`
  inside that env. Nothing else. This is the literal string `run_prodigy.yaml`
  declares as `engine.env`.
- `server` — `python=3.11`, `pip`, and the `protein-design-mcp` package
  itself installed via a **normal, non-editable** `pip install .` (not `-e`).
  Deliberately non-editable: an editable install would trivially "find" the
  manifest directory because it symlinks straight back into the checkout,
  proving nothing about `[tool.setuptools.package-data]`. A real, non-editable
  install is the thing that actually exercises that packaging path.

These two environments share no dependencies — `scoring` never sees `mcp`,
`server` never sees `prodigy-prot`/`freesasa`/`biopython`'s prodigy-specific
usage. That separation is verified by construction (two separate
`micromamba create -n <env>` blocks, no cross-install).

One build-time wrinkle: `freesasa` (a `prodigy-prot` dependency) ships no
manylinux wheel on PyPI, only macOS/Windows wheels plus an sdist, so `pip`
builds it from source on Linux. That needs a C/C++ toolchain, so the image
installs `build-essential` via `apt-get` as `root` before creating the
`scoring` env. This is the only reason the image needs a compiler at all;
everything else is pure-Python/wheels. Build time for the whole image was
**~74 seconds** end to end (image size 1.16 GB), well within "reasonably
quick — this is a proof of the contract, not a production image."

The manifest directory is verified at build time, not assumed: the final
`RUN` layer in `Dockerfile.envs` runs
`python -c "from protein_design_mcp.app import manifest_dir; ... assert 'run_prodigy.yaml' in files"`
against the real installed package inside the `server` env, and the build
log shows it passing:
```
manifest_dir: /opt/conda/envs/server/lib/python3.11/site-packages/protein_design_mcp/manifests
manifests found: ['run_prodigy.yaml']
```
So `[tool.setuptools.package-data] protein_design_mcp = ["manifests/*.yaml"]`
does carry `src/protein_design_mcp/manifests/run_prodigy.yaml` into a real
wheel install — confirmed, not assumed.

## 2. Test fixture: why 1BRS, not the repo's `two_chain_complex.pdb`

The repo's `tests/fixtures/test_pdbs/two_chain_complex.pdb` is a minimal
5-residue-per-chain synthetic structure. I ran PRODIGY against it directly
(outside the server, just to check whether it would be rejected) before
building the image: it was **not** rejected — PRODIGY ran and returned a
result — but the result is degenerate: exactly **1** intermolecular contact,
everything else zero, `binding_affinity_kcal_per_mol = -5.0`. That is a real
PRODIGY execution, but not a meaningful demonstration that the tool's
regression over interfacial contacts is doing anything.

Per the task's own guidance, I fetched a real two-chain complex instead:
**PDB 1BRS** (barnase–barstar), via
`curl https://files.rcsb.org/download/1BRS.pdb`, saved to
`tests/fixtures/test_pdbs/1BRS.pdb` (457 KB, committed). 1BRS contains three
barnase/barstar copies (chains A/D, B/E, C/F); I used **chains A and D**.
This gives a real, non-degenerate interface: 68 intermolecular contacts,
predicted affinity −11.2 kcal/mol. PRODIGY also printed gap warnings for
missing residues (normal for a real crystal structure) that had no effect on
the outcome.

## 3. Exact build and run commands

```bash
cd /home/jk661/projects/protein-design-mcp-dev

# Build (~74s)
docker build -f Dockerfile.envs -t protein-design-mcp:envs-proto .

# Run the full acceptance proof (default CMD)
docker run --rm protein-design-mcp:envs-proto
```

The acceptance driver is `scripts/live_proof_prodigy.py` (new file, part of
this deliverable). It does **not** call `ServerApp.call_tool` directly — it
fetches the real handler `mcp.server.Server` registered for
`types.CallToolRequest` (the same handler a real `tools/call` JSON-RPC
request would reach) out of `pdmcp.server.request_handlers`, builds a real
`types.CallToolRequest`/`CallToolRequestParams`, and awaits the handler:

```python
handler = pdmcp.server.request_handlers[types.CallToolRequest]
request = types.CallToolRequest(
    method="tools/call",
    params=types.CallToolRequestParams(name=name, arguments=arguments),
)
result = await handler(request)
```

This traverses the full chain the task specified: `mcp.server.Server`'s
registered handler → `ServerApp.call_tool` → `validate_and_fill` →
`_resolve_path_params` → `ADAPTERS["run_prodigy"]` → `build_args` →
`EnvDispatcher.run` → `micromamba run -n scoring prodigy <args>` →
`parse_output`.

I first ran this same script against the **host** environment (no Docker) as
a sanity check that the driver itself was correct, and got exactly the
expected proof that the gap the task describes is real:
```
"error": "could not start engine 'prodigy' in environment 'scoring':
[Errno 2] No such file or directory: 'micromamba'. ..."
```
confirming `micromamba` is genuinely absent on the host, before then getting
a real result inside the container.

## 4. Verbatim PRODIGY stdout (captured via the adapter's `CompletedRun`, success case)

This is what `EnvDispatcher.run` captured as `stdout` for
`micromamba run -n scoring prodigy tests/fixtures/test_pdbs/1BRS.pdb --selection A D --temperature 25.0`
inside the container (confirmed byte-for-byte identical to a direct run
outside the server, done as a pre-check):

```
[!] Structure contains gaps:
	A VAL3 < Fragment 0 > A ARG110
	B ALA1 < Fragment 1 > B ARG110
	C VAL3 < Fragment 2 > C ARG110
	D LYS1 < Fragment 3 > D THR63
	D GLY66 < Fragment 4 > D SER89
	E LYS2 < Fragment 5 > E THR63
	E GLY66 < Fragment 6 > E SER89
	F LYS1 < Fragment 7 > F SER89

[+] Executing 1 task(s) in total
##########################################
[+] Processing structure 1BRS_model0
[+] No. of intermolecular contacts: 68
[+] No. of charged-charged contacts: 11.0
[+] No. of charged-polar contacts: 12.0
[+] No. of charged-apolar contacts: 25.0
[+] No. of polar-polar contacts: 2.0
[+] No. of apolar-polar contacts: 11.0
[+] No. of apolar-apolar contacts: 7.0
[+] Percentage of apolar NIS residues: 32.55
[+] Percentage of charged NIS residues: 31.15
[++] Predicted binding affinity (kcal.mol-1):    -11.2
[++] Predicted dissociation constant (M) at 25.0˚C:  6.0e-09
```

## 5. Parsed result the server returned (through the full stack above)

```json
{
  "binding_affinity_kcal_per_mol": -11.2,
  "dissociation_constant_M": 6e-09,
  "intermolecular_contacts": 68,
  "caveat": "PRODIGY is calibrated on natural complexes and systematically mis-ranks de novo designed binders. Use it as a sanity floor, not as a ranking criterion."
}
```
`isError: False`. `binding_affinity_kcal_per_mol` is a genuine Python
`float` (`-11.2`), produced by `_AFFINITY_RE` matching real PRODIGY stdout,
not a fixture.

## 6. The three regex/CLI questions

1. **Do `_AFFINITY_RE`, `_KD_RE`, `_CONTACTS_RE` match real PRODIGY output?**
   Yes, all three matched the real stdout above **unchanged** — no regex
   edits, no fixture edits were needed.
   - `_AFFINITY_RE` (`binding affinity \(kcal\.mol-1\):\s*(-?[\d.]+)`) matched
     `Predicted binding affinity (kcal.mol-1):    -11.2` → `-11.2`.
   - `_KD_RE` (`dissociation constant \(M\)[^:]*:\s*([\d.eE+-]+)`) matched
     `Predicted dissociation constant (M) at 25.0˚C:  6.0e-09` → `6.0e-09`
     (the `[^:]*` correctly swallows `" at 25.0˚C"`, including the non-ASCII
     `˚` character, before the real colon).
   - `_CONTACTS_RE` (`intermolecular contacts:\s*(\d+)`) matched
     `No. of intermolecular contacts: 68` → `68`.
   The existing unit-test fixture in `tests/test_adapter_prodigy.py`
   (`SAMPLE_STDOUT`) is consistent with this real output's line shapes and
   needed no changes; the host suite's `test_adapter_prodigy.py` (10 tests)
   still passes unmodified.

2. **Does `build_args`' `--selection A B` form work against the real CLI?**
   Yes, confirmed both via direct CLI (`prodigy 1BRS.pdb --selection A D
   --temperature 25.0`) and via the full server round-trip. `prodigy --help`
   documents it as `--selection A B [A,B C ...]`, i.e. two single-character
   positional tokens after the flag — exactly what `build_args` emits.

3. **Does `--temperature` exist and behave as assumed?**
   Yes. `prodigy --help` lists `--temperature TEMPERATURE  Temperature (C)
   for Kd prediction`, and the real run above used `--temperature 25.0`
   (the manifest's default) and produced a Kd computed at "25.0˚C" as printed
   in the dissociation-constant line, confirming the flag is actually
   consumed and drives the reported temperature.

## 7. Failure paths (through the full server stack, in-container)

**Nonexistent input path** (`complex_pdb=/no/such/file.pdb`):
```json
{"error": "engine 'prodigy' exited with code 1.\n\nWorking directory preserved for diagnosis: /tmp/pdmcp-f6ae6d44752e"}
```
`isError: True`. This is `EngineError`, not a traceback — the caller gets a
clean, structured error. One caveat worth flagging (see §8): PRODIGY prints
`File /no/such/file.pdb does not exist` to **stdout**, not stderr, and
`EnvDispatcher`'s error message only echoes `stderr` — so this particular
message is less informative than it could be (no traceback, but no root
cause text either). This is a pre-existing property of `EnvDispatcher`, not
something introduced by this task; I did not change it since the task asked
me to verify and report, not redesign error plumbing.

**Malformed chain ID** (`chain_a="AB"`, multi-character):
```json
{"error": "run_prodigy.chain_a = 'AB' does not match the required format ^[A-Za-z0-9]$. Example: 'A'."}
```
`isError: True`. This is `ToolInputError` from `validate_and_fill`, caught
before any subprocess is spawned — exactly as designed. Clear, actionable,
names the parameter and the constraint.

**Well-formed but nonexistent chain ID** (`chain_b="Z"`, valid single
character, but chain Z is not in 1BRS): our schema validation passes (a
single alphanumeric character is a syntactically valid chain ID), so this
one reaches the engine, which runs and exits 0 but never prints an affinity
line:
```json
{"error": "adapter for run_prodigy failed: PRODIGY produced no binding affinity line. Output was:\n[!] Structure contains gaps:\n\t...\nError processing model: No contacts found for selection"}
```
`isError: True`. This is the generic `except Exception` branch in
`ServerApp.call_tool` catching `parse_output`'s `ValueError`, and it is
genuinely informative — it includes PRODIGY's own
`Error processing model: No contacts found for selection` in the tail of
the message the caller sees.

## 8. What surprised me

- **PRODIGY exits 0 on some real errors.** A nonexistent selection chain
  produces `returncode == 0` with an error message embedded in stdout
  (`Error processing model: No contacts found for selection`), not a
  non-zero exit. `EnvDispatcher` only treats non-zero exit as failure, so
  this case is caught downstream by `parse_output`'s missing-affinity-line
  check rather than by `EngineError`. The system still produces a correct,
  informative error, but through the "no affinity match" path rather than
  the "engine exited nonzero" path — worth knowing if anyone later assumes
  all engine-level failures surface as `EngineError`.
- **PRODIGY's own I/O split is inconsistent.** Its `does not exist` message
  for a missing input file goes to stdout, while a real subprocess failure
  would more conventionally use stderr. Combined with `EnvDispatcher` only
  echoing `stderr` in its message, the nonexistent-path error the caller
  sees is technically correct (exit code 1, clearly flagged as an error) but
  doesn't carry PRODIGY's own diagnostic text. Not a blocker, just an
  observation for future polish.
- Everything else worked essentially as designed on the first real run: the
  regexes needed zero changes, `--selection A B` and `--temperature` behave
  exactly as `build_args` assumes, and the manifest packaging worked without
  needing `PROTEIN_MCP_MANIFEST_DIR` as an escape hatch.
- `freesasa` (a transitive `prodigy-prot` dependency) has no Linux wheel on
  PyPI, requiring a C++ toolchain in the image. This added `build-essential`
  to the image but did not meaningfully affect build time (~74s total).

## 9. Host test suite

Before touching anything: `41 failed, 335 passed, 2 skipped` (captured via
`/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/ -q`).
After adding `Dockerfile.envs`, `scripts/live_proof_prodigy.py`, and
`tests/fixtures/test_pdbs/1BRS.pdb`: identical — `41 failed, 335 passed, 2
skipped`, and the sorted `FAILED` test-ID list is byte-for-byte identical to
the pre-change baseline (`diff` confirms no difference). No adapter code,
no regex, and no existing fixture needed to change.
