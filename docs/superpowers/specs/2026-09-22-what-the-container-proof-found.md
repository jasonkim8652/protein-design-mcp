# What the in-container proof found, and why nothing else could have

Written 2026-09-22, at the close of the GPU engine substrate work. This records the
defects that only appeared when the server was finally run as it ships, and the
practices that surfaced them — so the next person does not have to rediscover either.

## The headline

41 tools passed on the host. **30 of 41 passed inside the image.**

Every one of the 11 differences was invisible to unit tests, to adapter tests, and to
the live verification each wave ran — because every one of those runs was a workaround.
Agents copied manifests to scratch and rewrote `entry:` paths at the host checkout,
because the container did not exist yet. That proves the adapter logic. It does not
prove the thing an operator deploys.

Two facts make the container the only honest place to check:

- The image-internal environments (`scoring`, `md`, `mpnn`, `mmseqs`) exist **only**
  inside the image. On the bare host those cases die with `critical libmamba The given
  prefix does not exist`.
- The ~20 host environments are reached by mounting them at their **identical host
  paths**, which only happens inside the container.

## The defects, by what made them invisible

| Class | Cases | Why no earlier test could see it |
|---|---|---|
| `$HOME` mismatch | 4 | The container's home is `/home/mambauser`; engines resolve checkpoints via `Path.home()`. The files were mounted **correctly** and looked for in the wrong place. |
| Runtime shared library | 2 | `libnvrtc.so.12`, `libXrender.so.1` — loaded by the dynamic linker, so `discover_mounts` (which walks Python `sys.path`) structurally cannot find them. |
| Read-only mount vs. write path | 1 | The engine `mkdir`s inside its own source checkout. Writable on the host, read-only by design in the image. |
| Host/container user | 1 | A mode-640 file the host user reads through group membership and the image's default user cannot. |
| Missing `CUDA_HOME` | 1 | Only matters when the engine JIT-compiles a kernel. |
| numpy version drift | 1 | Image numpy 2.4.6 versus the host dev env's 2.2.6 — 2.x tightened scalar casting. **Our code, not the environment.** |
| Sibling container | 1 | `run_alphafold3` — accepted, see Ruling 4. |

## Two defects the proof found in *us*, not the engines

**Our own package was being mounted.** Eight manifests declared a mount of
`/home/jk661/projects/protein-design-mcp/src` — a *different checkout of this server*,
because a stale editable install put it on several engine environments' `sys.path`.
`discover_mounts` reported it honestly; the conclusion was wrong. Mounting it would have
shadowed the container's own code with another working tree — a failure presenting as
"the container is running old code". Fixed structurally (commit `61b0bb2`): the helper
now refuses any path providing `protein_design_mcp`.

That fix then **broke two tools** that had pointed at a host environment precisely
because they import the server package. The right answer was already in the image: only
the `server` environment has it. Both now declare `env: server`. Afterwards the derived
mount list dropped the host dev-env entry entirely — proof the dependency was removed
rather than the symptom hidden.

**Every adapter hand-casts numpy.** None of the fourteen uses `to_jsonable`, which
exists for exactly this and encodes a load-bearing detail: its `int` branch precedes its
numpy branch, safe **only** because no numpy integer subclasses Python `int` — unlike
`np.float64`, which subclasses `float`. Fourteen hand-written casts will not reproduce
that consistently. `run_chai1` was not a one-off; it was the first payload to trip
numpy 2.x. The others differ in exposure, not in kind.

## Practices that actually caught things

**Derive the deployment shape from the manifests.** `scripts/container_run.py` generates
the `docker run` command from what is registered. Building it surfaced three defects
before the image was ever built — the self-mount, a missing in-image environment
(`mmseqs`, declared by a manifest and created by nothing), and a `foldseek` binary at a
path that exists on the host and not in the image. A hand-maintained list goes stale
*silently*: a missing mount appears as `ModuleNotFoundError` deep inside an engine, not
as a startup error.

**Validate cases statically, not only by running them.** The coverage test checked that
a case *exists*. `run_boltzgen_filter`'s case passed `design_dir` after that parameter
was removed, and nothing noticed until the full proof ran behind a GPU queue — the most
expensive possible moment. A schema check over every case finds both halves of that
drift in under three seconds
(`tests/test_live_proof_cases_validate.py`).

**Assert invariants, never snapshots.** Two snapshot tests shipped and both were
replaced: one asserting the adapters directory held exactly four named modules, one
asserting a hardcoded tool list. Each would have broken on every tool addition — roughly
thirty more times — while catching nothing. Derive the expected set at runtime and assert
the property: "every manifest has an adapter and every adapter has a manifest" is a
better test than either.

**Guard the guard.** A check that silently stops matching passes vacuously. Both the
in-image environment check and the case-schema check carry a second test asserting the
parser still sees anything at all. A green result proving nothing is worse than a red one.

## Verification discipline that paid for itself

Live verification found **11 upstream engine bugs** that reading source did not: Boltz's
`--subsample_msa` help contradicting its own default; OpenFold3 silently discarding an
MSA whose basename it does not recognise, then dying much later with an `IndexError`;
ESMFold2's `mean_plddt` on the 0–100 scale rather than 0–1; BoltzGen's `--config`
override bypassing its own `huggingface:repo:file` resolution; `Filter.write_outdir`
reading the pre-refold `.cif`; FrameFlow and MultiFlow's `length_subset` silently
overriding `min/max_length`; MultiFlow's recursive glob double-counting its own
self-consistency copy; La-Proteina's config filename having to match `--config_name`
exactly, and its checkpoint-embedded `DATA_PATH` crashing even with metrics off; Genie 3
writing the full complex per binder sample.

Not one of those is visible from the source alone.

## Where "cannot" turned out to be "looked under the wrong name"

Three tools were reported unrunnable and all three were wrong:

- AlphaFold 3's image was present as `romerolabduke/alphafast:latest` — the benchmark
  scripts take the image as an argument, so no tag was searchable anywhere.
- A working PyRosetta was in the `BindCraft` environment. The 1.8 GB wheel at
  `/opt/pyrosetta_wheels` has **zero `.so` files** and can never import.
- `rc-foundry` "did not exist on PyPI" because the query ran under python 3.10 and the
  package requires 3.12. A version constraint and an absent package produce the same
  error text.

The lesson is narrow and practical: before recording that something is unavailable,
check the name it actually ships under, and check with the right interpreter.
