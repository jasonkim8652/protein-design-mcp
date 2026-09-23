# GPU engine substrate — design

Written 2026-09-22. Covers how GPU engines are dispatched, mounted and pinned. The
engine inventory (which tools ship, with what parameters) is a separate document; this
one fixes the substrate they all sit on.

Predecessor: `2026-09-21-atomistic-tool-refresh-design.md` (§5 environment) and
`2026-09-22-plan-3-carry-forward.md` (gate items G1–G4, model-facing defects M1–M2).

## 1. What changed from plan 2

Plan 2's substrate assumed one Docker image containing every engine's environment,
built by micromamba from `Dockerfile.envs`. That holds for the four CPU engines and
does not generalise:

- The GPU engines already exist as ~40 conda environments under
  `/home/jk661/.conda/envs`, several as editable installs against working checkouts.
- AlphaFold 3 needs ~2.9TB of local assets under `/opt/alphafold3_data` and ships its
  own image and run recipe.

Rebuilding either inside an image is not viable. The user chose to keep the container
boundary and **mount the host environments into it**.

## 2. The four facts this design is built on

Each was established by running it in a container on this host, not inferred.

### 2.1 Pin the GPU at the container boundary

Docker 29.6.2 exposes CDI devices `nvidia.com/gpu=0..7`.

```
docker run --device nvidia.com/gpu=7 ...   →   torch.cuda.device_count() == 1
```

The container sees exactly one GPU — an L40S with 47.7GB. **No `CUDA_VISIBLE_DEVICES`
plumbing is needed inside the container, and no engine can reach another GPU even if it
sets the variable itself.** This makes the user's "GPU 7 only" constraint structural
rather than a convention every wrapper script has to remember, and it is the reason
this design does not add a per-engine device field.

### 2.2 Conda environments must be mounted at their host path

Console scripts carry absolute shebangs:

```
$ head -1 ~/.conda/envs/boltz/bin/boltz
#!/home/jk661/.conda/envs/boltz/bin/python3.11
```

So `-v /home/jk661/.conda/envs/X:/home/jk661/.conda/envs/X:ro` is mandatory and
`-v ...:/envs/X:ro` breaks every console script in the environment. Read-only is
sufficient for import and execution.

### 2.3 An editable install needs its source checkout mounted too

This is the trap that cost the most time, and it fails in a way that looks like a
missing package rather than a missing mount. With only the env mounted:

```
/home/jk661/.conda/envs/boltz/bin/python -c "import boltz"
→ ModuleNotFoundError: No module named 'boltz'
```

although the interpreter runs and `import torch` succeeds. The cause is
`site-packages/__editable__.boltz-2.2.1.pth` pointing at
`/home/jk661/projects/lightning-boltz-dev/src`. Mounting that path as well makes the
import succeed.

Half of the surveyed environments are affected, and two more resolve their module
outside the env entirely:

| env | module resolves to | extra mount required |
|---|---|---|
| `boltz` | `~/projects/lightning-boltz-dev/src/boltz` | yes — editable |
| `pp1c` | `~/projects/protpardelle-1c/src/protpardelle` | yes — editable |
| `genie2`, `frameflow`, `multiflow` | editable, `.pth` uses a finder hook | yes — resolve per engine |
| `esmfold2` | `~/.local/lib/python3.12/site-packages/esm` | yes — **user-site**, not the env |
| `esmfold2` (user-site) | — | also a leak risk: user-site is shared by every python 3.12 env on this host |

**Requirement:** the mount set for an engine is discovered, not hand-written. A helper
resolves an engine's module origin and its `__editable__*.pth` targets and emits the
mount list, so a new engine cannot ship with a silently incomplete mount set.

### 2.4 `micromamba run -n` cannot reach a mounted environment

`MAMBA_ENVS_DIRS` and `CONDA_ENVS_DIRS` make the environment appear in
`micromamba env list`, but `run -n` still resolves only under the root prefix:

```
micromamba run -n boltz ...
→ critical libmamba The given prefix does not exist: "/opt/conda/envs/boltz"
```

Prefix form works: `micromamba run -p /home/jk661/.conda/envs/boltz python ...`, as does
invoking the environment's absolute python directly.

## 3. Schema changes

`EngineSpec` today is `repo, env, entry, stage`, and `EnvDispatcher` holds one
process-wide `runner`. Two engines cannot currently use different resolution strategies.

### 3.1 `EngineSpec.prefix`

A manifest names either `env` (a name resolved under the image's root prefix, as today)
or `prefix` (an absolute path to a mounted host environment). Declaring both is a load
error. A manifest naming `prefix` dispatches as
`micromamba run -p <prefix> <entry...>`; `env` keeps `micromamba run -n <env> <entry...>`.

This keeps the four CPU manifests untouched and requires no per-manifest runner field —
the two forms are distinguished by which key is present, not by a mode flag.

### 3.2 `EngineSpec.mounts`

The read-only host paths this engine needs beyond its prefix — editable source
checkouts, user-site directories, weight directories. Each is mounted at its own path.
Validated at load: absolute, existing, no `..`.

`mounts` declares intent; §2.3's helper generates the list and a test asserts the
declared set still matches what the environment actually resolves to, so an engine
reinstalled as non-editable (or newly editable) fails loudly rather than at call time.

### 3.3 Environment variables for the child process

`dispatch/env.py` calls `create_subprocess_exec` without `env=`, so the child inherits
the server's environment wholesale. Engines need per-engine variables — cache
directories above all, since several will otherwise write into a read-only mount or
into a shared `~/.cache`. Add an `env_vars` mapping on `EngineSpec`, merged over a copy
of `os.environ`, and point each engine's caches (`HF_HOME`, `TORCH_HOME`,
`XDG_CACHE_HOME`) into its scratch workdir by default.

This is deliberately NOT how the GPU is pinned — see §2.1.

## 4. The AF3 pattern, applied to every GPU engine

The user asked that AlphaFold 3's Docker method be the model. The ground truth is their
own checkout at `~/projects/af3-mmseqs-gpu`
(`benchmarks/run_inference_original_db.sh`), not the upstream README:

```
docker run --rm --gpus "device=$GPU" \
  -v "$DB_DIR:/db:ro" -v "$MODEL_DIR:/models:ro" \
  -v "$INPUT_JSON:/input.json:ro" -v "$TARGET_OUTPUT:/output" \
  "$IMAGE" python /run_alphafold.py \
    --json_path=/input.json --output_dir=/output --db_dir=/db --model_dir=/models
```

Four elements carry over to every GPU engine, not just AF3:

1. **One GPU named explicitly** — §2.1 does this with CDI.
2. **Read-only mounts at canonical paths for weights and databases.** Assets stay on
   the host; nothing large is ever baked into an image.
3. **A JSON job specification** rather than a long argv. This maps onto the manifest's
   parameter schema directly — the adapter serialises validated parameters to JSON and
   the wrapper script reads it.
4. **An explicit `--output_dir`.** This is the most valuable of the four for us: it
   removes by construction the "engine writes next to its input" problem that forced
   the staging mechanism for ipSAE. **Every GPU wrapper script must take an explicit
   output directory and write only there**, so `stage` stays unused for this class.

AF3 itself keeps running as its own container from its own image. Our server therefore
needs to launch a sibling container for that one engine; that is deferred to the AF3
task and is not a dependency of the other engines.

AF3 assets, to be used in place (the user asked for the local paths, not a fetch):
`/opt/alphafold3_data` — `mmseqs_db` 1.3T, `official_v3_fresh` 1.1T, `fasta_databases`
395G, `af3_databases` 98G, `weights` 2.1G.

## 5. Gate items folded in

Plan 3's carry-forward lists four gates. Two are prerequisites for this substrate and
must land before the first GPU manifest:

- **G1 — one malformed manifest removes every tool.** Reproduced: a bad reference in
  `run_mpnn.yaml` collapses `list_tools` to `['describe_tool']` and makes the unrelated
  `run_prodigy` "unknown tool". This plan multiplies the manifest count, so the loader
  must route a malformed manifest into `ToolRegistry`'s existing per-tool exclusion
  path instead of aborting the load.
- **G2 — `run.outputs` is discarded when `parse_output` raises.** GPU engines run for
  tens of minutes; losing collected paths to a parser error is expensive here in a way
  it was not for millisecond CPU engines.

G3 (pdbfixer unused) and G4 (`run_prodigy` missing `timeout_s`) are CPU-side and can
ride along, but they do not block.

M1 (`describe_tool` returns `isError=False` carrying an error) should land early: this
plan is where the tool count grows enough for `describe_tool` to become the model's
primary discovery surface.

## 6. Licence and redistribution constraints

Unchanged and binding: PyRosetta may not be redistributed; AlphaFold 3 weights are
non-commercial and may not be redistributed. Both are mounted from the host at runtime
and must never be copied into an image layer. `boltz` resolves to the user's own fork
at `~/projects/lightning-boltz-dev`, so the tool runs their modified code rather than
upstream — this should be stated in the tool's documentation so results are
attributable.

## 7. What this design deliberately does not do

- No per-engine GPU selection field. §2.1 settles the device at the container boundary;
  adding a field would create a second, weaker control that could disagree with it.
- No attempt to make conda environments relocatable. Same-path mounting is simpler and
  was proven; relocation is fragile for compiled extensions.
- No copying of engine environments into an image. The mount set is the contract.
