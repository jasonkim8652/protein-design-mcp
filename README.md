# Protein Design MCP Server

[![Docker Hub](https://img.shields.io/docker/v/jasonkim8652/protein-design-mcp?label=docker%20hub&logo=docker)](https://hub.docker.com/r/jasonkim8652/protein-design-mcp)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

An [MCP](https://modelcontextprotocol.io) server that gives an LLM agent **41 atomistic
protein-design tools**: 39 `run_*` tools that each run exactly one engine, plus
`describe_tool` and `get_job_status`.

Every step is a separate call. Generation, sequence design, folding, alignment and
scoring are never bundled, so the agent chooses each one and can be asked to defend the
choice.

---

## v2 is a breaking change — read this first

**If you are looking for `design_binder`, `predict_complex`, `predict_structure`,
`analyze_interface` or `score_stability`, you want `v1.0.0`** (image
`jeonghyeonkim8652/protein-design-mcp:1.0.0`, still published and untouched).

v2 removes all five. They were composites: one call that ran several engines behind a
fixed pipeline. Two things were wrong with that.

1. **They hid the decisions.** `design_binder` picked the generator, the sequence
   designer and the structure predictor for you, so an agent asking for a binder got one
   opinion with no way to argue with it.
2. **`design_binder` was wrong.** It assumed a chain order that does not hold, and
   returned **the target chain as its own design**. The bug is invisible from the
   outside — the output is a valid PDB of a real protein — and it is why this rewrite
   exists.

The replacement is not a like-for-like tool. It is the same work, written out:

```
run_interface_residues | run_epitope_scan        ->  hotspots
  -> run_rfdiffusion3_binder | run_boltzgen_design | run_genie3_binder | ...
  -> run_mpnn | run_boltzgen_inverse_fold
  -> run_esmfold2 | run_boltz | run_chai1 | ...   (MSA stated, never inherited)
  -> run_ipsae | run_prodigy | run_rosetta_interface
```

---

## The tools

Classified by **function**, not by which engine they came from — so an agent picking a
binder generator sees seven interchangeable options rather than seven product names.

| Category | n | What it does |
|---|---|---|
| `binder_generation` | 7 | generate a binder against a target |
| `monomer_generation` | 6 | generate a monomer or scaffold |
| `sequence_design` | 2 | design a sequence for a fixed backbone |
| `structure_prediction` | 11 | predict a structure |
| `msa` | 2 | build an alignment |
| `scoring` | 4 | score an existing structure |
| `run_analysis` | 4 | operate on a finished run's outputs |
| `target_analysis` | 2 | find where to bind |
| `preparation` | 1 | modify a structure before scoring |
| `meta` | 2 | `describe_tool`, `get_job_status` |

Engines include RFdiffusion 1/2/3, Genie 2/3, FrameFlow, MultiFlow, La-Proteina,
Protpardelle-1c, Proteina-Complexa, BoltzGen, ProteinMPNN, ESMFold2, Boltz-2, Chai-1,
Protenix v1, OpenFold3, Promera, RoseTTAFold3, AlphaFold 3, AlphaFold2-Multimer,
MMseqs2, ColabFold, ipSAE, PRODIGY, PyRosetta and OpenMM.

**[docs/TOOLS.md](docs/TOOLS.md) is the full list**, with every tool, its engine, and
the design rules summarised below. Per-tool reference pages are in
[docs/tools/](docs/tools/), one generated from each manifest.

### Two rules worth knowing before you call anything

**MSA is always supplied, never generated inside a folding tool.** No folding tool
builds its own alignment; it comes from the `msa` category or not at all. The `msa`
parameter has **no default** — `null` means run MSA-free, a path means use that
alignment, and omitting it is rejected. There is no `"auto"`, because a folding tool
that searches its own databases makes two models incomparable (the difference between
their outputs confounds the model with the alignment), and because several engines
default to a *remote* MSA server, which is how a novel design silently leaves the
machine.

**`chains` is never inferred.** Whether a prediction runs with the target present or on
the binder alone is the caller's decision, and the same binder predicted alone and in
complex are different experiments.

### Parameters

400 parameters across the 39 tools, mean 10.3 per tool. **Every one has a description**
saying what it does, what changes when it moves, and a sensible range; 75% carry an
`enum`, `minimum`/`maximum` or `pattern`. No engine flag is fixed outside the schema —
what the manifests pin is plumbing only (`PYTHONPATH`, cache locations, `CUDA_HOME`),
never a scientific choice.

---

## Running the 2.4.0 integrated image

The image contains isolated engine environments, engine code, CUDA toolkit
components, and redistributable public model weights. Engine executables and
editable source mappings use fixed in-image paths; no developer home or host
conda environment is required.
The all-engine image is approximately **218 GiB uncompressed**, before run
outputs and Docker's additional storage requirements.

The packaged runtime versions are listed in
[`docs/integrated-environments.json`](docs/integrated-environments.json);
public asset locations and upstream license references are recorded in
[`docs/integrated-assets.json`](docs/integrated-assets.json).

```bash
mkdir -p "$PWD/workspace"
docker run -i --rm --device=nvidia.com/gpu=0 --shm-size=16g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e TMPDIR="$PWD/workspace" -v "$PWD/workspace:$PWD/workspace" \
  jasonkim8652/protein-design-mcp:2.4.0
```

This starts the MCP stdio server. Keep the input/output workspace mounted at
its identical absolute path so returned artifact paths are readable by the
client. Choose an allocated GPU and provide a compatible NVIDIA host driver
and Docker GPU support.

### External assets

Only external databases and user-obtained licensed materials need additional
read-only mounts. The host folders can have arbitrary names and locations.

| Container destination | Required content | Tool |
|---|---|---|
| `/data/databases/mmseqs` | Prepared MMseqs database prefixes, indexes and `.dbtype` files | `run_mmseqs_search` |
| `/data/databases/colabfold` | ColabFold local search DBs matching the requested `db1`/`db3` | Local mode of `run_colabfold_search` |
| `/data/databases/tinyprot` | `ccd.lmdb/data.mdb` and `taxonomy.lmdb/data.mdb` | `run_promera` |
| `/data/models/alphafold3` | User-obtained `af3.bin` | `run_alphafold3` |
| `/data/licenses/pyrosetta` | Licensed `pyrosetta/` and companion `rosetta/`, compatible with Python 3.10/Linux | `run_rosetta_interface` |

For example, add `--mount type=bind,source=/your/af3-weights,target=/data/models/alphafold3,readonly`
before the image name. An empty directory is not a completed database or
weight installation. All parent directories must allow traversal and files
must be readable by the invoking account. The image imports only the licensed
PyRosetta packages from its mount, preserving the bundled dependencies.

Missing external assets exclude dependent tools with a startup explanation.
ColabFold remote search remains available without its optional local DB;
local mode checks its selected database files before launching. AF2-Multimer
includes all five multimer_v3 parameter sets and needs none of these external
assets when passed `msa: null` or a supplied A3M.

ProteinMEM provides an asset-path JSON template, a config generator, and
`scripts/check_runtime_paths.py --require-all` to inspect paths and discover
all 41 tools before a campaign. Discovery verifies availability declarations;
actual inference and database searches require separate smoke tests.

OpenMM adds missing terminal atoms such as OXT before hydrogens, and reports
`added_terminal_atoms`. Its energies are force-field potential energies;
`E_complex - E_binder - E_target` is a computational proxy, not a measured
binding free energy.

For campaign archival, set `PROTEIN_MCP_KEEP_WORKDIR=1` and place `TMPDIR`
on a writable workspace mount. Calls then retain all engine intermediates and
full `engine.stdout.log` / `engine.stderr.log` files, including partial output
on failure or timeout. Tool responses expose their paths in
`execution_artifacts`; callers can copy them into a campaign archive. Without
this option, successful scratch directories are removed after declared outputs
are collected. Retained work directories consume additional disk space.

## Building the image

`Dockerfile.envs` builds the core environments. `Dockerfile.integrated` adds
curated and relocated engine environments, source and public weights:

```bash
docker build -f Dockerfile.envs -t protein-design-mcp:2.4.0-core .
# After staging and auditing all engine environments and public assets:
python scripts/assemble_integrated_payload.py \
  --rootfs /your/staged/rootfs --output /your/prepared-payload/rootfs.tar
docker build -f Dockerfile.integrated \
  --build-context payload=/your/prepared-payload \
  -t jasonkim8652/protein-design-mcp:2.4.0 .
```

The staging helpers `scripts/prepare_integrated_envs.py` and
`scripts/prepare_integrated_assets.py` accept explicit machine-local input
inventories. They copy installed runtimes without changing source files,
relocate prefixes and editable installs, and allowlist public assets.
Restricted weights, PyRosetta distributions, credentials, unrelated caches
and user databases must be excluded **before** creating the archive; deleting
them in a later Docker layer would still distribute them. Preserve component
licenses and `/opt/models/licenses/upstream/ASSET-MANIFEST.json` with the
payload. AF3's bundled, pinned source revision retains its own CC BY-NC-SA
license; the server's Apache license does not replace component licenses.

`Dockerfile`, `Dockerfile.full`, `Dockerfile.lite`, `Dockerfile.colabfold` and
`Dockerfile.patch` are historical v1 recipes, not the integrated release.

---

## Adding a tool

A tool is one YAML manifest. No Python.

```
src/protein_design_mcp/manifests/run_<name>.yaml
```

The manifest is the single source for the MCP `Tool` (name, summary, `inputSchema`), the
`describe_tool` response, the dispatch entry, and the generated page in `docs/tools/`.
After editing one:

```bash
python scripts/generate_tool_docs.py     # tests/test_doc_generation.py fails if you forget
pytest tests/ -q
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Tests that need a GPU engine's host environment skip when it is absent. The suite also
derives and checks deployment facts — mount completeness, the container command's shape,
doc freshness — because those are the defects unit tests cannot see.

## License

Apache 2.0 for this server (see [LICENSE](LICENSE)). **The engines it dispatches to
carry their own licenses**, several of which are non-commercial or require a separate
grant; AlphaFold 3 weights and PyRosetta are mounted from the host rather than
distributed here for exactly that reason. Check each engine's terms before use.

## References

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [docs/TOOLS.md](docs/TOOLS.md) — the full tool list and the rules behind it
- [docs/tools/](docs/tools/) — generated reference page per tool
