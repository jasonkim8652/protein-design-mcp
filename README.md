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

## Running it

v2 runs as a container whose engines live in **host environments mounted read-only**,
rather than one image with everything pip-installed into a single Python. The engines
here disagree about torch, CUDA and numpy in ways no single environment resolves.

### Quick start — no mounts

```bash
docker run -i --rm jasonkim8652/protein-design-mcp:2.2.0
```

This speaks MCP over stdio immediately and gives you **7 tools**: the five engines baked
into the image (`run_prodigy`, `run_ipsae`, `run_openmm_minimize`, `run_mpnn`,
`run_rosetta_interface`) plus the two meta tools. The other 34 are excluded, each with a
reason you can read — a missing mount degrades the registry, it never crashes the
server.

### Full surface — with mounts

The mount list is **derived from the manifests, not maintained by hand**:

```bash
python scripts/container_run.py            # prints the exact `docker run` invocation
PROTEIN_DESIGN_GPU=3 python scripts/container_run.py
```

On the development machine that is 54 read-only mounts. Three properties are enforced by
[`tests/test_container_run.py`](tests/test_container_run.py) rather than by convention:

- **Exactly one GPU**, via `--device=nvidia.com/gpu=N`. The container sees one device, so
  an engine that ignores `CUDA_VISIBLE_DEVICES` still cannot reach another index. Not
  `--gpus all`.
- **Every mount read-only, at its identical host path.** An environment must be mounted
  where it believes it lives; editable installs and compiled extensions hardcode
  absolute paths.
- **The server's own package is never mounted**, so the container runs the installed
  server rather than silently picking up a host checkout.

`/var/run/docker.sock` is deliberately **not** mounted. Nothing in this server needs it,
and combined with an exposed HTTP port it would be an unauthenticated path to root.

### MCP client config

```json
{
  "mcpServers": {
    "protein-design": {
      "command": "docker",
      "args": ["run", "-i", "--rm",
               "--device=nvidia.com/gpu=0",
               "jasonkim8652/protein-design-mcp:2.2.0"]
    }
  }
}
```

Add the `-v` flags from `scripts/container_run.py` for the full tool surface.

### Optional: AlphaFold 3

`run_alphafold3` needs an ~8 GB venv and its databases mounted. Without them that one
tool is excluded and the other 40 work normally. AlphaFold 3's weights are not
redistributable, so they are mounted, never baked into the image — the same is true of
PyRosetta.

---

## Building the image

```bash
docker build -f Dockerfile.envs -t protein-design-mcp:envs .
```

One micromamba environment per engine, plus a thin `server` environment that shares no
dependencies with any of them. The server is installed **non-editable** on purpose, so
the build proves `manifests/*.yaml` actually ship inside the distribution rather than
being reachable only through a symlink back to the checkout.

The image's verification harness stays available as an explicit override:

```bash
docker run --rm <mounts...> protein-design-mcp:envs \
  micromamba run -n server python scripts/live_proof.py
```

`Dockerfile`, `Dockerfile.full`, `Dockerfile.lite`, `Dockerfile.colabfold` and
`Dockerfile.patch` build the **v1** all-in-one images, and are kept for reproducing
`v1.0.0` only.

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
