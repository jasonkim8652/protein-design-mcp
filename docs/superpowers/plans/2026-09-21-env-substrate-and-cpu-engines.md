# Environment Substrate and CPU Engine Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the one-engine proof-of-concept image into the real multi-environment substrate, close the three template gaps the whole-branch review identified, and land three more CPU engines that each exercise a part of the dispatch contract PRODIGY did not.

**Architecture:** `Dockerfile.envs` already proves the pattern — one micromamba environment per engine inside a single image, with the MCP server in its own thin environment dispatching by subprocess. This plan promotes that file to the project's real image, adds the two manifest fields every later engine needs (`outputs:` and `timeout_s`), and then adds three engines chosen because each forces a different part of the contract: `run_ipsae` takes a JSON input rather than a structure, `run_openmm_minimize` is the first engine that WRITES A FILE, and `run_mpnn` is the first with bundled weights and a model-variant enum.

**Tech Stack:** micromamba 2.9.0, Docker, Python >=3.10 (server) / 3.11 (engines), `mcp` SDK 1.25, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-atomistic-tool-refresh-design.md` (§5.1, §5.2, §5.3)
**Carry-forward:** `docs/superpowers/specs/2026-09-21-plan-2-carry-forward.md`

**Scope:** This is plan 2 of 4. It covers the environment substrate and the CPU engine tier. Plan 3 covers GPU engines that pip-install with downloadable weights (Chai-1, BoltzGen, Protenix v1, OpenFold3, ESMFold2, Promera, RFD3+RF3 via one `rc-foundry` install, Boltz-2). Plan 4 covers the heavy-asset engines (AlphaFold3 against the 3TB local mount, PyRosetta under the operator's licence) and the locally-checked-out generative models (Proteina-Complexa ×4, Genie 3, Protpardelle-1c, La-Proteina, FrameFlow, MultiFlow). Splitting this way keeps each plan's failure mode uniform: dependency resolution here, weight downloads in plan 3, mounts and licences in plan 4.

## Global Constraints

- Branch: `dev` in `/home/jk661/projects/protein-design-mcp-dev`. Never touch `/home/jk661/projects/protein-design-mcp` — a separate checkout holding uncommitted work.
- Server Python `>=3.10`. Engine environments pin their own Python; `ligandmpnn` requires `>=3.11,<3.12`.
- The host suite sits at **41 pre-existing failures** caused by the sandbox denying `stat()` on paths outside the working tree. It must stay at 41 with the same sorted FAILED list. Diff the list, never just the count.
- Manifests live in `src/protein_design_mcp/manifests/*.yaml`, packaged via `[tool.setuptools.package-data]`, overridable with `PROTEIN_MCP_MANIFEST_DIR`.
- Every manifest in a category with more than one member must contain the literal heading `## When to use this instead of the alternatives`. A manifest doc naming a sibling tool that does not yet exist must mark it as not yet available.
- Composite tools are never registered and never dispatchable; both properties derive from one filtered set.
- No tool may declare a parameter its engine silently ignores.
- Pin every engine dependency to an exact version. Floating versions in an image that 20 more engines will share is how a dependency solve stops being reproducible.
- Commit messages: single line under 72 chars, blank line, then the `Co-Authored-By:` trailer for the model doing the work.

---

## File Structure

| File | Responsibility |
|---|---|
| `Dockerfile.envs` | The project's real image. One `micromamba create -n <env>` block per engine, plus the `server` env. Modified, not replaced. |
| `src/protein_design_mcp/manifest/schema.py` | Gains `OutputSpec` and the `outputs:`/`timeout_s` manifest fields |
| `src/protein_design_mcp/dispatch/env.py` | Gains declared-output collection; workdir cleanup becomes conditional on outputs having been copied out |
| `src/protein_design_mcp/results.py` | New. Owns the persistent results directory that declared outputs are copied into, so a workdir can still be removed. |
| `src/protein_design_mcp/app.py` | Passes `manifest.timeout_s`; returns collected output paths |
| `src/protein_design_mcp/manifests/run_ipsae.yaml` | Interface-confidence metric from a PAE file |
| `src/protein_design_mcp/manifests/run_openmm_minimize.yaml` | Energy minimisation — first file-writing engine |
| `src/protein_design_mcp/manifests/run_mpnn.yaml` | Inverse folding, `model_type` enum over ProteinMPNN/Soluble/Ligand variants |
| `src/protein_design_mcp/adapters/{ipsae,openmm_minimize,mpnn}.py` | One adapter per engine, each `build_args(manifest, params)` / `parse_output(manifest, run)` |
| `scripts/live_proof.py` | Generalised from `scripts/live_proof_prodigy.py` — drives the real SDK handler for every registered tool |

---

### Task 1: Manifest gains `outputs:` and `timeout_s`

**Files:**
- Modify: `src/protein_design_mcp/manifest/schema.py`
- Test: `tests/test_manifest_schema.py`

**Interfaces:**
- Consumes: `Manifest`, `parse_manifest`, `ManifestError` (unchanged signatures)
- Produces: `OutputSpec(name: str, pattern: str, description: str)`; `Manifest.outputs: tuple[OutputSpec, ...]`; `Manifest.timeout_s: int`

The whole-branch review named both of these as one-field-now, 28-edits-later. `timeout_s` matters because PRODIGY finishes in milliseconds while a diffusion sampler runs for hours, and today a single process-wide `DEFAULT_TIMEOUT_S` covers both. `outputs:` matters because PRODIGY parses stdout, so the template currently teaches stdout parsing, while most later engines write files.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_manifest_schema.py
def test_outputs_default_to_empty():
    assert parse_manifest(MINIMAL).outputs == ()


def test_outputs_are_parsed():
    data = {
        **MINIMAL,
        "outputs": [
            {"name": "minimized_pdb", "pattern": "minimized.pdb",
             "description": "The relaxed structure."},
        ],
    }
    (out,) = parse_manifest(data).outputs
    assert out.name == "minimized_pdb"
    assert out.pattern == "minimized.pdb"
    assert out.description == "The relaxed structure."


def test_output_without_name_is_rejected():
    data = {**MINIMAL, "outputs": [{"pattern": "x.pdb"}]}
    with pytest.raises(ManifestError, match="name"):
        parse_manifest(data)


def test_output_without_pattern_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x"}]}
    with pytest.raises(ManifestError, match="pattern"):
        parse_manifest(data)


def test_absolute_output_pattern_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x", "pattern": "/etc/passwd"}]}
    with pytest.raises(ManifestError, match="relative"):
        parse_manifest(data)


def test_output_pattern_escaping_the_workdir_is_rejected():
    data = {**MINIMAL, "outputs": [{"name": "x", "pattern": "../escape.pdb"}]}
    with pytest.raises(ManifestError, match="relative"):
        parse_manifest(data)


def test_duplicate_output_names_are_rejected():
    data = {
        **MINIMAL,
        "outputs": [{"name": "x", "pattern": "a.pdb"},
                    {"name": "x", "pattern": "b.pdb"}],
    }
    with pytest.raises(ManifestError, match="duplicate"):
        parse_manifest(data)


def test_timeout_defaults_to_one_hour():
    assert parse_manifest(MINIMAL).timeout_s == 3600


def test_timeout_is_parsed():
    assert parse_manifest({**MINIMAL, "timeout_s": 120}).timeout_s == 120


def test_nonpositive_timeout_is_rejected():
    with pytest.raises(ManifestError, match="timeout_s"):
        parse_manifest({**MINIMAL, "timeout_s": 0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_manifest_schema.py -k "outputs or timeout" -v`
Expected: FAIL with `AttributeError: 'Manifest' object has no attribute 'outputs'`

- [ ] **Step 3: Write minimal implementation**

In `src/protein_design_mcp/manifest/schema.py`, add the dataclass next to `Requirements`:

```python
DEFAULT_TIMEOUT_S = 3600


@dataclass(frozen=True)
class OutputSpec:
    """A file an engine writes into its scratch directory.

    ``pattern`` is a path relative to the scratch directory. Declaring outputs
    is what lets the dispatcher collect results and then remove the workdir;
    an engine whose results are only on stdout declares none.
    """

    name: str
    pattern: str
    description: str = ""
```

Add the fields to `Manifest`:

```python
    outputs: tuple[OutputSpec, ...] = ()
    timeout_s: int = DEFAULT_TIMEOUT_S
```

Add the parser, and call it from `parse_manifest`:

```python
def _parse_outputs(data: Any, name: str) -> tuple[OutputSpec, ...]:
    if data is None:
        return ()
    if not isinstance(data, list):
        raise ManifestError(f"{name}: outputs must be a list")

    specs: list[OutputSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(data):
        label = f"{name}: outputs[{index}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{label} must be a mapping")

        out_name = entry.get("name")
        if not out_name:
            raise ManifestError(f"{label} is missing required key 'name'")
        if out_name in seen:
            raise ManifestError(f"{name}: duplicate output name {out_name!r}")
        seen.add(str(out_name))

        pattern = entry.get("pattern")
        if not pattern:
            raise ManifestError(f"{label} is missing required key 'pattern'")
        pattern = str(pattern)
        if pattern.startswith("/") or ".." in Path(pattern).parts:
            raise ManifestError(
                f"{label}: pattern {pattern!r} must be relative to the scratch "
                "directory and must not escape it"
            )

        specs.append(
            OutputSpec(
                name=str(out_name),
                pattern=pattern,
                description=str(entry.get("description", "")),
            )
        )
    return tuple(specs)


def _parse_timeout(data: Any, name: str) -> int:
    if data is None:
        return DEFAULT_TIMEOUT_S
    try:
        value = int(data)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name}: timeout_s must be an integer") from exc
    if value <= 0:
        raise ManifestError(f"{name}: timeout_s must be positive, got {value}")
    return value
```

`_parse_outputs` uses `Path`, so add `from pathlib import Path` to the imports. In `parse_manifest`'s `Manifest(...)` construction add:

```python
        outputs=_parse_outputs(data.get("outputs"), name),
        timeout_s=_parse_timeout(data.get("timeout_s"), name),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_manifest_schema.py -v`
Expected: PASS, all previous tests plus the 9 new ones

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/manifest/schema.py tests/test_manifest_schema.py
git commit -m "feat: add outputs and timeout_s to the manifest schema"
```

---

### Task 2: Collect declared outputs, then clean the workdir

**Files:**
- Create: `src/protein_design_mcp/results.py`
- Modify: `src/protein_design_mcp/dispatch/env.py`
- Test: `tests/test_results.py`, `tests/test_env_dispatcher.py`

**Interfaces:**
- Consumes: `OutputSpec` (Task 1), `CompletedRun`, `EngineError`
- Produces: `results_dir() -> Path`; `collect_outputs(specs, workdir, run_id) -> dict[str, str]`; `CompletedRun.outputs: dict[str, str]`; `EnvDispatcher.run(engine, args, *, timeout, outputs=())`

A workdir cannot be both removed on success and the place results live. The resolution: declared outputs are COPIED into a persistent results directory, and only then is the workdir removed. Undeclared files are discarded with the workdir, which is what makes `outputs:` load-bearing rather than decorative.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_results.py
import os
from pathlib import Path

import pytest

from protein_design_mcp.manifest.schema import OutputSpec
from protein_design_mcp.results import collect_outputs, results_dir


def test_results_dir_honours_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "r"))
    assert results_dir() == tmp_path / "r"


def test_results_dir_is_created(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "made"))
    results_dir().mkdir(parents=True, exist_ok=True)
    assert (tmp_path / "made").is_dir()


def test_declared_output_is_copied_out(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "minimized.pdb").write_text("ATOM\n")

    specs = (OutputSpec(name="minimized_pdb", pattern="minimized.pdb"),)
    collected = collect_outputs(specs, workdir, "run123")

    assert set(collected) == {"minimized_pdb"}
    copied = Path(collected["minimized_pdb"])
    assert copied.read_text() == "ATOM\n"
    assert not copied.is_relative_to(workdir)


def test_glob_pattern_collects_the_single_match(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    (workdir / "out_0.pdb").write_text("X\n")

    specs = (OutputSpec(name="design", pattern="out_*.pdb"),)
    assert Path(collect_outputs(specs, workdir, "r")["design"]).name == "out_0.pdb"


def test_missing_declared_output_raises_naming_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    specs = (OutputSpec(name="minimized_pdb", pattern="minimized.pdb"),)
    with pytest.raises(FileNotFoundError, match="minimized_pdb"):
        collect_outputs(specs, workdir, "r")


def test_no_specs_collects_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    workdir = tmp_path / "wd"
    workdir.mkdir()
    assert collect_outputs((), workdir, "r") == {}
```

```python
# append to tests/test_env_dispatcher.py
import sys
from pathlib import Path

from protein_design_mcp.manifest.schema import EngineSpec, OutputSpec


@pytest.mark.asyncio
async def test_declared_outputs_survive_workdir_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    script = "open('made.txt','w').write('hello')"

    result = await d.run(
        engine,
        ["-c", script],
        timeout=30,
        outputs=(OutputSpec(name="made", pattern="made.txt"),),
    )

    assert not result.workdir.exists(), "workdir should be removed on success"
    assert Path(result.outputs["made"]).read_text() == "hello"


@pytest.mark.asyncio
async def test_workdir_is_kept_when_a_declared_output_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PROTEIN_MCP_RESULTS_DIR", str(tmp_path / "res"))
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))

    with pytest.raises(EngineError, match="made"):
        await d.run(
            engine,
            ["-c", "pass"],
            timeout=30,
            outputs=(OutputSpec(name="made", pattern="made.txt"),),
        )


@pytest.mark.asyncio
async def test_run_without_outputs_still_removes_the_workdir(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    result = await d.run(engine, ["-c", "print('hi')"], timeout=30)
    assert result.outputs == {}
    assert not result.workdir.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_results.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.results'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/protein_design_mcp/results.py
"""The persistent home for files engines produce.

An engine writes into a scratch directory that is removed after the run. Any
file the caller needs must therefore be declared in the manifest's ``outputs:``
and copied out first. Undeclared files go away with the workdir, which is what
makes the declaration load-bearing rather than documentation.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import gettempdir

from protein_design_mcp.manifest.schema import OutputSpec


def results_dir() -> Path:
    """Where collected outputs live. Override with PROTEIN_MCP_RESULTS_DIR."""
    override = os.environ.get("PROTEIN_MCP_RESULTS_DIR")
    if override:
        return Path(override)
    return Path(gettempdir()) / "pdmcp-results"


def collect_outputs(
    specs: Sequence[OutputSpec],
    workdir: Path,
    run_id: str,
) -> dict[str, str]:
    """Copy each declared output out of ``workdir``. Returns name -> path.

    Raises FileNotFoundError naming the output whose pattern matched nothing,
    so the caller can keep the workdir for diagnosis.
    """
    if not specs:
        return {}

    destination = results_dir() / run_id
    destination.mkdir(parents=True, exist_ok=True)

    collected: dict[str, str] = {}
    for spec in specs:
        matches = sorted(workdir.glob(spec.pattern))
        if not matches:
            raise FileNotFoundError(
                f"declared output {spec.name!r} matched no file for pattern "
                f"{spec.pattern!r} in the engine's working directory"
            )
        source = matches[0]
        target = destination / source.name
        shutil.copy2(source, target)
        collected[spec.name] = str(target)
    return collected
```

In `src/protein_design_mcp/dispatch/env.py`, add `outputs: dict[str, str]` to `CompletedRun` (defaulting to an empty dict via `field(default_factory=dict)` — import `field` from `dataclasses`), add the `outputs` keyword to `run()`, and replace the success path's cleanup so it collects first:

```python
    async def run(
        self,
        engine: EngineSpec,
        args: Sequence[Any],
        *,
        timeout: float,
        outputs: Sequence[OutputSpec] = (),
    ) -> CompletedRun:
```

and, where the success path currently removes the workdir:

```python
        try:
            collected = collect_outputs(outputs, workdir, workdir.name)
        except FileNotFoundError as exc:
            raise EngineError(
                f"engine {engine.repo!r} exited successfully but did not produce "
                f"an expected output: {exc}\n\n"
                f"Working directory preserved for diagnosis: {workdir}"
            ) from exc

        shutil.rmtree(workdir, ignore_errors=True)
        return CompletedRun(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            workdir=workdir,
            outputs=collected,
        )
```

Import `shutil`, `OutputSpec` and `collect_outputs` at the top of `env.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_results.py tests/test_env_dispatcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/results.py src/protein_design_mcp/dispatch/env.py tests/test_results.py tests/test_env_dispatcher.py
git commit -m "feat: collect declared engine outputs before removing workdir"
```

---

### Task 3: Wire per-manifest timeout and outputs through the app

**Files:**
- Modify: `src/protein_design_mcp/app.py`
- Test: `tests/test_server_wiring.py`

**Interfaces:**
- Consumes: `Manifest.timeout_s`, `Manifest.outputs` (Task 1); `EnvDispatcher.run(..., outputs=)` (Task 2)
- Produces: successful tool results carry an `outputs` key when the manifest declares any

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_server_wiring.py
import json

import pytest

from protein_design_mcp.app import ServerApp
from protein_design_mcp.manifest.registry import ToolRegistry
from protein_design_mcp.manifest.schema import parse_manifest


class _RecordingDispatcher:
    """Captures what the app asked the dispatcher to do."""

    def __init__(self):
        self.timeout = None
        self.outputs = None

    async def run(self, engine, args, *, timeout, outputs=()):
        from pathlib import Path

        from protein_design_mcp.dispatch.env import CompletedRun

        self.timeout = timeout
        self.outputs = tuple(outputs)
        return CompletedRun(
            returncode=0, stdout="", stderr="", workdir=Path("/tmp"), outputs={}
        )


def _timeout_manifest():
    return parse_manifest(
        {
            "name": "run_prodigy",
            "category": "scoring",
            "engine": {"repo": "prodigy", "env": "scoring", "entry": ["prodigy"]},
            "summary": "Timeout probe.",
            "doc": "## What this is\nProbe.\n",
            "timeout_s": 45,
            "outputs": [{"name": "o", "pattern": "o.txt"}],
            "schema": {
                "complex_pdb": {"type": "string", "required": True,
                                "example": "c.pdb", "format": "path"},
                "chain_a": {"type": "string", "required": True, "example": "A"},
                "chain_b": {"type": "string", "required": True, "example": "B"},
                "temperature": {"type": "number", "default": 25.0},
            },
        }
    )


@pytest.mark.asyncio
async def test_manifest_timeout_reaches_the_dispatcher(tmp_path):
    dispatcher = _RecordingDispatcher()
    app = ServerApp(ToolRegistry([_timeout_manifest()]), dispatcher)
    pdb = tmp_path / "c.pdb"
    pdb.write_text("ATOM\n")

    await app.call_tool(
        "run_prodigy",
        {"complex_pdb": str(pdb), "chain_a": "A", "chain_b": "B"},
    )
    assert dispatcher.timeout == 45


@pytest.mark.asyncio
async def test_manifest_outputs_reach_the_dispatcher(tmp_path):
    dispatcher = _RecordingDispatcher()
    app = ServerApp(ToolRegistry([_timeout_manifest()]), dispatcher)
    pdb = tmp_path / "c.pdb"
    pdb.write_text("ATOM\n")

    await app.call_tool(
        "run_prodigy",
        {"complex_pdb": str(pdb), "chain_a": "A", "chain_b": "B"},
    )
    assert [o.name for o in dispatcher.outputs] == ["o"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_server_wiring.py -k "dispatcher" -v`
Expected: FAIL — `dispatcher.timeout` is the module-level `DEFAULT_TIMEOUT_S` (3600), not 45

- [ ] **Step 3: Write minimal implementation**

In `app.py`'s `call_tool`, replace the dispatcher invocation:

```python
            run = await self._dispatcher.run(
                manifest.engine,
                build_args(manifest, params),
                timeout=manifest.timeout_s,
                outputs=manifest.outputs,
            )
            payload = parse_output(manifest, run)
            if run.outputs:
                payload = {**payload, "outputs": run.outputs}
            return _ok(payload)
```

`DEFAULT_TIMEOUT_S` in `app.py` is now unused as a per-call value; keep it only if something else reads it, otherwise delete it and let `Manifest.timeout_s`'s own default govern.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_server_wiring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/app.py tests/test_server_wiring.py
git commit -m "feat: honour per-manifest timeout and declared outputs"
```

---

### Task 4: `run_ipsae` — a non-structure input

**Files:**
- Create: `src/protein_design_mcp/manifests/run_ipsae.yaml`, `src/protein_design_mcp/adapters/ipsae.py`
- Modify: `src/protein_design_mcp/app.py` (register in `ADAPTERS`), `src/protein_design_mcp/manifests/run_prodigy.yaml` (drop the "not yet implemented" marker now that this exists)
- Test: `tests/test_adapter_ipsae.py`

**Interfaces:**
- Consumes: `CompletedRun`, `validate_and_fill`, `load_manifests`
- Produces: `build_args(manifest, params) -> list[str]`, `parse_output(manifest, run) -> dict`

ipSAE is the field's standard interface-confidence discriminator — ProtDBench (arXiv 2605.04118) found it ~1.4× more precise than ipTM for identifying true binders. It is the first engine whose input is a predictor's JSON output rather than a structure, which is why it goes first: it proves the contract is not PDB-shaped.

`run_prodigy`'s doc currently says `run_ipsae` is "(not yet implemented)". Once this task lands, that is false, and the loader does not check it — remove the marker in the same commit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_ipsae.py
from pathlib import Path

import pytest

from protein_design_mcp.adapters.ipsae import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

SAMPLE_STDOUT = """\
Chn1 Chn2   ipSAE   ipTM_af  pDockQ   LIS
A    B      0.7213  0.6900   0.5412   0.3311
"""


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_ipsae")


def test_manifest_loads_and_is_not_composite():
    m = _manifest()
    assert m.category == "scoring"
    assert m.composite is False
    assert m.requires.gpu is False


def test_manifest_documents_the_choice_against_siblings():
    assert "## When to use this instead of the alternatives" in _manifest().doc


def test_build_args_passes_both_files_and_the_cutoffs():
    args = build_args(
        _manifest(),
        {"pae_json": "/tmp/pae.json", "structure": "/tmp/m.cif",
         "pae_cutoff": 10.0, "dist_cutoff": 10.0},
    )
    assert "/tmp/pae.json" in args
    assert "/tmp/m.cif" in args
    assert "10.0" in args


def test_parse_output_extracts_ipsae_for_the_chain_pair():
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="",
                     workdir=Path("/tmp")),
    )
    assert result["ipsae"] == pytest.approx(0.7213)
    assert result["iptm_af"] == pytest.approx(0.6900)
    assert result["pdockq"] == pytest.approx(0.5412)
    assert result["chain_pair"] == "A_B"


def test_parse_output_raises_when_no_row_is_present():
    with pytest.raises(ValueError, match="ipSAE"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="no table here", stderr="",
                         workdir=Path("/tmp")),
        )


def test_validation_rejects_a_non_json_pae_file():
    with pytest.raises(ToolInputError, match="pae_json"):
        validate_and_fill(_manifest(), {"pae_json": "notes.txt",
                                        "structure": "m.cif"})


def test_validation_fills_the_default_cutoffs():
    params = validate_and_fill(
        _manifest(), {"pae_json": "p.json", "structure": "m.cif"}
    )
    assert params["pae_cutoff"] == 10.0
    assert params["dist_cutoff"] == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_adapter_ipsae.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.adapters.ipsae'`

- [ ] **Step 3: Write minimal implementation**

```yaml
# src/protein_design_mcp/manifests/run_ipsae.yaml
name: run_ipsae
category: scoring
composite: false
timeout_s: 300

engine:
  repo: ipsae
  env: scoring
  entry: ["ipsae"]

requires:
  gpu: false

summary: >-
  Score how confident a structure predictor was about a protein-protein
  interface, from the PAE matrix it already produced (ipSAE). This is the
  field's standard discriminator between real and spurious binders and is the
  metric to rank designs by. It scores a prediction you already have: run a
  co-folding tool first and feed this its PAE output.

doc: |
  ## What this is
  ipSAE (interface Score from Aligned Errors) reads a structure predictor's
  predicted aligned error matrix and returns a per-chain-pair confidence in the
  interface itself, rather than in the fold as a whole.

  ## What it is for
  Ranking designed binders. Benchmarking across AF2-IG, Boltz-1, Boltz-2,
  Chai-1, ColabFold and Protenix found ipSAE roughly 1.4x more precise than
  ipTM at separating true binders from non-binders, and it remains the field
  standard as of late 2026.

  ## When to use this instead of the alternatives
  - `run_prodigy` returns an absolute binding free energy on a physical scale,
    but is calibrated on natural complexes and mis-ranks de novo designs. Use
    PRODIGY for a sanity floor, ipSAE for ranking.
  - The raw `iptm` a co-folding tool reports is the older, less precise form of
    the same idea. Prefer ipSAE when you have the PAE matrix.
  - ipSAE tells you nothing about the physics of the interface. For buried
    surface area, hydrogen bonds and shape complementarity you need a
    physics-based interface analysis, which is not yet implemented here.

  ## What you must supply
  The PAE JSON a predictor emitted, and the structure it goes with.

  ## What you get back
  `ipsae`, `iptm_af`, `pdockq` and the `chain_pair` they describe.

schema:
  pae_json:
    type: string
    format: path
    pattern: '\.json$'
    required: true
    description: PAE matrix JSON produced by a structure predictor.
    example: pae.json
  structure:
    type: string
    format: path
    pattern: '\.(pdb|cif)$'
    required: true
    description: The predicted structure the PAE matrix belongs to.
    example: model.cif
  pae_cutoff:
    type: number
    minimum: 1.0
    maximum: 30.0
    default: 10.0
    description: PAE cutoff in Angstroms.
    example: 10.0
  dist_cutoff:
    type: number
    minimum: 1.0
    maximum: 30.0
    default: 10.0
    description: Distance cutoff in Angstroms.
    example: 10.0
```

```python
# src/protein_design_mcp/adapters/ipsae.py
"""Adapter for ipSAE (DunbrackLab/IPSAE, PyPI `ipsae`).

ipSAE prints a whitespace-aligned table to stdout with one row per ordered
chain pair. We return the first row; a caller wanting a specific pair should
score that pair's structure.
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

# Chn1 Chn2  ipSAE  ipTM_af  pDockQ  LIS
_ROW_RE = re.compile(
    r"^\s*([A-Za-z0-9]+)\s+([A-Za-z0-9]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)",
    re.M,
)


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into ipSAE's argv."""
    return [
        str(params["pae_json"]),
        str(params["structure"]),
        str(params["pae_cutoff"]),
        str(params["dist_cutoff"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract the first chain-pair row from ipSAE's table."""
    for match in _ROW_RE.finditer(run.stdout):
        chain1, chain2, ipsae, iptm, pdockq = match.groups()
        if chain1.lower() == "chn1":
            continue
        return {
            "chain_pair": f"{chain1}_{chain2}",
            "ipsae": float(ipsae),
            "iptm_af": float(iptm),
            "pdockq": float(pdockq),
        }
    raise ValueError(
        "ipSAE produced no chain-pair row. Output was:\n"
        f"{run.stdout.strip()[-1000:]}"
    )
```

In `app.py`, add to `ADAPTERS`:

```python
from protein_design_mcp.adapters import ipsae as _ipsae

ADAPTERS = {
    "run_prodigy": (prodigy.build_args, prodigy.parse_output),
    "run_ipsae": (_ipsae.build_args, _ipsae.parse_output),
}
```

In `src/protein_design_mcp/manifests/run_prodigy.yaml`, change the `run_ipsae`
line in the comparison section from the "(not yet implemented)" phrasing to
present tense, since it now exists.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_adapter_ipsae.py tests/test_doc_generation.py -v`
Expected: PASS. `test_generated_docs_are_current` will fail until you regenerate:
`/home/jk661/.conda/envs/protein-design-mcp/bin/python scripts/generate_tool_docs.py`

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/manifests/ src/protein_design_mcp/adapters/ipsae.py src/protein_design_mcp/app.py tests/test_adapter_ipsae.py docs/tools/
git commit -m "feat: add run_ipsae interface confidence scoring"
```

---

### Task 5: `run_openmm_minimize` — the first engine that writes a file

**Files:**
- Create: `src/protein_design_mcp/manifests/run_openmm_minimize.yaml`, `src/protein_design_mcp/adapters/openmm_minimize.py`, `scripts/engines/openmm_minimize.py`
- Modify: `src/protein_design_mcp/app.py`
- Test: `tests/test_adapter_openmm.py`

**Interfaces:**
- Consumes: `OutputSpec`, `collect_outputs`, `CompletedRun.outputs`
- Produces: `build_args(manifest, params) -> list[str]`, `parse_output(manifest, run) -> dict`

This is the task that proves Tasks 1–3. OpenMM has no single-command CLI that does what we want, so the engine entry is a small script shipped with the package and run inside the `md` environment. That is itself the pattern most later engines need, so it is worth establishing here rather than in a GPU task.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_openmm.py
from pathlib import Path

import pytest

from protein_design_mcp.adapters.openmm_minimize import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

SAMPLE_STDOUT = """\
initial_potential_energy_kj_mol: 15234.7812
final_potential_energy_kj_mol: -8811.2044
iterations: 500
output_pdb: minimized.pdb
"""


def _manifest():
    return next(
        m for m in load_manifests(manifest_dir()) if m.name == "run_openmm_minimize"
    )


def test_manifest_declares_its_output():
    (out,) = _manifest().outputs
    assert out.name == "minimized_pdb"
    assert out.pattern == "minimized.pdb"


def test_manifest_sets_its_own_timeout():
    assert _manifest().timeout_s == 1800


def test_build_args_names_the_workdir_relative_output():
    args = build_args(
        _manifest(),
        {"input_pdb": "/tmp/in.pdb", "max_iterations": 500,
         "forcefield": "amber14"},
    )
    assert "/tmp/in.pdb" in args
    assert "minimized.pdb" in args
    assert "500" in args


def test_parse_output_reports_the_energy_change():
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=SAMPLE_STDOUT, stderr="",
                     workdir=Path("/tmp")),
    )
    assert result["initial_potential_energy_kj_mol"] == pytest.approx(15234.7812)
    assert result["final_potential_energy_kj_mol"] == pytest.approx(-8811.2044)
    assert result["energy_change_kj_mol"] == pytest.approx(-24045.9856)
    assert result["iterations"] == 500


def test_parse_output_raises_when_energies_are_absent():
    with pytest.raises(ValueError, match="energy"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout="nothing", stderr="",
                         workdir=Path("/tmp")),
        )


def test_validation_rejects_an_out_of_range_iteration_count():
    with pytest.raises(ToolInputError, match="max_iterations"):
        validate_and_fill(_manifest(), {"input_pdb": "in.pdb",
                                        "max_iterations": 0})


def test_validation_rejects_an_unknown_forcefield():
    with pytest.raises(ToolInputError, match="forcefield"):
        validate_and_fill(_manifest(), {"input_pdb": "in.pdb",
                                        "forcefield": "charmm99"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_adapter_openmm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.adapters.openmm_minimize'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/engines/openmm_minimize.py
"""Energy-minimise a structure with OpenMM. Runs inside the `md` environment.

Prints `key: value` lines the adapter parses, and writes the minimised
structure to the path given as the second argument, relative to the working
directory the dispatcher created.
"""

from __future__ import annotations

import argparse

from openmm import LangevinMiddleIntegrator, unit
from openmm.app import PDBFile, ForceField, Modeller, Simulation, HBonds, NoCutoff

FORCEFIELDS = {
    "amber14": ("amber14-all.xml", "amber14/tip3pfb.xml"),
    "charmm36": ("charmm36.xml", "charmm36/water.xml"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdb")
    parser.add_argument("output_pdb")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--forcefield", default="amber14", choices=sorted(FORCEFIELDS))
    args = parser.parse_args()

    pdb = PDBFile(args.input_pdb)
    forcefield = ForceField(*FORCEFIELDS[args.forcefield])
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    initial = simulation.context.getState(
        getEnergy=True
    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    simulation.minimizeEnergy(maxIterations=args.max_iterations)

    state = simulation.context.getState(getPositions=True, getEnergy=True)
    final = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    with open(args.output_pdb, "w") as handle:
        PDBFile.writeFile(simulation.topology, state.getPositions(), handle)

    print(f"initial_potential_energy_kj_mol: {initial:.4f}")
    print(f"final_potential_energy_kj_mol: {final:.4f}")
    print(f"iterations: {args.max_iterations}")
    print(f"output_pdb: {args.output_pdb}")


if __name__ == "__main__":
    main()
```

```yaml
# src/protein_design_mcp/manifests/run_openmm_minimize.yaml
name: run_openmm_minimize
category: scoring
composite: false
timeout_s: 1800

engine:
  repo: openmm_minimize
  env: md
  entry: ["python", "/app/scripts/engines/openmm_minimize.py"]

requires:
  gpu: false

outputs:
  - name: minimized_pdb
    pattern: minimized.pdb
    description: The energy-minimised structure.

summary: >-
  Relax a structure with OpenMM molecular mechanics, removing the clashes and
  strained geometry that generative models routinely produce. Run this before
  any physics-based scoring; scoring an unrelaxed model measures its clashes
  more than its interface. Returns the energy before and after, and writes the
  relaxed structure.

doc: |
  ## What this is
  Gradient-based energy minimisation under an Amber or CHARMM force field,
  using OpenMM. Hydrogens are added before minimising.

  ## What it is for
  Cleaning up a predicted or generated structure so that a physics-based score
  means something. De novo designs and diffusion outputs frequently contain
  atom clashes that dominate any energy term computed on them directly.

  ## When to use this instead of the alternatives
  - This is preparation, not scoring. It tells you the structure's internal
    energy improved; it says nothing about whether two chains bind.
  - For an interface score after relaxing, use `run_prodigy` for an absolute
    free energy or `run_ipsae` for predictor confidence.
  - Minimisation moves atoms. If you need the original coordinates preserved
    exactly, score the input instead of the output.

  ## What you must supply
  A PDB file. Multi-chain inputs are relaxed as one system.

  ## What you get back
  `initial_potential_energy_kj_mol`, `final_potential_energy_kj_mol`,
  `energy_change_kj_mol`, `iterations`, and under `outputs` the path to
  `minimized_pdb`.

schema:
  input_pdb:
    type: string
    format: path
    pattern: '\.pdb$'
    required: true
    description: Structure to minimise.
    example: model.pdb
  max_iterations:
    type: integer
    minimum: 1
    maximum: 10000
    default: 500
    description: Minimisation step limit.
    example: 500
  forcefield:
    type: string
    enum: ["amber14", "charmm36"]
    default: amber14
    description: Force field to minimise under.
    example: amber14
```

```python
# src/protein_design_mcp/adapters/openmm_minimize.py
"""Adapter for the OpenMM minimisation script run inside the `md` environment."""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

OUTPUT_NAME = "minimized.pdb"

_INITIAL_RE = re.compile(r"initial_potential_energy_kj_mol:\s*(-?[\d.]+)")
_FINAL_RE = re.compile(r"final_potential_energy_kj_mol:\s*(-?[\d.]+)")
_ITER_RE = re.compile(r"iterations:\s*(\d+)")


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the engine script's argv.

    The output path is relative, so the file lands in the dispatcher's scratch
    directory where the manifest's ``outputs:`` pattern can find it.
    """
    return [
        str(params["input_pdb"]),
        OUTPUT_NAME,
        "--max-iterations",
        str(params["max_iterations"]),
        "--forcefield",
        str(params["forcefield"]),
    ]


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract the energies the engine script printed."""
    initial = _INITIAL_RE.search(run.stdout)
    final = _FINAL_RE.search(run.stdout)
    if initial is None or final is None:
        raise ValueError(
            "OpenMM minimisation printed no energy lines. Output was:\n"
            f"{run.stdout.strip()[-1000:]}"
        )

    initial_value = float(initial.group(1))
    final_value = float(final.group(1))
    iterations = _ITER_RE.search(run.stdout)
    return {
        "initial_potential_energy_kj_mol": initial_value,
        "final_potential_energy_kj_mol": final_value,
        "energy_change_kj_mol": final_value - initial_value,
        "iterations": int(iterations.group(1)) if iterations else None,
    }
```

Register in `app.py`'s `ADAPTERS` as `"run_openmm_minimize"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_adapter_openmm.py -v`
then regenerate docs: `/home/jk661/.conda/envs/protein-design-mcp/bin/python scripts/generate_tool_docs.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/manifests/ src/protein_design_mcp/adapters/openmm_minimize.py scripts/engines/ src/protein_design_mcp/app.py tests/test_adapter_openmm.py docs/tools/
git commit -m "feat: add run_openmm_minimize, the first file-writing engine"
```

---

### Task 6: `run_mpnn` — bundled weights and a variant enum

**Files:**
- Create: `src/protein_design_mcp/manifests/run_mpnn.yaml`, `src/protein_design_mcp/adapters/mpnn.py`
- Modify: `src/protein_design_mcp/app.py`
- Test: `tests/test_adapter_mpnn.py`

**Interfaces:**
- Consumes: `CompletedRun`, `OutputSpec`
- Produces: `build_args(manifest, params) -> list[str]`, `parse_output(manifest, run) -> dict`

`dauparas/LigandMPNN` serves ProteinMPNN, SolubleMPNN and LigandMPNN from one codebase with weights in-repo, MIT licensed. One manifest with a `model_type` enum is therefore correct, and three separate tools would be wrong. The PyPI package `ligandmpnn==0.1.2` requires Python `>=3.11,<3.12`, which is why it gets its own environment.

The sweep found `UMA-Inverse` (CC BY 4.0) reporting better interface recovery, but it is a 3.3M-parameter single-paper self-reported result. Not substituted here; recorded as a candidate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_mpnn.py
import json
from pathlib import Path

import pytest

from protein_design_mcp.adapters.mpnn import build_args, parse_output
from protein_design_mcp.app import manifest_dir
from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.loader import load_manifests
from protein_design_mcp.validation import ToolInputError, validate_and_fill

SAMPLE_FASTA = """\
>input, T=0.1, seed=37, overall_confidence=0.4521, ligand_confidence=0.0
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ
>input, id=1, T=0.1, seed=37, overall_confidence=0.5310, ligand_confidence=0.0
MKTAYIAKQRQLSFVKSHFSRQLEERLGLIEVQ
>input, id=2, T=0.1, seed=37, overall_confidence=0.5502, ligand_confidence=0.0
MKTAYIARQRQLSFVKSHFSRQLEERLGLIEVQ
"""


def _manifest():
    return next(m for m in load_manifests(manifest_dir()) if m.name == "run_mpnn")


def test_manifest_is_sequence_design_and_declares_its_output():
    m = _manifest()
    assert m.category == "sequence_design"
    (out,) = m.outputs
    assert out.name == "designs_fasta"


def test_model_type_enum_covers_the_three_variants():
    enum = _manifest().schema["model_type"]["enum"]
    assert set(enum) == {"protein", "soluble", "ligand"}


def test_build_args_maps_model_type_to_the_checkpoint_flag():
    args = build_args(
        _manifest(),
        {"backbone_pdb": "/tmp/bb.pdb", "model_type": "soluble",
         "num_sequences": 8, "sampling_temp": 0.1, "seed": 37},
    )
    assert "soluble_mpnn" in args
    assert "/tmp/bb.pdb" in args
    assert "8" in args


def test_build_args_always_passes_a_seed_for_reproducibility():
    args = build_args(
        _manifest(),
        {"backbone_pdb": "/tmp/bb.pdb", "model_type": "protein",
         "num_sequences": 4, "sampling_temp": 0.1, "seed": 99},
    )
    assert "--seed" in args
    assert "99" in args


def test_parse_output_drops_the_native_input_sequence():
    """The first FASTA record is the input, not a design."""
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=SAMPLE_FASTA, stderr="",
                     workdir=Path("/tmp")),
    )
    assert result["num_designs"] == 2
    assert all(d["id"] is not None for d in result["designs"])
    assert "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ" not in [
        d["sequence"] for d in result["designs"]
    ]


def test_parse_output_reports_confidence_per_design():
    result = parse_output(
        _manifest(),
        CompletedRun(returncode=0, stdout=SAMPLE_FASTA, stderr="",
                     workdir=Path("/tmp")),
    )
    assert result["designs"][0]["overall_confidence"] == pytest.approx(0.5310)


def test_parse_output_raises_when_only_the_input_record_is_present():
    only_input = SAMPLE_FASTA.split(">input, id=1")[0]
    with pytest.raises(ValueError, match="no designs"):
        parse_output(
            _manifest(),
            CompletedRun(returncode=0, stdout=only_input, stderr="",
                         workdir=Path("/tmp")),
        )


def test_validation_rejects_an_unknown_model_type():
    with pytest.raises(ToolInputError, match="model_type"):
        validate_and_fill(_manifest(), {"backbone_pdb": "bb.pdb",
                                        "model_type": "rna"})


def test_validation_rejects_a_temperature_above_the_maximum():
    with pytest.raises(ToolInputError, match="sampling_temp"):
        validate_and_fill(_manifest(), {"backbone_pdb": "bb.pdb",
                                        "sampling_temp": 5.0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_adapter_mpnn.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'protein_design_mcp.adapters.mpnn'`

- [ ] **Step 3: Write minimal implementation**

```yaml
# src/protein_design_mcp/manifests/run_mpnn.yaml
name: run_mpnn
category: sequence_design
composite: false
timeout_s: 1800

engine:
  repo: mpnn
  env: mpnn
  entry: ["python", "-m", "ligandmpnn.run"]

requires:
  gpu: false

outputs:
  - name: designs_fasta
    pattern: "seqs/*.fa"
    description: Designed sequences in FASTA format.

summary: >-
  Design amino acid sequences for a fixed protein backbone (ProteinMPNN and its
  variants). This is the step between generating a backbone and predicting what
  that sequence actually folds into. Select the variant with model_type:
  protein for the standard model, soluble to avoid designing buried
  hydrophobics, ligand when a small molecule, nucleotide or metal is present.

doc: |
  ## What this is
  Inverse folding: given backbone coordinates, predict amino acid sequences
  likely to fold into them. One codebase serves three trained variants, chosen
  with `model_type`.

  ## What it is for
  Every generative backbone model produces coordinates without a sequence.
  This turns them into something you can express, and is the most mature and
  most validated step in the whole design pipeline.

  ## When to use this instead of the alternatives
  - `model_type: protein` is the default and correct for a bare protein
    backbone.
  - `model_type: soluble` is trained to avoid the exposed hydrophobic patches
    the standard model places when it has no membrane context. Prefer it for
    anything you intend to express in solution.
  - `model_type: ligand` conditions on non-protein atoms. Use it whenever a
    small molecule, nucleotide or metal sits in the structure; the standard
    model ignores them and will design a sequence that clashes.
  - After designing, fold the sequence back and check it matches the backbone
    you designed for. A sequence this tool likes is not automatically one that
    folds.

  ## What you must supply
  A backbone PDB. Side chains are ignored.

  ## What you get back
  `designs`, each with `id`, `sequence` and `overall_confidence`;
  `num_designs`; and under `outputs` the path to the FASTA file.

  ## Important caveat
  The first record the engine emits is the INPUT sequence, not a design. It is
  dropped here. Confidence is the model's own likelihood, not a prediction of
  experimental success.

schema:
  backbone_pdb:
    type: string
    format: path
    pattern: '\.(pdb|cif)$'
    required: true
    description: Backbone structure to design a sequence for.
    example: backbone.pdb
  model_type:
    type: string
    enum: ["protein", "soluble", "ligand"]
    default: protein
    description: Which trained variant to use.
    example: soluble
  num_sequences:
    type: integer
    minimum: 1
    maximum: 128
    default: 8
    description: How many sequences to sample.
    example: 8
  sampling_temp:
    type: number
    minimum: 0.0001
    maximum: 1.0
    default: 0.1
    description: Sampling temperature. Lower is more conservative.
    example: 0.1
  seed:
    type: integer
    minimum: 0
    default: 37
    description: Random seed. Fixed by default so runs are reproducible.
    example: 37
```

```python
# src/protein_design_mcp/adapters/mpnn.py
"""Adapter for dauparas/LigandMPNN (PyPI `ligandmpnn`).

One codebase serves ProteinMPNN, SolubleMPNN and LigandMPNN; `model_type`
selects the checkpoint family.

The engine writes FASTA whose FIRST record is the input sequence rather than a
design. Returning it as a design is a real bug this server previously shipped —
a caller trusting ``designs[0]`` got back what it put in.
"""

from __future__ import annotations

import re
from typing import Any

from protein_design_mcp.dispatch.env import CompletedRun
from protein_design_mcp.manifest.schema import Manifest

CHECKPOINT_FOR = {
    "protein": "protein_mpnn",
    "soluble": "soluble_mpnn",
    "ligand": "ligand_mpnn",
}

_ID_RE = re.compile(r"\bid=(\d+)")
_CONF_RE = re.compile(r"\boverall_confidence=([\d.]+)")


def build_args(manifest: Manifest, params: dict[str, Any]) -> list[str]:
    """Translate validated parameters into the engine's argv."""
    return [
        "--model_type",
        CHECKPOINT_FOR[str(params["model_type"])],
        "--pdb_path",
        str(params["backbone_pdb"]),
        "--out_folder",
        ".",
        "--batch_size",
        str(params["num_sequences"]),
        "--temperature",
        str(params["sampling_temp"]),
        "--seed",
        str(params["seed"]),
    ]


def _records(text: str) -> list[tuple[str, str]]:
    """Split FASTA into (header, sequence) pairs, preserving order."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:], []
        elif header is not None and line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def parse_output(manifest: Manifest, run: CompletedRun) -> dict[str, Any]:
    """Extract designs, dropping the input record the engine emits first."""
    designs = []
    for header, sequence in _records(run.stdout):
        id_match = _ID_RE.search(header)
        if id_match is None:
            # No id= means this is the native input sequence, not a design.
            continue
        conf = _CONF_RE.search(header)
        designs.append(
            {
                "id": int(id_match.group(1)),
                "sequence": sequence,
                "overall_confidence": float(conf.group(1)) if conf else None,
            }
        )

    if not designs:
        raise ValueError(
            "MPNN produced no designs — only the input record was present. "
            f"Output was:\n{run.stdout.strip()[-1000:]}"
        )

    return {"designs": designs, "num_designs": len(designs)}
```

Register in `app.py`'s `ADAPTERS` as `"run_mpnn"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_adapter_mpnn.py -v`
then regenerate docs.
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/manifests/ src/protein_design_mcp/adapters/mpnn.py src/protein_design_mcp/app.py tests/test_adapter_mpnn.py docs/tools/
git commit -m "feat: add run_mpnn inverse folding with variant selection"
```

---

### Task 7: Extend the image to four environments and prove all four engines live

**Files:**
- Modify: `Dockerfile.envs`
- Create: `scripts/live_proof.py`
- Delete: `scripts/live_proof_prodigy.py`
- Test: `tests/test_live_proof_script.py`

**Interfaces:**
- Consumes: every manifest and adapter from Tasks 4–6
- Produces: a container whose default command exercises every registered tool through the real SDK handler

`scripts/live_proof_prodigy.py` proved one engine. Generalise it: it should discover every registered tool, run the ones it has inputs for, and fail loudly on any tool it cannot exercise — so that adding an engine without adding a live check is visible rather than silent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_proof_script.py
"""The live-proof driver must cover every registered tool.

This runs on the host without Docker: it checks the driver's coverage map
against the registry, so an engine added without a live check fails here
rather than being silently unproven.
"""

import sys
from pathlib import Path

from protein_design_mcp.app import build_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_proof import CASES  # noqa: E402


def test_every_registered_tool_has_a_live_case():
    registered = {t.name for t in build_registry(device="cpu").tools()}
    covered = {case["tool"] for case in CASES}
    missing = registered - covered
    assert not missing, f"tools with no live-proof case: {sorted(missing)}"


def test_no_case_references_an_unregistered_tool():
    registered = {t.name for t in build_registry(device="cpu").tools()}
    covered = {case["tool"] for case in CASES}
    assert not covered - registered


def test_every_case_declares_expected_result_keys():
    for case in CASES:
        assert case["expect_keys"], f"{case['tool']} declares no expected keys"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_live_proof_script.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'live_proof'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/live_proof.py` by generalising `scripts/live_proof_prodigy.py`. Keep its existing mechanism for reaching the real handler — it fetches `server.request_handlers[types.CallToolRequest]` from the `mcp.server.Server` instance and awaits a real `types.CallToolRequest`. Replace its single hardcoded call with a table:

```python
CASES = [
    {
        "tool": "run_prodigy",
        "arguments": {
            "complex_pdb": "tests/fixtures/test_pdbs/1BRS.pdb",
            "chain_a": "A",
            "chain_b": "D",
        },
        "expect_keys": ["binding_affinity_kcal_per_mol", "intermolecular_contacts"],
    },
    {
        "tool": "run_openmm_minimize",
        "arguments": {
            "input_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "max_iterations": 50,
        },
        "expect_keys": ["energy_change_kj_mol", "outputs"],
    },
    {
        "tool": "run_mpnn",
        "arguments": {
            "backbone_pdb": "tests/fixtures/test_pdbs/mini_protein.pdb",
            "num_sequences": 2,
        },
        "expect_keys": ["designs", "num_designs"],
    },
    {
        "tool": "run_ipsae",
        "arguments": {
            "pae_json": "tests/fixtures/pae/example_pae.json",
            "structure": "tests/fixtures/test_pdbs/two_chain_complex.pdb",
        },
        "expect_keys": ["ipsae", "chain_pair"],
    },
    {
        "tool": "describe_tool",
        "arguments": {"category": "scoring"},
        "expect_keys": ["tools"],
    },
]
```

The driver iterates `CASES`, calls each through the real handler, asserts `isError` is False and every key in `expect_keys` is present, prints the parsed payload, and exits non-zero on the first failure.

`run_ipsae`'s case needs a PAE fixture. Create `tests/fixtures/pae/example_pae.json` containing a small square PAE matrix matching `two_chain_complex.pdb`'s residue count, in the `{"pae": [[...]]}` shape ipSAE reads. If ipSAE rejects a synthetic matrix, generate a real one by running a co-folding tool — but that is a plan 3 dependency, so instead reduce this case to the failure path (assert the error names the input) and record in your report that `run_ipsae`'s success path is unproven until plan 3 lands. Do not fabricate a passing result.

In `Dockerfile.envs`, add three environments following the documented pattern, pinning every version:

```dockerfile
RUN micromamba create -y -n md -c conda-forge python=3.11 pip && \
    micromamba run -n md pip install --no-cache-dir openmm==8.6.1 pdbfixer==1.11 && \
    micromamba clean --all --yes

RUN micromamba create -y -n mpnn -c conda-forge python=3.11 pip && \
    micromamba run -n mpnn pip install --no-cache-dir ligandmpnn==0.1.2 && \
    micromamba clean --all --yes
```

`ipsae` joins the existing `scoring` environment — it is pure Python with no dependencies that conflict with `prodigy-prot`:

```dockerfile
RUN micromamba run -n scoring pip install --no-cache-dir ipsae==1.0.1
```

Pin `prodigy-prot` explicitly in the `scoring` block while you are there; the sweep reported 2.2.1 as current but PyPI serves 2.4.0, so the floating install is already ambiguous. Use `prodigy-prot==2.4.0` and note the version in your report.

Copy the engine scripts into the image so `entry: ["python", "/app/scripts/engines/openmm_minimize.py"]` resolves:

```dockerfile
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts/ ./scripts/
```

Change the default command to the generalised driver.

- [ ] **Step 4: Build the image and run every live case**

```bash
docker build -f Dockerfile.envs -t protein-design-mcp:envs .
docker run --rm protein-design-mcp:envs
```

Expected: every case prints a parsed payload and the run ends with all checks passed. Then confirm the host suite:
`/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/ -q`
Expected: 41 failures, same sorted FAILED list.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile.envs scripts/ tests/
git rm scripts/live_proof_prodigy.py
git commit -m "feat: four-environment image with live proof for every tool"
```

---

### Task 8: Close the two carried-forward template defects

**Files:**
- Modify: `src/protein_design_mcp/app.py`, `src/protein_design_mcp/manifest/loader.py`
- Test: `tests/test_manifest_loader.py`, `tests/test_describe_tool.py`

**Interfaces:**
- Consumes: `_json_schema_for` (currently private to `registry.py`)
- Produces: `registry.json_schema_for(manifest) -> dict` (renamed public); loader rejects a doc naming an unknown tool

Two defects the whole-branch review carried forward, both cheap now and expensive at 29 manifests. C7: two code paths derive a `Tool` from a `Manifest` — `registry._json_schema_for` and a hand-inlined copy in `app.list_tools` for `describe_tool` — agreeing only by coincidence. C8: doc cross-references are unverified, so when a tool named "(not yet implemented)" ships, the doc silently misleads and no test fails. Task 4 already hit this manually for `run_ipsae`; this makes it structural.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_manifest_loader.py
def test_doc_naming_an_unknown_tool_is_rejected(tmp_path):
    body = SOLO.replace(
        "      PRODIGY.\n",
        "      PRODIGY. See run_does_not_exist for the alternative.\n",
    )
    _write(tmp_path, "run_prodigy.yaml", body)
    with pytest.raises(ManifestError, match="run_does_not_exist"):
        load_manifests(tmp_path)


def test_doc_naming_a_known_sibling_is_accepted(tmp_path):
    heading = SIBLING_DOC_HEADING
    a = SOLO.replace(
        "      PRODIGY.\n",
        f"      PRODIGY.\n\n      {heading}\n      Use run_ipsae to rank designs.\n",
    )
    b = a.replace("run_prodigy", "run_ipsae")
    _write(tmp_path, "a.yaml", a)
    _write(tmp_path, "b.yaml", b)
    assert len(load_manifests(tmp_path)) == 2


def test_doc_may_name_a_tool_marked_not_yet_implemented(tmp_path):
    body = SOLO.replace(
        "      PRODIGY.\n",
        "      PRODIGY. run_rosetta_interface (not yet implemented) will "
        "give the physics breakdown.\n",
    )
    _write(tmp_path, "run_prodigy.yaml", body)
    assert len(load_manifests(tmp_path)) == 1
```

```python
# append to tests/test_describe_tool.py
def test_describe_tool_schema_matches_the_registry_derivation():
    """The meta-tool's schema must come from the same code path as every other."""
    from protein_design_mcp.manifest.registry import json_schema_for
    from protein_design_mcp.meta_tools import DESCRIBE_TOOL_MANIFEST

    derived = json_schema_for(DESCRIBE_TOOL_MANIFEST)
    assert derived["additionalProperties"] is False
    assert "required" not in derived["properties"]["name"]
    assert "example" not in derived["properties"]["name"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/test_manifest_loader.py tests/test_describe_tool.py -v`
Expected: FAIL — the loader accepts unknown tool names, and `json_schema_for` is not importable

- [ ] **Step 3: Write minimal implementation**

In `registry.py`, rename `_json_schema_for` to `json_schema_for` (keeping a module-level alias is unnecessary; update its one internal caller). In `app.py`'s `list_tools`, delete the inlined schema block and use it:

```python
        tools.append(
            Tool(
                name=DESCRIBE_TOOL_MANIFEST.name,
                description=DESCRIBE_TOOL_MANIFEST.summary,
                inputSchema=json_schema_for(DESCRIBE_TOOL_MANIFEST),
            )
        )
```

In `loader.py`, add the cross-reference rule and call it from `load_manifests` after `_check_sibling_docs`:

```python
_TOOL_MENTION_RE = re.compile(r"\brun_[a-z0-9_]+\b")
_UNAVAILABLE_MARKER = "not yet implemented"


def _check_doc_references(manifests: list[Manifest]) -> None:
    """Every tool a doc names must exist, or be marked not yet implemented.

    Without this, a doc that says "use run_x instead" keeps saying it after
    run_x ships under a different name, or before it ships at all — and the
    model acts on it either way.
    """
    known = {m.name for m in manifests}
    for manifest in manifests:
        for line in manifest.doc.splitlines():
            for mentioned in _TOOL_MENTION_RE.findall(line):
                if mentioned in known or mentioned == manifest.name:
                    continue
                if _UNAVAILABLE_MARKER in line.lower():
                    continue
                raise ManifestError(
                    f"{manifest.name}: doc names {mentioned!r}, which is not a "
                    "known tool. Either fix the name, or mark it "
                    f"'({_UNAVAILABLE_MARKER})' on the same line."
                )
```

Add `import re` to `loader.py` if absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/ -q`
Expected: 41 failures, same sorted FAILED list; all new tests pass

- [ ] **Step 5: Commit**

```bash
git add src/protein_design_mcp/manifest/ src/protein_design_mcp/app.py tests/
git commit -m "feat: verify doc tool references, unify schema derivation"
```

---

### Task 9: Remove the orphaned `tools/` package

**Files:**
- Delete: `src/protein_design_mcp/tools/` (17 modules, 2061 lines), `tests/test_design_binder.py`, `tests/test_validate_design.py`, `tests/test_optimize.py`, `tests/test_hotspots.py`, `tests/test_tools.py`
- Modify: `docs/superpowers/specs/2026-09-21-plan-2-carry-forward.md`

**Interfaces:**
- Consumes: nothing
- Produces: nothing — this is pure removal

Plan 1 deliberately kept both the package and its tests, because deleting a test while its subject still ships converts covered code into silently dead code. Now they go together, which is the coherent change that was being deferred. Nothing live imports `protein_design_mcp.tools`; its `__init__.py` eagerly imports seven removed-tool modules, so it is an import hazard, not merely dead weight.

Before deleting, check whether `tools/hotspots.py` (536 lines of interface analysis) contains logic worth preserving under the new structure. If it does, extract it in a separate commit FIRST and say so in your report; otherwise record that you looked and found nothing worth keeping.

- [ ] **Step 1: Record the baseline**

```bash
/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/ -q 2>&1 | grep '^FAILED' | sort > /tmp/before.txt
wc -l /tmp/before.txt
```
Expected: 41

- [ ] **Step 2: Confirm nothing imports the package**

```bash
grep -rn "protein_design_mcp.tools\|from .tools\|from ..tools" --include="*.py" src/ scripts/ tests/ deploy/ | grep -v "^src/protein_design_mcp/tools/"
```
Expected: no output. If there IS output, stop and report it — something depends on this and the deletion is not safe.

- [ ] **Step 3: Delete**

```bash
git rm -r src/protein_design_mcp/tools/
git rm tests/test_design_binder.py tests/test_validate_design.py \
       tests/test_optimize.py tests/test_hotspots.py tests/test_tools.py
```

- [ ] **Step 4: Verify the failure count dropped by exactly the removed tests**

```bash
/home/jk661/.conda/envs/protein-design-mcp/bin/python -m pytest tests/ -q 2>&1 | grep '^FAILED' | sort > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
```
Expected: the diff shows ONLY removals, all from the five deleted files (25 of them: 7 from test_optimize, 17 from test_hotspots, 1 from test_design_binder). The remaining count should be 16. Any failure that is present in `after.txt` but not `before.txt` is a regression you introduced.

Also confirm the package still imports and lists its tools:
```bash
PYTHONPATH=src /home/jk661/.conda/envs/protein-design-mcp/bin/python -c "
import asyncio, protein_design_mcp.server as s
print([t.name for t in asyncio.run(s._app.list_tools())])"
```

- [ ] **Step 5: Commit**

Update the carry-forward doc to strike the "unreachable code scheduled for removal" section, then:

```bash
git add -A
git commit -m "refactor: remove the orphaned tools package and its tests"
```

---

## Self-Review

**Spec coverage.**

| Spec / carry-forward item | Task |
|---|---|
| §5.1 multi-environment image | 7 |
| §5.2 step 1 — inputs available to the engine | already live (central path resolution, plan 1 fix wave) |
| §5.2 step 3 — parse declared output files | 1, 2, 3, 5 |
| §5.3 per-engine resource limits | 1, 3 (`timeout_s`) |
| C7 duplicated schema derivation | 8 |
| C8 unverified doc cross-references | 8 |
| C11 single global timeout | 1, 3 |
| `outputs:` manifest field | 1, 2, 5 |
| Orphaned `tools/` removal | 9 |
| Engine tier: ipSAE, OpenMM, MPNN | 4, 5, 6 |

**Deliberately out of scope, carried to plans 3 and 4:** C9 (HTTP transport has no authentication — needs a real answer before anyone runs `--host 0.0.0.0`, and it is a security design question, not an engine task); weights and licence DETECTION (`available_weights`/`licensed` are still empty, so no manifest in this plan may declare `requires.weights` — the first that does belongs with plan 4's AlphaFold3 mount); the fix-soon minors from plan 1's ledger that touch error wording (`ToolRegistry`'s operator-voiced exclusion messages, `describe_tool(category="meta")`); and stdout/stderr buffering, which only bites on an engine that prints for an hour — plan 3's diffusion samplers are exactly that profile, so it belongs there.

**Placeholder scan.** No TBDs. Every code step carries runnable code. One conditional remains by design: Task 7's `run_ipsae` live case may not have a valid PAE fixture, and the instruction is explicit — reduce to the failure path and report it unproven rather than fabricate a passing result.

**Type consistency.** `OutputSpec(name, pattern, description)` from Task 1 is used unchanged in Tasks 2, 5, 6. `collect_outputs(specs, workdir, run_id) -> dict[str, str]` from Task 2 is called only by `EnvDispatcher.run`. `CompletedRun` gains `outputs: dict[str, str]` in Task 2 and is constructed with it in Tasks 4–6 tests. `build_args(manifest, params)` and `parse_output(manifest, run)` match the signature plan 1's fix wave established. `json_schema_for` is renamed in Task 8 and used in the same task; no earlier task references it by the old private name. `manifest_dir()` is imported from `app` in every adapter test, matching plan 1's final location.

**One ordering note for the executor.** Tasks 4, 5 and 6 each regenerate `docs/tools/`. If they are run out of order the staleness test will fail on whichever runs second until you regenerate — that is the guard working, not a defect.
