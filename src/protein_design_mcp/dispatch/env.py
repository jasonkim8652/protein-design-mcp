"""Run an engine inside its own conda environment.

Each engine pins a Python, numpy and torch version that conflict with the
others (spec §5.1), so every engine lives in its own micromamba environment
inside one image and is invoked as a subprocess. This generalises the
``conda run -n <env>`` branch that previously existed only in
``pipelines/boltz_runner.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from protein_design_mcp.manifest.schema import EngineSpec, OutputSpec
from protein_design_mcp.results import collect_outputs

_OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda error: out of memory")

# Some ``engine.prefix`` (host-mounted conda environment) dispatches read
# files that live under this host's real home directory -- checkpoints,
# weight caches (see e.g. run_promera.yaml/run_rf3.yaml/
# run_rfdiffusion3_binder.yaml's ``mounts:``, all under /home/jk661/...).
# The three engines below (promera's tinyprot taxonomy cache, foundry's
# checkpoint registry behind rf3/rfd3) resolve those paths via
# ``Path.home()``/``os.environ["HOME"]`` rather than an explicit flag this
# server could pass instead. Inside a container this PROCESS's own $HOME is
# whatever the container itself was started with (see
# ``scripts/container_run.py``'s own ``-e HOME=...``) -- not this host's
# real home directory -- so the file is genuinely present and correctly
# mounted, but the engine looks for it in the wrong place
# (task-13-report.md, "$HOME mismatch", 4 manifests found by the same
# in-container proof this fixes for good).
#
# Fixed HERE, once, for exactly these engines, rather than with a repeated
# ``env_vars: {HOME: ...}`` in each affected manifest -- a per-manifest fix
# is exactly the kind of thing a later engine with the same failure mode
# forgets to copy, silently reintroducing this bug one manifest at a time.
#
# NOT applied to every ``prefix``-dispatched engine, even though that was
# tried first and looked like the more general fix: it broke
# ``run_boltzgen_design`` in-container, which is ALSO prefix-dispatched but
# needs its OWN, container-default $HOME (``/tmp``, writable — see
# ``container_run.py``) for triton's own JIT kernel cache
# (``$HOME/.triton/cache``) — `/home/jk661` is a read-only mount in this
# image, so redirecting a WRITE-using engine's $HOME there instead of a
# READ-using one's breaks it outright (`triton.runtime.build` failing to
# create its cache dir). So this is an explicit allowlist of ``engine.repo``
# values known to need it, not everything dispatched via ``prefix`` —
# extending it for a future engine with the same read-side failure mode is
# still one line in one place, not a new manifest key to remember.
_HOME_OVERRIDE_REPOS = frozenset({"promera", "rf3", "rfd3"})
_HOST_HOME = "/home/jk661"


class EngineError(RuntimeError):
    """An engine subprocess failed, timed out, or could not be started."""


@dataclass(frozen=True)
class CompletedRun:
    returncode: int
    stdout: str
    stderr: str
    workdir: Path
    outputs: dict[str, str | list[str]] = field(default_factory=dict)


def _excerpt(text: str, head: int = 1200, tail: int = 1800) -> str:
    """Keep BOTH ends of an engine's stderr, not just the tail.

    A plain ``[-2000:]`` looks right and fails on a whole class of engine:
    a CLI that catches an inner error and re-raises ``CalledProcessError``
    puts its own full argv at the END of stderr, so the last 2000 characters
    are the command line and the actual traceback -- which is EARLIER -- is
    discarded. Proteina-Complexa does exactly this, and it made an
    in-container failure undiagnosable: the message named the command and
    not the cause, and the preserved workdir died with the --rm container.
    """
    text = text.strip()
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


class EnvDispatcher:
    """Build and execute ``micromamba run -n <env> <entry> <args>`` commands."""

    def __init__(
        self,
        runner: str | None = "micromamba",
        scratch_root: Path | None = None,
    ) -> None:
        self._runner = runner
        self._scratch_root = Path(scratch_root) if scratch_root else Path(gettempdir())

    def build_command(self, engine: EngineSpec, args: Sequence[Any]) -> list[str]:
        """Return the full argv. Pure — safe to assert on in tests.

        ``engine.prefix`` (an absolute path to a host-mounted environment)
        dispatches as ``run -p <prefix>``; ``engine.env`` (a name resolved
        under the image's root prefix) keeps ``run -n <env>``. Exactly one
        of the two is ever set — enforced at manifest load, not here.
        """
        prefix: list[str] = []
        if self._runner:
            if engine.prefix is not None:
                prefix = [self._runner, "run", "-p", engine.prefix]
            else:
                prefix = [self._runner, "run", "-n", engine.env]
        return [*prefix, *engine.entry, *(str(a) for a in args)]

    def _make_workdir(self) -> Path:
        workdir = self._scratch_root / f"pdmcp-{uuid.uuid4().hex[:12]}"
        workdir.mkdir(parents=True, exist_ok=False)
        return workdir

    def new_workdir(self) -> Path:
        """Create a fresh scratch directory ahead of a run.

        For a caller that must stage input files (see
        ``protein_design_mcp.staging.stage_inputs``) into the SAME
        directory ``run()`` will use as the subprocess's cwd. Staging has to
        happen before the engine's argv is built (a staged path replaces the
        original in that argv), which is itself before ``run()`` is called —
        so the workdir has to exist earlier than ``run()`` normally creates
        one. Pass the returned path back in as ``run(..., workdir=...)`` to
        make ``run()`` use this one instead of making its own.
        """
        return self._make_workdir()

    async def run(
        self,
        engine: EngineSpec,
        args: Sequence[Any],
        *,
        timeout: float,
        outputs: Sequence[OutputSpec] = (),
        workdir: Path | None = None,
    ) -> CompletedRun:
        """Execute the engine. Raises EngineError on any failure.

        ``workdir``, if given, must come from ``new_workdir()`` (typically
        after staging files into it) and is used as-is instead of a fresh
        directory being created here. Its lifecycle — preserved on failure,
        removed on success — is identical either way.
        """
        command = self.build_command(engine, args)
        if workdir is None:
            workdir = self._make_workdir()

        # env_vars is merged over a COPY of this process's environment —
        # never passed alone as env=, which would strip PATH and the
        # subprocess would not start (design §3.3). The cache variables are
        # defaults: they point into THIS run's scratch workdir so an engine
        # never writes into a read-only mount or collides with another
        # engine in a shared ~/.cache, but engine.env_vars — applied last —
        # can override any of them.
        # PER-ENGINE and PERSISTENT, deliberately NOT per-call. This used to
        # be `workdir / ".cache"`, which met the two goals below but threw the
        # cache away on every call: run_multiflow's self-consistency refold
        # re-downloaded ~8.5GB of ESMFold weights each time it ran, adding
        # minutes per call and hammering the network for nothing.
        # Namespacing by engine identity keeps both original properties — it
        # is writable scratch, so no engine writes into a read-only mount, and
        # two engines never collide in a shared ~/.cache.
        cache_key = re.sub(r"[^A-Za-z0-9_.-]", "_", engine.prefix or engine.env or "shared")
        cache_dir = self._scratch_root / ".pdmcp-engine-cache" / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess_env = {
            **os.environ,
            "HF_HOME": str(cache_dir / "huggingface"),
            "TORCH_HOME": str(cache_dir / "torch"),
            "XDG_CACHE_HOME": str(cache_dir),
            # Same reasoning as the three cache vars above, for triton's own
            # JIT kernel cache (defaults to $HOME/.triton/cache, see
            # triton.knobs — TRITON_CACHE_DIR overrides it directly).
            # Unconditional (not scoped to _HOME_OVERRIDE_REPOS below) —
            # regression found live: rf3/rfd3 (foundry env) both compile
            # triton kernels, and once their own $HOME is redirected to the
            # read-only /home/jk661 mount (for the checkpoint-path fix
            # below), `os.makedirs('/home/jk661/.triton')` fails with
            # PermissionError. Scoping this redirect to the SAME allowlist
            # as the HOME override would only be correct until the next
            # engine that both needs the HOME override AND uses triton;
            # redirecting it for everyone, like the other cache vars, is not
            # more work and closes that off for good.
            "TRITON_CACHE_DIR": str(cache_dir / "triton"),
            # See _HOME_OVERRIDE_REPOS's own comment: only the specific
            # engines known to resolve a mounted host path via Path.home()
            # get this — every other engine (prefix- or env-dispatched
            # alike) keeps whatever $HOME it already had.
            **({"HOME": _HOST_HOME} if engine.repo in _HOME_OVERRIDE_REPOS else {}),
            **engine.env_vars,
        }

        env_desc = (
            f"prefix {engine.prefix!r}"
            if engine.prefix is not None
            else f"environment {engine.env!r}"
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workdir),
                env=subprocess_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise EngineError(
                f"could not start engine {engine.repo!r} in {env_desc}: "
                f"{exc}. Check that the environment exists and that "
                f"{command[0]!r} is on PATH. Working directory preserved "
                f"for diagnosis: {workdir}"
            ) from exc

        try:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                with contextlib.suppress(asyncio.CancelledError):
                    await process.wait()
                raise EngineError(
                    f"engine {engine.repo!r} timed out after {timeout:.0f}s. "
                    "Reduce the sample count or raise the tool's timeout. "
                    f"Working directory preserved for diagnosis: {workdir}"
                ) from exc
        except BaseException:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                with contextlib.suppress(asyncio.CancelledError):
                    await process.wait()
            raise

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        if process.returncode != 0:
            lowered = stderr.lower()
            if any(marker in lowered for marker in _OOM_MARKERS):
                raise EngineError(
                    f"engine {engine.repo!r} ran out of GPU memory. Reduce the "
                    "number of samples, shorten the input, or use a smaller "
                    f"model variant.\n{_excerpt(stderr)}\n"
                    f"Working directory preserved for diagnosis: {workdir}"
                )
            raise EngineError(
                f"engine {engine.repo!r} exited with code {process.returncode}.\n"
                f"{_excerpt(stderr)}\n"
                f"Working directory preserved for diagnosis: {workdir}"
            )

        # Declared outputs must be copied out before the workdir is removed:
        # a workdir cannot be both cleaned up and the place results live. A
        # missing or ambiguous declared output, or any other collection
        # failure (permission denied, disk full, a pattern matching a
        # directory, ...), keeps the workdir (like the EngineError branches
        # above) so it can be inspected. OSError is caught rather than just
        # FileNotFoundError so every collection failure — not only a missing
        # file — produces the same diagnosable message.
        try:
            collected = collect_outputs(outputs, workdir, workdir.name)
        except OSError as exc:
            raise EngineError(
                f"engine {engine.repo!r} exited successfully but did not produce "
                f"an expected output: {exc}\n\n"
                f"Working directory preserved for diagnosis: {workdir}"
            ) from exc

        # Only a clean run's scratch directory is removed: a failed run
        # keeps its workdir (see the EngineError branches above) so it can
        # be inspected, since it may hold partial output or logs.
        shutil.rmtree(workdir, ignore_errors=True)

        return CompletedRun(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            workdir=workdir,
            outputs=collected,
        )
