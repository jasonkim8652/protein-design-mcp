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
from tempfile import TemporaryDirectory, gettempdir
from typing import Any

from protein_design_mcp.manifest.schema import EngineSpec, OutputSpec
from protein_design_mcp.results import collect_outputs

_OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda error: out of memory")

# Model and database locations are explicit per-engine variables. Preserve
# the caller's writable HOME; it is not a checkpoint lookup mechanism.


class EngineError(RuntimeError):
    """An engine subprocess failed, timed out, or could not be started."""

    def __init__(
        self, message: str, *, execution_artifacts: dict[str, str] | None = None,
        error_kind: str = "engine_error",
    ):
        super().__init__(message)
        self.execution_artifacts = execution_artifacts or {}
        self.error_kind = error_kind


@dataclass(frozen=True)
class CompletedRun:
    returncode: int
    stdout: str
    stderr: str
    workdir: Path
    outputs: dict[str, str | list[str]] = field(default_factory=dict)
    execution_artifacts: dict[str, str] = field(default_factory=dict)


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


class _OutputCapture:
    """Bound returned diagnostics while preserving complete retained log files.

    Keep the first and last MiB: adapters parse summary records near either
    end, and errors often put their cause before a long command-line tail.
    """

    _SIDE_LIMIT = 1024 * 1024

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.size = 0
        self.oom = False
        self.argument_error = False
        self._overlap = b"\n"

    def append(self, chunk: bytes) -> None:
        combined = self._overlap + chunk
        self.argument_error |= b"\nPROTEIN_MCP_ARGUMENT_ERROR:" in combined
        lowered = combined.lower()
        self.oom |= any(marker.encode() in lowered for marker in _OOM_MARKERS)
        self._overlap = combined[-64:]
        self.size += len(chunk)
        remaining = self._SIDE_LIMIT - len(self.head)
        self.head.extend(chunk[:remaining])
        self.tail.extend(chunk[remaining:])
        if len(self.tail) > self._SIDE_LIMIT:
            del self.tail[:-self._SIDE_LIMIT]

    def text(self) -> str:
        omitted = self.size - len(self.head) - len(self.tail)
        separator = f"\n... [{omitted} bytes omitted] ...\n".encode() if omitted else b""
        return (self.head + separator + self.tail).decode("utf-8", errors="replace")


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
        removed on success — is identical either way. Set
        PROTEIN_MCP_KEEP_WORKDIR=1 to retain all intermediates and full logs
        on both successful and failed calls for campaign archival.
        """
        if workdir is None:
            workdir = self._make_workdir()
        keep_workdir = os.environ.get("PROTEIN_MCP_KEEP_WORKDIR", "").lower() in {
            "1", "true", "yes",
        }
        artifacts = {"workdir": str(workdir)} if keep_workdir else {}
        # Tee subprocess pipes to disk, including partial output before a
        # timeout or cancellation. Keeping pipes preserves descendant lifecycle
        # semantics: the call cannot finish while children still hold them open.
        with contextlib.ExitStack() as log_files:
            # multiprocessing creates AF_UNIX sockets under tempfile.gettempdir().
            # Linux limits socket paths to 107 bytes, so a perfectly valid mounted
            # workspace can make worker queue transport fail silently. Keep all
            # temporary content in the retained workdir, accessed through a short,
            # private symlink for the lifetime of this process group only.
            try:
                temp_root = workdir / ".engine-tmp"
                temp_root.mkdir(exist_ok=True)
                if os.name == "posix" and len(os.fsencode(temp_root)) > 50:
                    alias_root = Path(log_files.enter_context(
                        TemporaryDirectory(prefix="pm-", dir="/tmp")
                    ))
                    alias = alias_root / "t"
                    alias.symlink_to(temp_root.resolve(), target_is_directory=True)
                    temp_root = alias
            except OSError as exc:
                raise EngineError(
                    f"could not prepare engine temporary directory: {exc}",
                    execution_artifacts=artifacts, error_kind="infrastructure_error",
                ) from exc
            stdout_target = stderr_target = asyncio.subprocess.PIPE
            if keep_workdir:
                for name in ("stdout", "stderr"):
                    artifacts[name] = str(workdir / f"engine.{name}.log")
                try:
                    # Unbuffered files fail at the write that exhausts storage,
                    # not later during close, which could mask the engine error.
                    stdout_target = log_files.enter_context(open(artifacts["stdout"], "wb", buffering=0))
                    stderr_target = log_files.enter_context(open(artifacts["stderr"], "wb", buffering=0))
                except OSError as exc:
                    raise EngineError(
                        f"could not open engine logs: {exc}. Working directory: {workdir}",
                        execution_artifacts=artifacts, error_kind="infrastructure_error",
                    ) from exc
            try:
                return await self._run(
                    engine, args, timeout=timeout, outputs=outputs,
                    workdir=workdir, execution_artifacts=artifacts,
                    stdout_target=stdout_target, stderr_target=stderr_target,
                    temp_root=temp_root,
                )
            except EngineError as exc:
                exc.execution_artifacts = artifacts
                raise

    async def _run(
        self, engine: EngineSpec, args: Sequence[Any], *, timeout: float,
        outputs: Sequence[OutputSpec], workdir: Path,
        execution_artifacts: dict[str, str], stdout_target: Any, stderr_target: Any,
        temp_root: Path,
    ) -> CompletedRun:
        command = self.build_command(engine, args)

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
            # JIT output stays writable even when model stores are read-only.
            "TRITON_CACHE_DIR": str(cache_dir / "triton"),
            # Applies to wrappers and their Python descendants, even through pipes.
            "PYTHONUNBUFFERED": "1",
            **engine.env_vars,
            # The alias controls only path spelling; bytes stay in this workdir.
            "TMPDIR": str(temp_root),
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
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

        stdout_capture, stderr_capture = _OutputCapture(), _OutputCapture()

        async def capture(stream, destination, output):
            log_error = None
            while chunk := await stream.read(65536):
                output.append(chunk)
                if execution_artifacts and log_error is None:
                    try:
                        remaining = memoryview(chunk)
                        while remaining:
                            written = destination.write(remaining)
                            if not written:
                                raise OSError("engine log write made no progress")
                            remaining = remaining[written:]
                        destination.flush()
                    except OSError as exc:
                        log_error = exc
                        # Stop workers immediately, but keep consuming both
                        # pipes: abandoning a full pipe can prevent wait() from
                        # completing even after the process has been killed.
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
            if log_error is not None:
                raise EngineError(
                    f"could not write engine log: {log_error}. "
                    f"Working directory preserved for diagnosis: {workdir}",
                    error_kind="infrastructure_error",
                ) from log_error

        # Shield the readers so timeout/cancellation kills the group first and
        # then drains its final diagnostics. This also retrieves every task's
        # result instead of leaking cancelled gather futures into the event loop.
        readers = [
            asyncio.create_task(capture(process.stdout, stdout_target, stdout_capture)),
            asyncio.create_task(capture(process.stderr, stderr_target, stderr_capture)),
            asyncio.create_task(process.wait()),
        ]
        completion = asyncio.gather(*readers)
        try:
            await asyncio.wait_for(asyncio.shield(completion), timeout=timeout)
        except BaseException as exc:
            # The leader can exit before its workers; always kill the group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.gather(*readers, return_exceptions=True)
            # Consume the original gather's exception as well.
            with contextlib.suppress(BaseException):
                await completion
            if isinstance(exc, asyncio.TimeoutError):
                raise EngineError(
                    f"engine {engine.repo!r} timed out after {timeout:g}s. "
                    "Reduce the sample count or raise the tool's timeout. "
                    f"\n{_excerpt(stderr_capture.text())}\n"
                    f"Working directory preserved for diagnosis: {workdir}",
                    error_kind="engine_timeout",
                ) from exc
            raise

        stdout = stdout_capture.text()
        stderr = stderr_capture.text()

        if process.returncode != 0:
            if stderr_capture.oom:
                raise EngineError(
                    f"engine {engine.repo!r} ran out of GPU memory. Reduce the "
                    "number of samples, shorten the input, or use a smaller "
                    f"model variant.\n{_excerpt(stderr)}\n"
                    f"Working directory preserved for diagnosis: {workdir}"
                )
            raise EngineError(
                f"engine {engine.repo!r} exited with code {process.returncode}.\n"
                f"{_excerpt(stderr)}\n"
                f"Working directory preserved for diagnosis: {workdir}",
                error_kind="argument_validation" if stderr_capture.argument_error else "engine_error",
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
        if not execution_artifacts:
            shutil.rmtree(workdir, ignore_errors=True)

        return CompletedRun(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            workdir=workdir,
            outputs=collected,
            execution_artifacts=execution_artifacts,
        )
