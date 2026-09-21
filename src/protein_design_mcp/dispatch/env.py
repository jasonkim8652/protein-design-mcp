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
import signal
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Sequence

from protein_design_mcp.manifest.schema import EngineSpec

_OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda error: out of memory")


class EngineError(RuntimeError):
    """An engine subprocess failed, timed out, or could not be started."""


@dataclass(frozen=True)
class CompletedRun:
    returncode: int
    stdout: str
    stderr: str
    workdir: Path


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
        """Return the full argv. Pure — safe to assert on in tests."""
        prefix: list[str] = []
        if self._runner:
            prefix = [self._runner, "run", "-n", engine.env]
        return [*prefix, *engine.entry, *(str(a) for a in args)]

    def _make_workdir(self) -> Path:
        workdir = self._scratch_root / f"pdmcp-{uuid.uuid4().hex[:12]}"
        workdir.mkdir(parents=True, exist_ok=False)
        return workdir

    async def run(
        self,
        engine: EngineSpec,
        args: Sequence[Any],
        *,
        timeout: float,
    ) -> CompletedRun:
        """Execute the engine. Raises EngineError on any failure."""
        command = self.build_command(engine, args)
        workdir = self._make_workdir()

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise EngineError(
                f"could not start engine {engine.repo!r} in environment "
                f"{engine.env!r}: {exc}. Check that the environment exists "
                f"and that {command[0]!r} is on PATH."
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
                    "Reduce the sample count or raise the tool's timeout."
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
                    f"model variant.\n{stderr.strip()[-2000:]}"
                )
            raise EngineError(
                f"engine {engine.repo!r} exited with code {process.returncode}.\n"
                f"{stderr.strip()[-2000:]}"
            )

        return CompletedRun(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            workdir=workdir,
        )
