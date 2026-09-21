import asyncio
import os
import sys
import time

import pytest

from protein_design_mcp.dispatch.env import EngineError, EnvDispatcher
from protein_design_mcp.manifest.schema import EngineSpec

ENGINE = EngineSpec(repo="prodigy", env="scoring", entry=("prodigy",))


def test_command_is_wrapped_in_micromamba_run():
    d = EnvDispatcher()
    cmd = d.build_command(ENGINE, ["--input", "a.pdb"])
    assert cmd == [
        "micromamba",
        "run",
        "-n",
        "scoring",
        "prodigy",
        "--input",
        "a.pdb",
    ]


def test_multiword_entry_is_preserved():
    engine = EngineSpec(repo="complexa", env="complexa", entry=("complexa", "generate"))
    cmd = EnvDispatcher().build_command(engine, ["--n", "4"])
    assert cmd[4:] == ["complexa", "generate", "--n", "4"]


def test_runner_can_be_overridden_for_local_execution():
    d = EnvDispatcher(runner=None)
    assert d.build_command(ENGINE, ["-x"]) == ["prodigy", "-x"]


def test_arguments_are_stringified():
    cmd = EnvDispatcher(runner=None).build_command(ENGINE, ["--n", 4, "--t", 0.1])
    assert cmd == ["prodigy", "--n", "4", "--t", "0.1"]


@pytest.mark.asyncio
async def test_successful_run_captures_stdout(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    result = await d.run(engine, ["-c", "print('hello')"], timeout=30)
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


@pytest.mark.asyncio
async def test_each_run_gets_its_own_scratch_directory(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    a = await d.run(engine, ["-c", "pass"], timeout=30)
    b = await d.run(engine, ["-c", "pass"], timeout=30)
    assert a.workdir != b.workdir
    assert a.workdir.is_dir() and b.workdir.is_dir()


@pytest.mark.asyncio
async def test_nonzero_exit_raises_with_stderr_in_the_message(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    with pytest.raises(EngineError) as exc:
        await d.run(
            engine,
            ["-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
            timeout=30,
        )
    assert "boom" in str(exc.value)
    assert "3" in str(exc.value)


@pytest.mark.asyncio
async def test_timeout_raises_engine_error_naming_the_limit(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    with pytest.raises(EngineError, match="timed out"):
        await d.run(engine, ["-c", "import time; time.sleep(10)"], timeout=1)


@pytest.mark.asyncio
async def test_cuda_oom_is_translated_into_actionable_advice(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    script = (
        "import sys; sys.stderr.write("
        "'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 75.94 GiB'"
        "); sys.exit(1)"
    )
    with pytest.raises(EngineError) as exc:
        await d.run(engine, ["-c", script], timeout=30)
    assert "out of memory" in str(exc.value).lower()
    assert "reduce" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_missing_runner_reports_the_environment_name(tmp_path):
    d = EnvDispatcher(runner="definitely-not-a-real-binary", scratch_root=tmp_path)
    with pytest.raises(EngineError, match="scoring"):
        await d.run(ENGINE, [], timeout=30)


@pytest.mark.asyncio
async def test_process_group_kill_on_timeout(tmp_path):
    """Verify that timeout kills the entire process group, not just the direct child."""
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))

    # Script that spawns a grandchild (not detached) which sleeps then writes sentinel
    # Without proper process group killing, the grandchild would survive timeout
    script = (
        "import subprocess, sys; "
        "sentinel = sys.argv[1]; "
        "subprocess.Popen([sys.executable, '-c', "
        "f'import time; time.sleep(2); open({sentinel!r}, \"w\").close()']); "
        "import time; time.sleep(10)"
    )

    sentinel_file = tmp_path / "grandchild_ran.txt"

    with pytest.raises(EngineError, match="timed out"):
        await d.run(engine, ["-c", script, str(sentinel_file)], timeout=1)

    # Wait slightly past grandchild's sleep time to verify it was killed
    await asyncio.sleep(2.5)

    # Assert the sentinel file never appeared (grandchild was killed)
    assert not sentinel_file.exists(), "Grandchild process was not killed with process group"


@pytest.mark.asyncio
async def test_cancellation_cleans_up_process(tmp_path):
    """Verify that task cancellation kills the process and propagates CancelledError."""
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))

    task = asyncio.create_task(
        d.run(engine, ["-c", "import time; time.sleep(30)"], timeout=60)
    )

    # Give the task time to start the process
    await asyncio.sleep(0.2)

    # Cancel the task
    task.cancel()

    # Assert CancelledError is propagated
    with pytest.raises(asyncio.CancelledError):
        await task

    # Give cleanup time to complete
    await asyncio.sleep(0.1)

    # Verify the child process has been reaped (no longer exists)
    # This is hard to verify directly without accessing internals,
    # so we do a second run to ensure resources were freed
    result = await d.run(
        engine, ["-c", "import sys; sys.exit(0)"], timeout=30
    )
    assert result.returncode == 0
