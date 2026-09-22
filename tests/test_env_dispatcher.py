import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from protein_design_mcp.dispatch.env import EngineError, EnvDispatcher
from protein_design_mcp.manifest.schema import EngineSpec, OutputSpec

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


@pytest.mark.asyncio
async def test_successful_run_removes_its_scratch_directory(tmp_path):
    """Regression: workdirs were never cleaned up after a successful run
    (verified live: pdmcp-321bf67fcbcd left behind after a run)."""
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    result = await d.run(engine, ["-c", "pass"], timeout=30)
    assert not result.workdir.exists()


@pytest.mark.asyncio
async def test_failed_run_preserves_its_scratch_directory_for_diagnosis(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    with pytest.raises(EngineError) as exc:
        await d.run(
            engine,
            ["-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
            timeout=30,
        )
    assert "preserved for diagnosis" in str(exc.value)
    workdirs = list(tmp_path.glob("pdmcp-*"))
    assert len(workdirs) == 1
    assert workdirs[0].is_dir()


@pytest.mark.asyncio
async def test_timed_out_run_preserves_its_scratch_directory_for_diagnosis(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    with pytest.raises(EngineError, match="preserved for diagnosis"):
        await d.run(engine, ["-c", "import time; time.sleep(10)"], timeout=1)
    workdirs = list(tmp_path.glob("pdmcp-*"))
    assert len(workdirs) == 1
    assert workdirs[0].is_dir()


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
    """Verify that task cancellation kills the process and propagates CancelledError.

    This test actively verifies the cancelled process is dead, not just inferred from
    dispatcher availability. If os.killpg() is removed from the except BaseException
    block, this test will fail.
    """
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))

    pid_file = tmp_path / "child_pid.txt"

    # Script that writes its own pid to a file then sleeps
    script = (
        "import os, time; "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )

    task = asyncio.create_task(
        d.run(engine, ["-c", script], timeout=60)
    )

    # Give the task time to start the process and write its pid
    await asyncio.sleep(0.3)

    # Cancel the task
    task.cancel()

    # Assert CancelledError is propagated
    with pytest.raises(asyncio.CancelledError):
        await task

    # Give cleanup time to complete
    await asyncio.sleep(0.1)

    # Read the child's pid from the file
    assert pid_file.exists(), "Child process did not write pid file"
    child_pid = int(pid_file.read_text().strip())

    # Verify the child process is actually dead by attempting to send signal 0
    # (signal 0 checks if the process exists without sending any signal)
    # If the process is dead, this raises ProcessLookupError
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


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


def test_new_workdir_creates_a_real_directory_under_scratch_root(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    workdir = d.new_workdir()
    assert workdir.is_dir()
    assert workdir.parent == tmp_path


@pytest.mark.asyncio
async def test_run_uses_a_pre_made_workdir_instead_of_creating_its_own(tmp_path):
    """The staging use case: a caller creates the workdir first (via
    new_workdir()), copies input files into it, THEN calls run() — run()
    must execute inside that exact directory, not a fresh one."""
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    workdir = d.new_workdir()
    (workdir / "staged_input.txt").write_text("pre-staged")

    result = await d.run(
        engine,
        ["-c", "print(open('staged_input.txt').read())"],
        timeout=30,
        workdir=workdir,
    )

    assert result.workdir == workdir
    assert result.stdout.strip() == "pre-staged"


@pytest.mark.asyncio
async def test_a_pre_made_workdir_is_still_preserved_on_failure(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    workdir = d.new_workdir()

    with pytest.raises(EngineError, match="preserved for diagnosis"):
        await d.run(
            engine,
            ["-c", "import sys; sys.exit(1)"],
            timeout=30,
            workdir=workdir,
        )

    assert workdir.exists()


@pytest.mark.asyncio
async def test_a_pre_made_workdir_is_still_removed_on_success(tmp_path):
    d = EnvDispatcher(runner=None, scratch_root=tmp_path)
    engine = EngineSpec(repo="py", env="unused", entry=(sys.executable,))
    workdir = d.new_workdir()

    result = await d.run(engine, ["-c", "pass"], timeout=30, workdir=workdir)

    assert result.workdir == workdir
    assert not workdir.exists()
