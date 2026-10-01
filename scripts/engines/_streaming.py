"""Stream wrapper subprocess output while retaining bounded error diagnostics.

Children share the wrapper's process group: EnvDispatcher owns timeout and
cancellation of the whole group, including workers surviving their parent.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence


def run_with_stderr_tail(command: Sequence[str]) -> subprocess.CompletedProcess:
    """Inherit stdout, tee stderr live, and return its last MiB for inspection."""
    process = subprocess.Popen(command, stderr=subprocess.PIPE)
    tail = bytearray()
    try:
        while chunk := process.stderr.read1(65536):
            sys.stderr.buffer.write(chunk)
            sys.stderr.buffer.flush()
            tail.extend(chunk)
            if len(tail) > 1024 * 1024:
                del tail[:-(1024 * 1024)]
        return subprocess.CompletedProcess(
            command, process.wait(), stderr=tail.decode("utf-8", errors="replace")
        )
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stderr.close()
