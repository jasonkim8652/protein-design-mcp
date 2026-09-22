"""Wrapper for RFdiffusion 1.1.0 (`run_rfdiffusion_binder`). Runs inside the
`SE3nv` environment, against the checkout at
`/file_server/data/jk661/pioneer/RFdiffusion` (see the manifest's `engine`
comment for why this needs an explicit PYTHONPATH rather than a plain
`import rfdiffusion`: the package's own editable-install metadata points at
a path that no longer exists on this host).

Two host-specific problems are worked around here, both CONFIRMED LIVE
(2026-09-22):

1. The stale editable install (see above) -- worked around by resolving
   `run_inference.py`'s location from the ALREADY-importable `rfdiffusion`
   package (which only imports at all because the manifest sets
   `PYTHONPATH` to the real checkout), rather than hardcoding the path a
   second time here.
2. `nvrtc: error: invalid value for --gpu-architecture (-arch)`, raised
   from e3nn's fused spherical-harmonics kernel the first time the SE(3)
   network runs on this host's L40S GPUs (sm_89) under this environment's
   CUDA 11.1 toolkit. Disabling PyTorch's JIT GPU fusion BEFORE importing
   anything that builds the network (the `torch._C._jit_*` calls below)
   avoids it; the `PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1` env var the manifest
   also sets was NOT sufficient on its own in testing -- both together are
   what was verified working.

Receives RFdiffusion's own Hydra-style `key=value` overrides as plain argv
(built by `adapters.rfdiffusion_binder.build_args`) and forwards them to
`run_inference.py` unchanged, with one exception: a relative
`inference.ckpt_override_path=...` value is rewritten to an absolute path
under the resolved repo root first. RFdiffusion resolves that path against
the subprocess's OWN cwd (this run's scratch workdir), not the repo, so a
relative override like `models/Complex_beta_ckpt.pt` -- the form the
adapter emits, since it has no reason to know the repo's absolute path --
would otherwise look for the checkpoint inside the empty scratch directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

import runpy  # noqa: E402  (must follow the JIT-fusion calls above)

import rfdiffusion  # noqa: E402  (import only succeeds once PYTHONPATH is set)

_REPO_ROOT = Path(rfdiffusion.__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "run_inference.py"
_CKPT_OVERRIDE_KEY = "inference.ckpt_override_path="


def _resolve_ckpt_override(arg: str) -> str:
    if not arg.startswith(_CKPT_OVERRIDE_KEY):
        return arg
    value = arg[len(_CKPT_OVERRIDE_KEY) :]
    if value and not value.startswith("/"):
        value = str(_REPO_ROOT / value)
    return f"{_CKPT_OVERRIDE_KEY}{value}"


def main() -> None:
    overrides = [_resolve_ckpt_override(a) for a in sys.argv[1:]]
    sys.argv = [str(_SCRIPT_PATH), *overrides]
    runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")


if __name__ == "__main__":
    main()
