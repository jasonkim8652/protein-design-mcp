"""Wrapper for MultiFlow's unconditional + ProteinMPNN-codesign sampler
(`run_multiflow`).

Forwards every argv it receives, unchanged, as Hydra overrides onto
MultiFlow's own `multiflow/experiments/inference_se3_flows.py -cn
inference_unconditional` -- the adapter (`adapters/multiflow.py`) builds the
override list; this wrapper's only job is the crash-tolerance behaviour
below, which cannot live in an adapter (it needs the subprocess's own
stderr and a post-hoc filesystem check).

**Known, upstream, documented limitation (confirmed live in
docs/superpowers/reviews/2026-09-22-gpu-engine-survey.md #10):** after
writing `sample.pdb` and the ProteinMPNN-codesigned sequence, MultiFlow's
own inference script UNCONDITIONALLY attempts a self-consistency ESMFold
refold of the codesigned sequence -- regardless of
`inference.also_fold_pmpnn_seq` (which this tool sets False anyway, since it
only gates a SECOND, entirely optional fold). That refold crashes with
`ModuleNotFoundError: No module named 'deepspeed'` in every environment on
this host, because the `multiflow` conda env's bundled `openfold` imports
`deepspeed`, which was never installed here. This is a real, upstream gap in
the environment, not something this wrapper can fix -- fixing it would mean
installing a large, GPU-build-sensitive dependency this wave has no mandate
to add. What this wrapper CAN do is not let a scoring step this tool never
asked for turn a successful generation into a reported failure: if the
subprocess exits non-zero, this checks whether the declared generation
outputs (`sample.pdb`) were nonetheless written, and if so, treats the run
as a success (this tool exposes generation only -- the self-consistency
refold's OWN output, had it worked, would not have been surfaced as this
tool's result anyway). A non-zero exit with NO outputs written is still a
real failure and is propagated as one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_INFERENCE_SCRIPT = "/home/jk661/projects/multiflow/multiflow/experiments/inference_se3_flows.py"
_PREDICT_DIR = "predict_out"  # fixed, relative -- see adapters/multiflow.py


def main() -> None:
    overrides = sys.argv[1:]
    cmd = [sys.executable, "-W", "ignore", _INFERENCE_SCRIPT, "-cn", "inference_unconditional", *overrides]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    samples_written = list(Path(_PREDICT_DIR).rglob("sample.pdb"))

    if proc.returncode != 0:
        if samples_written and "deepspeed" in proc.stderr.lower():
            sys.stderr.write(
                "\n[run_multiflow wrapper] engine exited non-zero from its "
                "own unconditional post-generation self-consistency refold "
                "(missing 'deepspeed' in this env -- documented, upstream "
                f"limitation), but {len(samples_written)} sample.pdb file(s) "
                "were written before that failure. Treating this call as a "
                "successful GENERATION (this tool does not expose the "
                "refold step).\n"
            )
            return
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
