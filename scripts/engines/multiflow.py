"""Wrapper for MultiFlow's unconditional + ProteinMPNN-codesign sampler
(`run_multiflow`).

Forwards every argv it receives, unchanged, as Hydra overrides onto
MultiFlow's own `multiflow/experiments/inference_se3_flows.py -cn
inference_unconditional` -- the adapter (`adapters/multiflow.py`) builds the
override list; this wrapper's job is (1) the crash-tolerance behaviour below
and (2) collecting the self-consistency refold's own scores when it
succeeds, neither of which can live in an adapter (both need the
subprocess's own stderr/exit code and a post-hoc filesystem check).

**Formerly a known, upstream, documented limitation -- now fixed (wave-I;
see the wave's report).** After writing `sample.pdb` and the
ProteinMPNN-codesigned sequence, MultiFlow's own inference script
UNCONDITIONALLY attempts a self-consistency ESMFold refold of the codesigned
sequence -- regardless of `inference.also_fold_pmpnn_seq` (which this tool
sets False anyway, since it only gates a SECOND, entirely optional fold).
That refold used to crash with `ModuleNotFoundError: No module named
'deepspeed'` (the `multiflow` conda env's bundled `openfold` imports it, and
it was never installed there) -- confirmed live, in the ORIGINAL `multiflow`
env, which was never modified. This tool now runs under `multiflow_fixed` (a
clone with `deepspeed` installed via the clone's own `bin/pip`, matched
against the clone's existing torch with no torch upgrade -- confirmed live),
where the refold genuinely completes: `sc_results.csv` and a real
ESMFold-refolded PDB are written per sample (confirmed live: bb_rmsd=0.637,
mean_plddt=81.96 for a real 20-residue sample). The crash-tolerance below is
kept as a DEFENSIVE fallback (e.g. if this tool is ever pointed back at an
env missing deepspeed) rather than removed: if the subprocess exits
non-zero, this checks whether the declared generation outputs (`sample.pdb`)
were nonetheless written, and if so, treats the run as a successful
GENERATION even though self-consistency scoring is then absent for that
run. A non-zero exit with NO outputs written is still a real failure and is
propagated as one.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

_INFERENCE_SCRIPT = "/home/jk661/projects/multiflow/multiflow/experiments/inference_se3_flows.py"
_PREDICT_DIR = "predict_out"  # fixed, relative -- see adapters/multiflow.py


def _collect_self_consistency(predict_dir: Path) -> dict[str, dict]:
    """(length_dir/sample_dir) -> {bb_rmsd, mean_plddt}, for every
    sc_results.csv the refold actually wrote. Absent entirely (empty dict)
    when the refold did not run for any sample -- e.g. the crash-tolerance
    branch below -- never an error: this is informational, not required.
    """
    summary: dict[str, dict] = {}
    for csv_path in predict_dir.rglob("sc_results.csv"):
        sample_dir = csv_path.parent
        key = f"{sample_dir.parent.name}/{sample_dir.name}"
        with open(csv_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        row = rows[0]
        try:
            summary[key] = {
                "bb_rmsd": float(row["bb_rmsd"]),
                "mean_plddt": float(row["mean_plddt"]),
            }
        except (KeyError, ValueError):
            # sc_results.csv exists but not in the expected shape -- skip
            # rather than fail the whole call over an optional score.
            continue
    return summary


def main() -> None:
    overrides = sys.argv[1:]
    cmd = [sys.executable, "-W", "ignore", _INFERENCE_SCRIPT, "-cn", "inference_unconditional", *overrides]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    predict_dir = Path(_PREDICT_DIR)
    samples_written = list(predict_dir.rglob("sample.pdb"))

    if proc.returncode != 0:
        if samples_written and "deepspeed" in proc.stderr.lower():
            sys.stderr.write(
                "\n[run_multiflow wrapper] engine exited non-zero from its "
                "own unconditional post-generation self-consistency refold "
                "(missing 'deepspeed' in this env), but "
                f"{len(samples_written)} sample.pdb file(s) were written "
                "before that failure. Treating this call as a successful "
                "GENERATION with no self-consistency score for this run "
                "(this env is missing deepspeed -- if you are on "
                "multiflow_fixed and see this, the clone's deepspeed "
                "install may have been reverted; see the manifest doc).\n"
            )
            (Path.cwd() / "self_consistency_summary.json").write_text(json.dumps({}, indent=2))
            return
        sys.exit(proc.returncode)

    # Success path: the refold ran (env has deepspeed) -- write whatever
    # self-consistency scores it produced. Always written, even if empty,
    # so the manifest's declared output exists on every successful call.
    summary = _collect_self_consistency(predict_dir)
    (Path.cwd() / "self_consistency_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
