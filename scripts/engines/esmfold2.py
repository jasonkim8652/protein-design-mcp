"""Fold one sequence with ESMFold2. Runs inside the `esmfold2` environment.

Prints `key: value` lines the adapter parses, and writes the predicted
structure to the path given as the second argument, relative to the working
directory the dispatcher created.

CRITICAL: this environment has a package-shadowing trap (see
run_esmfold2.yaml's `engine.env_vars.PYTHONNOUSERSITE` comment) -- without
PYTHONNOUSERSITE=1, `import esm` silently resolves to the WRONG package
(fair-esm 2.0.0 from ~/.local, Meta's old ESMFold v1) instead of this
environment's own esm==3.4.0 (EvolutionaryScale SDK, the real ESMFold2). The
check below verifies the loaded package actually lives inside THIS
environment before doing anything else, and fails loudly if not -- silently
running the wrong model is worse than crashing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _verify_correct_esm_package() -> None:
    import esm

    esm_location = Path(esm.__file__).resolve()
    env_prefix = Path(sys.prefix).resolve()
    if env_prefix not in esm_location.parents:
        raise RuntimeError(
            f"wrong 'esm' package loaded: {esm_location} is NOT inside this "
            f"environment's own prefix ({env_prefix}). This almost always "
            "means PYTHONNOUSERSITE=1 was not honored and "
            "~/.local/lib/python3.12/site-packages' fair-esm 2.0.0 (Meta's "
            "ESMFold v1) shadowed the environment's real esm==3.4.0 "
            "(EvolutionaryScale SDK, esm.models.esmfold2). Refusing to "
            "silently run the wrong model."
        )
    try:
        from esm.models.esmfold2 import EsmFold2Model  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"esm package at {esm_location} has no esm.models.esmfold2 -- "
            "this is not the EvolutionaryScale esm==3.4.0 SDK this tool "
            "requires."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("output_pdb")
    parser.add_argument("--num-recycles", type=int, required=True)
    parser.add_argument("--num-diffusion-samples", type=int, required=True)
    parser.add_argument("--num-sampling-steps", type=int, required=True)
    args = parser.parse_args()

    _verify_correct_esm_package()

    from esm.models.esmfold2 import EsmFold2Model

    model = EsmFold2Model.from_pretrained(
        "biohub/ESMFold2", device="cuda", local_files_only=True
    )

    output = model.infer_protein(
        args.sequence,
        num_loops=args.num_recycles,
        num_diffusion_samples=args.num_diffusion_samples,
        num_sampling_steps=args.num_sampling_steps,
    )
    pdb_str = model.output_to_pdb(output)

    with open(args.output_pdb, "w") as handle:
        handle.write(pdb_str)

    # Read the per-residue pLDDT back out of the CA atoms' own B-factor
    # column, which output_to_pdb() has already computed and masked
    # correctly -- reusing that instead of re-deriving the same masking
    # logic from the raw output tensors here.
    ca_bfactors: list[float] = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            ca_bfactors.append(float(line[60:66]))

    if not ca_bfactors:
        raise RuntimeError("ESMFold2 produced a PDB with no CA atoms")

    mean_plddt = sum(ca_bfactors) / len(ca_bfactors)

    print(f"mean_plddt: {mean_plddt:.4f}")
    print(f"num_residues: {len(ca_bfactors)}")
    print(f"sequence_length: {len(args.sequence)}")
    print(f"output_pdb: {args.output_pdb}")


if __name__ == "__main__":
    main()
