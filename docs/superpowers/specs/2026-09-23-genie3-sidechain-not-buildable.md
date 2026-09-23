# Why there is no `run_genie3_sidechain`

Written 2026-09-23 after building the tool and finding it cannot exist.

## The problem it was meant to solve

`predict_sidechain: true` is unreachable from both Genie 3 tools, for two
different reasons, and in each case it fails *after* the main stage has
finished — discarding a generation that was already paid for.

| tool | fails at | guard |
|---|---|---|
| `run_genie3_binder` | `workflow.py:202` | `assert config.dataset.source == "unconditional"` (binder runs with `target`) |
| `run_genie3_scaffold` | `workflow.py:204` | `assert ...sampler.predict_sequence` (pinned false: sequence design is `run_mpnn`'s job) |

That also leaves `run_genie3_binder`'s output unusable: its design chain comes
back as `UNK` with CA atoms only, which ProDy does not classify as protein, so
`run_mpnn` cannot design onto it either. The tool succeeds and produces
something nothing downstream can consume.

Splitting the side-chain pass into its own tool would have fixed both — §3.1's
own rule, since it is a step the engine performs separately — and would have
made it reachable from binder outputs, which the combined path forbids.

## Why it cannot be built

The side-chain stage reads PDBs out of `<paths.dataset>/pdbs/` and does not
care what produced them, so the stage itself is general. The obstacle is that
**`sidechain` is not a source a config can request**:

```python
# genie3/generation/config/data/sample_dataset.py
def get_sample_dataset_config(source):
    """source: Dataset type ('unconditional', 'motif', 'target')"""
    if source == "unconditional":  ...
    elif source == "motif":        ...
    elif source == "target":       ...
    else:
        logging.error(f"Invalid sample dataset source: {source}")
        exit(0)
```

`workflow.py` reaches the stage by mutating `config.dataset.source =
"sidechain"` at line 206 — *after* the config has been built, bypassing the
registry above. There is no public path to it.

Confirmed by building the wrapper and running it. Two upstream errors were
fixed along the way (`generation.dataset.datadir` is now `paths.dataset`;
checkpoint paths must be absolute because Genie 3 resolves config paths
against the process cwd), and the third is structural:

```
Invalid sample dataset source: sidechain
SystemExit: 0
```

## What was done instead

`predict_sidechain` was removed from both manifests, their adapters and their
docs, each explaining which assertion blocks it. `run_genie3_binder`'s output
description now states that its design chain is `UNK`/CA-only and therefore
invisible to `run_mpnn`, so a caller learns that before spending a GPU on it.

## What would make it possible

A patch to `get_sample_dataset_config` admitting `sidechain` with a
`{datadir, batch_size}` template. That is a change to the engine, not to this
server, and belongs upstream.
