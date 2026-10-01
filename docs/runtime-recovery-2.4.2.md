# Long workspace paths and BoltzGen inverse folding

BoltzGen's upstream `--only_inverse_fold` branch ignores `--num_workers`
and leaves the data module at four workers. The adapter now also sets
`--config inverse_folding data.num_workers=0`, which reaches that branch's
actual loader configuration. Model, sequence count and design choices are unchanged.

On Linux, multiprocessing resource sharing uses Unix-domain sockets with a
107-byte path limit. Long workspace paths inherited through TMPDIR can exceed
that limit, raising `AF_UNIX path too long` in a queue feeder thread and leaving
prediction waiting without GPU activity. Engine processes now use a private
short-lived symlink for TMPDIR/TEMP/TMP when needed. The link is in `/tmp`; all
temporary file content stays in `.engine-tmp` under the configured work directory.
The link is removed after the process group finishes, including failure. Retained
logs and artifacts keep their original absolute workspace paths.

The repair keeps the published environments and weights. Existing measured
campaigns retain their pinned evaluator protocol; only the design backend changes.
Original prompts and independent scoring are unchanged.

Regression checks exercise actual multiprocessing Listener creation under a
long workspace and preservation of temporary files after success and failure.
The inverse-folding check also requires the effective step-specific override.

## Verified publication and tests

Immutable published image:

```text
jasonkim8652/protein-design-mcp:2.4.2@sha256:d98c2aaee98907b10b16ea1325f68b8c07f15591ae5abfc7c9191c06fe105cb5
```

Source revision: `5600f845bcd6c7b13c941f4972f5bf6c6982134b`. Remote manifest config matches
local image `sha256:a39fca95a3a411c1a296da98d343f1e2615aa0fecde2c3b85b8f4fc4c348af04`; immutable pull passed.
Size: 285,324,851,640 bytes uncompressed (about 266 GiB);
194,775,616,926 compressed layer bytes; 106 layers.
Existing base layers can be reused.

- Server full suite: 1,537 passed, 30 skipped.
- Focused dispatcher, long-path and BoltzGen regressions: 90 passed.
- Actual image discovers all 41 tools with the configured external mounts.
- Actual inverse folding: two generated sequences/structures in 27.6 seconds;
  the resolved `inverse_folding.yaml` has `data.num_workers: 0`.
- Full-target complex refolding of both inverse-folded outputs: 130.6 seconds
  with 3 recycles, 200 sampling steps and one diffusion sample per design.
- Container IPC probe: multiprocessing socket creation passed from a workspace
  longer than the Unix socket limit, using a 50-byte socket path. The alias was
  removed and temporary content remained in the configured workspace.

These are runtime and handoff checks, not evidence of improved candidate scores
or completion of a three-round campaign. Existing independent measurements and
original design prompts remain unchanged.
