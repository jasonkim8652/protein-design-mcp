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
