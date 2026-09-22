"""Wrapper for ESM2-650M masked-marginal pseudo-log-likelihood scoring
(``run_esm_score``). Runs inside the ``esm_env`` environment (``esm`` ==
fair-esm, torch 2.4.0+cu121).

For each position in the input sequence, that position's token is replaced
with ``<mask>`` and the model's log-probability of the TRUE residue at that
position, under the masked context, is recorded (the standard "pseudo-
likelihood" definition for a masked language model -- NOT the cheaper
single-forward-pass "wildtype marginal" approximation some benchmark scripts
use instead; see the manifest's doc for the distinction). Masked variants
are batched (``batch_size`` positions per forward pass) purely for speed --
each position is still scored independently, with only that one position
masked, so batching never changes the result.

Reads one argv: a JSON object ``{"sequence": str, "batch_size": int}`` (see
``adapters/esm_score.py`` for its exact shape). Prints one JSON object as
the LAST line of stdout; nothing is written to disk (this tool declares no
``outputs:`` -- there is nothing to collect).
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    job = json.loads(sys.argv[1])
    sequence: str = job["sequence"]
    batch_size: int = job["batch_size"]

    import torch
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("seq", sequence)])
    tokens = tokens.to(device)

    length = len(sequence)
    mask_idx = alphabet.mask_idx
    # Token layout is [BOS, residue_0, residue_1, ..., residue_{L-1}, EOS] --
    # residue i sits at token index i + 1.
    true_token_ids = [tokens[0, i + 1].item() for i in range(length)]

    per_residue_ll: list[float] = [0.0] * length
    with torch.no_grad():
        for start in range(0, length, batch_size):
            end = min(start + batch_size, length)
            chunk = end - start

            batch_tokens = tokens.repeat(chunk, 1).clone()
            for row, position in enumerate(range(start, end)):
                batch_tokens[row, position + 1] = mask_idx

            logits = model(batch_tokens)["logits"]  # (chunk, L+2, vocab)
            log_probs = torch.log_softmax(logits, dim=-1)

            for row, position in enumerate(range(start, end)):
                per_residue_ll[position] = log_probs[
                    row, position + 1, true_token_ids[position]
                ].item()

    pseudo_log_likelihood = sum(per_residue_ll) / length

    result = {
        "pseudo_log_likelihood": pseudo_log_likelihood,
        "per_residue_log_likelihood": per_residue_ll,
        "sequence_length": length,
        "device": device,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
