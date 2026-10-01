#!/usr/bin/env python
"""Serialize an external designed backbone with native BoltzGen APIs, then fold.

Conversion runs on CPU without a model, checkpoint, or generated coordinates.
The native YAML parser/featurizer derives metadata and DesignWriter serializes
it. Only the subsequent ordinary folding pipeline invokes the confidence model.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser
from Bio.SeqUtils import seq1, seq3
import yaml


def prepare_structure(structure, design_chains, sequences, output):
    """Attach supplied sequences to observed backbones, discarding old sidechains."""
    path = Path(structure)
    parser = PDBParser(QUIET=True) if str(path).lower().endswith(('.pdb', '.pdb.gz')) else MMCIFParser(QUIET=True)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as handle:
        parsed = parser.get_structure('external', handle)
    models = list(parsed.get_models())
    if len(models) != 1:
        raise ValueError('exactly one structure model is required')
    model = models[0]
    if not design_chains or len(set(design_chains)) != len(design_chains) or set(design_chains) != set(sequences):
        raise ValueError('designed_sequences must map exactly the unique design_chains')
    for chain_id in design_chains:
        residues = [r for r in model[chain_id] if r.id[0] == ' ']
        sequence = sequences[chain_id]
        if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY') or len(sequence) != len(residues):
            raise ValueError(f'invalid designed sequence or length for chain {chain_id}')
        for residue, aa in zip(residues, sequence):
            if not all(atom in residue for atom in ('N', 'CA', 'C', 'O')):
                raise ValueError(f'chain {chain_id} requires complete N/CA/C/O backbone')
            residue.resname = seq3(aa).upper()
            for atom in list(residue):
                if atom.name not in ('N', 'CA', 'C', 'O'):
                    residue.detach_child(atom.id)
    all_sequences = {}
    for chain in model:
        residues = list(chain)
        if any(r.id[0] != ' ' or seq1(r.resname) == 'X' for r in residues):
            raise ValueError('external folding supports canonical protein chains only; use native mode for ligands or modified residues')
        all_sequences[chain.id] = ''.join(seq1(r.resname) for r in residues)
    writer = MMCIFIO()
    writer.set_structure(parsed)
    writer.save(str(output))
    return all_sequences


def convert_external(structure, design_chains, sequences, workdir, moldir):
    """Round-trip actual parsed structure features through BoltzGen's writer."""
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.task.predict.data_from_yaml import Dataset, PredictionDataset, collate
    from boltzgen.task.predict.writer import DesignWriter

    workdir = Path(workdir)
    prepared = workdir / 'external_input.cif'
    all_sequences = prepare_structure(structure, design_chains, sequences, prepared)
    # Biopython writes atom_site only. Gemmi supplies the polymer/entity tables
    # required by BoltzGen's native mmCIF parser, from these observed residues.
    import gemmi
    converted = gemmi.read_structure(str(prepared))
    # BoltzGen reads label_asym_id; Biopython otherwise gives that field A/B
    # even when auth_asym_id is X/Y. Keep both in the caller's namespace here.
    for chain in converted[0]:
        for residue in chain:
            residue.subchain = chain.name
    converted.setup_entities()
    for chain in converted[0]:
        entity = converted.get_entity_of(chain.get_polymer())
        entity.full_sequence = [res.name for res in chain]
    converted.assign_label_seq_id()
    converted.make_mmcif_document().write_file(str(prepared))
    input_spec = workdir / 'external_input.yaml'
    input_spec.write_text(yaml.safe_dump({'entities': [{'file': {
        'path': str(prepared.resolve()),
        'design': [{'chain': {'id': chain}} for chain in design_chains],
    }}]}, sort_keys=False))
    dataset = PredictionDataset(
        dataset=Dataset(yaml_path=str(input_spec), tokenizer=Tokenizer(atomize_modified_residues=False), featurizer=Featurizer()),
        canonicals=load_canonicals(moldir), moldir=Path(moldir), design=False,
        extra_features=["structure"],
    )
    features = dataset[0]
    input_structure = features.pop('structure')
    # Native writers preserve asym_id order but assign their own chain names.
    ordered_sequences = {str(chain['name']): all_sequences[str(chain['name'])]
                         for chain in input_structure.chains}
    batch = collate([features])
    # Writer accepts the same features produced by the model; here serialize
    # the observed input features directly. No inference or coordinate synthesis.
    prediction = dict(batch)
    prediction['coords'] = batch['coords'][0]
    prediction['exception'] = False
    writer = DesignWriter(output_dir=str(workdir / 'generated_files'), res_atoms_only=False,
                          atom14=False, write_native=False)
    writer.write_on_batch_end(prediction=prediction, batch=batch, sample_id='external')
    if writer.failed or not (workdir / 'generated_files/external_0.npz').is_file():
        raise RuntimeError('BoltzGen failed to serialize the external design')
    output_cif = workdir / 'generated_files/external_0.cif'
    chain_mapping = _map_serialized_chains(ordered_sequences, output_cif)
    (workdir / 'chain_mapping.json').write_text(json.dumps({
        'input_to_generated': chain_mapping,
        'input_design_chains': list(design_chains),
    }, indent=2) + '\n')
    # The CLI validates this portable spec even for fold/analyze. Actual design
    # identities and conditioning remain in the native CIF/NPZ, not this spec.
    spec = workdir / 'design_spec.yaml'
    spec.write_text(yaml.safe_dump({'entities': [
        {'protein': {'id': chain_mapping[chain], 'sequence': sequence}}
        for chain, sequence in ordered_sequences.items()
    ]}, sort_keys=False))
    return spec


def _map_serialized_chains(input_sequences, output_cif):
    """Match the native writer's preserved chain order, checking each sequence."""
    model = MMCIFParser(QUIET=True).get_structure('serialized', output_cif)[0]
    output_sequences = {chain.id: ''.join(seq1(r.resname) for r in chain) for chain in model}
    if list(input_sequences.values()) != list(output_sequences.values()):
        raise ValueError('Native serialized chain count/order/sequences differ from the input')
    return dict(zip(input_sequences, output_sequences))


def record_refold_mapping(mapping_path, refold_cif, design_only):
    """Record actual refold names, including chain renaming after target removal."""
    mapping_path = Path(mapping_path)
    mapping = json.loads(mapping_path.read_text())
    generated = MMCIFParser(QUIET=True).get_structure(
        'generated', mapping_path.parent / 'generated_files/external_0.cif')[0]
    input_sequences = {
        original: ''.join(seq1(r.resname) for r in generated[output])
        for original, output in mapping['input_to_generated'].items()
        if not design_only or original in mapping['input_design_chains']
    }
    mapping['input_to_refolded'] = _map_serialized_chains(input_sequences, refold_cif)
    mapping_path.write_text(json.dumps(mapping, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('design_spec', nargs='?')
    parser.add_argument('--structure')
    parser.add_argument('--design-chains')
    parser.add_argument('--designed-sequences')
    parser.add_argument('--passthrough', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.structure:
        from boltzgen.cli.boltzgen import get_artifact_path
        reference = args.passthrough[args.passthrough.index('--moldir') + 1]
        moldir = get_artifact_path(SimpleNamespace(force_download=False, models_token=None, cache=None), reference, repo_type='dataset')
        spec = convert_external(args.structure, json.loads(args.design_chains),
                                json.loads(args.designed_sequences), Path.cwd(), moldir)
    else:
        original = Path(args.design_spec).resolve()
        content = yaml.safe_load(original.read_text())
        for entity in content.get('entities', []):
            body = entity.get('file')
            if isinstance(body, dict) and body.get('path') and not Path(body['path']).is_absolute():
                body['path'] = str((original.parent / body['path']).resolve())
        spec = Path.cwd() / 'design_spec.yaml'
        spec.write_text(yaml.safe_dump(content, sort_keys=False))
    result = subprocess.run(['boltzgen', 'run', str(spec), *args.passthrough])
    if args.structure and result.returncode == 0:
        step = args.passthrough[args.passthrough.index('--steps') + 1]
        design_only = step == 'design_folding'
        folder = 'refold_design_cif' if design_only else 'refold_cif'
        record_refold_mapping(Path.cwd() / 'chain_mapping.json',
                              Path.cwd() / 'generated_files' / folder / 'external_0.cif',
                              design_only)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
