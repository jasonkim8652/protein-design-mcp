"""External sequence/backbone serialization into BoltzGen's native fold input."""
import importlib.util
import json
import os
from pathlib import Path

from Bio.PDB import MMCIFParser
import numpy as np
import pytest
import yaml

def _external_pdb(path):
    rows = []
    for i, (chain, resname) in enumerate([('A', 'ALA'), ('B', 'GLY')]):
        for j, atom in enumerate(['N', 'CA', 'C', 'O']):
            rows.append(f'ATOM  {i * 4 + j + 1:5d} {atom:^4s} {resname} {chain}{1:4d}    {float(i * 10 + j):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{20.:6.2f}          {atom[0]:>2s}\n')
    path.write_text(''.join(rows) + 'END\n')
    return str(path)



def _engine():
    path = Path(__file__).parents[1] / 'scripts/engines/boltzgen_fold.py'
    assert path.is_file(), 'fold conversion engine is required'
    spec = importlib.util.spec_from_file_location('boltzgen_fold_engine', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('compressed', [False, True])
def test_external_replacement_retains_backbone_and_target_sequence(tmp_path, compressed):
    source = _external_pdb(tmp_path / 'input.pdb')
    if compressed:
        import gzip
        compressed_path = tmp_path / 'input.pdb.gz'
        with gzip.open(compressed_path, 'wt') as handle:
            handle.write(Path(source).read_text())
        source = str(compressed_path)
    output = tmp_path / 'converted.cif'
    result = _engine().prepare_structure(source, ['B'], {'B': 'W'}, output)
    model = MMCIFParser(QUIET=True).get_structure('out', output)[0]
    assert model['A'][1].resname == 'ALA'
    assert model['B'][1].resname == 'TRP'
    assert set(atom.name for atom in model['B'][1]) == {'N', 'CA', 'C', 'O'}
    np.testing.assert_allclose(model['B'][1]['CA'].coord, [11., 0., 0.])
    assert result == {'A': 'A', 'B': 'W'}


@pytest.mark.parametrize("input_chains", [("A", "B"), ("X", "Y")])
def test_native_writer_roundtrip_preserves_sequence_design_mask_and_unresolved_sidechains(tmp_path, input_chains):
    pytest.importorskip('boltzgen')
    moldir = os.environ.get('BOLTZGEN_TEST_MOLDIR')
    if not moldir:
        pytest.skip('set BOLTZGEN_TEST_MOLDIR to run the native CPU integration check')
    source = _external_pdb(tmp_path / 'input.pdb')
    source_path = Path(source)
    source_path.write_text(source_path.read_text().replace('ALA A', 'ALA ' + input_chains[0]).replace('GLY B', 'GLY ' + input_chains[1]))
    engine = _engine()
    spec = engine.convert_external(source, [input_chains[1]], {input_chains[1]: 'W'}, tmp_path, moldir)
    mapping = json.loads((tmp_path / 'chain_mapping.json').read_text())
    assert mapping['input_to_generated'] == {input_chains[0]: 'A', input_chains[1]: 'B'}
    output = tmp_path / 'generated_files/external_0.cif'
    model = MMCIFParser(QUIET=True).get_structure('out', output)[0]
    assert [r.resname for r in model['A']] == ['ALA']
    assert [r.resname for r in model['B']] == ['TRP']
    assert set(a.name for a in model['B'][1]) == {'N', 'CA', 'C', 'O'}
    np.testing.assert_allclose(model['B'][1]['CA'].coord - model['A'][1]['CA'].coord, [10., 0., 0.])
    with np.load(output.with_suffix('.npz')) as metadata:
        assert metadata['design_mask'].tolist() == [0., 1.]
    assert yaml.safe_load(spec.read_text())['entities'] == [{'protein': {'id': 'A', 'sequence': 'A'}}, {'protein': {'id': 'B', 'sequence': 'W'}}]
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.tokenize.tokenizer import Tokenizer
    from boltzgen.task.predict.data_from_generated import FromGeneratedDataset
    for design_only, expected_length in [(False, 2), (True, 1)]:
        dataset = FromGeneratedDataset(
            generated_paths=[output], metadata_paths=[output.with_suffix('.npz')],
            native_paths=[output], moldir=Path(moldir), canonicals=load_canonicals(moldir),
            tokenizer=Tokenizer(atomize_modified_residues=False), featurizer=Featurizer(),
            target_templates=True, return_designfolding=design_only,
            extra_mol_dir=tmp_path / 'generated_files/molecules_out_dir',
        )
        feature = dataset[0]
        assert len(feature['design_mask']) == expected_length
        assert feature['design_mask'].sum().item() == 1
        assert feature['atom_resolved_mask'].sum().item() == (4 if design_only else 8)
        assert feature['res_type'].argmax(-1)[-1].item() == 19  # TRP in native token vocabulary


def test_refold_mapping_tracks_design_only_chain_renaming(tmp_path):
    mapping = tmp_path / 'chain_mapping.json'
    mapping.write_text(json.dumps({'input_to_generated': {'X': 'A', 'Y': 'B'}, 'input_design_chains': ['Y']}))
    generated = tmp_path / 'generated_files'
    generated.mkdir()
    source = _external_pdb(tmp_path / 'input.pdb')
    _engine().prepare_structure(source, ['B'], {'B': 'W'}, generated / 'external_0.cif')
    refold = tmp_path / 'refold.cif'
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('refold', generated / 'external_0.cif')
    structure[0].detach_child('A')
    structure[0]['B'].id = 'A'
    from Bio.PDB import MMCIFIO
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(refold))
    _engine().record_refold_mapping(mapping, refold, design_only=True)
    assert json.loads(mapping.read_text())['input_to_refolded'] == {'Y': 'A'}
