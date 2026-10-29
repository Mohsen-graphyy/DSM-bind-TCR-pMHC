from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from contextlib import nullcontext
from unittest.mock import patch
from argparse import Namespace
from pathlib import Path

import torch

from bindenergy.apps.tcr_pmhc.train import check_head_gradients
from bindenergy.data.tcr_pmhc import initialize_dataset_database


class ReferencePilotTests(unittest.TestCase):
    def test_training_records_baseline_and_keeps_initial_if_no_improvement(self):
        from bindenergy.apps.tcr_pmhc.train import train_model
        class ToyEnergy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.W_o = torch.nn.Linear(1, 1)
                self.U_o = torch.nn.Linear(1, 1)
            def forward(self, binder, target, use_sidechain=True, return_diagnostics=False):
                prediction = (self.W_o(torch.ones(1)) + self.U_o(torch.ones(1))).sum()
                signal = (prediction - 1).square()
                loss = signal + 1
                return {'loss': loss, 'signal_loss': signal, 'zero_force_loss': loss.new_tensor(1.)} if return_diagnostics else loss
        class Dataset:
            def __init__(self, *args):
                pass
            def __len__(self):
                return 2
            def iter_batches(self, batch_size, indices=None):
                for i in (indices if indices is not None else range(2)):
                    yield [i]
            def close(self):
                pass
        class Embeddings:
            metadata = {'dimension': 1}
            def __init__(self, *args):
                pass
            def require_sequences(self, *args):
                pass
            def close(self):
                pass
        for learning_rate, intermediate_only in ((0., False), (.001, False), (.001, True)):
            with self.subTest(learning_rate=learning_rate, intermediate_only=intermediate_only), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = ToyEnergy()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(.1)
                architecture = Namespace(esm_size=1)
                checkpoint = (model.state_dict(), {}, architecture)
                init = root / 'original.pt'
                torch.save(checkpoint, init)
                (root / 'train.sqlite').write_bytes(b'train fixture')
                (root / 'validation.sqlite').write_bytes(b'validation fixture')
                args = Namespace(learning_rate=learning_rate, anneal_rate=1., resume=None,
                    train_db=root / 'train.sqlite', validation_db=root / 'validation.sqlite', test_db=None,
                    embeddings=Path('embeddings.sqlite'), chain_cache_size=1, embedding_cache_size=1,
                    batch_size=1, epochs=1, patience=0, seed=7, validation_batches=0, test_batches=0,
                    no_sidechain=False, max_train_pairs=None, clip_norm=1., checkpoint_every_batches=1,
                    output_dir=root, init_checkpoint=init, recurrence_backend='reference', validate_every_batches=1)
                evaluations = iter((2., 1.5, 2.5))
                def controlled_validation(*positional, **keywords):
                    value = next(evaluations)
                    details = keywords.get('details', positional[9] if len(positional) > 9 else None)
                    details.update(loss=value, zero_force_loss=1.)
                    return value
                with patch('bindenergy.apps.tcr_pmhc.train.TCRpMHCPairDataset', Dataset), \
                     patch('bindenergy.apps.tcr_pmhc.train.EmbeddingStore', Embeddings), \
                     patch('bindenergy.apps.tcr_pmhc.train.make_model_batch', return_value=(None, None)), \
                     (patch('bindenergy.apps.tcr_pmhc.train.evaluate_loss', side_effect=controlled_validation) if intermediate_only else nullcontext()):
                    self.assertEqual(train_model(args, model, architecture, checkpoint, torch.device('cpu')), 0)
                report = json.loads((root / 'final_metrics.json').read_text())
                self.assertEqual(report['best_checkpoint_is_initial'], learning_rate == 0)
                self.assertEqual(report['baseline_zero_force_loss'], 1.)
                self.assertTrue((root / 'head_gradient_checks.json').is_file())
                if learning_rate:
                    self.assertLess(report['best_validation_loss'], report['baseline_validation_loss'])
                    self.assertTrue(all(value > 0 for value in report['best_head_max_abs_updates'].values()))
                if intermediate_only:
                    from bindenergy.apps.tcr_pmhc.train import torch_load
                    best = torch_load(root / 'best.pt', torch.device('cpu'))
                    self.assertEqual(best['history'], [])
                    self.assertEqual(report['best_selection']['kind'], 'within_epoch')
                    self.assertEqual(best['training_state']['next_position'], 1)
                    self.assertEqual(len(json.loads((root / 'validation_history.json').read_text())), 2)
                    uninterrupted = torch_load(root / 'last.pt', torch.device('cpu'))
                    resumed_model = ToyEnergy()
                    resumed_model.load_state_dict(best['model_state'])
                    args.resume = root / 'best.pt'
                    evaluations = iter((2.5,))
                    with patch('bindenergy.apps.tcr_pmhc.train.TCRpMHCPairDataset', Dataset), \
                         patch('bindenergy.apps.tcr_pmhc.train.EmbeddingStore', Embeddings), \
                         patch('bindenergy.apps.tcr_pmhc.train.make_model_batch', return_value=(None, None)), \
                         patch('bindenergy.apps.tcr_pmhc.train.evaluate_loss', side_effect=controlled_validation):
                        train_model(args, resumed_model, architecture, best, torch.device('cpu'))
                    resumed = torch_load(root / 'last.pt', torch.device('cpu'))
                    for name, value in uninterrupted['model_state'].items():
                        self.assertTrue(torch.equal(value, resumed['model_state'][name]), name)

    def test_energy_head_guard_detects_missing_and_nonfinite_gradients(self):
        model = torch.nn.Module()
        model.W_o = torch.nn.Linear(2, 2)
        model.U_o = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(RuntimeError, 'Missing mixed gradient'):
            check_head_gradients(model)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        self.assertEqual(len(check_head_gradients(model)), 4)
        model.U_o.bias.grad[0] = float('nan')
        with self.assertRaises(FloatingPointError):
            check_head_gradients(model)

    def test_subset_preserves_pair_data_and_excludes_unused_chains(self):
        script = Path(__file__).resolve().parents[2] / 'scripts' / 'pack_reference_pilot.py'
        spec = importlib.util.spec_from_file_location('pack_reference_pilot_for_test', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.sqlite'
            connection = initialize_dataset_database(source, {'format_version': 1, 'patch_size': 50})
            for i in range(4):
                sid = f'S{i}'
                connection.execute('INSERT INTO structures VALUES (?,?,?,?)', (sid, 'train', 'source.pdb', 1))
                for chain in ('A', 'M', 'unused'):
                    connection.execute('INSERT INTO chains VALUES (?,?,?,?,?,?)', (sid, chain, 'AA', 'hash', 2, b'coordinates'))
                connection.execute('INSERT INTO pairs(structure_id,split,binder_chain,target_chain,binder_indices,binder_count,target_indices,target_count,contact_count,min_distance,source_pdb) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (sid, 'train', 'A', 'M', b'indices', 2, b'indices', 2, 10, 2., 'source.pdb'))
            connection.commit()
            connection.close()
            target = root / 'subset.sqlite'
            manifest = module.export_subset(source, target, 2, 7, 'train')
            with closing(sqlite3.connect(target)) as output:
                self.assertEqual(output.execute('SELECT COUNT(*) FROM pairs').fetchone()[0], 2)
                self.assertEqual(output.execute('SELECT COUNT(*) FROM chains').fetchone()[0], 4)
                self.assertEqual(output.execute('SELECT DISTINCT coords FROM chains').fetchone()[0], b'coordinates')
                self.assertEqual(output.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(len(manifest['selected_source_pair_ids']), 2)
            with self.assertRaises(FileExistsError):
                module.export_subset(source, target, 2, 7, 'train')


if __name__ == '__main__':
    unittest.main()
